//! Laser-supported detonation: deposition, the plasma column, the driver
//! (M6c, step 4).
//!
//! This is the layer where the beam and the gas dynamics finally couple.
//! [`euler1d`](crate::euler1d) knows nothing about lasers and
//! [`plasmaprops`](crate::plasmaprops) knows nothing about hydrodynamics; both
//! of those are deliberate (`docs/M6C_SPEC.md`, gate decision 4), and this
//! module is what joins them.
//!
//! # Geometry and the sign of the march
//!
//! The column's `x` axis is the beam axis. The laser sits beyond `x_min`, so
//! **the beam travels in `+x`**; the absorption wave runs *back up the beam*
//! toward the laser, so **the front travels in `−x`**. Everything upstream of
//! the front (`x < x_front`) is cold, transparent air, which is why the front
//! sees the full incident intensity `S` — exactly the quantity Raizer's closed
//! form is written in.
//!
//! The spec writes the beam attenuation as `dI/dx = +α_pl·I`. That sign is a
//! slip: marching in the direction of travel, an *absorbing* medium gives
//! `dI/dx = −α_pl·I`, and the `+` would amplify the beam. The physics the spec
//! describes everywhere else — Beer–Lambert decay, energy conserved between the
//! two halves — is the `−` version, which is what is implemented here and what
//! G5 closes. `docs/M6C_SPEC.md` is amended to match.
//!
//! # Deposition is discretely conservative, on purpose
//!
//! Rather than sample `α·I` at the cell centre, each cell takes the intensity
//! it actually removes from the beam:
//!
//! ```text
//! I_{k+1} = I_k · exp(−α_k·dx)
//! q_k     = (I_k − I_{k+1}) / dx          (W/m³)
//! ```
//!
//! so `Σ q_k·dx` is *identically* the absorbed intensity `I_in − I_out`, with no
//! quadrature error. That is what lets G5 close to round-off instead of to a
//! discretization tolerance, and for small `α·dx` it reduces to `α·I` as it must.
//!
//! # The coupling cadence
//!
//! Strang, matching the propagator's own splitting discipline and for the same
//! reason — it is what keeps the coupled scheme 2nd order:
//!
//! ```text
//! source(dt/2) → hydro(dt) → recompute α, I → source(dt/2)
//! ```
//!
//! [`Euler1d::add_energy`](crate::euler1d::Euler1d::add_energy) is the source
//! half; [`Euler1d::step`](crate::euler1d::Euler1d::step) is the hydro half.
//!
//! # Two absorption models, for two different jobs
//!
//! - [`Absorption::GreyThreshold`] — the **verification** model: a fixed `α`
//!   wherever the gas is hot enough to be plasma, zero elsewhere. No table, no
//!   interpolation, nothing that can drift. This is what G3 and G5 run on, and
//!   it is the model the closed form assumes.
//! - [`Absorption::InverseBremsstrahlung`] — the **production** model:
//!   free-free absorption with `n_e` from the frozen equilibrium table.
//!
//! # What this module does not do
//!
//! It does not run the 3-D propagator. [`LsdColumn`] is the 1-D coupled driver,
//! which is all G3/G5 need and all the closed forms are written for;
//! [`PlasmaColumn`] exposes the resulting `α(x)` profile to the propagator as a
//! read-only [`Medium`], and the sub-cycled propagator↔hydro loop (with the
//! `propagation_every` convergence check the spec requires) lands with the CLI
//! case. Per D7 the plasma couples to the beam through **absorption only** —
//! there is no Drude index perturbation here, and `δn ≡ 0`.

use anyhow::{Result, bail};
use ndarray::Array2;

use crate::euler1d::{Boundary, Euler1d, IdealGas, Primitive};
use crate::medium::Medium;
use crate::plasmaprops::{PlasmaTable, SECOND_IONIZATION_K, Z_BAR};

/// Planck constant (J·s).
const H_PLANCK: f64 = 6.626_070_15e-34;
/// Boltzmann constant (J/K).
const K_B: f64 = 1.380_649e-23;
/// Speed of light in vacuum (m/s).
const C_LIGHT: f64 = 299_792_458.0;

/// Free-free absorption prefactor in SI.
///
/// Rybicki & Lightman (1979) Eq. 5.18b gives the thermal free-free coefficient
/// as `3.7×10⁸ · T^(−1/2) Z² n_e n_i ν^(−3) (1 − e^(−hν/kT)) ḡ` in CGS — `α` in
/// 1/cm and number densities in 1/cm³. Converting to SI multiplies by 10² for
/// `cm⁻¹ → m⁻¹` and by 10⁻¹² for the two number densities, i.e. by 10⁻¹⁰.
const C_FREE_FREE: f64 = 3.7e-2;

/// Minimum cells across the absorption length `1/α` for the deposition to be
/// resolved rather than dumped into a cell.
const MIN_CELLS_PER_ABSORPTION_LENGTH: f64 = 4.0;

/// Largest fraction of the domain the absorption length may occupy before the
/// deposition is volumetric and the detonation analogy fails — that is the LSC
/// regime, which is out of scope.
const MAX_ABSORPTION_LENGTH_FRACTION: f64 = 0.25;

/// Smallest fraction of the incident beam that must survive the cold gas and
/// actually reach the front.
const MIN_FRACTION_REACHING_FRONT: f64 = 0.9;

/// Pressure ratio across the front below which "strong detonation" — the
/// ordering `p₁ ≫ p₀` the closed form assumes — is not a fair description.
const MIN_STRONG_DETONATION_PRESSURE_RATIO: f64 = 10.0;

/// Raizer's closed form for the LSD front velocity (m/s):
/// `D = [2(γ² − 1)·S/ρ₀]^(1/3)`.
///
/// `s` is the absorbed intensity at the front (W/m²) and `rho_0` the undisturbed
/// density (kg/m³).
///
/// **This lives here, next to the model, and not in
/// [`validate`](crate::validate) — deliberately.** `validate` holds *independent*
/// reference solutions; this expression is not one. It is the Chapman–Jouguet
/// construction with the chemical heat release replaced by `q = S/(ρ₀D)`, which
/// is the same construction the deposition model above is built from. Checking
/// the solver against it is **verification** — that HLLC, the Strang-split
/// source, and the EOS together reproduce a nontrivial self-similar solution —
/// and never validation. Filing it under `validate` would quietly assert
/// otherwise, which is precisely the M6a "D5" trap (`docs/M6C_SPEC.md`,
/// § "The circularity, stated up front"). The physics gate is G4, the
/// parameter-free exponent.
pub fn raizer_lsd_velocity(gas: &IdealGas, s: f64, rho_0: f64) -> f64 {
    (2.0 * (gas.gamma * gas.gamma - 1.0) * s / rho_0).cbrt()
}

/// What to do when the plasma runs hotter than the table's singly-ionized range.
///
/// Above [`SECOND_IONIZATION_K`] the frozen table understates `n_e`, and so
/// understates the free-free absorption that the front runs on — see the
/// limitation note in [`plasmaprops`](crate::plasmaprops). Neither answer here
/// is silent: one stops, the other records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IonizationCeiling {
    /// Bail, naming the temperature. The default, and the right answer when a
    /// number is going to be quoted.
    Refuse,
    /// Continue, and set [`LsdColumn::used_singly_ionized_approximation`]. For
    /// runs that knowingly cross the ceiling — the CJ state behind a strong LSD
    /// front sits at 1.5–2.5×10⁴ K, so a demonstration run legitimately does.
    Flag,
}

/// How the gas absorbs the beam.
pub enum Absorption {
    /// Verification model: a grey plasma. `alpha` (1/m) wherever the specific
    /// internal energy exceeds `e_ignite` (J/kg), zero below it.
    ///
    /// Crude, and that is the point: it introduces nothing that can drift
    /// between grid refinements, so the residual G3 measures is the solver's
    /// and not the property table's. Keying on internal energy rather than
    /// temperature keeps it well defined under the constant-`γ` EOS, which has
    /// no gas constant to convert with.
    ///
    /// The threshold does **not** set the wave speed. A Chapman–Jouguet
    /// velocity depends on the *total* heat release, not on where or how fast
    /// it is released; `e_ignite` only has to sit far enough above ambient that
    /// undisturbed air is transparent and far enough below the shocked state
    /// that the front stays lit.
    GreyThreshold {
        /// Absorption coefficient in the plasma (1/m).
        alpha: f64,
        /// Specific internal energy above which the gas absorbs (J/kg).
        e_ignite: f64,
    },
    /// Production model: thermal inverse bremsstrahlung, with `n_e` from the
    /// frozen equilibrium table.
    ///
    /// ```text
    /// α_IB = C · Z̄² n_e n_i T^(−1/2) ν^(−3) (1 − e^(−hν/kT)) · ḡ
    /// ```
    ///
    /// with `n_i = n_e` by quasi-neutrality and `Z̄ ≡ 1` (both structural — see
    /// [`plasmaprops`](crate::plasmaprops)). The Gaunt factor `ḡ` is a
    /// dimensionless multiplier of order unity; a proper Gaunt table is out of
    /// scope, and it does not need to be in scope, because `ḡ` enters M6c's
    /// physics gate as a *coefficient* and no coefficient can shift the `1/3`
    /// exponent G4 measures.
    ///
    /// Note one honest inconsistency: the hydro carries a constant-`γ` ideal
    /// gas while the ionization comes from the equilibrium table, so `T` here is
    /// the table's inversion of the hydro's `(ρ, p)` rather than a
    /// self-consistent EOS. Closing that loop is the spec's "production mode"
    /// EOS, which is a separate change.
    InverseBremsstrahlung {
        /// Vacuum wavelength (m).
        wavelength: f64,
        /// Gaunt factor `ḡ` (dimensionless, order 1).
        gaunt: f64,
        /// The frozen equilibrium property table.
        table: PlasmaTable,
        /// What to do above [`SECOND_IONIZATION_K`].
        ceiling: IonizationCeiling,
    },
}

/// One cell's absorption, plus the temperature that produced it when a model
/// has one to report.
#[derive(Debug, Clone, Copy)]
struct AbsorptionSample {
    alpha: f64,
    temperature: Option<f64>,
}

impl Absorption {
    /// Absorption coefficient (1/m) for the state `w` under the EOS `gas`.
    ///
    /// Public so a caller can ask what a closure it is *not* running would say —
    /// the demonstration run evaluates the inverse-bremsstrahlung closure at its
    /// own measured post-front state to report the grid that closure would
    /// demand, rather than asserting a number for it.
    pub fn coefficient(&self, gas: &IdealGas, w: Primitive) -> Result<f64> {
        Ok(self.sample(gas, w)?.alpha)
    }

    /// Absorption coefficient (1/m) for the state `w` under the EOS `gas`.
    fn sample(&self, gas: &IdealGas, w: Primitive) -> Result<AbsorptionSample> {
        match self {
            Self::GreyThreshold { alpha, e_ignite } => {
                let e = gas.specific_internal_energy(w.rho, w.p);
                Ok(AbsorptionSample {
                    alpha: if e > *e_ignite { *alpha } else { 0.0 },
                    temperature: None,
                })
            }
            Self::InverseBremsstrahlung {
                wavelength,
                gaunt,
                table,
                ceiling,
            } => {
                let t = table.temperature(w.rho, w.p)?;
                if *ceiling == IonizationCeiling::Refuse
                    && PlasmaTable::is_singly_ionized_approximation(t)
                {
                    bail!(
                        "lsd: plasma reached {t:.0} K, above the table's singly-ionized \
                         ceiling of {SECOND_IONIZATION_K:.0} K, where n_e (and so α_IB) \
                         is understated; use IonizationCeiling::Flag to proceed knowingly"
                    );
                }
                let props = table.at(t, w.p)?;
                let nu = C_LIGHT / wavelength;
                let x = H_PLANCK * nu / (K_B * t);
                let alpha = C_FREE_FREE * gaunt * Z_BAR * Z_BAR * props.n_e * props.n_i()
                    / (t.sqrt() * nu * nu * nu)
                    * (-(-x).exp_m1());
                Ok(AbsorptionSample {
                    alpha,
                    temperature: Some(t),
                })
            }
        }
    }
}

/// Where and how hard to light the initial spark.
///
/// The spec's open question 4 offers two ignition paths: seed a hot spot
/// (clean — decouples the velocity gates from M6a) or trigger from
/// `AirBreakdown` (physical — but inherits M6a's explicitly ungated absolute
/// threshold). This is the seeded one, which is what G3 and G4 use so their
/// residuals measure gas dynamics and not M6a's level.
#[derive(Debug, Clone, Copy)]
pub struct SeededIgnition {
    /// Centre of the hot spot (m).
    pub centre: f64,
    /// Full width of the hot spot (m).
    pub width: f64,
    /// Pressure inside it (Pa). Density is left at ambient, so this is a pure
    /// energy deposit — the numerical analogue of a spark.
    pub pressure: f64,
}

/// A 1-D laser-supported detonation: gas dynamics, beam attenuation, and the
/// deposition that couples them.
pub struct LsdColumn {
    hydro: Euler1d,
    absorption: Absorption,
    incident: f64,
    ambient: Primitive,
    deposited: f64,
    initial_energy: f64,
    hottest: f64,
    above_ceiling: bool,
}

impl LsdColumn {
    /// Build a column of `n_cells` over `[0, length]` filled with `ambient`,
    /// with a seeded hot spot, driven by an incident intensity `incident`
    /// (W/m²) entering at `x = 0`.
    pub fn seeded(
        gas: IdealGas,
        n_cells: usize,
        length: f64,
        ambient: Primitive,
        ignition: SeededIgnition,
        absorption: Absorption,
        incident: f64,
    ) -> Result<Self> {
        if !(incident > 0.0 && incident.is_finite()) {
            bail!("lsd: incident intensity must be positive and finite, got {incident}");
        }
        if !(length > 0.0 && length.is_finite()) {
            bail!("lsd: domain length must be positive and finite, got {length}");
        }
        if !(ignition.width > 0.0 && ignition.width.is_finite()) || ignition.pressure <= ambient.p {
            bail!(
                "lsd: the seed must be a hot spot — width {} m, pressure {} Pa vs ambient {} Pa",
                ignition.width,
                ignition.pressure,
                ambient.p
            );
        }
        let half = 0.5 * ignition.width;
        let hydro = Euler1d::from_fn(
            gas,
            Boundary::Transmissive,
            0.0,
            length / n_cells as f64,
            n_cells,
            |x| {
                if (x - ignition.centre).abs() <= half {
                    Primitive {
                        p: ignition.pressure,
                        ..ambient
                    }
                } else {
                    ambient
                }
            },
        )?;
        let initial_energy = hydro.total_energy();
        Ok(Self {
            hydro,
            absorption,
            incident,
            ambient,
            deposited: 0.0,
            initial_energy,
            hottest: f64::NEG_INFINITY,
            above_ceiling: false,
        })
    }

    /// The gas dynamics, read-only.
    pub fn hydro(&self) -> &Euler1d {
        &self.hydro
    }

    /// Incident intensity at `x = 0` (W/m²).
    pub fn incident_intensity(&self) -> f64 {
        self.incident
    }

    /// Set the incident intensity (W/m²) — the seam a propagator-driven outer
    /// loop writes through.
    pub fn set_incident_intensity(&mut self, incident: f64) -> Result<()> {
        if !(incident > 0.0 && incident.is_finite()) {
            bail!("lsd: incident intensity must be positive and finite, got {incident}");
        }
        self.incident = incident;
        Ok(())
    }

    /// Absorption coefficient per cell (1/m).
    pub fn alpha_profile(&mut self) -> Result<Vec<f64>> {
        let gas = self.hydro.gas();
        let mut alpha = Vec::with_capacity(self.hydro.len());
        for w in self.hydro.primitives() {
            let s = self.absorption.sample(&gas, w)?;
            if let Some(t) = s.temperature {
                self.hottest = self.hottest.max(t);
                self.above_ceiling |= PlasmaTable::is_singly_ionized_approximation(t);
            }
            alpha.push(s.alpha);
        }
        Ok(alpha)
    }

    /// Beam intensity at each cell's **upstream face** (W/m²), plus the
    /// intensity leaving the far end. Length `n + 1`; element 0 is the incident
    /// intensity by construction.
    pub fn intensity_profile(&mut self) -> Result<Vec<f64>> {
        let alpha = self.alpha_profile()?;
        let dx = self.hydro.dx();
        let mut i = self.incident;
        let mut profile = Vec::with_capacity(alpha.len() + 1);
        profile.push(i);
        for a in alpha {
            i *= (-a * dx).exp();
            profile.push(i);
        }
        Ok(profile)
    }

    /// Absorbed power density per cell (W/m³).
    ///
    /// `q_k = (I_k − I_{k+1})/dx`, so the sum over the domain times `dx` is
    /// exactly the intensity the beam lost — see the module note.
    pub fn deposition(&mut self) -> Result<Vec<f64>> {
        let dx = self.hydro.dx();
        let i = self.intensity_profile()?;
        Ok(i.windows(2).map(|w| (w[0] - w[1]) / dx).collect())
    }

    /// Stable step for the **coupled** system (s).
    ///
    /// [`Euler1d::stable_dt`] sizes the step from the current wave speeds, but
    /// the Strang sandwich deposits energy *before* the flux update runs: the
    /// leading half-step raises `p`, and so raises `c`, and so shrinks the true
    /// CFL limit below what the pre-deposition state advertises. Handing the
    /// hydro that larger step is not a rounding matter — its CFL guard refuses
    /// it outright, which is the guard working correctly.
    ///
    /// So predict the post-deposition pressure. A source `q` acting for `dt/2`
    /// adds `(γ−1)·q·dt/2` to the pressure, which gives `dt` on both sides;
    /// iterating is a contraction (a larger `dt` gives a larger `c` gives a
    /// smaller `dt`), and a handful of passes settles it far below the 1e-12
    /// slack the guard allows.
    pub fn stable_dt(&mut self) -> Result<f64> {
        let q = self.deposition()?;
        let gas = self.hydro.gas();
        let w = self.hydro.primitives();
        let (dx, cfl) = (self.hydro.dx(), self.hydro.cfl());
        let mut dt = self.hydro.stable_dt()?;
        for _ in 0..8 {
            let speed = w
                .iter()
                .zip(&q)
                .map(|(c, &qi)| {
                    let p = c.p + (gas.gamma - 1.0) * qi * 0.5 * dt;
                    c.u.abs() + gas.sound_speed(c.rho, p.max(c.p))
                })
                .fold(0.0, f64::max);
            let next = cfl * dx / speed;
            let converged = (next - dt).abs() <= 1e-10 * dt;
            dt = next;
            if converged {
                break;
            }
        }
        if !(dt > 0.0 && dt.is_finite()) {
            bail!(
                "lsd: no stable coupled step at t = {:.4e} s (got dt = {dt})",
                self.hydro.time()
            );
        }
        // The prediction is a linearisation, so shave a hair off rather than
        // sit exactly on the limit.
        Ok(dt * (1.0 - 1e-9))
    }

    /// Advance one coupled step of `dt` with Strang splitting.
    ///
    /// `dt` must satisfy the coupled CFL bound — see
    /// [`stable_dt`](Self::stable_dt), which is what
    /// [`advance_to`](Self::advance_to) sizes with.
    pub fn advance(&mut self, dt: f64) -> Result<()> {
        self.deposit(0.5 * dt)?;
        self.hydro.step(dt)?;
        self.deposit(0.5 * dt)?;
        Ok(())
    }

    /// One source half-step: deposit for `dt` and bank the energy.
    fn deposit(&mut self, dt: f64) -> Result<()> {
        let q = self.deposition()?;
        let dx = self.hydro.dx();
        self.hydro.add_energy(dt, |i, _| q[i])?;
        self.deposited += dt * dx * q.iter().sum::<f64>();
        Ok(())
    }

    /// Run to `t_end` (s), sizing each step from the current wave speeds.
    pub fn advance_to(&mut self, t_end: f64) -> Result<()> {
        while self.hydro.time() < t_end {
            let dt = self.stable_dt()?.min(t_end - self.hydro.time());
            self.advance(dt)?;
        }
        Ok(())
    }

    /// Position of the front (m): the leading — that is, laser-side — edge of
    /// the pressure rise, at the half-maximum level and linearly interpolated
    /// between cells for sub-cell resolution.
    ///
    /// `None` before a front exists, which here means the peak pressure has not
    /// yet reached twice ambient.
    pub fn front_position(&self) -> Option<f64> {
        let w = self.hydro.primitives();
        let p_max = w.iter().map(|c| c.p).fold(f64::NEG_INFINITY, f64::max);
        if p_max < 2.0 * self.ambient.p {
            return None;
        }
        let level = self.ambient.p + 0.5 * (p_max - self.ambient.p);
        let i = w.iter().position(|c| c.p >= level)?;
        if i == 0 {
            // The front has run off the laser-side boundary; there is nothing
            // left to interpolate against.
            return Some(self.hydro.x_centre(0));
        }
        let f = (level - w[i - 1].p) / (w[i].p - w[i - 1].p);
        Some(self.hydro.x_centre(i - 1) + f * self.hydro.dx())
    }

    /// Advance by `span` (s) and return the mean front speed over it (m/s),
    /// **positive toward the laser** — the sign convention `D` is quoted in.
    pub fn measure_front_speed(&mut self, span: f64) -> Result<f64> {
        let Some(x0) = self.front_position() else {
            bail!(
                "lsd: no front to track at t = {:.4e} s (peak pressure has not \
                 reached twice ambient)",
                self.hydro.time()
            );
        };
        let t0 = self.hydro.time();
        self.advance_to(t0 + span)?;
        let Some(x1) = self.front_position() else {
            bail!("lsd: the front vanished during the {span:.4e} s measurement window");
        };
        Ok((x0 - x1) / (self.hydro.time() - t0))
    }

    /// Total optical depth of the column, `τ = Σ α_k·dx` (dimensionless).
    pub fn optical_depth(&mut self) -> Result<f64> {
        let dx = self.hydro.dx();
        Ok(self.alpha_profile()?.iter().sum::<f64>() * dx)
    }

    /// Fraction of the beam that survives the whole column, `exp(−τ)`.
    ///
    /// This is the **shielding** the plasma imposes on everything downstream of
    /// it, and it is the one number the D7 coupling ultimately delivers to a
    /// propagator: the column is transversely uniform, so the transmitted
    /// fraction is `exp(−τ)` exactly, with no dependence on the beam profile or
    /// the transverse grid. Marching a field through [`PlasmaColumn`] reproduces
    /// it rather than discovering it — which is why the gate on that path is a
    /// Beer–Lambert closed-form check (the M2 precedent) and not a new claim.
    pub fn transmitted_fraction(&mut self) -> Result<f64> {
        Ok((-self.optical_depth()?).exp())
    }

    /// Cumulative laser energy absorbed by the column (J/m²).
    pub fn deposited_energy(&self) -> f64 {
        self.deposited
    }

    /// Relative closure of the energy budget (G5): absorbed laser energy versus
    /// the domain's energy gain.
    ///
    /// The boundary flux term is *verified* rather than accounted: with
    /// transmissive ends and undisturbed ambient gas at both of them, the energy
    /// flux `(E + p)u` is identically zero there, so the whole budget is
    /// `ΔE_domain = E_absorbed`. [`boundaries_undisturbed`](Self::boundaries_undisturbed)
    /// is what checks that premise instead of assuming it — call it alongside
    /// this, or the number is meaningless.
    pub fn energy_residual(&self) -> f64 {
        let gained = self.hydro.total_energy() - self.initial_energy;
        (gained - self.deposited).abs() / self.deposited.abs().max(f64::MIN_POSITIVE)
    }

    /// Whether both end cells are still at their ambient state, so no energy has
    /// left the domain. The premise [`energy_residual`](Self::energy_residual)
    /// rests on.
    pub fn boundaries_undisturbed(&self) -> bool {
        let w = self.hydro.primitives();
        let quiet = |c: &Primitive| {
            (c.p - self.ambient.p).abs() <= 1e-9 * self.ambient.p
                && (c.rho - self.ambient.rho).abs() <= 1e-9 * self.ambient.rho
                && c.u == 0.0
        };
        quiet(&w[0]) && quiet(&w[w.len() - 1])
    }

    /// Hottest temperature the absorption model reported (K), or `None` for a
    /// model that has no temperature — [`Absorption::GreyThreshold`] keys on
    /// internal energy and never forms one.
    pub fn hottest_temperature(&self) -> Option<f64> {
        self.hottest.is_finite().then_some(self.hottest)
    }

    /// Whether the run has queried the property table above
    /// [`SECOND_IONIZATION_K`], where it is a singly-ionized approximation
    /// rather than a prediction. Only ever true under
    /// [`IonizationCeiling::Flag`]; [`IonizationCeiling::Refuse`] stops first.
    pub fn used_singly_ionized_approximation(&self) -> bool {
        self.above_ceiling
    }

    /// Fraction of the incident beam that survives the gas upstream of the
    /// front and actually reaches it.
    ///
    /// "The front" here is the leading edge of the **strongly absorbing**
    /// region — the first cell whose `α` reaches half the column's maximum —
    /// and not [`front_position`](Self::front_position)'s pressure half-maximum.
    /// The two are deliberately different, and using the pressure front here
    /// would be a category error: in a detonation the reaction zone *leads* the
    /// pressure peak, so beam energy absorbed between the two is the front
    /// doing its job, not the front starving. Keying on `α` measures what the
    /// check is named for — extinction in the gas *ahead* of the absorbing
    /// zone.
    ///
    /// Note this is ~1 by construction under
    /// [`Absorption::GreyThreshold`], where cold gas is exactly transparent.
    /// That is the honest answer for that model, not a strong check; the
    /// quantity earns its keep under
    /// [`Absorption::InverseBremsstrahlung`], where a long weakly ionized
    /// precursor genuinely can eat the beam before it arrives.
    pub fn fraction_reaching_front(&mut self) -> Result<f64> {
        let alpha = self.alpha_profile()?;
        let alpha_max = alpha.iter().copied().fold(0.0, f64::max);
        if alpha_max <= 0.0 {
            return Ok(1.0);
        }
        let Some(k) = alpha.iter().position(|&a| a >= 0.5 * alpha_max) else {
            return Ok(1.0);
        };
        let profile = self.intensity_profile()?;
        Ok(profile[k] / self.incident)
    }

    /// Assert the run is in the LSD regime the model is written for, in the M4
    /// Péclet spirit: refuse, don't mis-model. Each failure names the number
    /// that is wrong and what it would take to fix it.
    ///
    /// Checks the spec's three validity conditions — the absorption layer is
    /// resolved and thin against the domain, the beam is not extinguished
    /// before the front, and the front is a strong detonation.
    pub fn check_regime(&mut self) -> Result<()> {
        let dx = self.hydro.dx();
        let length = dx * self.hydro.len() as f64;
        let alpha_max = self.alpha_profile()?.into_iter().fold(0.0, f64::max);
        if alpha_max <= 0.0 {
            bail!(
                "lsd: nothing absorbs at t = {:.4e} s — there is no front, and the \
                 detonation model has nothing to describe",
                self.hydro.time()
            );
        }
        let absorption_length = 1.0 / alpha_max;
        let cells = absorption_length / dx;
        if cells < MIN_CELLS_PER_ABSORPTION_LENGTH {
            bail!(
                "lsd: the absorption length 1/α = {absorption_length:.4e} m spans only \
                 {cells:.2} cells (dx = {dx:.4e} m); the deposition is unresolved. Refine \
                 to dx ≤ {:.4e} m, or lower α.",
                absorption_length / MIN_CELLS_PER_ABSORPTION_LENGTH
            );
        }
        if absorption_length > MAX_ABSORPTION_LENGTH_FRACTION * length {
            bail!(
                "lsd: the absorption length 1/α = {absorption_length:.4e} m is {:.0}% of the \
                 {length:.4e} m domain; the deposition is volumetric rather than a front, \
                 which is the LSC regime and out of scope",
                100.0 * absorption_length / length
            );
        }
        let reaching = self.fraction_reaching_front()?;
        if reaching < MIN_FRACTION_REACHING_FRONT {
            bail!(
                "lsd: only {:.1}% of the beam reaches the front — it is being extinguished \
                 upstream, which the 1-D model handles badly",
                100.0 * reaching
            );
        }
        let p_max = self
            .hydro
            .primitives()
            .iter()
            .map(|c| c.p)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio = p_max / self.ambient.p;
        if ratio < MIN_STRONG_DETONATION_PRESSURE_RATIO {
            bail!(
                "lsd: peak pressure is only {ratio:.2}× ambient, below the {:.0}× that makes \
                 `p₁ ≫ p₀` a fair description; the strong-detonation closed form does not \
                 apply here",
                MIN_STRONG_DETONATION_PRESSURE_RATIO
            );
        }
        Ok(())
    }
}

/// The plasma column seen by the propagator: **pure Beer–Lambert absorption**,
/// no index perturbation.
///
/// This is decision D7 made concrete. The column supplies `α(z)` per slab and
/// `δn ≡ 0`, which is what keeps M6c clear of the near-critical failure that
/// broke M6b as specified: a paraxial split-step envelope cannot carry the
/// Drude index of a plasma at these densities, so M6c does not ask it to.
///
/// Read-only by construction — it is built from a snapshot of the hydro state
/// and never writes back. The outer loop rebuilds it after each hydro step.
pub struct PlasmaColumn {
    n: usize,
    alpha: Vec<f64>,
}

impl PlasmaColumn {
    /// A column of `alpha` (1/m), one entry per z-slab, on an `n × n`
    /// transverse grid.
    pub fn new(n: usize, alpha: Vec<f64>) -> Result<Self> {
        if n == 0 || alpha.is_empty() {
            bail!(
                "lsd: empty plasma column ({n}×{n} grid, {} slabs)",
                alpha.len()
            );
        }
        if let Some((i, a)) = alpha
            .iter()
            .enumerate()
            .find(|(_, a)| !(a.is_finite() && **a >= 0.0))
        {
            bail!("lsd: slab {i} has non-physical extinction {a} 1/m");
        }
        Ok(Self { n, alpha })
    }

    /// Snapshot the current absorption profile of `column`.
    pub fn from_column(column: &mut LsdColumn, n: usize) -> Result<Self> {
        Self::new(n, column.alpha_profile()?)
    }

    /// Snapshot `column` onto `n_slabs` coarser slabs, returning the column and
    /// the slab thickness `dz` (m) the propagator **must** be given for it.
    ///
    /// The hydro resolves the absorption layer with thousands of 10 µm cells;
    /// handing the propagator one FFT slab per hydro cell would cost thousands
    /// of transforms to reproduce a number the column already knows. Binning is
    /// exact rather than approximate here: a slab's `α` is the arithmetic mean
    /// over its cells, so `α_slab·dz = Σ α_k·dx` and the optical depth — the
    /// only thing a transversely uniform absorber can affect — is preserved to
    /// round-off. (Gated by `plasma_column_resampling_preserves_optical_depth`.)
    ///
    /// `n_cells` must be an exact multiple of `n_slabs`; a ragged binning would
    /// give slabs of differing thickness, which the propagator's single `dz`
    /// cannot express, and silently mis-weighting them is precisely the kind of
    /// quiet error this returns an `Err` for instead.
    pub fn from_column_resampled(
        column: &mut LsdColumn,
        n: usize,
        n_slabs: usize,
    ) -> Result<(Self, f64)> {
        let alpha = column.alpha_profile()?;
        let dx = column.hydro().dx();
        if n_slabs == 0 || n_slabs > alpha.len() || !alpha.len().is_multiple_of(n_slabs) {
            bail!(
                "lsd: cannot bin {} hydro cells onto {n_slabs} slabs — need a divisor \
                 of {} in 1..={}",
                alpha.len(),
                alpha.len(),
                alpha.len()
            );
        }
        let per = alpha.len() / n_slabs;
        let binned: Vec<f64> = alpha
            .chunks_exact(per)
            .map(|c| c.iter().sum::<f64>() / per as f64)
            .collect();
        Ok((Self::new(n, binned)?, per as f64 * dx))
    }
}

impl Medium for PlasmaColumn {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        // Not an unreachable stub: the medium is linear, so the propagator does
        // call this. D7's "absorption only" *is* δn ≡ 0.
        Array2::zeros((self.n, self.n))
    }

    fn extinction(&self, z_slab: usize) -> Option<Array2<f64>> {
        let a = *self.alpha.get(z_slab)?;
        (a > 0.0).then(|| Array2::from_elem((self.n, self.n), a))
    }

    fn slab_count(&self) -> Option<usize> {
        Some(self.alpha.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMBIENT: Primitive = Primitive {
        rho: 1.225,
        u: 0.0,
        p: 101_325.0,
    };

    fn grey() -> Absorption {
        Absorption::GreyThreshold {
            alpha: 1e4,
            e_ignite: 2e6,
        }
    }

    fn column(n: usize) -> LsdColumn {
        LsdColumn::seeded(
            IdealGas::AIR,
            n,
            1e-2,
            AMBIENT,
            SeededIgnition {
                centre: 7e-3,
                width: 5e-4,
                pressure: 3e7,
            },
            grey(),
            1e11,
        )
        .unwrap()
    }

    #[test]
    fn raizer_velocity_matches_the_worked_example() {
        // docs/M6C_SPEC.md § "The CJ state": γ = 1.4, S = 1e11 W/m²,
        // ρ₀ = 1.225 kg/m³ → 5.4 km/s.
        let d = raizer_lsd_velocity(&IdealGas::AIR, 1e11, 1.225);
        assert!((d - 5390.0).abs() < 10.0, "D = {d} m/s");
    }

    #[test]
    fn raizer_velocity_has_the_one_third_exponents() {
        // The relation G4 gates, checked here as an algebra unit test on the
        // closed form itself — not as a claim about the solver.
        let gas = IdealGas::AIR;
        let d = raizer_lsd_velocity(&gas, 1e11, 1.225);
        let d_10s = raizer_lsd_velocity(&gas, 1e12, 1.225);
        let d_10rho = raizer_lsd_velocity(&gas, 1e11, 12.25);
        assert!((d_10s / d - 10f64.cbrt()).abs() < 1e-12);
        assert!((d_10rho / d - 0.1f64.cbrt()).abs() < 1e-12);
    }

    #[test]
    fn cold_air_is_transparent_and_hot_air_absorbs() {
        let mut c = column(256);
        let alpha = c.alpha_profile().unwrap();
        let hot: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert!(!hot.is_empty(), "the seed does not absorb");
        // Only the seed absorbs, and it sits where it was put.
        for &i in &hot {
            let x = c.hydro().x_centre(i);
            assert!((x - 7e-3).abs() <= 3e-4, "cell at x = {x} m should be cold");
        }
    }

    #[test]
    fn deposition_sums_to_the_beam_energy_removed() {
        // The discrete conservation the module note claims: Σ q·dx is exactly
        // I_in − I_out, to round-off, with no quadrature error.
        let mut c = column(256);
        let profile = c.intensity_profile().unwrap();
        let removed = profile[0] - profile[profile.len() - 1];
        let q = c.deposition().unwrap();
        let summed = q.iter().sum::<f64>() * c.hydro().dx();
        assert!(
            (summed - removed).abs() / removed < 1e-13,
            "Σq·dx = {summed:.9e} vs I_in − I_out = {removed:.9e}"
        );
    }

    #[test]
    fn the_beam_only_decays_and_starts_at_the_incident_intensity() {
        let mut c = column(256);
        let profile = c.intensity_profile().unwrap();
        assert_eq!(profile[0], c.incident_intensity());
        for w in profile.windows(2) {
            assert!(w[1] <= w[0], "the beam grew: {:?} → {:?}", w[0], w[1]);
        }
        // The seed is optically thick: α·w = 1e4 × 5e-4 = 5 optical depths, so
        // e^(−5) ≈ 7e-3 of the beam survives it. Once the wave runs, the plasma
        // it leaves behind makes that far smaller.
        assert!(profile[profile.len() - 1] / profile[0] < 1e-2);
    }

    #[test]
    fn the_front_runs_toward_the_laser() {
        let mut c = column(512);
        let x0 = c.front_position().unwrap();
        c.advance_to(2e-7).unwrap();
        let x1 = c.front_position().unwrap();
        assert!(x1 < x0, "the front moved downstream: {x0} → {x1}");
        // And at a speed of the right order — the gate on the number is G3.
        let d = (x0 - x1) / c.hydro().time();
        assert!((2e3..2e4).contains(&d), "D = {d} m/s is not LSD-like");
    }

    #[test]
    fn an_unresolved_absorption_layer_is_refused() {
        // 1/α = 100 µm against dx = 1 cm/16 = 625 µm: well under the four cells
        // the deposition needs.
        let mut c = column(16);
        let err = c.check_regime().unwrap_err().to_string();
        assert!(err.contains("unresolved"), "unexpected error: {err}");
        assert!(err.contains("Refine to"), "no remedy offered: {err}");
    }

    #[test]
    fn a_volumetric_absorber_is_refused_as_out_of_scope() {
        // 1/α = 5 mm on a 1 cm domain: no front, this is the LSC regime.
        let mut c = LsdColumn::seeded(
            IdealGas::AIR,
            512,
            1e-2,
            AMBIENT,
            SeededIgnition {
                centre: 7e-3,
                width: 5e-4,
                pressure: 3e7,
            },
            Absorption::GreyThreshold {
                alpha: 2e2,
                e_ignite: 2e6,
            },
            1e11,
        )
        .unwrap();
        let err = c.check_regime().unwrap_err().to_string();
        assert!(err.contains("LSC regime"), "unexpected error: {err}");
    }

    #[test]
    fn a_healthy_run_passes_the_regime_check() {
        let mut c = column(512);
        c.advance_to(1e-7).unwrap();
        c.check_regime().unwrap();
        assert!(c.boundaries_undisturbed());
    }

    #[test]
    fn degenerate_setups_are_refused() {
        let seed = SeededIgnition {
            centre: 7e-3,
            width: 5e-4,
            pressure: 3e7,
        };
        let build = |incident, length, seed| {
            LsdColumn::seeded(IdealGas::AIR, 64, length, AMBIENT, seed, grey(), incident)
        };
        assert!(build(-1.0, 1e-2, seed).is_err(), "negative intensity");
        assert!(build(f64::NAN, 1e-2, seed).is_err(), "NaN intensity");
        assert!(build(1e11, -1e-2, seed).is_err(), "negative length");
        assert!(
            build(
                1e11,
                1e-2,
                SeededIgnition {
                    pressure: AMBIENT.p,
                    ..seed
                }
            )
            .is_err(),
            "a seed at ambient pressure is not a spark"
        );
    }

    #[test]
    fn inverse_bremsstrahlung_is_the_right_order_for_an_lsd_front() {
        // ~10²⁴ 1/m³ at ~2×10⁴ K absorbs a 10.6 µm beam over ~1 mm. This pins
        // the unit conversion in C_FREE_FREE, which is the part of the formula
        // that is easy to get wrong by ten orders of magnitude.
        let ib = Absorption::InverseBremsstrahlung {
            wavelength: 10.6e-6,
            gaunt: 1.0,
            table: PlasmaTable::load().unwrap(),
            ceiling: IonizationCeiling::Flag,
        };
        let table = PlasmaTable::load().unwrap();
        let props = table.at(18_000.0, 1e7).unwrap();
        let w = Primitive {
            rho: props.rho,
            u: 0.0,
            p: 1e7,
        };
        let a = ib.sample(&IdealGas::AIR, w).unwrap();
        assert!(
            (1e2..1e5).contains(&a.alpha),
            "α_IB = {:.4e} 1/m is not front-like",
            a.alpha
        );
        assert!((a.temperature.unwrap() - 18_000.0).abs() < 1.0);
    }

    #[test]
    fn inverse_bremsstrahlung_scales_as_the_formula_says() {
        let table = PlasmaTable::load().unwrap();
        let make = |gaunt| Absorption::InverseBremsstrahlung {
            wavelength: 10.6e-6,
            gaunt,
            table: PlasmaTable::load().unwrap(),
            ceiling: IonizationCeiling::Flag,
        };
        let props = table.at(15_000.0, 1e6).unwrap();
        let w = Primitive {
            rho: props.rho,
            u: 0.0,
            p: 1e6,
        };
        let gas = IdealGas::AIR;
        let a1 = make(1.0).sample(&gas, w).unwrap().alpha;
        let a2 = make(2.0).sample(&gas, w).unwrap().alpha;
        assert!((a2 / a1 - 2.0).abs() < 1e-12, "ḡ must enter linearly");
        // Cold air is transparent by ~20 orders of magnitude.
        let cold = table.at(300.0, 1e5).unwrap();
        let a_cold = make(1.0)
            .sample(
                &gas,
                Primitive {
                    rho: cold.rho,
                    u: 0.0,
                    p: 1e5,
                },
            )
            .unwrap()
            .alpha;
        assert!(a_cold < 1e-20 * a1, "cold air absorbs: {a_cold:.3e} 1/m");
    }

    #[test]
    fn the_second_ionization_ceiling_is_enforced() {
        let table = PlasmaTable::load().unwrap();
        let props = table.at(25_000.0, 1e7).unwrap();
        let w = Primitive {
            rho: props.rho,
            u: 0.0,
            p: 1e7,
        };
        let gas = IdealGas::AIR;

        let refuse = Absorption::InverseBremsstrahlung {
            wavelength: 10.6e-6,
            gaunt: 1.0,
            table: PlasmaTable::load().unwrap(),
            ceiling: IonizationCeiling::Refuse,
        };
        let err = refuse.sample(&gas, w).unwrap_err().to_string();
        assert!(err.contains("singly-ionized ceiling"), "unexpected: {err}");
        assert!(err.contains("25000") || err.contains("2500"), "no T: {err}");

        let flag = Absorption::InverseBremsstrahlung {
            wavelength: 10.6e-6,
            gaunt: 1.0,
            table: PlasmaTable::load().unwrap(),
            ceiling: IonizationCeiling::Flag,
        };
        assert!(flag.sample(&gas, w).is_ok(), "Flag must proceed");
    }

    #[test]
    fn plasma_column_is_absorption_only() {
        let mut c = column(64);
        let col = PlasmaColumn::from_column(&mut c, 8).unwrap();
        assert_eq!(col.slab_count(), Some(64));
        // D7: no index perturbation, ever.
        assert!(col.index_perturbation(0).iter().all(|&v| v == 0.0));
        assert_eq!(col.index_perturbation(0).dim(), (8, 8));
        // Cold slabs are lossless; the seed is not.
        assert!(col.extinction(0).is_none(), "cold air should not absorb");
        let hot = (0..64).filter_map(|j| col.extinction(j)).count();
        assert!(hot > 0, "no absorbing slab");
        // Off the end of the column is None, not a panic.
        assert!(col.extinction(64).is_none());
    }

    #[test]
    fn plasma_column_resampling_preserves_optical_depth() {
        // The invariant that licenses coarsening at all: a transversely uniform
        // absorber acts on the beam only through Σα·dx, so binning must carry
        // it exactly, not approximately.
        let mut c = column(512);
        c.advance_to(1e-7).unwrap();
        let tau = c.optical_depth().unwrap();
        for n_slabs in [512usize, 256, 128, 64, 8] {
            let (col, dz) = PlasmaColumn::from_column_resampled(&mut c, 4, n_slabs).unwrap();
            let binned: f64 = (0..n_slabs)
                .filter_map(|j| col.extinction(j).map(|a| a[[0, 0]] * dz))
                .sum();
            assert!(
                (binned - tau).abs() <= 1e-12 * tau,
                "{n_slabs} slabs: τ = {binned:.12e} vs {tau:.12e}"
            );
        }
        // A ragged binning is refused rather than silently mis-weighted.
        assert!(PlasmaColumn::from_column_resampled(&mut c, 4, 7).is_err());
        assert!(PlasmaColumn::from_column_resampled(&mut c, 4, 0).is_err());
        assert!(PlasmaColumn::from_column_resampled(&mut c, 4, 1024).is_err());
    }

    #[test]
    fn transmitted_fraction_is_exp_minus_tau() {
        let mut c = column(256);
        let tau = c.optical_depth().unwrap();
        let t = c.transmitted_fraction().unwrap();
        assert!(tau > 0.0, "the seed should be optically thick");
        assert!((t - (-tau).exp()).abs() < 1e-15);
        // And it matches the column's own Beer–Lambert march, which is the
        // consistency that lets the demo quote one from the other.
        let profile = c.intensity_profile().unwrap();
        let marched = profile[profile.len() - 1] / profile[0];
        assert!(
            (t / marched - 1.0).abs() < 1e-12,
            "exp(−τ) = {t:.9e} vs the marched {marched:.9e}"
        );
    }

    #[test]
    fn plasma_column_refuses_nonsense() {
        assert!(PlasmaColumn::new(0, vec![1.0]).is_err(), "empty grid");
        assert!(PlasmaColumn::new(8, vec![]).is_err(), "no slabs");
        assert!(PlasmaColumn::new(8, vec![-1.0]).is_err(), "negative α");
        assert!(PlasmaColumn::new(8, vec![f64::NAN]).is_err(), "NaN α");
    }
}
