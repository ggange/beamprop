//! 0-D optical-breakdown threshold kernel (M6a).
//!
//! A point in the gas sees an intensity history `I(t)`; its electron density
//! `n_e` avalanches by inverse-bremsstrahlung cascade ionization, seeded by
//! multiphoton ionization and drained by attachment and diffusion:
//!
//! ```text
//! dn_e/dt = (ν_i(I,p) − ν_att(p) − ν_diff(p)) · n_e + S_mpi(I,p)
//! ```
//!
//! Breakdown is declared when `n_e` reaches the criterion density within the
//! pulse. The kernel is a **driver-callable pure function** (no propagator, no
//! `Medium`): the later M6a.2 (Monte-Carlo ignition clouds) and M6c (LSD
//! trigger) rungs evaluate the same [`AirBreakdown`] per point. The governing
//! model, constants, and — importantly — which checks are physics gates versus
//! integrator unit tests are pinned in `docs/M6A_SPEC.md`.
//!
//! Cascade ionization is driven by the **net** power — inverse-bremsstrahlung
//! heating minus the inelastic excitation losses the electron pays climbing to
//! `U_i`. Both scale `∝ p`, so their difference gives `I_thr(p)` a constant
//! high-pressure plateau on top of the `1/p` avalanche term.
//!
//! The mean electron energy is **not** an input: the default
//! [`CascadeModel::SelfConsistentClimb`] eliminates it by solving the climb
//! `dε/dt = heating − δ_eff·ν_m·ε` exactly, leaving `δ_eff` as the only free
//! constant. Growth is **logistic**, not exponential: ionization consumes the
//! neutrals it feeds on, so `n_e` saturates at full ionization instead of
//! running away.
//!
//! The absolute threshold *level* depends on several order-of-magnitude
//! constants (`U_i`, `D_e`, `Λ`, `n_bd`, `δ_eff`) and is deliberately **not**
//! validated; published thresholds scatter 3–10× across labs, and this model
//! sits 4.8–7.2× above T&T.
//!
//! **Where the comparison stands (paper obtained 2026-07-25).** The kernel
//! models the collisional cascade only. Thiyagarajan & Thompson's *measured*
//! curve is therefore the wrong target — it contains multiphoton ionization
//! too (they quote 88 % cascade / 12 % MPI at 760 Torr, MPI-dominant below
//! 100 Torr, and MPI-correct the data before comparing it to theory). The
//! apples-to-apples reference is their cascade closed form, Eq. 4, which at
//! 1064 nm is **flat in pressure**: the `λ⁻²` term is `1.94×10⁵` against a
//! `p²` term that never exceeds 6.9.
//!
//! Against that reference the kernel does reasonably — 4.1–5.1× high in level
//! for this model, 1.3–3.2× for [`CascadeModel::FixedMeanEnergy`], gated by
//! `tt2012_cascade_theory_reference`. Against the raw measurement it does not,
//! and cannot: no cascade-only model can produce the measured `n = 0.329` when
//! accepted cascade theory says `n ≈ 0`. That gate
//! (`tt2012_threshold_slope_matches_measurement`) is red on purpose.
//!
//! The bracket survives every correction so far: `n = 0.095` here and `0.468`
//! for the fixed-`⟨ε⟩` limit, straddling 0.329. It has now held through the
//! seed-window fix (0.356 → 0.127) and the focal-geometry fix (→ 0.095), which
//! is the main reason to trust it over either endpoint. Analysis in
//! `docs/M6A_SPEC.md`; external gates in `tests/validation.rs`.

use std::f64::consts::PI;

use anyhow::{Result, bail};

// --- Physical constants (SI; docs/M6A_SPEC.md) --------------------------------

/// Elementary charge (C).
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Electron mass (kg).
const M_E: f64 = 9.109_383_701_5e-31;
/// Speed of light (m/s).
const C_LIGHT: f64 = 299_792_458.0;
/// Vacuum permittivity (F/m).
const EPS0: f64 = 8.854_187_812_8e-12;
/// Reduced Planck constant (J·s).
const HBAR: f64 = 1.054_571_817e-34;
/// Reference pressure, 1 atm (Pa).
const P_REF: f64 = 101_325.0;

/// Largest growth exponent `β·dt` the integrator evaluates; `exp` overflows
/// above ~709. See [`AirBreakdown::advance`].
const MAX_EXPONENT: f64 = 700.0;

/// Bisection bracket for [`AirBreakdown::threshold_intensity`] (W/m²).
const I_BRACKET_LO: f64 = 1e12;
const I_BRACKET_HI: f64 = 1e22;

/// How the inelastic energy loss enters the cascade rate.
///
/// The two variants are the parameter-light limits of the same energy balance,
/// and they **bracket** the measured threshold slope: 0.551 and 0.127 against
/// T&T's 0.329. Neither reproduces it; the bracket is the claim. See
/// `docs/M6A_SPEC.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeModel {
    /// Loss evaluated at a fixed mean electron energy `⟨ε⟩`:
    /// `ν_i = max(0, heating − δ_eff·ν_m·⟨ε⟩)/U_i`.
    ///
    /// `⟨ε⟩` is a free constant, literature-bounded to ≈2–5 eV. Retained as
    /// the steeper end of the bracket (`n = 0.551`) and for the unit tests that
    /// need a cascade rate linear in intensity; not the default.
    FixedMeanEnergy,
    /// `⟨ε⟩` eliminated by solving the climb exactly.
    ///
    /// An electron's energy obeys `dε/dt = heating − δ_eff·ν_m·ε`, a linear
    /// ODE with solution `ε(t) = ε_∞(1 − e^{−t/t_r})`, `ε_∞ = heating/(δ_eff·ν_m)`
    /// and `t_r = 1/(δ_eff·ν_m)`. Ionization happens when the climb reaches
    /// `U_i`, so
    ///
    /// ```text
    /// ν_i = δ_eff·ν_m / ln(ε_∞/(ε_∞ − U_i)),    zero if ε_∞ ≤ U_i
    /// ```
    ///
    /// No free `⟨ε⟩` at all, and still exactly `∝ p` (the scaling the external
    /// `E_eff` gate confirms). **The default**, chosen for parameter parsimony
    /// — *not* for agreement: it gives `n = 0.127` against a measured 0.329,
    /// which is the flatter side of the bracket and misses by 2.6×.
    ///
    /// Its idealization is the flip side of its parsimony, and it is the likely
    /// reason it is too flat: putting every electron on the *mean* trajectory
    /// makes the threshold artificially sharp at `ε_∞ = U_i`, where the climb
    /// time diverges logarithmically. A real energy distribution has a tail
    /// that ionizes earlier, which would soften the plateau and steepen the
    /// slope back toward the data. That is the open question M6a hands forward.
    SelfConsistentClimb,
}

/// A dry-air optical-breakdown point model at one wavelength.
///
/// Construct with [`AirBreakdown::air_1064nm`] for the pinned M6a case, or
/// [`AirBreakdown::new`] to vary the geometry. All rate methods are pure
/// functions of `(intensity, pressure)`; the struct holds only fixed gas and
/// laser parameters.
#[derive(Debug, Clone, Copy)]
pub struct AirBreakdown {
    /// Laser angular frequency `ω = 2πc/λ` (rad/s).
    omega: f64,
    /// Effective ionization energy `U_i` (J).
    u_ion: f64,
    /// Electron-neutral collision-frequency slope `ν_m = k_m·p` (s⁻¹·Pa⁻¹).
    k_m: f64,
    /// Free-electron diffusion coefficient at `P_REF` (m²/s); scales as `1/p`.
    d_e_ref: f64,
    /// Fractional electron energy loss per collision, `δ_eff` (dimensionless).
    /// Literature ≈ 0.01–0.05 for air above ~1 eV; **never tuned to data**.
    delta_eff: f64,
    /// Mean electron energy `⟨ε⟩` (J), used only by
    /// [`CascadeModel::FixedMeanEnergy`]. Literature ≈ 2–5 eV.
    mean_energy: f64,
    /// Which limit of the energy balance drives the cascade.
    cascade_model: CascadeModel,
    /// Pulse-integration half-window, in FWHMs. A converged model must give a
    /// threshold independent of this; see [`AirBreakdown::peak_ne`].
    window_half_widths: f64,
    /// Diffusion length `Λ` (m), from the focal geometry — **pinned, not fit**.
    lambda_diff: f64,
    /// Dissociative-attachment rate coefficient `e + O₂ → O⁻ + O` (m³/s).
    k_att_2body: f64,
    /// Three-body attachment rate coefficient `e + O₂ + M → O₂⁻ + M` (m⁶/s).
    /// Dominant channel in air at atmospheric density; scales as `p²`.
    k_att_3body: f64,
    /// O₂ fraction of dry air by number.
    f_o2: f64,
    /// Per-neutral multiphoton ionization rate at [`Self::mpi_i_ref`] (s⁻¹).
    /// Zero disables the MPI source entirely (the default).
    mpi_rate_ref: f64,
    /// Reference intensity for the MPI rate (W/m²).
    mpi_i_ref: f64,
    /// Number of photons `K` for multiphoton ionization.
    k_photons: i32,
    /// Breakdown criterion density `n_bd` (m⁻³).
    n_bd: f64,
    /// Neutral-density-per-pressure `N/p` at reference temperature (m⁻³·Pa⁻¹),
    /// i.e. `1/(k_B·T)`.
    n_over_p: f64,
    /// Initial seed density `n_e0` (m⁻³): one electron in the focal volume.
    n_seed: f64,
}

impl AirBreakdown {
    /// General constructor. `wavelength` (m), `u_ion_ev` effective ionization
    /// energy (eV), `lambda_diff` diffusion length (m, from geometry),
    /// `temperature` (K) for the neutral density, `focal_volume` (m³) for the
    /// single-electron seed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wavelength: f64,
        u_ion_ev: f64,
        lambda_diff: f64,
        temperature: f64,
        focal_volume: f64,
    ) -> Result<Self> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("wavelength must be positive and finite, got {wavelength}");
        }
        if !(lambda_diff > 0.0 && lambda_diff.is_finite()) {
            bail!("diffusion length must be positive and finite, got {lambda_diff}");
        }
        if !(temperature > 0.0 && temperature.is_finite()) {
            bail!("temperature must be positive and finite, got {temperature}");
        }
        if !(focal_volume > 0.0 && focal_volume.is_finite()) {
            bail!("focal volume must be positive and finite, got {focal_volume}");
        }
        if !(u_ion_ev > 0.0 && u_ion_ev.is_finite()) {
            bail!("ionization energy must be positive and finite, got {u_ion_ev} eV");
        }
        const K_B: f64 = 1.380_649e-23;
        let omega = 2.0 * PI * C_LIGHT / wavelength;
        let photon_energy = HBAR * omega;
        let u_ion = u_ion_ev * E_CHARGE;
        // Photons to reach the ionization potential, ⌈U_ion/ħω⌉.
        let k_photons = (u_ion / photon_energy).ceil() as i32;
        Ok(Self {
            omega,
            u_ion,
            k_m: 3.9e7,
            d_e_ref: 2.0e-1,
            // δ_eff = 0.02, ⟨ε⟩ = 3 eV — the CENTRE of the literature ranges
            // (δ_eff ≈ 0.01–0.05 for air above ~1 eV, ⟨ε⟩ ≈ 2–5 eV), not a
            // value chosen to improve agreement. The ~12× spread in these
            // ranges is why the external slope gate is an envelope test rather
            // than a point comparison. See docs/M6A_SPEC.md.
            delta_eff: 0.02,
            mean_energy: 3.0 * E_CHARGE,
            cascade_model: CascadeModel::SelfConsistentClimb,
            window_half_widths: 2.0,
            lambda_diff,
            // Attachment from measured rate coefficients, not an s⁻¹Pa⁻¹ fudge
            // (Kossyi et al. 1992; Itikawa 2009). The three-body channel is the
            // dominant one at atmospheric density. Both are ~150x smaller than
            // the order-of-magnitude constant they replaced, which makes
            // attachment negligible against diffusion and the growth
            // requirement — see docs/M6A_SPEC.md.
            k_att_2body: 1.0e-17,
            k_att_3body: 1.0e-43,
            f_o2: 0.21,
            // MPI source off by default: the seed electron (n_e0 = 1/V_focal)
            // is the initial condition, and the avalanche multiplies it. The
            // continuous multiphoton source is the swappable term (docs Open
            // Question 2); a physical σ_K would be supplied when it is enabled.
            mpi_rate_ref: 0.0,
            mpi_i_ref: 1.0,
            k_photons,
            n_bd: 1.0e23,
            n_over_p: 1.0 / (K_B * temperature),
            n_seed: 1.0 / focal_volume,
        })
    }

    /// The pinned M6a case: 1064 nm in dry air at 288 K, with the focal
    /// geometry of Thiyagarajan & Thompson 2012 taken from **their Eq. 5**.
    ///
    /// The focus is **divergence-limited**, not diffraction-limited: a 1 cm
    /// beam with 1 mrad divergence through an `f` = 4 cm lens gives
    /// `r₀ = f·α/2 = 20 µm` and a depth of focus `l₀ = 0.414·(α/d)·f² = 66 µm`.
    /// (The Rayleigh range of a *diffraction-limited* 20 µm waist would be
    /// 1.2 mm — 36× longer — so assuming a diffraction-limited filament here
    /// is wrong in the other direction.) The diffusion length of that cylinder
    /// is their Eq. 5,
    ///
    /// ```text
    /// (1/Λ)² = (π/l₀)² + (2.405/r₀)²   →   Λ = 7.74 µm
    /// ```
    ///
    /// matching the `Λ = 8 µm` the paper states. An earlier version modelled
    /// the focus as a *sphere* with `Λ = r₀/π = 6.37 µm`, which overstated
    /// `ν_diff` by 1.48×.
    pub fn air_1064nm() -> Self {
        let r_focus = 20e-6_f64; // f·α/2, f = 4 cm, α = 1 mrad
        let l_axial = 0.414 * (1e-3 / 1e-2) * 0.04_f64.powi(2); // = 66.2 µm
        let lambda_diff = 1.0 / ((PI / l_axial).powi(2) + (2.405 / r_focus).powi(2)).sqrt();
        let focal_volume = PI * r_focus * r_focus * l_axial;
        // Unwrap: all arguments are known-good constants.
        Self::new(1064e-9, 12.06, lambda_diff, 288.0, focal_volume).unwrap()
    }

    /// Rebuild with a different inelastic-loss parameterisation, from the two
    /// physical quantities it lumps: the fractional energy loss per collision
    /// `δ_eff` and the mean electron energy `⟨ε⟩` (eV).
    ///
    /// Exists so the external gate can sweep the **literature ranges**
    /// (`δ_eff ≈ 0.01–0.05`, `⟨ε⟩ ≈ 2–5 eV`) and test that the measurement
    /// falls inside the resulting envelope. It is not a tuning knob: the
    /// default sits at the centre of those ranges and the gate never selects a
    /// value that improves agreement.
    pub fn with_inelastic_loss(mut self, delta_eff: f64, mean_energy_ev: f64) -> Self {
        self.delta_eff = delta_eff;
        self.mean_energy = mean_energy_ev * E_CHARGE;
        self
    }

    /// Enable the multiphoton source, calibrated to Thiyagarajan & Thompson's
    /// own MPI estimate (their Sec. II A): `I_B(MPI) = 4.42×10⁹ W/cm²` with
    /// `S = 14` photons, from `U_i = 15.6 eV` for air at `ħω = 1.165 eV`.
    ///
    /// The calibration reads their number as its own definition — at
    /// `I = I_B(MPI)`, multiphoton ionization alone reaches the breakdown
    /// criterion within one pulse — giving `W_ref = n_bd/(N·τ)`.
    ///
    /// `K = 14` here rather than the 11 that `U_i = 12.06 eV` (O₂) implies,
    /// because the calibration must be self-consistent with the photon count
    /// the paper used. The cascade keeps 12.06 eV, the lowest ionization
    /// channel available to a collisional electron.
    ///
    /// **Off by default, and see `tt2012_mpi_calibration_undershoots_the_data`
    /// before switching it on**: their MPI threshold sits 45× *below* their own
    /// measured threshold, so a rate anchored to it predicts breakdown far too
    /// early.
    pub fn with_tt2012_mpi(mut self, fwhm: f64, pressure: f64) -> Self {
        const I_B_MPI: f64 = 4.42e9 * 1e4; // W/cm² → W/m²
        self.k_photons = 14;
        self.mpi_i_ref = I_B_MPI;
        self.mpi_rate_ref = self.n_bd / (self.n_over_p * pressure * fwhm);
        self
    }

    /// Select which limit of the energy balance drives the cascade — see
    /// [`CascadeModel`]. The default is [`CascadeModel::SelfConsistentClimb`].
    pub fn with_cascade_model(mut self, cascade_model: CascadeModel) -> Self {
        self.cascade_model = cascade_model;
        self
    }

    /// Set the pulse-integration half-window in FWHMs (default 2.0, at which
    /// the Gaussian is `1.5×10⁻⁵` of peak).
    ///
    /// Exists so `threshold_is_window_independent` can vary it. Results must
    /// not depend on it; if they do, the model is being propped up by its
    /// integration bounds rather than by its physics.
    pub fn with_window(mut self, window_half_widths: f64) -> Self {
        self.window_half_widths = window_half_widths;
        self
    }

    /// Equilibrium electron energy `ε_∞ = heating/(δ_eff·ν_m)` (J) — the energy
    /// at which inelastic losses balance inverse-bremsstrahlung heating.
    ///
    /// Independent of pressure to the `(ν_m/ω)²` correction, since heating and
    /// loss both scale `∝ p`. [`CascadeModel::SelfConsistentClimb`] ionizes only
    /// where this exceeds `U_i`, which is what fixes its threshold plateau.
    pub fn equilibrium_energy(&self, intensity: f64, pressure: f64) -> f64 {
        self.heating_power(intensity, pressure) / (self.delta_eff * self.k_m * pressure)
    }

    /// Initial seed density `n_e0` (m⁻³): one electron in the focal volume.
    pub fn seed_density(&self) -> f64 {
        self.n_seed
    }

    /// Breakdown criterion density `n_bd` (m⁻³) — the level `n_e` must reach
    /// within the pulse for breakdown to be declared.
    pub fn criterion_density(&self) -> f64 {
        self.n_bd
    }

    /// Neutral number density `N = p/(k_B·T)` (m⁻³) — the ceiling on `n_e`,
    /// since ionization consumes the neutrals it feeds on.
    pub fn neutral_density(&self, pressure: f64) -> f64 {
        self.n_over_p * pressure
    }

    /// Inverse-bremsstrahlung heating power absorbed per electron (W).
    ///
    /// `P = (e²·I)/(m_e·c·ε₀) · ν_m/(ν_m²+ω²)` with `ν_m = k_m·p`. In the
    /// optical regime `ν_m ≪ ω` this is `∝ I·p`.
    pub fn heating_power(&self, intensity: f64, pressure: f64) -> f64 {
        let nu_m = self.k_m * pressure;
        let p_abs = E_CHARGE * E_CHARGE * intensity / (M_E * C_LIGHT * EPS0);
        p_abs * nu_m / (nu_m * nu_m + self.omega * self.omega)
    }

    /// Inelastic energy-loss power per electron (W), `L = L′·p`.
    ///
    /// Vibrational and electronic excitation of N₂/O₂ drain the electron on its
    /// climb to `U_i`. Scales `∝ p` because it goes as the collision frequency.
    pub fn inelastic_loss_power(&self, pressure: f64) -> f64 {
        self.delta_eff * self.k_m * pressure * self.mean_energy
    }

    /// Cascade (avalanche) ionization frequency `ν_i(I, p)` (s⁻¹).
    ///
    /// Driven by the **net** power — heating minus inelastic loss — since an
    /// electron that cannot outrun its excitation losses never reaches `U_i`:
    ///
    /// ```text
    /// ν_i = max(0, heating(I,p) − L(p)) / U_i
    /// ```
    ///
    /// Both terms scale `∝ p`, so `ν_i ∝ p` at fixed `I` (which is what the
    /// external `E_eff` gate confirms), while the `I`-dependence is *affine*,
    /// not linear: there is a finite intensity below which no cascade runs at
    /// all. That offset is what gives `I_thr(p)` its high-pressure plateau.
    pub fn cascade_rate(&self, intensity: f64, pressure: f64) -> f64 {
        match self.cascade_model {
            CascadeModel::FixedMeanEnergy => {
                let net =
                    self.heating_power(intensity, pressure) - self.inelastic_loss_power(pressure);
                net.max(0.0) / self.u_ion
            }
            CascadeModel::SelfConsistentClimb => {
                let eps_inf = self.equilibrium_energy(intensity, pressure);
                if eps_inf <= self.u_ion {
                    // Losses cap the electron below the ionization potential:
                    // the climb never completes, however long the pulse.
                    return 0.0;
                }
                // t_climb = t_r·ln(ε_∞/(ε_∞ − U_i)), t_r = 1/(δ_eff·ν_m).
                let nu_relax = self.delta_eff * self.k_m * pressure;
                nu_relax / (eps_inf / (eps_inf - self.u_ion)).ln()
            }
        }
    }

    /// Total loss frequency `ν_att + ν_diff` (s⁻¹).
    ///
    /// Attachment has two channels, both from measured rate coefficients:
    /// dissociative `k₂·n_O₂ ∝ p` and three-body `k₃·n_O₂·n ∝ p²`, the latter
    /// dominant at atmospheric density. Diffusion goes as `1/p` (free-electron
    /// `D_e ∝ 1/p` over the fixed diffusion length `Λ`).
    ///
    /// At 760 Torr the balance is attachment 6.7e7, diffusion 4.9e9 — i.e.
    /// **attachment is negligible here**, and the threshold is set by diffusion
    /// and by the finite-pulse growth requirement.
    ///
    /// Like the other rate methods this is a bare pure function and assumes
    /// `pressure > 0`; `p = 0` gives an infinite diffusion loss and `p < 0` a
    /// sign-flipped one. The `Result`-returning entry point
    /// [`threshold_intensity`](Self::threshold_intensity) is where pressure is
    /// validated.
    pub fn loss_rate(&self, pressure: f64) -> f64 {
        let n_neutral = self.n_over_p * pressure;
        let n_o2 = self.f_o2 * n_neutral;
        let nu_att = self.k_att_2body * n_o2 + self.k_att_3body * n_o2 * n_neutral;
        let d_e = self.d_e_ref * (P_REF / pressure);
        let nu_diff = d_e / (self.lambda_diff * self.lambda_diff);
        nu_att + nu_diff
    }

    /// Multiphoton seed rate `S_mpi` (m⁻³·s⁻¹), written about a reference
    /// intensity rather than as a bare cross-section:
    ///
    /// ```text
    /// S_mpi = N · W_ref · (I/I_ref)^K
    /// ```
    ///
    /// The `σ_K·I^K` form is unusable here: at `K = 14` and SI intensities it
    /// overflows `f64` well inside the bisection bracket (`I^14 > 10³⁰⁸` for
    /// `I > 10²²`), while `σ_K` itself underflows to ~`10⁻¹⁸⁶`. The ratio form
    /// keeps every intermediate in range.
    pub fn mpi_source(&self, intensity: f64, pressure: f64) -> f64 {
        if self.mpi_rate_ref == 0.0 {
            return 0.0;
        }
        let n_neutral = self.n_over_p * pressure;
        n_neutral * self.mpi_rate_ref * (intensity / self.mpi_i_ref).powi(self.k_photons)
    }

    /// Advance `n_e` by one time-slice `dt` at constant intensity, using the
    /// exact exponential solution of the linear rate ODE:
    /// `n_e' = n_e·e^{β dt} + S·expm1(β dt)/β` with `β = ν_i − ν_loss`,
    /// `S = S_mpi`. `expm1` keeps the `β → 0` limit (`n_e + S·dt`) accurate.
    /// The growth exponent is saturated at [`MAX_EXPONENT`] so a violently
    /// over-threshold pulse stays monotone rather than overflowing to `NaN`.
    pub fn advance(&self, n_e: f64, intensity: f64, pressure: f64, dt: f64) -> f64 {
        let nu_i = self.cascade_rate(intensity, pressure);
        let beta = nu_i - self.loss_rate(pressure);
        let s = self.mpi_source(intensity, pressure);
        // Ionization consumes neutrals, so the cascade cannot outrun the gas it
        // feeds on: `b = ν_i/N` makes the growth logistic and caps `n_e` at full
        // ionization. Without it the equation is linear and `n_e` runs away —
        // it reached 10^40 m⁻³ in the shipped `breakdown` case, 10^14 times the
        // neutral density and 10^13 times critical density, i.e. pure
        // extrapolation dressed as a result.
        let n_neutral = self.n_over_p * pressure;
        let b = if n_neutral > 0.0 {
            nu_i / n_neutral
        } else {
            0.0
        };

        let bdt = beta * dt;

        if s == 0.0 {
            if n_e <= 0.0 {
                return 0.0;
            }
            // Exact solution of `dn/dt = β·n − b·n²` (logistic/Bernoulli),
            // written as
            //     n' = n / (e^{−β dt} + b·n·(1 − e^{−β dt})/β)
            // rather than the textbook `β n e^{βdt} / (β + b n (e^{βdt} − 1))`.
            // Two reasons, both load-bearing. The textbook form overflows to
            // `inf/inf = NaN` far above threshold. And `e^{−β dt}` *underflows*
            // harmlessly to 0 for a strong cascade, leaving `n' → β/b` — the
            // saturation density — so no exponent clamp is needed on the
            // growing side. Clamping there would corrupt the saturation limit
            // itself (it made `peak_ne` non-monotonic in intensity: 0.1·N at
            // 10²⁴ W/m² between 1.0·N at 10²⁰ and 10³⁰).
            // Still exact, still step-size independent.
            let neg = (-bdt).min(MAX_EXPONENT); // only β < 0 can overflow `exp`
            let em = neg.exp();
            // (1 − e^{−β dt})/β, with the removable singularity at β → 0.
            let factor = if bdt.abs() < 1e-12 {
                dt
            } else {
                -neg.exp_m1() / beta
            };
            let denom = em + b * n_e * factor;
            return if denom > 0.0 && denom.is_finite() {
                n_e / denom
            } else {
                n_e * bdt.min(MAX_EXPONENT).exp()
            };
        }

        // With the multiphoton source the equation is Riccati and has no
        // elementary closed form. Fall back to the exact *linear* step with the
        // depletion factor frozen at the slice start — first-order in `dt`, and
        // documented as such (`σ_K = 0` by default, so the exact path above is
        // the one every gate exercises).
        let rate = beta - b * n_e;
        let d = (rate * dt).clamp(-MAX_EXPONENT, MAX_EXPONENT);
        // expm1(x)/x → 1 as x → 0; guard the exact-zero and tiny cases.
        let source_factor = if d.abs() < 1e-12 {
            dt
        } else {
            d.exp_m1() / rate
        };
        let next = n_e * d.exp() + s * source_factor;
        // Full ionization is the ceiling on this path too.
        if n_neutral > 0.0 {
            next.min(n_neutral)
        } else {
            next
        }
    }

    /// Peak electron density reached over a Gaussian temporal pulse of peak
    /// intensity `i_peak` (W/m²) and full-width-half-maximum `fwhm` (s) at
    /// pressure `p`, sampled with `n_steps` slices over
    /// `[−w·fwhm, +w·fwhm]`, `w` = [`window_half_widths`](Self::with_window).
    ///
    /// `n_e` is floored at the seed density throughout, and that floor is
    /// **physics, not hygiene**. Without it the model is not window-independent:
    /// losses grind the seed down during the quiet arm before the pulse
    /// arrives — at 760 Torr, `ν_loss ≈ 5×10⁹ s⁻¹` decays it by `e^-60` over
    /// 12 ns — so the avalanche must first climb back out of a hole that the
    /// arbitrary choice of `w` dug. That is meaningless as physics (`e^-60`
    /// times one electron is `10⁻²⁶` of an electron in a volume that holds
    /// either one or none) and it made the threshold depend on `w`, which the
    /// `threshold_is_window_independent` gate now forbids. The floor states the
    /// modelling assumption plainly: one seed electron is available in the
    /// focal volume when the pulse arrives.
    pub fn peak_ne(&self, i_peak: f64, fwhm: f64, pressure: f64, n_steps: usize) -> f64 {
        let half = self.window_half_widths * fwhm;
        let dt = 2.0 * half / n_steps as f64;
        // I(t) = I_pk·exp(−4 ln2 (t/fwhm)²); t centred on the pulse peak.
        let c = 4.0 * std::f64::consts::LN_2 / (fwhm * fwhm);
        let mut n_e = self.n_seed;
        let mut peak = n_e;
        for step in 0..n_steps {
            let t = -half + (step as f64 + 0.5) * dt;
            let intensity = i_peak * (-c * t * t).exp();
            n_e = self.advance(n_e, intensity, pressure, dt).max(self.n_seed);
            if n_e > peak {
                peak = n_e;
            }
        }
        peak
    }

    /// Whether a pulse of peak intensity `i_peak` breaks the gas down at
    /// pressure `p` (peak `n_e` reaches the criterion density `n_bd`).
    pub fn breaks_down(&self, i_peak: f64, fwhm: f64, pressure: f64, n_steps: usize) -> bool {
        self.peak_ne(i_peak, fwhm, pressure, n_steps) >= self.n_bd
    }

    /// Breakdown threshold peak intensity `I_thr(p)` (W/m²) for a `fwhm` pulse,
    /// by bisection in log-intensity.
    ///
    /// The bracket is **fixed** at [`I_BRACKET_LO`] … [`I_BRACKET_HI`]
    /// (1e12 … 1e22 W/m²) — it is not widened. Errors, rather than looping or
    /// returning an endpoint, if the bracket does not straddle threshold; at
    /// low pressure the diffusion branch climbs towards the ceiling (≈3e21 W/m²
    /// at 10 Torr for the pinned focus), so a change to `D_e,ref` or `Λ` can
    /// push points out of range.
    pub fn threshold_intensity(&self, fwhm: f64, pressure: f64, n_steps: usize) -> Result<f64> {
        if !(pressure > 0.0 && pressure.is_finite()) {
            bail!("pressure must be positive and finite, got {pressure}");
        }
        let mut lo = I_BRACKET_LO;
        let mut hi = I_BRACKET_HI;
        if self.breaks_down(lo, fwhm, pressure, n_steps) {
            bail!("threshold below bracket floor {lo:.1e} W/m² at p = {pressure:.1} Pa");
        }
        if !self.breaks_down(hi, fwhm, pressure, n_steps) {
            bail!("threshold above bracket ceiling {hi:.1e} W/m² at p = {pressure:.1} Pa");
        }
        // Bisect in log-intensity: threshold spans orders of magnitude.
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.breaks_down(mid, fwhm, pressure, n_steps) {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi / lo < 1.0 + 1e-6 {
                break;
            }
        }
        Ok((lo * hi).sqrt())
    }

    /// Threshold curve `I_thr(p)` over `n` log-spaced pressures in
    /// `[p_min, p_max]` (Pa). Returns `(pressure, threshold)` pairs; a pressure
    /// whose threshold escapes the bracket is skipped with its error dropped
    /// (the sweep is a survey, not a gate).
    pub fn pressure_sweep(
        &self,
        p_min: f64,
        p_max: f64,
        n: usize,
        fwhm: f64,
        n_steps: usize,
    ) -> Vec<(f64, f64)> {
        (0..n)
            .filter_map(|i| {
                let frac = i as f64 / (n - 1).max(1) as f64;
                let p = p_min * (p_max / p_min).powf(frac);
                self.threshold_intensity(fwhm, p, n_steps)
                    .ok()
                    .map(|it| (p, it))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::loglog_slope;

    const TORR: f64 = 133.322_368_4; // Pa per Torr

    /// Pinned pressure range of the M6a slope gate (Pa) — see
    /// `high_pressure_threshold_slope_is_cascade_limited` and
    /// docs/M6A_SPEC.md. Brackets the T&T measurement range; the fitted
    /// exponent is only defined relative to it.
    const GATE_P_LO: f64 = 300.0 * TORR;
    const GATE_P_HI: f64 = 2000.0 * TORR;

    fn model() -> AirBreakdown {
        AirBreakdown::air_1064nm()
    }

    // --- Integrator unit tests (NOT physics validation — closed-form limits of
    // the rate ODE, docs/M6A_SPEC.md § Integrator unit tests). ----------------

    #[test]
    fn pure_cascade_is_exponential_growth() {
        // ν_i only: a hand-made model with no losses, no MPI. Compare the
        // stepped solution to n_e0·exp(ν_i·t).
        let m = model();
        let p = 760.0 * TORR;
        let intensity = 1e15; // W/m²; keeps ν_i·t modest (no exp overflow)
        let nu_i = m.cascade_rate(intensity, p);
        // Suppress losses and source for this limit by evaluating advance with
        // a model whose loss and MPI are zero: build a bespoke one.
        let bare = AirBreakdown {
            k_att_2body: 0.0,
            k_att_3body: 0.0,
            d_e_ref: 0.0,
            mpi_rate_ref: 0.0,
            ..m
        };
        let dt = 1e-11;
        let steps = 50;
        let mut n_e = 1.0;
        for _ in 0..steps {
            n_e = bare.advance(n_e, intensity, p, dt);
        }
        let analytic = (nu_i * dt * steps as f64).exp();
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "cascade {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn pure_loss_is_exponential_decay() {
        let m = model();
        let p = 760.0 * TORR;
        // No cascade (zero intensity), no MPI: pure loss decay.
        let loss = m.loss_rate(p);
        let bare = AirBreakdown {
            mpi_rate_ref: 0.0,
            ..m
        };
        let dt = 1e-10;
        let steps = 30;
        let mut n_e = 1e20;
        for _ in 0..steps {
            n_e = bare.advance(n_e, 0.0, p, dt);
        }
        let analytic = 1e20 * (-loss * dt * steps as f64).exp();
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "loss {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn mpi_only_seeding_is_linear() {
        // β = 0 (no cascade, no loss), constant source S: n_e(t) = n_e0 + S·t.
        let m = model();
        let p = 760.0 * TORR;
        let s = 1e30; // m⁻³ s⁻¹
        let balanced = AirBreakdown {
            k_att_2body: 0.0,
            k_att_3body: 0.0,
            d_e_ref: 0.0,
            ..m
        };
        // Force β=0 by giving cascade zero intensity and injecting S via a
        // constant source: emulate with advance at I=0 but nonzero mpi by
        // choosing the reference so that S_mpi(I0) = s exactly.
        let i0: f64 = 1e17;
        let src = AirBreakdown {
            mpi_rate_ref: s / (balanced.n_over_p * p),
            mpi_i_ref: i0,
            ..balanced
        };
        // At I0 the cascade is nonzero, so also null it to isolate β=0: use a
        // separate check via advance math with cascade suppressed.
        let src0 = AirBreakdown {
            u_ion: f64::INFINITY, // ν_i → 0
            ..src
        };
        let dt = 1e-11;
        let steps = 40;
        let mut n_e = 0.0;
        for _ in 0..steps {
            n_e = src0.advance(n_e, i0, p, dt);
        }
        let analytic = s * dt * steps as f64;
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "mpi {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn balance_point_is_linear_from_seed() {
        // β=0 with a nonzero seed: n_e(t) = n_e0 + S·t.
        let m = model();
        let p = 760.0 * TORR;
        let n0 = 1e18;
        let s = 5e29;
        let i0: f64 = 1e17;
        let bal = AirBreakdown {
            u_ion: f64::INFINITY, // no cascade → β = −loss; cancel loss too
            k_att_2body: 0.0,
            k_att_3body: 0.0,
            d_e_ref: 0.0,
            mpi_rate_ref: s / (m.n_over_p * p),
            mpi_i_ref: i0,
            ..m
        };
        let dt = 1e-11;
        let steps = 25;
        let mut n_e = n0;
        for _ in 0..steps {
            n_e = bal.advance(n_e, i0, p, dt);
        }
        let analytic = n0 + s * dt * steps as f64;
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "balance {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn slice_refinement_is_consistent() {
        // The exact per-slice solution must be step-size independent for
        // constant intensity: one step of dt equals m steps of dt/m.
        let m = model();
        let p = 500.0 * TORR;
        let intensity = 2e17;
        let dt = 2e-10;
        let coarse = m.advance(1e15, intensity, p, dt);
        let mut fine = 1e15;
        let sub = 64;
        for _ in 0..sub {
            fine = m.advance(fine, intensity, p, dt / sub as f64);
        }
        assert!(
            (coarse / fine - 1.0).abs() < 1e-9,
            "coarse {coarse:e} vs fine {fine:e}"
        );
    }

    // --- Model behaviour + physics-gate observable ---------------------------

    #[test]
    fn cascade_is_affine_in_intensity_and_linear_in_pressure() {
        // FixedMeanEnergy only: SelfConsistentClimb is nonlinear in I by design.
        // Both heating and inelastic loss carry the same factor p, so at fixed
        // intensity ν_i ∝ p EXACTLY — which is the scaling the external E_eff
        // gate confirms, and it survives the inelastic term untouched.
        let m = model().with_cascade_model(CascadeModel::FixedMeanEnergy);
        let p = 300.0 * TORR;
        let i = 1e17;
        // Linear only to the (ν_m/ω)² correction of the IB Lorentzian, ~1e-6.
        assert!((m.cascade_rate(i, 2.0 * p) / m.cascade_rate(i, p) - 2.0).abs() < 1e-4);

        // In intensity the rate is AFFINE, not linear: equal intensity steps
        // give equal rate steps, but the line does not pass through the origin.
        let (a, b, c) = (
            m.cascade_rate(i, p),
            m.cascade_rate(2.0 * i, p),
            m.cascade_rate(3.0 * i, p),
        );
        assert!(((b - a) / (c - b) - 1.0).abs() < 1e-12, "not affine in I");
        // ν_i = k·(I − I₀) with I₀ > 0, so doubling I more than doubles ν_i.
        // Exactly 2.0 would mean the inelastic offset had gone missing.
        assert!(
            b / a > 2.0,
            "ν_i/I is linear through the origin ({:.6}) — inelastic offset missing",
            b / a
        );
    }

    #[test]
    fn cascade_shuts_off_below_the_inelastic_loss() {
        // The new physics in one assertion: there is a finite intensity below
        // which heating cannot outrun excitation losses and no cascade runs at
        // all. This offset is what gives I_thr(p) a high-pressure plateau; the
        // model could not produce one without it.
        let m = model().with_cascade_model(CascadeModel::FixedMeanEnergy);
        let p = P_REF;
        let i_cut = m.inelastic_loss_power(p) / (m.heating_power(1.0, p));
        assert!(i_cut > 0.0 && i_cut.is_finite());
        assert_eq!(m.cascade_rate(0.5 * i_cut, p), 0.0);
        assert!(m.cascade_rate(2.0 * i_cut, p) > 0.0);
        // At the cut the two powers balance by construction.
        let net = m.heating_power(i_cut, p) - m.inelastic_loss_power(p);
        assert!(net.abs() < 1e-12 * m.inelastic_loss_power(p));
    }

    #[test]
    fn threshold_brackets_and_breaks_down_above() {
        let m = model();
        let p = 760.0 * TORR;
        let fwhm = 6e-9;
        let it = m.threshold_intensity(fwhm, p, 400).unwrap();
        assert!(it.is_finite() && it > 0.0);
        // Just above threshold breaks down; just below does not.
        assert!(m.breaks_down(it * 1.2, fwhm, p, 400));
        assert!(!m.breaks_down(it * 0.8, fwhm, p, 400));
    }

    #[test]
    fn inelastic_loss_envelope_brackets_the_slope() {
        // MODEL-CONSISTENCY GATE over the LITERATURE RANGE of the one lumped
        // constant the inelastic term adds (δ_eff ≈ 0.01–0.05, ⟨ε⟩ ≈ 2–5 eV,
        // ~12× in L′). The model's slope is an envelope, not a point, and this
        // pins that envelope so it cannot drift silently:
        //
        //   δ=0.01 ⟨ε⟩=2 → n=0.785     δ=0.05 ⟨ε⟩=5 → n=0.187
        //
        // Measured n = 0.329 is inside THIS envelope — but note this is the
        // FixedMeanEnergy variant, not the default, and note also that the
        // measurement is NOT a cascade-only observable (it contains MPI). The
        // default SelfConsistentClimb envelope is [0.023, 0.231]. The
        // apples-to-apples target is T&T's own cascade closed form, gated by
        // tt2012_cascade_theory_reference in tests/validation.rs.
        let mut ns = Vec::new();
        for delta in [0.01, 0.02, 0.05] {
            for ev in [2.0, 3.0, 5.0] {
                let m = AirBreakdown::air_1064nm()
                    .with_cascade_model(CascadeModel::FixedMeanEnergy)
                    .with_inelastic_loss(delta, ev);
                let c = m.pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400);
                assert_eq!(c.len(), 8, "sweep lost points at δ={delta}, ⟨ε⟩={ev}");
                ns.push(-loglog_slope(&c).unwrap());
            }
        }
        let lo = ns.iter().cloned().fold(f64::MAX, f64::min);
        let hi = ns.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (0.17..=0.21).contains(&lo) && (0.76..=0.81).contains(&hi),
            "literature envelope moved: n ∈ [{lo:.3}, {hi:.3}], expected ≈[0.187, 0.785]"
        );
        // The whole envelope must sit below the no-inelastic-loss floor of 1.74
        // — that improvement is the point of the term.
        assert!(
            hi < 1.5,
            "envelope top {hi:.3} did not improve on n = 1.737"
        );
    }

    #[test]
    fn self_consistent_climb_eliminates_the_mean_energy() {
        // The whole point of the self-consistent variant: ⟨ε⟩ is not an input.
        // Solving dε/dt = heating − δ_eff·ν_m·ε removes it, leaving only δ_eff.
        // If ⟨ε⟩ ever leaks back into this path, this fails.
        let sweep = |ev: f64| {
            AirBreakdown::air_1064nm()
                .with_inelastic_loss(0.02, ev)
                .with_cascade_model(CascadeModel::SelfConsistentClimb)
                .pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400)
        };
        let (a, b) = (sweep(2.0), sweep(5.0));
        assert_eq!(a.len(), 8);
        let (sa, sb) = (loglog_slope(&a).unwrap(), loglog_slope(&b).unwrap());
        assert!(
            (sa - sb).abs() < 1e-9,
            "⟨ε⟩ still affects the self-consistent path: {sa:.6} vs {sb:.6}"
        );
        // And it must keep ν_i ∝ p, which the external E_eff gate confirms.
        let m = AirBreakdown::air_1064nm().with_cascade_model(CascadeModel::SelfConsistentClimb);
        let (p, i) = (500.0 * TORR, 2e16);
        let ratio = m.cascade_rate(i, 2.0 * p) / m.cascade_rate(i, p);
        assert!(
            (ratio - 2.0).abs() < 1e-3,
            "self-consistent ν_i is no longer ∝ p: {ratio:.6}"
        );
    }

    #[test]
    fn the_two_cascade_models_bracket_the_measurement() {
        // THE DURABLE CLAIM of M6a, and the only slope statement that survived
        // the 2026-07-25 audit. The two limits of the same energy balance sit
        // either side of T&T's n = 0.329: FixedMeanEnergy (⟨ε⟩ free,
        // literature-bounded) is steeper at 0.468; SelfConsistentClimb (⟨ε⟩
        // eliminated) is flatter at 0.095. Neither reproduces the measurement;
        // the truth lies between, which is what a mean-energy treatment of a
        // process governed by the tail of the energy distribution should give.
        //
        // Recording the bracket keeps either from being quietly "improved" into
        // the other's territory — and note the bracket is what remained TRUE
        // when the seed-window artifact was removed and both endpoints moved
        // (0.800 → 0.551 → 0.468 and 0.356 → 0.127 → 0.095 as the seed-window
        // artifact and then the focal geometry were corrected).
        let slope_of = |model| {
            let c = AirBreakdown::air_1064nm()
                .with_cascade_model(model)
                .pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400);
            assert_eq!(c.len(), 8);
            -loglog_slope(&c).unwrap()
        };
        let fixed = slope_of(CascadeModel::FixedMeanEnergy);
        let selfc = slope_of(CascadeModel::SelfConsistentClimb);
        const MEASURED: f64 = 0.329;
        assert!(
            (0.45..=0.49).contains(&fixed),
            "FixedMeanEnergy slope moved: {fixed:.3}, expected ≈0.468"
        );
        assert!(
            (0.08..=0.11).contains(&selfc),
            "SelfConsistentClimb slope moved: {selfc:.3}, expected ≈0.095"
        );
        assert!(
            selfc < MEASURED && MEASURED < fixed,
            "the models no longer bracket the measurement: \
             {selfc:.3} .. {MEASURED} .. {fixed:.3}"
        );
    }

    #[test]
    fn high_pressure_threshold_slope_lies_between_analytic_limits() {
        // PHYSICS GATE, model-consistency form. Bracketed by two CLOSED-FORM
        // limits rather than a fitted band. Solving the avalanche criterion —
        // cascade must outrun losses and clear the n_seed → n_bd growth
        // requirement G = ln(n_bd/n_seed)/τ within the pulse — gives
        //
        //     I_thr(p) = [ν_att(p) + ν_diff(p) + G] / (A·p),   A ≡ ν_i/(I·p)
        //
        // With attachment from measured rate coefficients it is negligible here
        // (6.7e7 vs 4.9e9 for diffusion at 760 Torr), leaving two terms whose
        // exponents are exact:
        //
        //   * growth-limited, losses → 0 : I_thr = G/(A·p)        ∝ p^-1
        //   * diffusion-limited          : I_thr = ν_diff/(A·p)   ∝ p^-2
        //     (ν_diff = D_e,ref·(P_REF/p)/Λ² ∝ 1/p)
        //
        // Those two bracketed the model to n ∈ [1, 2] BEFORE the inelastic-loss
        // term existed (observed then: n = 1.737). The term adds a third,
        // pressure-independent contribution — a genuine plateau — which drags
        // the slope below that old floor, to n = 0.356 with the default
        // SelfConsistentClimb (0.800 with FixedMeanEnergy). So this test now
        // gates the *combined* form:
        //
        //     I_thr(p) = L′/h + U_i·(ν_diff + ν_att + G)/(h·p)
        //
        // whose exponent runs from 0 (plateau-dominated) to 2 (diffusion), and
        // the assertion is that the default sits in the plateau-influenced
        // regime n ∈ (0, 1) rather than the old loss-only regime n ≥ 1.
        //
        // Neither bracket is universal: they hold only while attachment is
        // negligible. Three-body attachment is ∝ p², contributing I_thr ∝ +p,
        // so far above this window the model leaves them entirely — the slope
        // turns positive above ~10^4 Torr. That is why the range is pinned.
        //
        // This is the model's own consistency, NOT agreement with experiment:
        // T&T measure n = 0.33, outside this interval entirely. That gap is the
        // real M6a finding and is gated separately in tests/validation.rs.
        let m = model();
        let n_points = 8;
        let curve = m.pressure_sweep(GATE_P_LO, GATE_P_HI, n_points, 6e-9, 400);
        assert_eq!(
            curve.len(),
            n_points,
            "gate sweep lost points to the bisection bracket: {curve:?}"
        );
        let slope = loglog_slope(&curve).unwrap();
        assert!(
            (-1.0..0.0).contains(&slope),
            "high-p threshold slope {slope:.3} over {:.0}–{:.0} Torr; with the inelastic-loss \
             plateau the model must sit in n∈(0,1), below the loss-only floor of n=1",
            GATE_P_LO / TORR,
            GATE_P_HI / TORR
        );
    }

    #[test]
    fn attachment_is_negligible_against_diffusion_at_one_atmosphere() {
        // Guards the finding that motivated switching to measured attachment
        // coefficients: the order-of-magnitude K_a·p it replaced was ~150x too
        // large and was the only thing flattening the modelled slope towards
        // the data. If a future edit reinstates a dominant attachment term,
        // this fails and points at docs/M6A_SPEC.md rather than letting the
        // slope quietly drift back into apparent agreement.
        let m = model();
        let p = P_REF;
        let n = m.n_over_p * p;
        let n_o2 = m.f_o2 * n;
        let nu_att = m.k_att_2body * n_o2 + m.k_att_3body * n_o2 * n;
        let nu_diff = m.d_e_ref * (P_REF / p) / (m.lambda_diff * m.lambda_diff);
        assert!(
            nu_att < 0.05 * nu_diff,
            "attachment {nu_att:e} is not negligible against diffusion {nu_diff:e}"
        );
        // Three-body is the dominant attachment channel at atmospheric density.
        assert!(m.k_att_3body * n_o2 * n > 0.1 * m.k_att_2body * n_o2);
    }

    #[test]
    fn tt2012_mpi_calibration_undershoots_the_data() {
        // Why the multiphoton source stays OFF by default, recorded as a
        // measurement rather than an opinion.
        //
        // T&T give an MPI threshold of 4.42e9 W/cm² at 760 Torr (their
        // Sec. II A) and separately measure breakdown at 2.06e11 — a factor 47
        // higher. So MPI *alone*, read as a breakdown criterion, should have
        // ignited the gas at 4.42e9, and it demonstrably did not. Anchoring a
        // rate coefficient to that number inherits the problem: the threshold
        // collapses to ≈5.5e9 W/cm², ~37× below what the same paper measures,
        // and contradicts its own accounting of 88 % cascade / 12 % MPI at
        // 760 Torr.
        //
        // The conclusion is about the paper's MPI estimate, not about MPI: it
        // is an order-of-magnitude significance indicator (Nelson's flux-density
        // criterion, whose constant C the paper never states), not a rate. A
        // real σ_K from multiphoton cross-section data is the open item.
        let p = 760.0 * TORR;
        let cascade = model();
        let with_mpi = cascade.with_tt2012_mpi(6e-9, p);
        let i_cascade = cascade.threshold_intensity(6e-9, p, 400).unwrap();
        let i_mpi = with_mpi.threshold_intensity(6e-9, p, 400).unwrap();
        const MEASURED: f64 = 2.06e11 * 1e4; // W/m²

        assert!(
            i_mpi < i_cascade / 100.0,
            "MPI no longer dominates the cascade: {i_mpi:.3e} vs {i_cascade:.3e}"
        );
        assert!(
            i_mpi < MEASURED / 20.0,
            "T&T-calibrated MPI threshold {i_mpi:.3e} is no longer far below the \
             measured {MEASURED:.3e}; recheck the calibration before enabling it"
        );
        // And the default must keep it off.
        assert_eq!(model().mpi_source(1e16, p), 0.0);
    }

    #[test]
    fn threshold_is_window_independent() {
        // NUMERICAL-HYGIENE GATE, added 2026-07-25 after this exact defect
        // shipped. The pulse at |t| = 2·FWHM is 1.5e-5 of peak, so widening the
        // integration window must not move the threshold. It used to: losses
        // ground the seed down during the quiet arm (e^-60 at 760 Torr over
        // 12 ns), so the avalanche had to climb out of a hole whose depth was
        // set by an arbitrary integration bound. The threshold rose 11% from
        // w = 2 to w = 4 and never converged.
        //
        // Worse, that decay is pressure-dependent, so it manufactured SLOPE:
        // removing it moved the default model from n = 0.356 to n = 0.127
        // against a measured 0.329. The external slope gate had been green on
        // the strength of a bookkeeping artifact.
        let base = model();
        let at = |w: f64| {
            base.with_window(w)
                .threshold_intensity(6e-9, 760.0 * TORR, 400)
                .unwrap()
        };
        let widths = [1.0, 1.5, 2.0, 3.0, 4.0];
        let vals: Vec<f64> = widths.iter().map(|&w| at(w)).collect();
        let lo = vals.iter().cloned().fold(f64::MAX, f64::min);
        let hi = vals.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            hi / lo < 1.01,
            "threshold depends on the integration window: {lo:.4e}..{hi:.4e} \
             over w ∈ {widths:?} ({:.1}% spread)",
            100.0 * (hi / lo - 1.0)
        );
    }

    #[test]
    fn cascade_saturates_at_full_ionization() {
        // The gas cannot be more than fully ionized. Before the logistic term
        // the rate equation was linear in n_e and ran to 1e40 m^-3 in the
        // shipped `breakdown` case — 1e14 times the neutral density, 1e13 times
        // critical density at 1064 nm — with the visible ceiling being a
        // plotting clamp rather than any physics.
        let m = model();
        let p = P_REF;
        let n_neutral = m.neutral_density(p);
        let mut prev = 0.0;
        for i in [1e17, 1e18, 1e20, 1e22, 1e24, 1e26, 1e30] {
            let ne = m.peak_ne(i, 6e-9, p, 400);
            assert!(ne.is_finite(), "peak n_e not finite at I = {i:e}");
            assert!(
                ne <= n_neutral * 1.001,
                "n_e = {ne:e} exceeds full ionization N = {n_neutral:e} at I = {i:e}"
            );
            // Monotone in intensity: the clamp used to break this, giving 0.1·N
            // at 1e24 between 1.0·N at 1e20 and 1e30.
            assert!(
                ne >= prev * 0.999,
                "peak n_e non-monotonic in intensity at I = {i:e}: {ne:e} < {prev:e}"
            );
            prev = ne;
        }
    }

    #[test]
    fn far_above_threshold_stays_broken_down() {
        // Monotonicity guard: the growth exponent saturates instead of
        // overflowing, so an absurdly over-threshold pulse must still report
        // breakdown. Unsaturated, exp(β·dt) → inf made n_e NaN and every
        // `n_e > peak` comparison false, so this returned *false*.
        let m = model();
        let p = 760.0 * TORR;
        for i_peak in [1e20, 1e22, 1e24, 1e30] {
            let n_e = m.peak_ne(i_peak, 6e-9, p, 400);
            assert!(!n_e.is_nan(), "peak n_e is NaN at I = {i_peak:e}");
            assert!(
                m.breaks_down(i_peak, 6e-9, p, 400),
                "no breakdown reported at I = {i_peak:e} (peak n_e = {n_e:e})"
            );
        }
    }

    #[test]
    fn threshold_out_of_bracket_errors_cleanly() {
        // A pressure so low that diffusion loss pushes threshold past the
        // bracket ceiling must error, not loop.
        let m = model();
        let fwhm = 6e-9;
        // 0.01 Torr: enormous diffusion loss.
        let r = m.threshold_intensity(fwhm, 0.01 * TORR, 400);
        assert!(r.is_err());
    }

    #[test]
    fn photon_count_is_eleven_at_1064nm() {
        // ⌈12.06 eV / 1.166 eV⌉ = 11.
        assert_eq!(model().k_photons, 11);
    }
}
