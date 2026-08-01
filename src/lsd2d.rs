//! Axisymmetric laser-supported detonation: a **finite-diameter** beam, so the
//! shocked gas can relieve sideways (M6d, step 5).
//!
//! This is `lsd.rs` with the one assumption M6c could not drop. There the beam
//! is infinitely wide by construction — a planar slab has no transverse
//! direction — and `docs/M6C_SPEC.md` § G7 rests its whole case on that:
//! "a planar 1-D code has **no radial relief** […] it is the one effect the
//! geometry has removed by assumption". Here the beam has a radius `R_b`, the
//! gas can escape across it, and the front slows down. How much is the
//! milestone's headline measurement.
//!
//! # What is reused rather than rebuilt
//!
//! [`Absorption`] (including the `GreyThreshold` verification closure and M6c's
//! reasoning for it), [`IonizationCeiling`], [`raizer_lsd_velocity`], the Strang
//! cadence, and the discretely conservative deposition
//! `q_k = (I_k − I_{k+1})/Δx` that lets the energy budget close to round-off
//! instead of to a quadrature tolerance. The absorption closures are functions
//! of `(ρ, p)` alone, so they carry over to two dimensions untouched.
//!
//! # The beam is a bundle of independent pencils
//!
//! One Beer–Lambert march per ring, no refraction and no diffraction
//! (`docs/M6D_SPEC.md` gate decision 5). Because each ring's march is
//! separately conservative, the `r`-weighted budget still closes exactly.
//!
//! Making the beam bend in the radially structured plasma would turn the
//! coupling two-way, which `docs/M6C_SPEC.md` open question 3 already reserves
//! for a later milestone with its own gate. It is stated here, not smuggled in.
//!
//! # Relief comes from the beam being finite, not the domain
//!
//! With the outer boundary well outside `R_b` the lateral rarefaction is
//! entirely interior, so the budget closes with no boundary-flux accounting at
//! all and [`Lsd2dColumn::check_regime`] refuses the configurations where that
//! stops being true. When relief *does* reach the wall — the long demonstration
//! runs — [`Euler2d::escaped_energy`] accounts for it.

use anyhow::{Result, bail};

use crate::euler1d::{IdealGas, Primitive};
use crate::euler2d::{Boundary2d, Conserved2d, Euler2d, Geometry, Primitive2d};
use crate::lsd::Absorption;

/// Minimum cells across the beam radius.
///
/// Below this the deposition profile is a staircase and the relief deficit
/// would be a mesh artefact rather than a measurement — the number G15 exists
/// to pin is exactly the one this protects.
pub const MIN_CELLS_PER_BEAM_RADIUS: f64 = 8.0;

/// Minimum domain radius, in beam radii.
///
/// Relief has to come from the beam being finite. If the outer wall is close
/// enough to participate, the measured deficit is partly the boundary's.
pub const MIN_DOMAIN_RADII: f64 = 3.0;

/// Radial shape of the incident beam.
#[derive(Debug, Clone, Copy)]
pub enum BeamProfile {
    /// Uniform out to `radius`, zero beyond.
    ///
    /// The gates use this: `R_b` is then unambiguous and the diameter effect
    /// reads cleanly against it. Place `R_b` on a cell face — `check_regime`
    /// does not enforce that, but a top-hat edge cutting a cell in half is a
    /// good way to give `δ` a spurious grid dependence.
    TopHat {
        /// Beam radius (m).
        radius: f64,
    },
    /// `exp(−(r/radius)^(2·order))` — a soft-edged beam for the demonstration
    /// run, where the profile is a picture rather than a measurement.
    SuperGaussian {
        /// `1/e` radius (m).
        radius: f64,
        /// Super-Gaussian order; 1 is an ordinary Gaussian.
        order: u32,
    },
}

impl BeamProfile {
    /// Fraction of the peak incident intensity at radius `r`.
    pub fn weight(&self, r: f64) -> f64 {
        match *self {
            Self::TopHat { radius } => {
                if r <= radius {
                    1.0
                } else {
                    0.0
                }
            }
            Self::SuperGaussian { radius, order } => (-(r / radius).powi(2 * order as i32)).exp(),
        }
    }

    /// Nominal beam radius (m).
    pub fn radius(&self) -> f64 {
        match *self {
            Self::TopHat { radius } | Self::SuperGaussian { radius, .. } => radius,
        }
    }
}

/// Where and how hard to light the initial spark.
///
/// M6c's [`SeededIgnition`](crate::lsd::SeededIgnition) with a radius. That
/// radius is a **second free parameter beside the pressure**, so M6c's
/// seed-independence discipline (G3c) has to cover it too — otherwise the
/// headline deficit sits on an unexamined knob.
#[derive(Debug, Clone, Copy)]
pub struct SeededIgnition2d {
    /// Axial centre of the hot spot (m).
    pub centre_x: f64,
    /// Full axial width of the hot spot (m).
    pub width_x: f64,
    /// Radius of the hot spot (m).
    pub radius: f64,
    /// Pressure inside it (Pa). Density stays ambient, so this is a pure energy
    /// deposit — the numerical analogue of a spark.
    pub pressure: f64,
}

/// An axisymmetric laser-supported detonation: gas dynamics, a finite-diameter
/// beam, and the deposition that couples them.
pub struct Lsd2dColumn {
    hydro: Euler2d,
    absorption: Absorption,
    /// Peak incident intensity at `x = 0` (W/m²).
    incident: f64,
    beam: BeamProfile,
    ambient: Primitive2d,
    deposited: f64,
    initial_energy: f64,
}

impl Lsd2dColumn {
    /// Build a column of `n_r × n_x` cells over `[0, radius] × [0, length]`,
    /// filled with `ambient`, with a seeded hot spot, driven by a beam of peak
    /// intensity `incident` (W/m²) entering at `x = 0` and travelling `+x`.
    ///
    /// The front then runs toward the laser, i.e. toward **decreasing** `x`, as
    /// in M6c.
    #[allow(clippy::too_many_arguments)]
    pub fn seeded(
        gas: IdealGas,
        n_r: usize,
        n_x: usize,
        radius: f64,
        length: f64,
        ambient: Primitive2d,
        ignition: SeededIgnition2d,
        absorption: Absorption,
        incident: f64,
        beam: BeamProfile,
    ) -> Result<Self> {
        Self::seeded_with_geometry(
            gas,
            Geometry::Axisymmetric,
            n_r,
            n_x,
            radius,
            length,
            ambient,
            ignition,
            absorption,
            incident,
            beam,
        )
    }

    /// [`seeded`](Self::seeded) with the geometry chosen explicitly.
    ///
    /// `Geometry::Planar` is the **control**, not a physical model: it is the
    /// same coupled driver with the geometric source switched off, which is what
    /// separates "the wave is transversely unstable" from "the axisymmetric
    /// terms are injecting noise". Gate G14 uses it for exactly that.
    #[allow(clippy::too_many_arguments)]
    pub fn seeded_with_geometry(
        gas: IdealGas,
        geometry: Geometry,
        n_r: usize,
        n_x: usize,
        radius: f64,
        length: f64,
        ambient: Primitive2d,
        ignition: SeededIgnition2d,
        absorption: Absorption,
        incident: f64,
        beam: BeamProfile,
    ) -> Result<Self> {
        if !(incident > 0.0 && incident.is_finite()) {
            bail!("lsd2d: incident intensity must be positive and finite, got {incident}");
        }
        if !(length > 0.0 && length.is_finite() && radius > 0.0 && radius.is_finite()) {
            bail!("lsd2d: domain must be positive and finite, got {radius} m x {length} m");
        }
        if !(ignition.width_x > 0.0
            && ignition.width_x.is_finite()
            && ignition.radius > 0.0
            && ignition.radius.is_finite())
            || ignition.pressure <= ambient.p
        {
            bail!(
                "lsd2d: the seed must be a hot spot — {} m x {} m, pressure {} Pa vs \
                 ambient {} Pa",
                ignition.radius,
                ignition.width_x,
                ignition.pressure,
                ambient.p
            );
        }
        let half = 0.5 * ignition.width_x;
        let hydro = Euler2d::from_fn(
            gas,
            geometry,
            [
                Boundary2d::Axis,
                Boundary2d::Transmissive,
                Boundary2d::Transmissive,
                Boundary2d::Transmissive,
            ],
            0.0,
            0.0,
            radius / n_r as f64,
            length / n_x as f64,
            n_r,
            n_x,
            |r, x| {
                if (x - ignition.centre_x).abs() <= half && r <= ignition.radius {
                    Primitive2d {
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
            beam,
            ambient,
            deposited: 0.0,
            initial_energy,
        })
    }

    /// The gas dynamics, read-only.
    pub fn hydro(&self) -> &Euler2d {
        &self.hydro
    }

    /// The gas dynamics, mutably — for the step controller and the CLI case.
    pub fn hydro_mut(&mut self) -> &mut Euler2d {
        &mut self.hydro
    }

    /// The beam's radial profile.
    pub fn beam(&self) -> BeamProfile {
        self.beam
    }

    /// Peak incident intensity at `x = 0` (W/m²).
    pub fn incident_intensity(&self) -> f64 {
        self.incident
    }

    /// The 1-D state an absorption closure sees.
    ///
    /// Every closure in [`Absorption`] is a function of `(ρ, p)` alone — the
    /// grey threshold keys on specific internal energy, inverse bremsstrahlung
    /// on the table's inversion of `(ρ, p)` — so the velocity component is
    /// irrelevant and is passed as zero rather than invented.
    fn as_1d(w: Primitive2d) -> Primitive {
        Primitive {
            rho: w.rho,
            u: 0.0,
            p: w.p,
        }
    }

    /// Absorption coefficient per cell (1/m), row-major.
    pub fn alpha_profile(&self) -> Result<Vec<f64>> {
        let gas = self.hydro.gas();
        self.hydro
            .primitives()
            .into_iter()
            .map(|w| self.absorption.coefficient(&gas, Self::as_1d(w)))
            .collect()
    }

    /// Absorbed power density per cell (W/m³), row-major.
    ///
    /// One independent Beer–Lambert march per ring, starting from
    /// `S·b(r_j)`. `q = (I_k − I_{k+1})/Δx` per cell, so each ring's deposition
    /// sums exactly to the intensity that ring's pencil lost.
    pub fn deposition(&self) -> Result<Vec<f64>> {
        let alpha = self.alpha_profile()?;
        let (n_r, n_x, dx) = (self.hydro.n_r(), self.hydro.n_x(), self.hydro.dx());
        let mut q = vec![0.0; n_r * n_x];
        for j in 0..n_r {
            let mut i_beam = self.incident * self.beam.weight(self.hydro.r_centre(j));
            for i in 0..n_x {
                let k = j * n_x + i;
                let next = i_beam * (-alpha[k] * dx).exp();
                q[k] = (i_beam - next) / dx;
                i_beam = next;
            }
        }
        Ok(q)
    }

    /// Beam intensity at each cell's **upstream face** (W/m²), row-major — what
    /// the renderer draws as "the beam being eaten", per ring.
    pub fn intensity_profile(&self) -> Result<Vec<f64>> {
        let alpha = self.alpha_profile()?;
        let (n_r, n_x, dx) = (self.hydro.n_r(), self.hydro.n_x(), self.hydro.dx());
        let mut out = vec![0.0; n_r * n_x];
        for j in 0..n_r {
            let mut i_beam = self.incident * self.beam.weight(self.hydro.r_centre(j));
            for i in 0..n_x {
                out[j * n_x + i] = i_beam;
                i_beam *= (-alpha[j * n_x + i] * dx).exp();
            }
        }
        Ok(out)
    }

    /// Ring volume in the `r dr` measure, matching the hydro's own metric.
    fn ring_volume(&self, j: usize) -> f64 {
        let dr = self.hydro.dr();
        let lo = j as f64 * dr;
        0.5 * ((lo + dr) * (lo + dr) - lo * lo)
    }

    /// Deposition integrated over the domain in the `r dr dx` measure (W).
    fn deposition_rate(&self, q: &[f64]) -> f64 {
        let (n_r, n_x, dx) = (self.hydro.n_r(), self.hydro.n_x(), self.hydro.dx());
        (0..n_r)
            .map(|j| {
                let vol = self.ring_volume(j) * dx;
                (0..n_x).map(|i| q[j * n_x + i]).sum::<f64>() * vol
            })
            .sum()
    }

    /// Stable step for the **coupled** system (s).
    ///
    /// M6c's predictor, unchanged in substance: the Strang sandwich deposits
    /// energy *before* the flux update, so the leading half-step raises `p`,
    /// raises `c`, and shrinks the true CFL limit below what the pre-deposition
    /// state advertises. Handing the hydro that larger step is refused outright
    /// by its guard, which is the guard working.
    pub fn stable_dt(&self) -> Result<f64> {
        let q = self.deposition()?;
        let gas = self.hydro.gas();
        let w = self.hydro.primitives();
        let (dr, dx, cfl) = (self.hydro.dr(), self.hydro.dx(), self.hydro.cfl());
        let mut dt = self.hydro.stable_dt()?;
        for _ in 0..8 {
            let rate = w
                .iter()
                .zip(&q)
                .map(|(c, &qi)| {
                    let p = c.p + (gas.gamma - 1.0) * qi * 0.5 * dt;
                    let sound = gas.sound_speed(c.rho, p.max(c.p));
                    ((c.u_r.abs() + sound) / dr).max((c.u_x.abs() + sound) / dx)
                })
                .fold(0.0, f64::max);
            let next = cfl / rate;
            let converged = (next - dt).abs() <= 1e-10 * dt;
            dt = next;
            if converged {
                break;
            }
        }
        if !(dt > 0.0 && dt.is_finite()) {
            bail!(
                "lsd2d: no stable coupled step at t = {:.4e} s (got dt = {dt})",
                self.hydro.time()
            );
        }
        Ok(dt * (1.0 - 1e-9))
    }

    /// Advance one coupled step of `dt` with Strang splitting.
    pub fn advance(&mut self, dt: f64) -> Result<()> {
        self.deposit(0.5 * dt)?;
        self.hydro.step(dt)?;
        self.deposit(0.5 * dt)?;
        Ok(())
    }

    /// One source half-step: deposit for `dt` and bank the energy.
    fn deposit(&mut self, dt: f64) -> Result<()> {
        let q = self.deposition()?;
        let n_x = self.hydro.n_x();
        self.hydro.add_energy(dt, |j, i, _, _| q[j * n_x + i])?;
        self.deposited += dt * self.deposition_rate(&q);
        Ok(())
    }

    /// Run to `t_end` (s), sizing each step from the current wave speeds.
    pub fn advance_to(&mut self, t_end: f64) -> Result<()> {
        while self.hydro.time() < t_end {
            let dt = self.stable_dt()?.min(t_end - self.hydro.time());
            if dt <= 0.0 || !dt.is_finite() {
                break;
            }
            self.advance(dt)?;
        }
        Ok(())
    }

    /// Position of the front on ring `j` (m): the laser-side edge of the
    /// pressure rise at half maximum, linearly interpolated.
    ///
    /// Same construction as M6c's, so the on-axis number is directly comparable
    /// to the 1-D one.
    pub fn front_position_at(&self, j: usize) -> Option<f64> {
        let n_x = self.hydro.n_x();
        let p = |i: usize| self.hydro.cell(j, i).to_primitive(&self.hydro.gas()).p;
        let p_max = (0..n_x).map(p).fold(f64::NEG_INFINITY, f64::max);
        if p_max < 2.0 * self.ambient.p {
            return None;
        }
        let level = self.ambient.p + 0.5 * (p_max - self.ambient.p);
        let i = (0..n_x).find(|&i| p(i) >= level)?;
        if i == 0 {
            return Some(self.hydro.x_centre(0));
        }
        let f = (level - p(i - 1)) / (p(i) - p(i - 1));
        Some(self.hydro.x_centre(i - 1) + f * self.hydro.dx())
    }

    /// Position of the front on the axis (m) — the number `D_2D` is measured
    /// from, and the one directly comparable to M6c.
    pub fn front_position(&self) -> Option<f64> {
        self.front_position_at(0)
    }

    /// How far the front at the beam edge lags the front on the axis (m).
    ///
    /// Positive when the axis leads, which is what relief produces: the edge of
    /// the beam is where gas escapes sideways, so it is driven least. The
    /// curvature *is* relief made visible, and it is a diagnostic and a figure
    /// rather than a gate.
    pub fn front_lag_at_beam_edge(&self) -> Option<f64> {
        let edge = self.beam.radius();
        let j = ((edge / self.hydro.dr()).floor() as usize).min(self.hydro.n_r() - 1);
        Some(self.front_position_at(j)? - self.front_position()?)
    }

    /// Advance by `span` (s) and return the mean on-axis front speed over it
    /// (m/s), **positive toward the laser**.
    pub fn measure_front_speed(&mut self, span: f64) -> Result<f64> {
        let Some(x0) = self.front_position() else {
            bail!(
                "lsd2d: no front to track at t = {:.4e} s (peak pressure has not reached \
                 twice ambient)",
                self.hydro.time()
            );
        };
        let t0 = self.hydro.time();
        self.advance_to(t0 + span)?;
        let Some(x1) = self.front_position() else {
            bail!("lsd2d: the front vanished during the {span:.4e} s measurement window");
        };
        Ok((x0 - x1) / (self.hydro.time() - t0))
    }

    /// Energy the beam has deposited so far, in the `r dr dx` measure (J/rad).
    pub fn deposited_energy(&self) -> f64 {
        self.deposited
    }

    /// Relative closure of the energy budget:
    /// `|(E_now + escaped − E_0) − deposited| / deposited`.
    ///
    /// The escape term is what lets this close on a run where relief actually
    /// reaches the wall, rather than only on the runs where nothing happens.
    pub fn energy_residual(&self) -> f64 {
        if self.deposited <= 0.0 {
            return 0.0;
        }
        let gained = self.hydro.total_energy() + self.hydro.escaped_energy() - self.initial_energy;
        (gained - self.deposited).abs() / self.deposited
    }

    /// Whether the **laser-side** end plane is still undisturbed.
    ///
    /// Only `x_min` is checked, and that is the physically meaningful choice
    /// rather than a relaxation. The front runs toward the laser, so `x_min` is
    /// the plane it approaches and the one whose disturbance would contaminate
    /// the speed. The downstream plane `x_max` is where the seed's own blast
    /// leaves, and it always does: a spark radiates both ways, and the outward
    /// half has no reason to stay in the domain. Reporting that as
    /// contamination would be crying wolf on every run.
    pub fn axial_boundaries_undisturbed(&self) -> bool {
        (0..self.hydro.n_r()).all(|j| !self.disturbed(self.hydro.cell(j, 0)))
    }

    /// Whether the downstream plane is still undisturbed — information, not a
    /// validity condition. See [`axial_boundaries_undisturbed`](Self::axial_boundaries_undisturbed).
    pub fn downstream_undisturbed(&self) -> bool {
        let n_x = self.hydro.n_x();
        (0..self.hydro.n_r()).all(|j| !self.disturbed(self.hydro.cell(j, n_x - 1)))
    }

    /// Whether the outermost ring is still undisturbed.
    ///
    /// **Not a validity condition on its own**, and saying so is the point. A
    /// radially uniform seed — which is what these runs use, so that no radial
    /// structure is present before the beam creates it — disturbs the rim at
    /// `t = 0` by construction. What matters is whether the *wall* is doing the
    /// relief, and that is answered by widening the domain: measured, the
    /// deficit is 21.1 % / 21.3 % / 21.3 % at domain radii of 3, 5 and 8 beam
    /// radii, i.e. converged by five and not a boundary effect.
    pub fn rim_undisturbed(&self) -> bool {
        let n_r = self.hydro.n_r();
        (0..self.hydro.n_x()).all(|i| !self.disturbed(self.hydro.cell(n_r - 1, i)))
    }

    fn disturbed(&self, c: Conserved2d) -> bool {
        let tol = 1e-6;
        let w = c.to_primitive(&self.hydro.gas());
        (w.p - self.ambient.p).abs() > tol * self.ambient.p
            || (w.rho - self.ambient.rho).abs() > tol * self.ambient.rho
    }

    /// Both of the above.
    pub fn boundaries_undisturbed(&self) -> bool {
        self.axial_boundaries_undisturbed()
            && self.rim_undisturbed()
            && self.downstream_undisturbed()
    }

    /// Refuse, don't mis-model — M6c's `check_regime` plus the three ways an
    /// axisymmetric run can produce a plausible wrong deficit.
    pub fn check_regime(&self) -> Result<()> {
        let (dr, dx) = (self.hydro.dr(), self.hydro.dx());
        let length = dx * self.hydro.n_x() as f64;
        let domain_radius = dr * self.hydro.n_r() as f64;
        let alpha_max = self.alpha_profile()?.into_iter().fold(0.0, f64::max);
        if alpha_max <= 0.0 {
            bail!(
                "lsd2d: nothing absorbs at t = {:.4e} s — there is no front, and the \
                 detonation model has nothing to describe",
                self.hydro.time()
            );
        }
        let absorption_length = 1.0 / alpha_max;
        let cells = absorption_length / dx;
        if cells < 5.0 {
            bail!(
                "lsd2d: the absorption length 1/α = {absorption_length:.4e} m spans only \
                 {cells:.2} cells (dx = {dx:.4e} m); the deposition is unresolved"
            );
        }
        if absorption_length > 0.1 * length {
            bail!(
                "lsd2d: the absorption length is {:.0}% of the {length:.4e} m domain; the \
                 deposition is volumetric rather than a front, which is the LSC regime \
                 and out of scope",
                100.0 * absorption_length / length
            );
        }

        // The three that are new in two dimensions.
        let r_b = self.beam.radius();
        let beam_cells = r_b / dr;
        if beam_cells < MIN_CELLS_PER_BEAM_RADIUS {
            bail!(
                "lsd2d: the beam radius {r_b:.4e} m spans only {beam_cells:.2} cells \
                 (dr = {dr:.4e} m), below the {MIN_CELLS_PER_BEAM_RADIUS} needed. The \
                 deposition profile is a staircase, and the relief deficit measured from \
                 it would be a property of the mesh"
            );
        }
        if domain_radius < MIN_DOMAIN_RADII * r_b {
            bail!(
                "lsd2d: the domain radius {domain_radius:.4e} m is under {MIN_DOMAIN_RADII}x \
                 the beam radius {r_b:.4e} m; relief must come from the beam being finite, \
                 not from the wall"
            );
        }
        if self.front_position().is_none() {
            bail!(
                "lsd2d: no front on the axis at t = {:.4e} s",
                self.hydro.time()
            );
        }
        // A curved front can leave through x_min on the axis while the edge is
        // still inside, and the on-axis measurement would silently be reading
        // the boundary.
        if let Some(x) = self.front_position()
            && x <= 2.0 * dx
        {
            bail!(
                "lsd2d: the on-axis front has reached x = {x:.4e} m, within two cells of \
                 the laser-side boundary; the measurement is reading the boundary"
            );
        }
        Ok(())
    }
}
