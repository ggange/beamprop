//! 1-D compressible Euler solver: HLLC + MUSCL-Hancock (M6c, step 1).
//!
//! ```text
//! ∂U/∂t + ∂F(U)/∂x = 0
//!
//! U = (ρ, ρu, E)ᵀ,   F = (ρu, ρu² + p, (E + p)u)ᵀ,   E = p/(γ−1) + ½ρu²
//! ```
//!
//! **There is no laser physics in this module, deliberately.** `docs/M6C_SPEC.md`
//! (gate decision 4) keeps the Riemann core laser-agnostic so its verification
//! gates — Sod against the exact Riemann solution (G1) and observed 2nd order on
//! smooth flow (G2) — test a general-purpose solver against standard closed
//! forms, uncontaminated by the model under test. Laser deposition, ignition,
//! and the plasma column live one layer up in `lsd.rs`; this module only exposes
//! [`Euler1d::step_with_source`] as the seam they attach to.
//!
//! Numerics, per the spec:
//!
//! - **HLLC** approximate Riemann solver with Einfeldt/Davis wave-speed
//!   estimates — the contact wave is resolved, which a plain HLL would smear,
//!   and the LSD front is exactly a contact-bearing structure.
//! - **MUSCL-Hancock** with a minmod limiter: 2nd order in smooth flow, TVD
//!   across the shock.
//! - **CFL asserted per step** from the current wave speeds, never a fixed `dt`.
//! - **Positivity guard** after every update. A negative `ρ` or `p` *bails with
//!   the cell and step*; it is never clamped. Clamping is how M6a's `n_e`
//!   runaway hid, and the failure-mode table in the spec pins this as loud.
//!
//! The EOS here is the ideal gas at constant `γ` — the verification mode of the
//! spec's two, the one the closed forms G1–G3 assume. The production table EOS
//! attaches later (D8) without touching the flux routines.

use anyhow::{Result, bail};

/// Ghost cells per side. MUSCL-Hancock reconstructs a slope from the immediate
/// neighbours and then solves a Riemann problem at each interface, so the
/// stencil reaches two cells beyond the domain.
pub const N_GHOST: usize = 2;

/// Ideal-gas EOS at constant `γ` (the spec's verification mode).
#[derive(Debug, Clone, Copy)]
pub struct IdealGas {
    /// Ratio of specific heats.
    pub gamma: f64,
}

impl IdealGas {
    /// Dry air at moderate temperature.
    pub const AIR: Self = Self { gamma: 1.4 };

    /// Sound speed `c = √(γp/ρ)` (m/s).
    pub fn sound_speed(&self, rho: f64, p: f64) -> f64 {
        (self.gamma * p / rho).sqrt()
    }

    /// Specific internal energy `e = p/((γ−1)ρ)` (J/kg).
    pub fn specific_internal_energy(&self, rho: f64, p: f64) -> f64 {
        p / ((self.gamma - 1.0) * rho)
    }
}

/// Primitive state `(ρ, u, p)` — SI: kg/m³, m/s, Pa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primitive {
    /// Density (kg/m³).
    pub rho: f64,
    /// Velocity (m/s).
    pub u: f64,
    /// Pressure (Pa).
    pub p: f64,
}

/// Conserved state `(ρ, ρu, E)` — SI: kg/m³, kg/(m²·s), J/m³.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conserved {
    /// Density (kg/m³).
    pub rho: f64,
    /// Momentum density `ρu` (kg/(m²·s)).
    pub mom: f64,
    /// Total energy density `E = ρe + ½ρu²` (J/m³).
    pub energy: f64,
}

impl Conserved {
    fn zip(self, other: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self {
            rho: f(self.rho, other.rho),
            mom: f(self.mom, other.mom),
            energy: f(self.energy, other.energy),
        }
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rho: f(self.rho),
            mom: f(self.mom),
            energy: f(self.energy),
        }
    }
}

impl Primitive {
    /// Convert to conserved variables under `gas`.
    pub fn to_conserved(self, gas: &IdealGas) -> Conserved {
        Conserved {
            rho: self.rho,
            mom: self.rho * self.u,
            energy: self.p / (gas.gamma - 1.0) + 0.5 * self.rho * self.u * self.u,
        }
    }
}

impl Conserved {
    /// Convert to primitive variables under `gas`.
    ///
    /// Unchecked: callers that can produce a non-physical state run
    /// [`Euler1d::assert_physical`] first, which is where the loud bail lives.
    pub fn to_primitive(self, gas: &IdealGas) -> Primitive {
        let u = self.mom / self.rho;
        Primitive {
            rho: self.rho,
            u,
            p: (gas.gamma - 1.0) * (self.energy - 0.5 * self.mom * u),
        }
    }
}

/// Physical flux `F(U)`.
pub(crate) fn flux(gas: &IdealGas, u: Conserved) -> Conserved {
    let w = u.to_primitive(gas);
    Conserved {
        rho: u.mom,
        mom: u.mom * w.u + w.p,
        energy: (u.energy + w.p) * w.u,
    }
}

/// What the solver does at the domain ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Zero-gradient outflow — the Sod setup, and the LSD domain's ends.
    Transmissive,
    /// Wrap-around, for the smooth-flow convergence gate.
    Periodic,
}

/// `true` for a strictly positive, finite number — and **false for NaN**.
///
/// That is the point of routing every validity check through it: written as
/// `!is_positive(x)`, a NaN state is refused, where the natural `x <= 0.0`
/// would wave it through.
pub(crate) fn is_positive(x: f64) -> bool {
    x > 0.0 && x.is_finite()
}

/// Minmod limiter on two slopes: the TVD choice, and the one that makes the
/// Sod contact and shock monotone.
pub(crate) fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

/// A uniform-mesh 1-D Euler domain and its state.
#[derive(Debug, Clone)]
pub struct Euler1d {
    gas: IdealGas,
    boundary: Boundary,
    dx: f64,
    x_min: f64,
    /// Interior cells only; ghosts are materialised per step.
    cells: Vec<Conserved>,
    cfl: f64,
    time: f64,
    step_count: usize,
}

impl Euler1d {
    /// Build from cell-centred primitive states on a uniform mesh of spacing
    /// `dx` starting at `x_min`.
    pub fn new(
        gas: IdealGas,
        boundary: Boundary,
        x_min: f64,
        dx: f64,
        initial: &[Primitive],
    ) -> Result<Self> {
        if initial.len() < 2 * N_GHOST {
            bail!(
                "euler1d: {} cells is below the {} the MUSCL stencil needs",
                initial.len(),
                2 * N_GHOST
            );
        }
        if !is_positive(dx) {
            bail!("euler1d: dx must be positive, got {dx}");
        }
        for (i, w) in initial.iter().enumerate() {
            if !is_positive(w.rho) || !is_positive(w.p) {
                bail!(
                    "euler1d: non-physical initial state in cell {i}: ρ = {}, p = {}",
                    w.rho,
                    w.p
                );
            }
        }
        Ok(Self {
            gas,
            boundary,
            dx,
            x_min,
            cells: initial.iter().map(|w| w.to_conserved(&gas)).collect(),
            cfl: 0.8,
            time: 0.0,
            step_count: 0,
        })
    }

    /// Sample `initial(x)` at the `n` cell centres of `[x_min, x_min + n·dx]`.
    pub fn from_fn(
        gas: IdealGas,
        boundary: Boundary,
        x_min: f64,
        dx: f64,
        n: usize,
        initial: impl Fn(f64) -> Primitive,
    ) -> Result<Self> {
        let states: Vec<Primitive> = (0..n)
            .map(|i| initial(x_min + (i as f64 + 0.5) * dx))
            .collect();
        Self::new(gas, boundary, x_min, dx, &states)
    }

    /// Courant number used to size each step. Capped at 0.8 (the spec's bound);
    /// values above it are rejected rather than quietly reduced.
    pub fn set_cfl(&mut self, cfl: f64) -> Result<()> {
        if !is_positive(cfl) || cfl > 0.8 {
            bail!("euler1d: CFL must be in (0, 0.8], got {cfl}");
        }
        self.cfl = cfl;
        Ok(())
    }

    /// Courant number in force.
    pub fn cfl(&self) -> f64 {
        self.cfl
    }

    /// Elapsed simulated time (s).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Completed hydro steps.
    pub fn steps(&self) -> usize {
        self.step_count
    }

    /// Number of interior cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the domain has no interior cells (never true after
    /// [`Euler1d::new`], which enforces the stencil width).
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Mesh spacing (m).
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// The EOS in force.
    pub fn gas(&self) -> IdealGas {
        self.gas
    }

    /// Centre coordinate of cell `i` (m).
    pub fn x_centre(&self, i: usize) -> f64 {
        self.x_min + (i as f64 + 0.5) * self.dx
    }

    /// Conserved state, interior cells.
    pub fn cells(&self) -> &[Conserved] {
        &self.cells
    }

    /// Primitive state, interior cells.
    pub fn primitives(&self) -> Vec<Primitive> {
        self.cells
            .iter()
            .map(|u| u.to_primitive(&self.gas))
            .collect()
    }

    /// Domain-integrated total energy `∫E dx` (J/m²) — the budget G5 closes.
    pub fn total_energy(&self) -> f64 {
        self.cells.iter().map(|u| u.energy).sum::<f64>() * self.dx
    }

    /// Domain-integrated mass `∫ρ dx` (kg/m²).
    pub fn total_mass(&self) -> f64 {
        self.cells.iter().map(|u| u.rho).sum::<f64>() * self.dx
    }

    /// Largest signal speed `max(|u| + c)` over the domain (m/s).
    pub fn max_wave_speed(&self) -> f64 {
        self.cells
            .iter()
            .map(|&u| {
                let w = u.to_primitive(&self.gas);
                w.u.abs() + self.gas.sound_speed(w.rho, w.p)
            })
            .fold(0.0, f64::max)
    }

    /// Stable step from the current wave speeds: `dt = CFL·dx/max(|u| + c)`.
    pub fn stable_dt(&self) -> Result<f64> {
        let s = self.max_wave_speed();
        if !is_positive(s) {
            bail!(
                "euler1d: no finite wave speed at step {} (max |u| + c = {s})",
                self.step_count
            );
        }
        Ok(self.cfl * self.dx / s)
    }

    /// Bail loudly on any non-physical cell, naming it and the step. The spec's
    /// failure-mode table: detect every update, never clamp.
    ///
    /// `first_index` is the **interior** index of `cells[0]`, because not every
    /// caller passes the interior array: the MUSCL stage checks a padded slice
    /// that starts one cell *before* the domain. Without it the bail names the
    /// wrong cell, which defeats the point of bailing loudly — the index is
    /// what you go and look at. Negative indices and indices ≥ `len()` are
    /// ghost cells and are reported as such.
    fn assert_physical(&self, cells: &[Conserved], stage: &str, first_index: isize) -> Result<()> {
        for (k, &u) in cells.iter().enumerate() {
            let w = u.to_primitive(&self.gas);
            if !is_positive(w.rho) || !is_positive(w.p) || !w.u.is_finite() {
                let i = k as isize + first_index;
                let ghost = if i < 0 || i >= self.cells.len() as isize {
                    " (ghost)"
                } else {
                    ""
                };
                bail!(
                    "euler1d: non-physical state after {stage} at step {}, cell {i}{ghost} \
                     (x = {:.6e} m, t = {:.6e} s): ρ = {:.6e}, u = {:.6e}, p = {:.6e}",
                    self.step_count,
                    self.x_min + (i as f64 + 0.5) * self.dx,
                    self.time,
                    w.rho,
                    w.u,
                    w.p
                );
            }
        }
        Ok(())
    }

    /// Interior cells padded with [`N_GHOST`] ghosts per side under the
    /// boundary condition.
    fn with_ghosts(&self) -> Vec<Conserved> {
        let n = self.cells.len();
        let mut padded = Vec::with_capacity(n + 2 * N_GHOST);
        for g in 0..N_GHOST {
            padded.push(match self.boundary {
                Boundary::Transmissive => self.cells[0],
                Boundary::Periodic => self.cells[n - N_GHOST + g],
            });
        }
        padded.extend_from_slice(&self.cells);
        for g in 0..N_GHOST {
            padded.push(match self.boundary {
                Boundary::Transmissive => self.cells[n - 1],
                Boundary::Periodic => self.cells[g],
            });
        }
        padded
    }
}

/// HLLC flux across the interface between left state `ul` and right `ur`.
///
/// Wave speeds are the Einfeldt/Davis estimates built on the Roe-averaged
/// velocity and sound speed, bounded by the one-sided characteristics:
/// `S_L = min(u_L − c_L, ũ − c̃)`, `S_R = max(u_R + c_R, ũ + c̃)`. Those are
/// the standard choice that keeps HLLC positivity-preserving for the
/// isolated Riemann problem (Toro §10.5–10.6).
///
/// A free function rather than a method, and `pub(crate)`, so that
/// `euler2d`'s directional sweeps can call the *same* Riemann
/// solver instead of carrying a second copy of it. That module needs one extra
/// fact, and it can read it off the returned flux rather than from a wider
/// signature: **the sign of the mass flux identifies the upwind side of the
/// contact.** In the star branches `F_ρ = ρ*_K·S*` with `ρ*_K > 0`, so
/// `sign(F_ρ) = sign(S*)`; in the supersonic branches `S_L ≥ 0` forces
/// `u_L > c_L > 0` and `S_R ≤ 0` forces `u_R < −c_R < 0`, so the sign still
/// points at the donor side. A transverse momentum component therefore rides
/// along as `F_ρ · v_upwind`, exactly (Toro §10.5: the transverse velocity is
/// constant across the acoustic waves within each star state).
pub(crate) fn hllc_flux(gas: &IdealGas, ul: Conserved, ur: Conserved) -> Conserved {
    let wl = ul.to_primitive(gas);
    let wr = ur.to_primitive(gas);
    let cl = gas.sound_speed(wl.rho, wl.p);
    let cr = gas.sound_speed(wr.rho, wr.p);

    // Roe averages (density-weighted), for the Einfeldt bound.
    let sl_rho = wl.rho.sqrt();
    let sr_rho = wr.rho.sqrt();
    let u_roe = (sl_rho * wl.u + sr_rho * wr.u) / (sl_rho + sr_rho);
    let hl = (ul.energy + wl.p) / wl.rho;
    let hr = (ur.energy + wr.p) / wr.rho;
    let h_roe = (sl_rho * hl + sr_rho * hr) / (sl_rho + sr_rho);
    let c_roe2 = (gas.gamma - 1.0) * (h_roe - 0.5 * u_roe * u_roe);
    let c_roe = if c_roe2 > 0.0 { c_roe2.sqrt() } else { 0.0 };

    let s_l = (wl.u - cl).min(u_roe - c_roe);
    let s_r = (wr.u + cr).max(u_roe + c_roe);

    if s_l >= 0.0 {
        return flux(gas, ul);
    }
    if s_r <= 0.0 {
        return flux(gas, ur);
    }

    // Contact speed (Toro Eq. 10.37).
    let ml = wl.rho * (s_l - wl.u);
    let mr = wr.rho * (s_r - wr.u);
    let s_star = (wr.p - wl.p + ml * wl.u - mr * wr.u) / (ml - mr);

    // Star state on the upwind side, and its flux F_K + S_K(U*_K − U_K).
    let star = |u: Conserved, w: Primitive, s_k: f64| -> Conserved {
        let coef = w.rho * (s_k - w.u) / (s_k - s_star);
        Conserved {
            rho: coef,
            mom: coef * s_star,
            energy: coef
                * (u.energy / w.rho + (s_star - w.u) * (s_star + w.p / (w.rho * (s_k - w.u)))),
        }
    };
    if s_star >= 0.0 {
        let u_star = star(ul, wl, s_l);
        flux(gas, ul).zip(u_star.zip(ul, |a, b| a - b), |f, du| f + s_l * du)
    } else {
        let u_star = star(ur, wr, s_r);
        flux(gas, ur).zip(u_star.zip(ur, |a, b| a - b), |f, du| f + s_r * du)
    }
}

impl Euler1d {
    /// MUSCL-Hancock interface fluxes for the whole domain: minmod-limited
    /// slopes, boundary-extrapolated values, a half-step evolution of each, then
    /// one HLLC solve per interface. Returns `n + 1` fluxes.
    fn interface_fluxes(&self, dt: f64) -> Result<Vec<Conserved>> {
        let padded = self.with_ghosts();
        let n_pad = padded.len();
        let half = 0.5 * dt / self.dx;

        // Reconstruct and half-step-evolve every cell that has both neighbours
        // (all but the outermost ghost on each side).
        let mut left_face = vec![padded[0]; n_pad];
        let mut right_face = vec![padded[0]; n_pad];
        for i in 1..n_pad - 1 {
            let back = padded[i].zip(padded[i - 1], |a, b| a - b);
            let fwd = padded[i + 1].zip(padded[i], |a, b| a - b);
            let slope = back.zip(fwd, minmod);
            let l = padded[i].zip(slope, |u, d| u - 0.5 * d);
            let r = padded[i].zip(slope, |u, d| u + 0.5 * d);
            // Hancock predictor: evolve the two faces by dt/2 with their own
            // fluxes. This is what lifts the scheme to 2nd order in time.
            let df = flux(&self.gas, l).zip(flux(&self.gas, r), |a, b| a - b);
            left_face[i] = l.zip(df.map(|d| half * d), |a, b| a + b);
            right_face[i] = r.zip(df.map(|d| half * d), |a, b| a + b);
        }
        // The slice starts at padded index 1, which is the last ghost cell —
        // interior index `1 - N_GHOST`.
        let first = 1 - N_GHOST as isize;
        self.assert_physical(
            &left_face[1..n_pad - 1],
            "MUSCL half-step (left face)",
            first,
        )?;
        self.assert_physical(
            &right_face[1..n_pad - 1],
            "MUSCL half-step (right face)",
            first,
        )?;

        // Interface j sits between padded cells N_GHOST-1+j and N_GHOST+j.
        Ok((0..=self.cells.len())
            .map(|j| {
                hllc_flux(
                    &self.gas,
                    right_face[N_GHOST - 1 + j],
                    left_face[N_GHOST + j],
                )
            })
            .collect())
    }

    /// Advance the homogeneous system by `dt`.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.step_with_source(dt, |_, _| 0.0)
    }

    /// Advance by `dt` with a volumetric energy source `source(i, x)` in W/m³,
    /// applied over the step.
    ///
    /// This is the seam the LSD driver attaches to — `α_pl(x)·I(x)` and nothing
    /// else enters the hydro. Kept here rather than in `lsd.rs` so the update
    /// and its positivity guard stay in one place; the module still knows
    /// nothing about where the energy comes from.
    ///
    /// # This method alone is only 1st-order accurate
    ///
    /// It folds the source into the conservative update, which is Godunov
    /// splitting. **Measured** on smooth flow with an active source, refining
    /// `dx` and `dt` together at fixed CFL: this gives observed order
    /// **0.88 / 1.02 / 1.07**, against **1.99 / 2.03 / 1.99** for the Strang
    /// sandwich
    ///
    /// ```text
    /// add_energy(dt/2) → step(dt) → recompute the source → add_energy(dt/2)
    /// ```
    ///
    /// which is what [`LsdColumn::advance`](crate::lsd::LsdColumn::advance)
    /// does and what gate G2b holds to. Reaching for this method because it is
    /// one call instead of four silently costs an order of accuracy, and
    /// nothing about the result looks wrong — it just converges half as fast.
    /// Prefer the driver; use this only where 1st order is genuinely acceptable
    /// and say so at the call site.
    pub fn step_with_source(&mut self, dt: f64, source: impl Fn(usize, f64) -> f64) -> Result<()> {
        if !is_positive(dt) {
            bail!(
                "euler1d: non-positive step dt = {dt} at step {}",
                self.step_count
            );
        }
        let limit = self.stable_dt()?;
        if dt > limit * (1.0 + 1e-12) {
            bail!(
                "euler1d: dt = {dt:.6e} s violates CFL at step {} (limit {limit:.6e} s, \
                 CFL = {}, dx = {:.6e} m)",
                self.step_count,
                self.cfl,
                self.dx
            );
        }

        let fluxes = self.interface_fluxes(dt)?;
        let lambda = dt / self.dx;
        let updated: Vec<Conserved> = (0..self.cells.len())
            .map(|i| {
                let div = fluxes[i + 1].zip(fluxes[i], |a, b| a - b);
                let mut u = self.cells[i].zip(div, |c, d| c - lambda * d);
                u.energy += dt * source(i, self.x_centre(i));
                u
            })
            .collect();

        self.assert_physical(&updated, "conservative update", 0)?;
        self.cells = updated;
        self.time += dt;
        self.step_count += 1;
        Ok(())
    }

    /// Apply the volumetric energy source `source(i, x)` (W/m³) for `dt`,
    /// **without** the flux update.
    ///
    /// This is the source half of a Strang split: the driver calls it for
    /// `dt/2`, then [`step`](Self::step) for `dt`, then it again for `dt/2`,
    /// which is what keeps the coupled scheme 2nd order (the spec's cadence,
    /// and the propagator's own discipline one dimension up). Deliberately does
    /// **not** advance [`time`](Self::time) or [`steps`](Self::steps) — a Strang
    /// sandwich is one step, not three — and deliberately does not check CFL,
    /// which constrains the flux update and not this.
    ///
    /// The positivity guard still runs: a source that empties a cell bails with
    /// the cell and step, exactly as [`step_with_source`](Self::step_with_source)
    /// does.
    pub fn add_energy(&mut self, dt: f64, source: impl Fn(usize, f64) -> f64) -> Result<()> {
        if !dt.is_finite() || dt < 0.0 {
            bail!(
                "euler1d: bad source interval dt = {dt} at step {}",
                self.step_count
            );
        }
        let updated: Vec<Conserved> = (0..self.cells.len())
            .map(|i| {
                let mut u = self.cells[i];
                u.energy += dt * source(i, self.x_centre(i));
                u
            })
            .collect();
        self.assert_physical(&updated, "energy source", 0)?;
        self.cells = updated;
        Ok(())
    }

    /// Run to `t_end`, sizing every step from the current wave speeds and
    /// clipping the last one to land exactly on `t_end`.
    pub fn advance_to(&mut self, t_end: f64) -> Result<()> {
        while self.time < t_end {
            let dt = self.stable_dt()?.min(t_end - self.time);
            self.step(dt)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sod_left() -> Primitive {
        Primitive {
            rho: 1.0,
            u: 0.0,
            p: 1.0,
        }
    }

    fn sod_right() -> Primitive {
        Primitive {
            rho: 0.125,
            u: 0.0,
            p: 0.1,
        }
    }

    fn sod(n: usize) -> Euler1d {
        Euler1d::from_fn(
            IdealGas::AIR,
            Boundary::Transmissive,
            0.0,
            1.0 / n as f64,
            n,
            |x| {
                if x < 0.5 { sod_left() } else { sod_right() }
            },
        )
        .unwrap()
    }

    #[test]
    fn primitive_conserved_roundtrip() {
        let gas = IdealGas::AIR;
        let w = Primitive {
            rho: 1.3,
            u: -420.0,
            p: 2.5e5,
        };
        let back = w.to_conserved(&gas).to_primitive(&gas);
        assert!((back.rho - w.rho).abs() < 1e-15);
        assert!((back.u - w.u).abs() < 1e-9);
        assert!((back.p - w.p).abs() / w.p < 1e-15);
    }

    #[test]
    fn uniform_flow_is_a_fixed_point() {
        // A constant state carries zero flux difference: the discrete operator
        // must leave it untouched to round-off. Catches sign and indexing slips
        // that a shock test can mask.
        let w = Primitive {
            rho: 1.2,
            u: 300.0,
            p: 1e5,
        };
        let mut s =
            Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 64, |_| w).unwrap();
        let dt = s.stable_dt().unwrap();
        s.step(dt).unwrap();
        for got in s.primitives() {
            assert!(
                (got.rho - w.rho).abs() / w.rho < 1e-14,
                "ρ drifted: {got:?}"
            );
            assert!(
                (got.u - w.u).abs() / w.u.abs() < 1e-12,
                "u drifted: {got:?}"
            );
            assert!((got.p - w.p).abs() / w.p < 1e-12, "p drifted: {got:?}");
        }
    }

    #[test]
    fn hllc_reduces_to_upwind_in_supersonic_flow() {
        // Both wave speeds one-signed → the flux is the upwind physical flux.
        let s = sod(16);
        let gas = IdealGas::AIR;
        let fast_right = Primitive {
            rho: 1.0,
            u: 2000.0,
            p: 1e5,
        }
        .to_conserved(&gas);
        let f = hllc_flux(&s.gas, fast_right, fast_right);
        let exact = flux(&gas, fast_right);
        assert!((f.rho - exact.rho).abs() / exact.rho.abs() < 1e-14);
        assert!((f.mom - exact.mom).abs() / exact.mom.abs() < 1e-14);
        assert!((f.energy - exact.energy).abs() / exact.energy.abs() < 1e-14);
    }

    #[test]
    fn periodic_advection_conserves_mass_and_energy() {
        let mut s = Euler1d::from_fn(
            IdealGas::AIR,
            Boundary::Periodic,
            0.0,
            1.0 / 128.0,
            128,
            |x| Primitive {
                rho: 1.0 + 0.2 * (2.0 * std::f64::consts::PI * x).sin(),
                u: 1.0,
                p: 1.0,
            },
        )
        .unwrap();
        let (m0, e0) = (s.total_mass(), s.total_energy());
        s.advance_to(0.2).unwrap();
        assert!((s.total_mass() - m0).abs() / m0 < 1e-13, "mass drift");
        assert!((s.total_energy() - e0).abs() / e0 < 1e-13, "energy drift");
    }

    #[test]
    fn sod_stays_monotone_and_bounded() {
        // TVD: no new extrema outside the initial range. Cheap here; the
        // quantitative accuracy gate (G1) lives in tests/validation.rs.
        let mut s = sod(200);
        s.advance_to(0.2).unwrap();
        for w in s.primitives() {
            assert!(
                w.rho > 0.9 * sod_right().rho && w.rho < 1.1 * sod_left().rho,
                "ρ out of range: {w:?}"
            );
            assert!(
                w.p > 0.0 && w.p < 1.1 * sod_left().p,
                "p out of range: {w:?}"
            );
        }
    }

    #[test]
    fn energy_source_lands_exactly_in_the_budget() {
        // With no flux gradient, a uniform source adds exactly Ṡ·dt·L.
        let w = Primitive {
            rho: 1.0,
            u: 0.0,
            p: 1e5,
        };
        let mut s =
            Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 32, |_| w).unwrap();
        let e0 = s.total_energy();
        let dt = s.stable_dt().unwrap();
        let s_dot = 1e9;
        s.step_with_source(dt, |_, _| s_dot).unwrap();
        let expected = e0 + s_dot * dt * 32.0 * 1e-3;
        assert!((s.total_energy() - expected).abs() / expected < 1e-14);
    }

    #[test]
    fn add_energy_deposits_without_advancing_the_clock() {
        // The source half-step of the driver's Strang split: energy lands, the
        // flux update does not run, and the step counter does not move.
        let w = Primitive {
            rho: 1.0,
            u: 0.0,
            p: 1e5,
        };
        let mut s =
            Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 32, |_| w).unwrap();
        let e0 = s.total_energy();
        s.add_energy(1e-9, |_, _| 1e12).unwrap();
        let expected = e0 + 1e12 * 1e-9 * 32.0 * 1e-3;
        assert!((s.total_energy() - expected).abs() / expected < 1e-14);
        assert_eq!(s.time(), 0.0, "add_energy advanced the clock");
        assert_eq!(s.steps(), 0, "add_energy counted a step");
        // Zero interval is a no-op, not an error: an empty Strang half-step.
        s.add_energy(0.0, |_, _| 1e12).unwrap();
        assert!((s.total_energy() - expected).abs() / expected < 1e-14);
    }

    #[test]
    fn add_energy_that_empties_a_cell_bails_loudly() {
        let mut s = Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 8, |_| {
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 1e5,
            }
        })
        .unwrap();
        let err = s.add_energy(1e-9, |_, _| -1e15).unwrap_err().to_string();
        assert!(err.contains("non-physical"), "unexpected error: {err}");
        assert!(err.contains("energy source"), "stage not named: {err}");
    }

    #[test]
    fn the_positivity_bail_names_the_cell_it_means() {
        // The whole point of bailing loudly is that the index is the one you go
        // and look at. The MUSCL stage checks a slice that starts one cell
        // *before* the domain, so it passes a negative first index; getting
        // that offset wrong silently misreports every blow-up.
        let s = sod(16);
        let gas = IdealGas::AIR;
        let good = Primitive {
            rho: 1.0,
            u: 0.0,
            p: 1.0,
        }
        .to_conserved(&gas);
        let mut cells = vec![good; 5];
        cells[3] = Conserved { rho: -1.0, ..good };

        // Interior array: element 3 is cell 3.
        let err = s
            .assert_physical(&cells, "test", 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cell 3"), "wrong cell named: {err}");
        assert!(
            err.contains(&format!("{:.6e}", s.x_centre(3))),
            "x does not match the named cell: {err}"
        );
        assert!(!err.contains("(ghost)"), "cell 3 is interior: {err}");

        // MUSCL slice: element 3 is interior cell 3 + (1 - N_GHOST) = 2.
        let err = s
            .assert_physical(&cells, "test", 1 - N_GHOST as isize)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cell 2"), "wrong cell named: {err}");
        assert!(
            err.contains(&format!("{:.6e}", s.x_centre(2))),
            "x does not match the named cell: {err}"
        );

        // And a genuine ghost is labelled rather than passed off as interior.
        let mut ghost = vec![good; 2];
        ghost[0] = Conserved { rho: -1.0, ..good };
        let err = s
            .assert_physical(&ghost, "test", 1 - N_GHOST as isize)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cell -1 (ghost)"), "ghost not flagged: {err}");
    }

    #[test]
    fn cfl_violation_bails() {
        let mut s = sod(64);
        let dt = s.stable_dt().unwrap();
        let err = s.step(10.0 * dt).unwrap_err().to_string();
        assert!(err.contains("violates CFL"), "unexpected error: {err}");
    }

    #[test]
    fn negative_initial_pressure_is_refused() {
        let err = Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 8, |_| {
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: -1.0,
            }
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-physical"), "unexpected error: {err}");
    }

    #[test]
    fn source_that_empties_a_cell_bails_loudly() {
        // A large negative source drives p < 0; the guard must name the cell
        // and the step rather than clamp.
        let mut s = Euler1d::from_fn(IdealGas::AIR, Boundary::Periodic, 0.0, 1e-3, 8, |_| {
            Primitive {
                rho: 1.0,
                u: 0.0,
                p: 1e5,
            }
        })
        .unwrap();
        let dt = s.stable_dt().unwrap();
        let err = s
            .step_with_source(dt, |_, _| -1e5 / dt * 10.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-physical"), "unexpected error: {err}");
        assert!(err.contains("cell"), "error does not name the cell: {err}");
    }
}
