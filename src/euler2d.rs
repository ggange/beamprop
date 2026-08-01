//! 2-D compressible Euler solver, planar or axisymmetric: HLLC + MUSCL-Hancock
//! with dimensional splitting (M6d, step 2).
//!
//! ```text
//! ∂U/∂t + (1/r^m)·∂(r^m·F_r)/∂r + ∂F_x/∂x = 0      m = 0 planar, 1 axisymmetric
//!
//! U   = (ρ, ρu_r, ρu_x, E)ᵀ
//! F_r = (ρu_r, ρu_r² + p, ρu_r u_x, (E + p)u_r)ᵀ
//! F_x = (ρu_x, ρu_r u_x, ρu_x² + p, (E + p)u_x)ᵀ
//! E   = p/(γ−1) + ½ρ(u_r² + u_x²)
//! ```
//!
//! **There is no laser physics in this module, deliberately** — `docs/M6D_SPEC.md`
//! gate decision 4, carried forward verbatim from M6c. The coupled axisymmetric
//! LSD column lives one layer up in `lsd2d.rs`; this module exposes
//! [`Euler2d::step_with_source`] and [`Euler2d::add_energy`] as the seams it
//! attaches to.
//!
//! # The Riemann solver is reused, not reimplemented
//!
//! Each directional sweep is the 1-D MUSCL-Hancock update, calling
//! `euler1d::hllc_flux` itself. A sweep carries a transverse momentum component
//! the 1-D state does not have, and it rides along exactly, with no second
//! Riemann solver:
//!
//! - **Pack** by removing the transverse kinetic energy, `E_∥ = E − ½ρv_t²`, so
//!   the 1-D solver sees a state whose pressure is the true 2-D pressure.
//! - **Unpack** with `F_{ρv_t} = F_ρ·v_t` and `F_E = F_{E_∥} + ½v_t²·F_ρ`, both
//!   exact identities (Toro §10.5: the transverse velocity is constant across
//!   the acoustic waves within each star state).
//! - The **upwind side** is read off `sign(F_ρ)`; see `hllc_flux`'s own comment
//!   for why that is exactly the side of the contact.
//!
//! # The axis is not a special case
//!
//! The `1/r` that makes axisymmetric codes delicate never appears. Cells are
//! annuli, interfaces carry area `A ∝ r`, and the axis interface has `A = 0`, so
//! nothing crosses `r = 0` by construction. The geometric source is written as
//! the **same expression** as the pressure part of the flux difference, so for a
//! radially uniform state the two cancel to *bit* precision and such a state is
//! an exact fixed point of the radial operator. See [`Geometry`] and
//! [`Euler2d::sweep_line`].
//!
//! # Splitting
//!
//! Strang, with the radial sweep outside: `R(dt/2) → X(dt) → R(dt/2)`. The order
//! is not arbitrary — it makes the planar, radially uniform limit an exact
//! identity in `R`, so `X(dt)` is bit-for-bit `Euler1d::step`, which is what
//! gate G9 asserts. Sweep order is never alternated between steps; that is 2nd
//! order only on average and would make the convergence gate read noise.
//!
//! Numerics otherwise follow `euler1d`: minmod-limited MUSCL-Hancock, CFL
//! asserted per step from the current wave speeds in **both** directions, and a
//! positivity guard that **bails with the cell and stage, never clamps**.

use anyhow::{Result, bail};

use crate::euler1d::{Conserved, IdealGas, N_GHOST, flux, hllc_flux, is_positive, minmod};

/// Which sweep a directional helper is serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    /// Along the beam axis `x`. Never area-weighted: on an annular cell the
    /// axial faces and the cell volume carry the same ring area, and it
    /// cancels.
    Axial,
    /// Along `r`. Area-weighted when [`Geometry::Axisymmetric`].
    Radial,
}

/// Planar slab or axisymmetric annuli.
///
/// The two differ only in the interface areas and cell volumes the radial sweep
/// uses, which is the point of routing both through one code path: the planar
/// case is then demonstrably the same arithmetic, and gate G9 can assert
/// bit-identity against `Euler1d` rather than a tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// `m = 0`. Radial interfaces have unit area; cell volume is `Δr`.
    Planar,
    /// `m = 1`. The radial interface at `r` has area `∝ r`; the annulus between
    /// `r_−` and `r_+` has volume `∝ (r_+² − r_−²)/2`.
    Axisymmetric,
}

impl Geometry {
    /// Area of the radial interface at radius `r`, in units where the planar
    /// case is 1.
    ///
    /// The axis interface (`r = 0`) therefore has **zero** area, which is what
    /// removes the `1/r` singularity: no flux can cross the axis, and no
    /// division by a vanishing radius is ever performed.
    fn face_area(self, r: f64) -> f64 {
        match self {
            Self::Planar => 1.0,
            Self::Axisymmetric => r,
        }
    }

    /// Volume of the ring `[r_−, r_+]`, in the same units.
    ///
    /// `(r_+² − r_−²)/2` rather than `r_j·Δr`: with this choice
    /// `(A_+ − A_−)/V` is *analytically* `1/r_j` at the arithmetic cell centre,
    /// so the geometric source is the exact volume average of `1/r` and is
    /// finite in the innermost ring without a floor or an epsilon.
    fn cell_volume(self, r_minus: f64, r_plus: f64) -> f64 {
        match self {
            Self::Planar => r_plus - r_minus,
            Self::Axisymmetric => 0.5 * (r_plus * r_plus - r_minus * r_minus),
        }
    }
}

/// What the solver does at a domain edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary2d {
    /// Zero-gradient outflow.
    Transmissive,
    /// Solid wall: normal momentum reflected, everything else mirrored.
    Reflective,
    /// The symmetry axis `r = 0`. Identical to [`Reflective`](Self::Reflective)
    /// in code, and named separately because it is a statement about the
    /// *coordinates* rather than about a physical wall — and because getting
    /// its parity wrong is the classic axisymmetric failure (gate G13).
    Axis,
}

impl Boundary2d {
    /// Whether a ghost fill mirrors the interior with a sign flip on the normal
    /// momentum.
    fn mirrors(self) -> bool {
        matches!(self, Self::Reflective | Self::Axis)
    }
}

/// Primitive state `(ρ, u_r, u_x, p)` — SI: kg/m³, m/s, m/s, Pa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primitive2d {
    /// Density (kg/m³).
    pub rho: f64,
    /// Radial velocity (m/s).
    pub u_r: f64,
    /// Axial velocity (m/s).
    pub u_x: f64,
    /// Pressure (Pa).
    pub p: f64,
}

/// Conserved state `(ρ, ρu_r, ρu_x, E)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conserved2d {
    /// Density (kg/m³).
    pub rho: f64,
    /// Radial momentum density (kg/(m²·s)).
    pub mom_r: f64,
    /// Axial momentum density (kg/(m²·s)).
    pub mom_x: f64,
    /// Total energy density (J/m³).
    pub energy: f64,
}

impl Conserved2d {
    fn zip(self, other: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self {
            rho: f(self.rho, other.rho),
            mom_r: f(self.mom_r, other.mom_r),
            mom_x: f(self.mom_x, other.mom_x),
            energy: f(self.energy, other.energy),
        }
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rho: f(self.rho),
            mom_r: f(self.mom_r),
            mom_x: f(self.mom_x),
            energy: f(self.energy),
        }
    }

    /// `a·self − b·other`, componentwise: the area-weighted flux difference.
    ///
    /// Written this way — areas multiplying *inside* the difference — rather
    /// than as `(self − other)` scaled afterwards, because with unit areas
    /// `1.0 * F` is exactly `F` and the whole expression collapses to
    /// `euler1d`'s. That is what makes the planar path bit-identical.
    fn weighted_difference(self, a: f64, other: Self, b: f64) -> Self {
        Self {
            rho: a * self.rho - b * other.rho,
            mom_r: a * self.mom_r - b * other.mom_r,
            mom_x: a * self.mom_x - b * other.mom_x,
            energy: a * self.energy - b * other.energy,
        }
    }

    /// Convert to primitive variables under `gas`.
    ///
    /// Unchecked, like its 1-D counterpart: callers that can produce a
    /// non-physical state run the guard first.
    pub fn to_primitive(self, gas: &IdealGas) -> Primitive2d {
        let u_r = self.mom_r / self.rho;
        let u_x = self.mom_x / self.rho;
        Primitive2d {
            rho: self.rho,
            u_r,
            u_x,
            p: (gas.gamma - 1.0) * (self.energy - 0.5 * (self.mom_r * u_r + self.mom_x * u_x)),
        }
    }
}

impl Primitive2d {
    /// Convert to conserved variables under `gas`.
    pub fn to_conserved(self, gas: &IdealGas) -> Conserved2d {
        Conserved2d {
            rho: self.rho,
            mom_r: self.rho * self.u_r,
            mom_x: self.rho * self.u_x,
            energy: self.p / (gas.gamma - 1.0)
                + 0.5 * self.rho * (self.u_r * self.u_r + self.u_x * self.u_x),
        }
    }
}

/// Split a 2-D state into the 1-D triple the shared Riemann solver takes, plus
/// the transverse velocity that rides along.
///
/// The transverse kinetic energy is **removed** from the energy, so the packed
/// state's pressure under `euler1d`'s own `to_primitive` is the true 2-D
/// pressure. Without that the sweep would silently solve a different problem —
/// every flux would carry a pressure short of the transverse contribution.
fn pack(u: Conserved2d, dir: Dir) -> (Conserved, f64) {
    let (par, tr) = match dir {
        Dir::Axial => (u.mom_x, u.mom_r),
        Dir::Radial => (u.mom_r, u.mom_x),
    };
    let v_t = tr / u.rho;
    (
        Conserved {
            rho: u.rho,
            mom: par,
            energy: u.energy - 0.5 * tr * v_t,
        },
        v_t,
    )
}

/// Reassemble a directional flux from the 1-D solver's answer and the upwind
/// transverse velocity. Both identities are exact — see the module header.
fn unpack_flux(f: Conserved, v_t: f64, dir: Dir) -> Conserved2d {
    let mom_t = f.rho * v_t;
    let energy = f.energy + 0.5 * v_t * mom_t;
    match dir {
        Dir::Axial => Conserved2d {
            rho: f.rho,
            mom_r: mom_t,
            mom_x: f.mom,
            energy,
        },
        Dir::Radial => Conserved2d {
            rho: f.rho,
            mom_r: f.mom,
            mom_x: mom_t,
            energy,
        },
    }
}

/// Physical flux `F_dir(U)`.
fn dir_flux(gas: &IdealGas, u: Conserved2d, dir: Dir) -> Conserved2d {
    let (c, v_t) = pack(u, dir);
    unpack_flux(flux(gas, c), v_t, dir)
}

/// HLLC flux across an interface, the transverse component carried by the
/// upwind side of the contact.
fn dir_hllc(gas: &IdealGas, ul: Conserved2d, ur: Conserved2d, dir: Dir) -> Conserved2d {
    let (cl, v_tl) = pack(ul, dir);
    let (cr, v_tr) = pack(ur, dir);
    let f = hllc_flux(gas, cl, cr);
    let v_t = if f.rho >= 0.0 { v_tl } else { v_tr };
    unpack_flux(f, v_t, dir)
}

/// Radius of a radial interface, with ghost interfaces **reflected about the
/// wall** rather than continued past it.
///
/// This is a correctness requirement, not a refinement, and it is the one place
/// where axisymmetry differs from a Cartesian code in a way that is easy to get
/// silently wrong.
///
/// A mirroring boundary works because the ghost's reconstruction is the exact
/// mirror of the interior cell's, so the interface Riemann problem is
/// symmetric, its contact speed is zero, and no mass crosses. In cylindrical
/// geometry that symmetry also has to hold for the *metric*: the mirror of the
/// ring `[r_w − Δr, r_w]` is the ring `[r_w, r_w + Δr]` only if the ghost's
/// faces carry the mirrored areas. Continuing the radius past the wall instead
/// gives the ghost a larger area than its mirror, the Hancock predictor then
/// evolves the two sides differently, the contact speed is no longer zero, and
/// the wall leaks. **Measured with the naive metric: 3.3×10⁻⁴ of the mass over
/// 30 steps of the closed-box test, and growing.**
///
/// At the axis `r_w = 0`, so this reduces to `|r|` — including the zero-area
/// face the first ring and its ghost share.
fn ghost_face_radius(raw: f64, r_min: f64, r_max: f64, lo: Boundary2d, hi: Boundary2d) -> f64 {
    if lo.mirrors() && raw < r_min {
        2.0 * r_min - raw
    } else if hi.mirrors() && raw > r_max {
        2.0 * r_max - raw
    } else {
        raw
    }
}

/// Pressure of a 2-D state, obtained through the *packed* 1-D state so the
/// geometric source and the momentum flux are built from the same number by the
/// same route.
fn pressure(gas: &IdealGas, u: Conserved2d, dir: Dir) -> f64 {
    pack(u, dir).0.to_primitive(gas).p
}

/// A uniform-mesh 2-D Euler domain and its state.
///
/// Cells are stored row-major: ring `j`, axial cell `i`, at `j·n_x + i`.
#[derive(Debug, Clone)]
pub struct Euler2d {
    gas: IdealGas,
    geometry: Geometry,
    bc_r_min: Boundary2d,
    bc_r_max: Boundary2d,
    bc_x_min: Boundary2d,
    bc_x_max: Boundary2d,
    n_r: usize,
    n_x: usize,
    dr: f64,
    dx: f64,
    r_min: f64,
    x_min: f64,
    cells: Vec<Conserved2d>,
    /// Interface areas, `n_r + 1` of them, for the radial sweep.
    r_areas: Vec<f64>,
    /// Ring volumes in the `r dr` measure, `n_r` of them.
    r_volumes: Vec<f64>,
    cfl: f64,
    time: f64,
    step_count: usize,
    escaped_energy: f64,
}

impl Euler2d {
    /// Build from cell-centred primitive states on a uniform `(r, x)` mesh.
    ///
    /// `initial` is row-major: `initial[j·n_x + i]`. `boundaries` is
    /// `[r_min, r_max, x_min, x_max]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gas: IdealGas,
        geometry: Geometry,
        boundaries: [Boundary2d; 4],
        r_min: f64,
        x_min: f64,
        dr: f64,
        dx: f64,
        n_r: usize,
        n_x: usize,
        initial: &[Primitive2d],
    ) -> Result<Self> {
        if n_r < 2 * N_GHOST || n_x < 2 * N_GHOST {
            bail!(
                "euler2d: {n_r} x {n_x} cells is below the {} per direction the MUSCL stencil needs",
                2 * N_GHOST
            );
        }
        if initial.len() != n_r * n_x {
            bail!(
                "euler2d: got {} initial states for a {n_r} x {n_x} mesh",
                initial.len()
            );
        }
        if !is_positive(dr) || !is_positive(dx) {
            bail!("euler2d: dr and dx must be positive, got dr = {dr}, dx = {dx}");
        }
        if r_min < 0.0 {
            bail!("euler2d: r_min must be non-negative, got {r_min}");
        }
        if geometry == Geometry::Axisymmetric && r_min == 0.0 && boundaries[0] != Boundary2d::Axis {
            bail!(
                "euler2d: an axisymmetric domain reaching r = 0 must use Boundary2d::Axis there, \
                 got {:?}",
                boundaries[0]
            );
        }
        for (k, w) in initial.iter().enumerate() {
            if !is_positive(w.rho) || !is_positive(w.p) {
                bail!(
                    "euler2d: non-physical initial state in cell (j = {}, i = {}): ρ = {}, p = {}",
                    k / n_x,
                    k % n_x,
                    w.rho,
                    w.p
                );
            }
        }
        // Padded metrics: index `k` is padded cell `k`, i.e. interior ring
        // `k − N_GHOST`. Ghosts carry their **own** areas rather than borrowing
        // a neighbour's — see `ghost_face_radius` for why that is a correctness
        // requirement and not a refinement.
        let r_max = r_min + n_r as f64 * dr;
        let face_radius = |k: isize| {
            ghost_face_radius(
                r_min + k as f64 * dr,
                r_min,
                r_max,
                boundaries[0],
                boundaries[1],
            )
        };
        let r_areas = (0..=(n_r + 2 * N_GHOST))
            .map(|k| geometry.face_area(face_radius(k as isize - N_GHOST as isize)))
            .collect();
        let r_volumes = (0..(n_r + 2 * N_GHOST))
            .map(|k| {
                let lo = face_radius(k as isize - N_GHOST as isize);
                let hi = face_radius(k as isize + 1 - N_GHOST as isize);
                geometry.cell_volume(lo.min(hi), lo.max(hi))
            })
            .collect();
        Ok(Self {
            gas,
            geometry,
            bc_r_min: boundaries[0],
            bc_r_max: boundaries[1],
            bc_x_min: boundaries[2],
            bc_x_max: boundaries[3],
            n_r,
            n_x,
            dr,
            dx,
            r_min,
            x_min,
            cells: initial.iter().map(|w| w.to_conserved(&gas)).collect(),
            r_areas,
            r_volumes,
            cfl: 0.8,
            time: 0.0,
            step_count: 0,
            escaped_energy: 0.0,
        })
    }

    /// Sample `initial(r, x)` at the cell centres.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fn(
        gas: IdealGas,
        geometry: Geometry,
        boundaries: [Boundary2d; 4],
        r_min: f64,
        x_min: f64,
        dr: f64,
        dx: f64,
        n_r: usize,
        n_x: usize,
        initial: impl Fn(f64, f64) -> Primitive2d,
    ) -> Result<Self> {
        let states: Vec<Primitive2d> = (0..n_r * n_x)
            .map(|k| {
                let (j, i) = (k / n_x, k % n_x);
                initial(r_min + (j as f64 + 0.5) * dr, x_min + (i as f64 + 0.5) * dx)
            })
            .collect();
        Self::new(
            gas, geometry, boundaries, r_min, x_min, dr, dx, n_r, n_x, &states,
        )
    }

    /// Courant number used to size each step. Capped at 0.8 as in `euler1d`;
    /// values above it are rejected rather than quietly reduced.
    pub fn set_cfl(&mut self, cfl: f64) -> Result<()> {
        if !is_positive(cfl) || cfl > 0.8 {
            bail!("euler2d: CFL must be in (0, 0.8], got {cfl}");
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

    /// Rings.
    pub fn n_r(&self) -> usize {
        self.n_r
    }

    /// Axial cells.
    pub fn n_x(&self) -> usize {
        self.n_x
    }

    /// Radial spacing (m).
    pub fn dr(&self) -> f64 {
        self.dr
    }

    /// Axial spacing (m).
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// The EOS in force.
    pub fn gas(&self) -> IdealGas {
        self.gas
    }

    /// The geometry in force.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    /// Centre radius of ring `j` (m).
    pub fn r_centre(&self, j: usize) -> f64 {
        self.r_min + (j as f64 + 0.5) * self.dr
    }

    /// Centre coordinate of axial cell `i` (m).
    pub fn x_centre(&self, i: usize) -> f64 {
        self.x_min + (i as f64 + 0.5) * self.dx
    }

    /// Conserved state, interior cells, row-major.
    pub fn cells(&self) -> &[Conserved2d] {
        &self.cells
    }

    /// Conserved state of cell `(j, i)`.
    pub fn cell(&self, j: usize, i: usize) -> Conserved2d {
        self.cells[j * self.n_x + i]
    }

    /// Primitive state, interior cells, row-major.
    pub fn primitives(&self) -> Vec<Primitive2d> {
        self.cells
            .iter()
            .map(|u| u.to_primitive(&self.gas))
            .collect()
    }

    /// Domain integral of a per-cell quantity in the discrete `r dr dx`
    /// measure — the measure in which mass, axial momentum and energy are
    /// exactly conserved.
    fn integrate(&self, f: impl Fn(Conserved2d) -> f64) -> f64 {
        (0..self.n_r)
            .map(|j| {
                let vol = self.r_volumes[N_GHOST + j] * self.dx;
                (0..self.n_x)
                    .map(|i| f(self.cells[j * self.n_x + i]))
                    .sum::<f64>()
                    * vol
            })
            .sum()
    }

    /// Domain-integrated mass `∫ρ r dr dx`.
    pub fn total_mass(&self) -> f64 {
        self.integrate(|u| u.rho)
    }

    /// Domain-integrated total energy `∫E r dr dx`.
    pub fn total_energy(&self) -> f64 {
        self.integrate(|u| u.energy)
    }

    /// Domain-integrated axial momentum `∫ρu_x r dr dx` — conserved.
    pub fn total_axial_momentum(&self) -> f64 {
        self.integrate(|u| u.mom_x)
    }

    /// Domain-integrated radial momentum `∫ρu_r r dr dx`.
    ///
    /// **Deliberately not conserved** in axisymmetry: the geometric source is a
    /// real term, and an implementation that conserved this would be wrong.
    /// Gate G12 asserts both halves of that sentence.
    pub fn total_radial_momentum(&self) -> f64 {
        self.integrate(|u| u.mom_r)
    }

    /// Energy that has left through transmissive boundaries, in the same
    /// measure as [`total_energy`](Self::total_energy).
    ///
    /// Without this the budget is only closable on runs where nothing escapes,
    /// which is exactly the case that proves nothing — hence G12's second leg.
    pub fn escaped_energy(&self) -> f64 {
        self.escaped_energy
    }

    /// Largest signal speed in each direction, `max(|u_d| + c)` (m/s).
    pub fn max_wave_speeds(&self) -> (f64, f64) {
        self.cells.iter().fold((0.0f64, 0.0f64), |(sr, sx), &u| {
            let w = u.to_primitive(&self.gas);
            let c = self.gas.sound_speed(w.rho, w.p);
            (sr.max(w.u_r.abs() + c), sx.max(w.u_x.abs() + c))
        })
    }

    /// Stable step from the current wave speeds in **both** directions:
    /// `dt = CFL / max(s_r/Δr, s_x/Δx)`.
    pub fn stable_dt(&self) -> Result<f64> {
        let (s_r, s_x) = self.max_wave_speeds();
        let rate = (s_r / self.dr).max(s_x / self.dx);
        if !is_positive(rate) {
            bail!(
                "euler2d: no finite wave speed at step {} (max |u_r| + c = {s_r}, \
                 max |u_x| + c = {s_x})",
                self.step_count
            );
        }
        Ok(self.cfl / rate)
    }

    /// Bail loudly on any non-physical cell, naming the cell, its physical
    /// position, the step **and the stage**. Never clamps.
    ///
    /// The stage matters more here than in 1-D because there are three places a
    /// state can go bad and they mean different things: an `r-sweep` failure
    /// points at the axis or the geometric source, an `x-sweep` failure at the
    /// same physics `euler1d` would have hit, and a `geometric source` failure
    /// at the genuinely new mode — that source adds radial momentum at fixed
    /// total energy, so it raises kinetic energy and can drive `p < 0` in the
    /// innermost rings of a strong converging flow.
    fn assert_physical(
        &self,
        cells: &[Conserved2d],
        stage: &str,
        locate: impl Fn(usize) -> (usize, usize),
    ) -> Result<()> {
        for (k, &u) in cells.iter().enumerate() {
            let w = u.to_primitive(&self.gas);
            if !is_positive(w.rho) || !is_positive(w.p) || !w.u_r.is_finite() || !w.u_x.is_finite()
            {
                let (j, i) = locate(k);
                bail!(
                    "euler2d: non-physical state after {stage} at step {}, cell \
                     (j = {j}, i = {i}) (r = {:.6e} m, x = {:.6e} m, t = {:.6e} s): \
                     ρ = {:.6e}, u_r = {:.6e}, u_x = {:.6e}, p = {:.6e}",
                    self.step_count,
                    self.r_centre(j.min(self.n_r - 1)),
                    self.x_centre(i.min(self.n_x - 1)),
                    self.time,
                    w.rho,
                    w.u_r,
                    w.u_x,
                    w.p
                );
            }
        }
        Ok(())
    }

    /// Pad a line of states with [`N_GHOST`] ghosts per side.
    ///
    /// `Reflective` and `Axis` mirror the line and **flip the sign of the
    /// normal momentum**. That odd parity is the whole axis condition, and
    /// getting it wrong is the classic axisymmetric failure: an even-parity
    /// fill produces a thin, artificially hot, under-dense column on the axis
    /// that grows with time and looks exactly like a physical result. Gate G13
    /// checks it against a deliberate even-parity contrast, which is why
    /// `mirror_flips_normal_momentum` is a parameter rather than a constant.
    fn pad_line(
        &self,
        line: &[Conserved2d],
        dir: Dir,
        lo: Boundary2d,
        hi: Boundary2d,
        mirror_flips_normal_momentum: bool,
    ) -> Vec<Conserved2d> {
        let n = line.len();
        let flip = |u: Conserved2d| {
            if !mirror_flips_normal_momentum {
                return u;
            }
            match dir {
                Dir::Axial => Conserved2d {
                    mom_x: -u.mom_x,
                    ..u
                },
                Dir::Radial => Conserved2d {
                    mom_r: -u.mom_r,
                    ..u
                },
            }
        };
        let mut padded = Vec::with_capacity(n + 2 * N_GHOST);
        for g in 0..N_GHOST {
            // Ghost g (outermost first) mirrors interior cell N_GHOST-1-g.
            padded.push(if lo.mirrors() {
                flip(line[N_GHOST - 1 - g])
            } else {
                line[0]
            });
        }
        padded.extend_from_slice(line);
        for g in 0..N_GHOST {
            padded.push(if hi.mirrors() {
                flip(line[n - 1 - g])
            } else {
                line[n - 1]
            });
        }
        padded
    }

    /// One MUSCL-Hancock sweep along `dir` over a single line of cells.
    ///
    /// `areas` holds the `n + 1` interface areas and `volumes` the `n` cell
    /// volumes. For every planar sweep these are `1.0` and `h`, and the
    /// arithmetic below then reduces **exactly** to `Euler1d`'s: `1.0 * F` is
    /// `F`, and `dt / h` is the same `lambda`. That is what makes gate G9 an
    /// equality rather than a tolerance, and it is why the area weights
    /// multiply *inside* the flux difference while the volume divides *through
    /// the scalar*.
    ///
    /// Returns the updated line and the energy that crossed its two end
    /// interfaces, per unit of the transverse measure.
    #[allow(clippy::too_many_arguments)]
    fn sweep_line(
        &self,
        line: &[Conserved2d],
        dt: f64,
        dir: Dir,
        lo: Boundary2d,
        hi: Boundary2d,
        areas: &[f64],
        volumes: &[f64],
        geometric: bool,
        mirror_flips: bool,
        locate: impl Fn(usize) -> (usize, usize) + Copy,
    ) -> Result<(Vec<Conserved2d>, f64)> {
        let n = line.len();
        let padded = self.pad_line(line, dir, lo, hi, mirror_flips);
        let n_pad = padded.len();
        let gas = &self.gas;

        let mut left_face = vec![padded[0]; n_pad];
        let mut right_face = vec![padded[0]; n_pad];
        let mut half_cell = vec![padded[0]; n_pad];
        for k in 1..n_pad - 1 {
            let back = padded[k].zip(padded[k - 1], |a, b| a - b);
            let fwd = padded[k + 1].zip(padded[k], |a, b| a - b);
            let slope = back.zip(fwd, minmod);
            let l = padded[k].zip(slope, |u, d| u - 0.5 * d);
            let r = padded[k].zip(slope, |u, d| u + 0.5 * d);

            // Each padded cell carries its OWN metric, ghosts included. Letting
            // ghosts borrow the nearest interior cell's areas is a real bug and
            // not a harmless approximation: a reflective wall works because the
            // ghost's reconstruction is the exact mirror of the interior cell's,
            // and a mismatched metric in the Hancock predictor destroys that
            // antisymmetry, so the wall leaks. Measured before the fix: 3.4e-4
            // of the mass, growing, on the closed-box test below.
            let (a_lo, a_hi, vol) = (areas[k], areas[k + 1], volumes[k]);
            let half = 0.5 * dt / vol;

            // Hancock predictor. `a_lo·F(l) − a_hi·F(r)` mirrors `euler1d`'s
            // `F(l) − F(r)` and is bit-identical to it when the areas are 1.
            let mut df =
                dir_flux(gas, l, dir).weighted_difference(a_lo, dir_flux(gas, r, dir), a_hi);
            if geometric {
                // The geometric source uses the **cell** pressure, written as
                // `a_hi·p − a_lo·p` so that for a radially uniform state it is
                // the exact negative of the pressure part of the flux
                // difference above and the pair vanishes bit-exactly.
                //
                // Using the *face* pressures here instead — `a_hi·p_R −
                // a_lo·p_L` — is well-balanced too, and wrong. Expanded against
                // the flux difference the pressure terms then cancel
                // identically, deleting the pressure gradient from the
                // predictor and leaving a leading-order error. It costs an
                // order: measured 0.86 / 1.12 against the planar path's
                // 1.71 / 1.89 on the same problem, which is how it was found.
                let p_cell = pressure(gas, padded[k], dir);
                df.mom_r += a_hi * p_cell - a_lo * p_cell;
            }
            let increment = df.map(|d| half * d);
            left_face[k] = l.zip(increment, |a, b| a + b);
            right_face[k] = r.zip(increment, |a, b| a + b);
            // The half-step cell average, for a time-centred geometric source.
            half_cell[k] = left_face[k].zip(right_face[k], |a, b| 0.5 * (a + b));
        }
        let stage = match dir {
            Dir::Axial => "x-sweep MUSCL half-step",
            Dir::Radial => "r-sweep MUSCL half-step",
        };
        let ghost_locate = |k: usize| locate(k.saturating_sub(1));
        self.assert_physical(&left_face[1..n_pad - 1], stage, ghost_locate)?;
        self.assert_physical(&right_face[1..n_pad - 1], stage, ghost_locate)?;

        let fluxes: Vec<Conserved2d> = (0..=n)
            .map(|k| {
                dir_hllc(
                    gas,
                    right_face[N_GHOST - 1 + k],
                    left_face[N_GHOST + k],
                    dir,
                )
            })
            .collect();

        // Interior cell k is padded cell N_GHOST + k, and its two interfaces are
        // padded faces N_GHOST + k and N_GHOST + k + 1.
        let g = N_GHOST;
        let mut escaped = 0.0;
        if lo == Boundary2d::Transmissive {
            escaped -= fluxes[0].energy * areas[g] * dt;
        }
        if hi == Boundary2d::Transmissive {
            escaped += fluxes[n].energy * areas[g + n] * dt;
        }

        let updated: Vec<Conserved2d> = (0..n)
            .map(|k| {
                let lambda = dt / volumes[g + k];
                let mut div =
                    fluxes[k + 1].weighted_difference(areas[g + k + 1], fluxes[k], areas[g + k]);
                if geometric {
                    // Same grouping once more, and subtracted *inside* the
                    // divergence rather than added to the result afterwards: a
                    // uniform state then gives `div.mom_r == 0.0` exactly, so
                    // `u − λ·0` is `u`, bit for bit. Adding the source as a
                    // separate term would leave `(u − y) + y`, which is not
                    // generally `u`.
                    let p_src = pressure(gas, half_cell[g + k], dir);
                    div.mom_r -= areas[g + k + 1] * p_src - areas[g + k] * p_src;
                }
                line[k].zip(div, |c, d| c - lambda * d)
            })
            .collect();
        let stage = match dir {
            Dir::Axial => "x-sweep conservative update",
            Dir::Radial => "r-sweep conservative update",
        };
        self.assert_physical(&updated, stage, locate)?;
        Ok((updated, escaped))
    }

    /// Sweep every ring along `x`.
    fn sweep_x(&mut self, dt: f64) -> Result<()> {
        // Unit areas everywhere, ghosts included: on an annular cell the axial
        // faces and the cell volume both carry the ring area, and it cancels.
        let areas = vec![1.0; self.n_x + 1 + 2 * N_GHOST];
        let volumes = vec![self.dx; self.n_x + 2 * N_GHOST];
        let mut escaped = 0.0;
        for j in 0..self.n_r {
            let line: Vec<Conserved2d> = self.cells[j * self.n_x..(j + 1) * self.n_x].to_vec();
            let (updated, esc) = self.sweep_line(
                &line,
                dt,
                Dir::Axial,
                self.bc_x_min,
                self.bc_x_max,
                &areas,
                &volumes,
                false,
                true,
                |i| (j, i),
            )?;
            // The axial end faces of ring `j` carry its ring area.
            escaped += esc * self.r_volumes[N_GHOST + j];
            self.cells[j * self.n_x..(j + 1) * self.n_x].copy_from_slice(&updated);
        }
        self.escaped_energy += escaped;
        Ok(())
    }

    /// Sweep every axial column along `r`.
    fn sweep_r(&mut self, dt: f64, mirror_flips: bool, well_balanced: bool) -> Result<()> {
        let geometric = well_balanced && self.geometry == Geometry::Axisymmetric;
        let areas = self.r_areas.clone();
        let volumes = self.r_volumes.clone();
        let mut escaped = 0.0;
        for i in 0..self.n_x {
            let line: Vec<Conserved2d> = (0..self.n_r)
                .map(|j| self.cells[j * self.n_x + i])
                .collect();
            let (updated, esc) = self.sweep_line(
                &line,
                dt,
                Dir::Radial,
                self.bc_r_min,
                self.bc_r_max,
                &areas,
                &volumes,
                geometric,
                mirror_flips,
                |j| (j, i),
            )?;
            // The radial faces of column `i` span its axial extent.
            escaped += esc * self.dx;
            for (j, &u) in updated.iter().enumerate() {
                self.cells[j * self.n_x + i] = u;
            }
        }
        self.escaped_energy += escaped;
        Ok(())
    }

    /// Advance the homogeneous system by `dt` — Strang, radial sweep outside.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.step_inner(dt, true, true)
    }

    /// [`step`](Self::step) with the axis ghost parity deliberately broken.
    ///
    /// Exists **only** so gate G13 has a non-vacuous contrast: without a run
    /// that gets the parity wrong, "the axis does not heat" is a claim no
    /// measurement has ever been able to fail. Never call it from a model.
    pub fn step_with_even_parity_axis_for_gate_contrast(&mut self, dt: f64) -> Result<()> {
        self.step_inner(dt, false, true)
    }

    /// [`step`](Self::step) with the geometric source **split out of the sweep**
    /// and applied as a separate forward-Euler update.
    ///
    /// The deliberately worse scheme, and it exists for the same reason
    /// `euler1d::step_with_source` does: an order gate that has never seen a
    /// first-order run cannot tell 2nd order from a measurement too coarse to
    /// resolve the difference. Gate G11 requires this contrast to read ≈1 while
    /// the real scheme reads ≈2.
    ///
    /// It also loses well-balancedness — a radially uniform state stops being a
    /// fixed point — which is the concrete reason the production path folds the
    /// source into the sweep instead.
    pub fn step_with_split_geometric_source_for_gate_contrast(&mut self, dt: f64) -> Result<()> {
        self.step_inner(dt, true, false)
    }

    fn step_inner(&mut self, dt: f64, mirror_flips: bool, well_balanced: bool) -> Result<()> {
        if !is_positive(dt) {
            bail!(
                "euler2d: non-positive step dt = {dt} at step {}",
                self.step_count
            );
        }
        let limit = self.stable_dt()?;
        if dt > limit * (1.0 + 1e-12) {
            bail!(
                "euler2d: dt = {dt:.6e} s violates CFL at step {} (limit {limit:.6e} s, \
                 CFL = {}, dr = {:.6e} m, dx = {:.6e} m)",
                self.step_count,
                self.cfl,
                self.dr,
                self.dx
            );
        }
        self.sweep_r(0.5 * dt, mirror_flips, well_balanced)?;
        self.sweep_x(dt)?;
        self.sweep_r(0.5 * dt, mirror_flips, well_balanced)?;
        if !well_balanced && self.geometry == Geometry::Axisymmetric {
            self.apply_split_geometric_source(dt)?;
        }
        self.time += dt;
        self.step_count += 1;
        Ok(())
    }

    /// The geometric source as a stand-alone forward-Euler update — the
    /// contrast path only. See
    /// [`step_with_split_geometric_source_for_gate_contrast`](Self::step_with_split_geometric_source_for_gate_contrast).
    fn apply_split_geometric_source(&mut self, dt: f64) -> Result<()> {
        for j in 0..self.n_r {
            let r = self.r_centre(j);
            for i in 0..self.n_x {
                let k = j * self.n_x + i;
                let p = self.cells[k].to_primitive(&self.gas).p;
                self.cells[k].mom_r += dt * p / r;
            }
        }
        let cells = self.cells.clone();
        self.assert_physical(&cells, "split geometric source", |k| {
            (k / self.n_x, k % self.n_x)
        })
    }

    /// Apply a volumetric energy source `source(j, i, r, x)` in W/m³ for `dt`,
    /// **without** the flux update — the source half of a Strang split.
    pub fn add_energy(
        &mut self,
        dt: f64,
        source: impl Fn(usize, usize, f64, f64) -> f64,
    ) -> Result<()> {
        if !dt.is_finite() || dt < 0.0 {
            bail!("euler2d: non-finite or negative source dt = {dt}");
        }
        for j in 0..self.n_r {
            for i in 0..self.n_x {
                let q = source(j, i, self.r_centre(j), self.x_centre(i));
                self.cells[j * self.n_x + i].energy += dt * q;
            }
        }
        let cells = self.cells.clone();
        self.assert_physical(&cells, "energy source", |k| (k / self.n_x, k % self.n_x))
    }

    /// Advance by `dt` with the source folded into the step.
    ///
    /// # This method alone is only 1st-order accurate
    ///
    /// Exactly as in `euler1d`: folding the source into the update is Godunov
    /// splitting. The Strang sandwich —
    /// `add_energy(dt/2) → step(dt) → recompute → add_energy(dt/2)` — is what
    /// keeps the coupled scheme 2nd order, and it is what `lsd2d` does. This
    /// method exists as the cheap contrast gate G11 needs, and for callers
    /// where 1st order is genuinely acceptable and said so at the call site.
    pub fn step_with_source(
        &mut self,
        dt: f64,
        source: impl Fn(usize, usize, f64, f64) -> f64,
    ) -> Result<()> {
        self.step(dt)?;
        self.add_energy(dt, source)
    }

    /// March to `t_end` at the stable step.
    pub fn advance_to(&mut self, t_end: f64) -> Result<()> {
        while self.time < t_end {
            let dt = self.stable_dt()?.min(t_end - self.time);
            if !is_positive(dt) {
                break;
            }
            self.step(dt)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: IdealGas = IdealGas { gamma: 1.4 };

    fn ambient(u_r: f64, u_x: f64) -> Primitive2d {
        Primitive2d {
            rho: 1.2256,
            u_r,
            u_x,
            p: 101_325.0,
        }
    }

    fn uniform(geometry: Geometry, n_r: usize, n_x: usize, u_x: f64) -> Euler2d {
        Euler2d::from_fn(
            AIR,
            geometry,
            [
                Boundary2d::Axis,
                Boundary2d::Transmissive,
                Boundary2d::Transmissive,
                Boundary2d::Transmissive,
            ],
            0.0,
            0.0,
            1e-4,
            1e-4,
            n_r,
            n_x,
            |_, _| ambient(0.0, u_x),
        )
        .expect("uniform setup")
    }

    /// The geometric source is the exact volume average of `1/r`.
    ///
    /// `(A_+ − A_−)/V = 2/(r_+ + r_−) = 1/r_j` analytically, with `r_j` the
    /// arithmetic cell centre. This is what makes the innermost ring finite
    /// without a floor: at `r_1 = Δr/2` the value is `2/Δr`, large but not
    /// singular.
    #[test]
    fn the_geometric_source_is_the_volume_average_of_one_over_r() {
        let g = Geometry::Axisymmetric;
        let dr = 1e-4;
        for j in 0..8 {
            let lo = j as f64 * dr;
            let hi = lo + dr;
            let ratio = (g.face_area(hi) - g.face_area(lo)) / g.cell_volume(lo, hi);
            let centre = 0.5 * (lo + hi);
            assert!(
                (ratio - 1.0 / centre).abs() / (1.0 / centre) < 1e-15,
                "ring {j}: (A+ − A−)/V = {ratio:.17e} but 1/r_j = {:.17e}",
                1.0 / centre
            );
        }
        // And the axis interface carries no area at all, which is why no flux
        // can cross r = 0 and why there is no 1/r in the code.
        assert_eq!(g.face_area(0.0), 0.0);
    }

    /// **G13(i)** — a radially uniform state is a fixed point of the
    /// axisymmetric operator.
    ///
    /// Well-balancedness. The geometric source and the pressure part of the
    /// radial flux difference are written as the same floating-point
    /// expression, so they cancel exactly and the innermost ring does not
    /// spontaneously heat or evacuate.
    ///
    /// Measured: mass and both momenta are **bit-identical** after 20 steps.
    /// Energy is not, and the reason is inherited rather than new — HLLC's star
    /// state computes `ρ·(E/ρ)`, which is within one ulp of `E` but not equal
    /// to it, so a uniform state carries a round-off-level radial energy flux
    /// that the area weights fail to cancel. It does not grow: see the bound
    /// asserted below.
    #[test]
    fn a_radially_uniform_state_is_a_fixed_point_of_the_axisymmetric_operator() {
        let mut s = uniform(Geometry::Axisymmetric, 12, 8, 300.0);
        let before = s.cells().to_vec();
        let e0 = s.total_energy();
        let dt = s.stable_dt().expect("dt");
        for _ in 0..20 {
            s.step(dt).expect("step");
        }
        for (k, (&a, &b)) in before.iter().zip(s.cells()).enumerate() {
            assert_eq!(
                (a.rho, a.mom_r, a.mom_x),
                (b.rho, b.mom_r, b.mom_x),
                "cell {k}: a radially uniform state moved in mass or momentum, so the \
                 geometric source is not cancelling the pressure flux exactly"
            );
        }
        let drift = (s.total_energy() - e0).abs() / e0;
        assert!(
            drift < 1e-13,
            "uniform-state energy drift {drift:.3e} over 20 steps is above round-off; \
             the radial operator is doing something to a state it should not touch"
        );
    }

    /// The planar operator leaves a uniform state alone in **every** component,
    /// including energy: with unit areas the flux difference is `F − F`, which
    /// is exactly zero whatever round-off `F` itself carries.
    #[test]
    fn a_uniform_state_is_a_bit_exact_fixed_point_in_planar_geometry() {
        let mut s = uniform(Geometry::Planar, 12, 8, 300.0);
        let before = s.cells().to_vec();
        let dt = s.stable_dt().expect("dt");
        for _ in 0..20 {
            s.step(dt).expect("step");
        }
        assert_eq!(
            before,
            s.cells(),
            "planar uniform state is not a fixed point"
        );
    }

    /// Packing removes the transverse kinetic energy so the 1-D solver sees the
    /// true pressure, and unpacking restores the state.
    #[test]
    fn pack_removes_transverse_kinetic_energy_and_round_trips() {
        let u = ambient(120.0, -450.0).to_conserved(&AIR);
        for dir in [Dir::Axial, Dir::Radial] {
            let (packed, v_t) = pack(u, dir);
            let p_2d = u.to_primitive(&AIR).p;
            let p_1d = packed.to_primitive(&AIR).p;
            assert!(
                (p_1d - p_2d).abs() / p_2d < 1e-14,
                "{dir:?}: packed pressure {p_1d:.6e} != 2-D pressure {p_2d:.6e}"
            );
            let restored = packed.energy + 0.5 * u.rho * v_t * v_t;
            assert!((restored - u.energy).abs() / u.energy < 1e-14);
        }
    }

    /// The axis fill mirrors and flips the radial momentum. Odd parity is the
    /// whole axis condition.
    #[test]
    fn the_axis_ghost_fill_flips_radial_momentum() {
        let s = uniform(Geometry::Axisymmetric, 8, 8, 0.0);
        let line: Vec<Conserved2d> = (0..8)
            .map(|j| {
                Primitive2d {
                    rho: 1.0 + j as f64,
                    u_r: 10.0 + j as f64,
                    u_x: 3.0,
                    p: 1e5,
                }
                .to_conserved(&AIR)
            })
            .collect();
        let padded = s.pad_line(
            &line,
            Dir::Radial,
            Boundary2d::Axis,
            Boundary2d::Transmissive,
            true,
        );
        // Ghost N_GHOST-1 mirrors interior 0, ghost N_GHOST-2 mirrors interior 1.
        for g in 0..N_GHOST {
            let ghost = padded[N_GHOST - 1 - g];
            assert_eq!(ghost.rho, line[g].rho);
            assert_eq!(ghost.mom_r, -line[g].mom_r);
            assert_eq!(ghost.mom_x, line[g].mom_x);
        }
        // And the contrast the gate needs: with the flip disabled the radial
        // momentum is mirrored *even*, which is the classic wall-heating bug.
        let broken = s.pad_line(
            &line,
            Dir::Radial,
            Boundary2d::Axis,
            Boundary2d::Transmissive,
            false,
        );
        assert_eq!(broken[N_GHOST - 1].mom_r, line[0].mom_r);
    }

    #[test]
    fn cfl_above_the_cap_is_refused() {
        let mut s = uniform(Geometry::Planar, 8, 8, 0.0);
        assert!(s.set_cfl(0.9).is_err());
        assert!(s.set_cfl(0.0).is_err());
        assert!(s.set_cfl(0.4).is_ok());
    }

    #[test]
    fn a_step_beyond_the_cfl_limit_is_refused() {
        let mut s = uniform(Geometry::Planar, 8, 8, 0.0);
        let dt = s.stable_dt().expect("dt");
        let err = s.step(dt * 2.0).expect_err("must refuse");
        assert!(format!("{err}").contains("violates CFL"), "got: {err}");
    }

    #[test]
    fn a_non_physical_initial_state_is_refused() {
        let bad = Euler2d::from_fn(
            AIR,
            Geometry::Planar,
            [Boundary2d::Transmissive; 4],
            0.0,
            0.0,
            1e-3,
            1e-3,
            8,
            8,
            |_, _| Primitive2d {
                rho: -1.0,
                u_r: 0.0,
                u_x: 0.0,
                p: 1e5,
            },
        );
        assert!(bad.is_err());
    }

    /// An axisymmetric domain that reaches `r = 0` must say so. Silently
    /// treating the axis as an outflow would leak mass through a face of zero
    /// area and is exactly the kind of thing that produces a plausible wrong
    /// answer.
    #[test]
    fn an_axisymmetric_domain_at_the_axis_requires_the_axis_boundary() {
        let bad = Euler2d::from_fn(
            AIR,
            Geometry::Axisymmetric,
            [Boundary2d::Transmissive; 4],
            0.0,
            0.0,
            1e-3,
            1e-3,
            8,
            8,
            |_, _| ambient(0.0, 0.0),
        );
        assert!(
            bad.is_err(),
            "the axis boundary requirement is not enforced"
        );
    }

    /// Mass and axial momentum are conserved in the `r dr dx` measure; radial
    /// momentum is not, and must not be.
    ///
    /// Measured, closed box, 30 steps: mass 3.30e-16, energy 1.36e-16 relative,
    /// radial momentum 2.76e-7 (i.e. not zero, which is the point).
    ///
    /// **This test earned its place by failing.** With ghost cells borrowing the
    /// nearest interior cell's areas — the obvious implementation — it read
    /// 3.3e-4 and growing, because a reflective wall is only exact when the
    /// ghost's *metric* is the mirror of the interior cell's too. See
    /// [`ghost_face_radius`].
    #[test]
    fn the_geometric_source_breaks_radial_momentum_conservation_only() {
        let mut s = Euler2d::from_fn(
            AIR,
            Geometry::Axisymmetric,
            [
                Boundary2d::Axis,
                Boundary2d::Reflective,
                Boundary2d::Reflective,
                Boundary2d::Reflective,
            ],
            0.0,
            0.0,
            1e-4,
            1e-4,
            16,
            16,
            |r, _| {
                let hot = r < 4e-4;
                Primitive2d {
                    rho: 1.2256,
                    u_r: 0.0,
                    u_x: 0.0,
                    p: if hot { 1e6 } else { 1e5 },
                }
            },
        )
        .expect("blob setup");
        let (m0, e0, px0) = (s.total_mass(), s.total_energy(), s.total_axial_momentum());
        let dt = s.stable_dt().expect("dt");
        for _ in 0..30 {
            let dt = s.stable_dt().expect("dt").min(dt);
            s.step(dt).expect("step");
        }
        assert!(
            (s.total_mass() - m0).abs() / m0 < 1e-13,
            "mass not conserved"
        );
        assert!(
            (s.total_energy() - e0).abs() / e0 < 1e-13,
            "energy not conserved"
        );
        assert!(
            (s.total_axial_momentum() - px0).abs() < 1e-9 * m0,
            "axial momentum not conserved"
        );
        assert!(
            s.total_radial_momentum().abs() > 0.0,
            "radial momentum is conserved, which means the geometric source is missing"
        );
    }
}
