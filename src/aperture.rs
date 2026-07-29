//! What a finite aperture does to a beam: the focal-intensity estimator and the
//! pupil phase statistics (M6a.2).
//!
//! # There is no focal grid here, on purpose
//!
//! The quantity M6a.2 needs is the **peak focal intensity** of a beam that has
//! crossed turbulence — the thing that decides whether a spark lights. The
//! obvious way to get it is to focus the field and read the peak, and that does
//! not work: turbulence is resolved on a centimetre grid over a kilometre path
//! while the focal spot is micrometres across, so one grid cannot carry both
//! (`docs/M6A2_SPEC.md` § "The units mismatch").
//!
//! It is also unnecessary. In the Fraunhofer regime the focal field is the
//! Fourier transform of the aperture field, so the **on-axis** focal amplitude
//! is its DC component — a plain integral over the pupil:
//!
//! ```text
//! U_focus(0) = (1/(λf)) · ∫∫ U(x, y) dx dy
//! ```
//!
//! Every quantity below is built from that integral, evaluated on the aperture
//! grid the propagator already produced. This is **exact** given Fraunhofer and
//! a thin lens, not a small-aberration approximation; the familiar Maréchal form
//! `S ≈ exp(−σ_φ²)` is a weak-aberration *limit* of it, which is why that is
//! gated as a limit rather than used as the definition.
//!
//! # Two different degradations, kept apart
//!
//! [`Aperture::coherent_sum`] carries amplitude as well as phase, so a beam that
//! has scintillated loses focal intensity even with a flat wavefront. Calling
//! the result "the Strehl ratio" would therefore be wrong. Two quantities are
//! reported instead and never conflated:
//!
//! - [`Aperture::focal_intensity_ratio`] — against the same beam through vacuum.
//!   **This is what feeds the ignition test**: it is the total degradation, phase
//!   and amplitude together.
//! - [`Aperture::phase_only_strehl`] — against the same beam's own amplitude with
//!   the phase flattened. Isolates the wavefront contribution, as a diagnostic.
//!
//! # Phase statistics are taken on screens, not on propagated fields
//!
//! [`Aperture::residual_phase_variance`] takes a real phase array, not a
//! [`Field`]. Deliberate: the phase of a propagated field is only recoverable
//! modulo 2π, and at the `D/r₀` these gates run at it wraps many times, so a
//! variance computed from `arg(u)` would be measuring the wrapping. A phase
//! screen is unwrapped by construction, and it is also exactly what Noll's
//! coefficients describe.

use anyhow::{Result, bail};
use ndarray::Array2;

use crate::field::Field;
use crate::grid::Grid;

/// Minimum samples across the aperture diameter for the pupil integrals to mean
/// anything.
const MIN_SAMPLES_ACROSS: f64 = 16.0;

/// Which low-order terms to project out before measuring residual phase
/// variance.
///
/// The distinction is not cosmetic — it is the difference between a gate that
/// depends on the outer scale and one that does not. See
/// [`Aperture::residual_phase_variance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiltRemoval {
    /// Remove the mean only (Noll's Δ₁, `1.0299·(D/r₀)^(5/3)` for Kolmogorov).
    PistonOnly,
    /// Remove mean and both tilts (Noll's Δ₃, `0.134·(D/r₀)^(5/3)`).
    PistonTipTilt,
}

/// A circular pupil of diameter `diameter` (m), centred on a [`Grid`].
#[derive(Debug, Clone, Copy)]
pub struct Aperture {
    grid: Grid,
    diameter: f64,
    /// Inclusive index bounds of the pupil's bounding box, precomputed so the
    /// pupil walk does not scan the whole grid. See [`Aperture::for_each`].
    lo: usize,
    hi: usize,
}

impl Aperture {
    /// Build a circular aperture of `diameter` metres on `grid`.
    ///
    /// Refuses an aperture that does not fit the grid or is too coarsely
    /// sampled to integrate over — both would return a number, and a wrong one,
    /// which is the failure this project bails on rather than reports.
    pub fn new(grid: Grid, diameter: f64) -> Result<Self> {
        if !(diameter > 0.0 && diameter.is_finite()) {
            bail!("aperture: diameter must be positive and finite, got {diameter}");
        }
        if diameter > grid.extent() {
            bail!(
                "aperture: diameter {diameter:.4e} m exceeds the grid's {:.4e} m extent; \
                 the pupil would be clipped by the domain rather than by itself",
                grid.extent()
            );
        }
        let across = diameter / grid.dx;
        if across < MIN_SAMPLES_ACROSS {
            bail!(
                "aperture: diameter {diameter:.4e} m spans only {across:.1} samples \
                 (dx = {:.4e} m); refine to dx ≤ {:.4e} m",
                grid.dx,
                diameter / MIN_SAMPLES_ACROSS
            );
        }
        // The pupil's bounding box in index space. `coord(i) = (i − n/2)·dx`,
        // so `|x| ≤ D/2` bounds `i` to `n/2 ± D/(2·dx)`; widen by one sample
        // each way so the box provably contains the disc under any rounding.
        let half = 0.5 * diameter / grid.dx;
        let centre = grid.n as f64 / 2.0;
        let lo = (centre - half).floor().max(0.0) as usize;
        let hi = ((centre + half).ceil() as usize + 1).min(grid.n - 1);
        Ok(Self {
            grid,
            diameter,
            lo,
            hi,
        })
    }

    /// Aperture diameter (m).
    pub fn diameter(&self) -> f64 {
        self.diameter
    }

    /// Samples across the diameter.
    pub fn samples_across(&self) -> f64 {
        self.diameter / self.grid.dx
    }

    /// Whether sample `(iy, ix)` lies inside the pupil.
    fn contains(&self, iy: usize, ix: usize) -> bool {
        let (x, y) = (self.grid.coord(ix), self.grid.coord(iy));
        x * x + y * y <= 0.25 * self.diameter * self.diameter
    }

    /// Visit every in-pupil sample as `(x, y, index)`.
    ///
    /// Walks the pupil's **bounding box**, not the whole grid. The membership
    /// test is unchanged, so the visited set is identical — this only stops the
    /// scan from asking about samples that cannot possibly be inside. It
    /// matters because a pupil is usually a small part of its grid: at the
    /// geometry the Noll gates run (`D` = 0.25 m, `dx` = 2 mm, `n` = 512) the
    /// full scan tests 262 144 samples to reach 12 272, and the box tests
    /// 15 625 — 17× fewer. Every pupil integral in this module goes through
    /// here, and several call it twice, so it compounds.
    fn for_each(&self, mut f: impl FnMut(f64, f64, (usize, usize))) {
        for iy in self.lo..=self.hi {
            for ix in self.lo..=self.hi {
                if self.contains(iy, ix) {
                    f(self.grid.coord(ix), self.grid.coord(iy), (iy, ix));
                }
            }
        }
    }

    /// Number of samples inside the pupil.
    pub fn sample_count(&self) -> usize {
        let mut n = 0;
        self.for_each(|_, _, _| n += 1);
        n
    }

    /// `∫∫ U dA` over the pupil — the complex coherent sum, and, up to `1/(λf)`,
    /// the on-axis focal amplitude.
    ///
    /// Returned as `(re, im)` rather than a `Complex64` so callers that only
    /// want `|·|²` do not have to reach for the complex crate.
    pub fn coherent_sum(&self, field: &Field) -> (f64, f64) {
        let da = self.grid.dx * self.grid.dx;
        let (mut re, mut im) = (0.0, 0.0);
        self.for_each(|_, _, idx| {
            let u = field.u[idx];
            re += u.re;
            im += u.im;
        });
        (re * da, im * da)
    }

    /// `|∫∫ U dA|²` — proportional to the on-axis focal intensity, with the
    /// `1/(λf)²` factor left to the caller since it cancels in every ratio.
    pub fn focal_power(&self, field: &Field) -> f64 {
        let (re, im) = self.coherent_sum(field);
        re * re + im * im
    }

    /// `∫∫ |U| dA` — the coherent sum this beam would give if its phase were
    /// flat, i.e. the best its own amplitude distribution can do.
    pub fn incoherent_sum(&self, field: &Field) -> f64 {
        let da = self.grid.dx * self.grid.dx;
        let mut s = 0.0;
        self.for_each(|_, _, idx| s += field.u[idx].norm());
        s * da
    }

    /// Peak focal intensity relative to a reference beam through the same
    /// aperture — normally the same launch through vacuum.
    ///
    /// **This is the quantity that feeds the ignition test.** It is the total
    /// degradation: wavefront error *and* amplitude scintillation, because the
    /// pupil field carries both. It is not the Strehl ratio, and the module note
    /// says why that distinction is kept.
    pub fn focal_intensity_ratio(&self, field: &Field, reference: &Field) -> Result<f64> {
        let denom = self.focal_power(reference);
        if !(denom > 0.0 && denom.is_finite()) {
            bail!("aperture: the reference beam carries no focal power ({denom})");
        }
        Ok(self.focal_power(field) / denom)
    }

    /// Phase-only Strehl ratio `|∫U dA|² / (∫|U| dA)²`.
    ///
    /// Normalised against the beam's *own* amplitude, so scintillation divides
    /// out and what is left is the wavefront contribution alone. Diagnostic:
    /// [`focal_intensity_ratio`](Self::focal_intensity_ratio) is what the physics
    /// uses.
    ///
    /// Exactly 1 for a flat wavefront and never above it, whatever the
    /// amplitude does — the triangle inequality, and gated as such.
    pub fn phase_only_strehl(&self, field: &Field) -> Result<f64> {
        let denom = self.incoherent_sum(field);
        if !(denom > 0.0 && denom.is_finite()) {
            bail!("aperture: no power inside the pupil ({denom})");
        }
        Ok(self.focal_power(field) / (denom * denom))
    }

    /// Residual phase variance (rad²) over the pupil after projecting out the
    /// low-order terms `mode` names.
    ///
    /// The reference values are Noll's, for Kolmogorov statistics over a
    /// circular pupil:
    ///
    /// ```text
    /// PistonOnly     σ_φ² = 1.0299 · (D/r₀)^(5/3)
    /// PistonTipTilt  σ_φ² = 0.134  · (D/r₀)^(5/3)
    /// ```
    ///
    /// Both coefficients are parameter-free. **They are not equally gateable**:
    /// piston-removed variance is dominated by the largest scales, so it is
    /// strongly outer-scale dependent and only reaches Noll's number as
    /// `L₀/D → ∞`, whereas tip/tilt removal strips exactly those terms and
    /// leaves a coefficient that is flat from `L₀/D ≳ 40`. Measured on this
    /// repo's screens: 0.1352 tip/tilt-removed against Noll's 0.134, and
    /// 0.34 → 1.01 piston-removed as `L₀/D` runs 10 → 20 000. Gated by
    /// `noll_*` in `tests/validation.rs`.
    ///
    /// Takes an unwrapped phase array — a screen — not a [`Field`]; see the
    /// module note on why `arg(u)` cannot be used here.
    pub fn residual_phase_variance(&self, phase: &Array2<f64>, mode: TiltRemoval) -> Result<f64> {
        if phase.dim() != (self.grid.n, self.grid.n) {
            bail!(
                "aperture: phase array is {:?}, expected {:?}",
                phase.dim(),
                (self.grid.n, self.grid.n)
            );
        }
        if phase.dim() != (self.grid.n, self.grid.n) {
            bail!(
                "aperture: phase array is {:?}, expected {:?}",
                phase.dim(),
                (self.grid.n, self.grid.n)
            );
        }
        let (a, b, c, n) = match mode {
            TiltRemoval::PistonOnly => {
                let (mut n, mut sp) = (0.0f64, 0.0f64);
                self.for_each(|_, _, idx| {
                    n += 1.0;
                    sp += phase[idx];
                });
                if n < 1.0 {
                    bail!("aperture: no samples in the pupil");
                }
                (sp / n, 0.0, 0.0, n)
            }
            TiltRemoval::PistonTipTilt => self.fit_plane(phase)?,
        };

        let mut var = 0.0;
        self.for_each(|x, y, idx| {
            let r = phase[idx] - a - b * x - c * y;
            var += r * r;
        });
        Ok(var / n)
    }

    /// Intensity-weighted mean wavefront tilt over the pupil, as the angle pair
    /// `(θx, θy)` in radians.
    ///
    /// A tilt of `θ` steers the focal spot to `f·θ`, which is the ignition-point
    /// wander [`focal_wander`](Self::focal_wander) reports. Computed from the
    /// same plane fit as
    /// [`residual_phase_variance`](Self::residual_phase_variance): the fitted
    /// gradient `∂φ/∂x` is a wavevector, and `θx = (∂φ/∂x)/k`.
    pub fn mean_tilt(&self, phase: &Array2<f64>, wavelength: f64) -> Result<(f64, f64)> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("aperture: wavelength must be positive and finite, got {wavelength}");
        }
        let (bx, by) = self.fit_gradient(phase)?;
        let k = 2.0 * std::f64::consts::PI / wavelength;
        Ok((bx / k, by / k))
    }

    /// Displacement of the focal spot (m) caused by the pupil's mean tilt, for a
    /// lens of focal length `focal_length`.
    pub fn focal_wander(
        &self,
        phase: &Array2<f64>,
        wavelength: f64,
        focal_length: f64,
    ) -> Result<(f64, f64)> {
        if !(focal_length > 0.0 && focal_length.is_finite()) {
            bail!("aperture: focal length must be positive and finite, got {focal_length}");
        }
        let (tx, ty) = self.mean_tilt(phase, wavelength)?;
        Ok((focal_length * tx, focal_length * ty))
    }

    /// Intensity-weighted mean wavefront tilt of a **propagated field**, as the
    /// angle pair `(θx, θy)` in radians — and so, times `f`, where its focal
    /// spot lands.
    ///
    /// # Why this is not `mean_tilt` on `arg(u)`
    ///
    /// A propagated field's phase is only recoverable modulo 2π and wraps many
    /// times across the pupil at any interesting `D/r₀`, so fitting a plane to
    /// `arg(u)` would fit the wrapping. The *local* gradient does not need the
    /// phase at all:
    ///
    /// ```text
    /// ∇φ = Im(U* ∇U) / |U|²
    /// ```
    ///
    /// and the intensity-weighted mean of it is exactly the first moment of the
    /// focal-plane intensity (the Fourier shift theorem, in moment form):
    ///
    /// ```text
    /// ⟨∇φ⟩ = ∫ Im(U* ∇U) dA / ∫ |U|² dA
    /// ```
    ///
    /// so the `|U|²` denominators cancel and no division by a small amplitude
    /// ever happens. This is the focal-spot centroid, computed on the pupil.
    ///
    /// Samples whose neighbours fall off the grid are skipped; the pupil sits
    /// well inside the guard band in any run the propagator accepts.
    pub fn mean_tilt_of_field(&self, field: &Field, wavelength: f64) -> Result<(f64, f64)> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("aperture: wavelength must be positive and finite, got {wavelength}");
        }
        if field.grid.n != self.grid.n {
            bail!(
                "aperture: field is {}² but the pupil is on {}²",
                field.grid.n,
                self.grid.n
            );
        }
        let n = self.grid.n;
        let inv_2dx = 0.5 / self.grid.dx;
        let (mut gx, mut gy, mut weight) = (0.0f64, 0.0f64, 0.0f64);
        self.for_each(|_, _, (iy, ix)| {
            if ix == 0 || iy == 0 || ix + 1 >= n || iy + 1 >= n {
                return;
            }
            let u = field.u[[iy, ix]];
            let dx = (field.u[[iy, ix + 1]] - field.u[[iy, ix - 1]]) * inv_2dx;
            let dy = (field.u[[iy + 1, ix]] - field.u[[iy - 1, ix]]) * inv_2dx;
            // Im(U* ∂U) — the |U|² that would divide it cancels against the
            // weight, so it is never formed.
            gx += (u.conj() * dx).im;
            gy += (u.conj() * dy).im;
            weight += u.norm_sqr();
        });
        if !(weight > 0.0 && weight.is_finite()) {
            bail!("aperture: no power inside the pupil to weight a tilt with ({weight})");
        }
        let k = 2.0 * std::f64::consts::PI / wavelength;
        Ok((gx / weight / k, gy / weight / k))
    }

    /// Focal-spot displacement (m) of a propagated field, for a lens of focal
    /// length `focal_length`. See [`mean_tilt_of_field`](Self::mean_tilt_of_field).
    pub fn focal_wander_of_field(
        &self,
        field: &Field,
        wavelength: f64,
        focal_length: f64,
    ) -> Result<(f64, f64)> {
        if !(focal_length > 0.0 && focal_length.is_finite()) {
            bail!("aperture: focal length must be positive and finite, got {focal_length}");
        }
        let (tx, ty) = self.mean_tilt_of_field(field, wavelength)?;
        Ok((focal_length * tx, focal_length * ty))
    }

    /// Least-squares fit of `φ ≈ a + b·x + c·y` over the pupil, returning
    /// `(a, b, c, sample_count)`.
    ///
    /// The full 3×3 solve, not the symmetric shortcut: `Grid::coord` is
    /// `(i − n/2)·dx`, so the samples run one further negative than positive and
    /// neither `Σx` nor `Σxy` vanishes exactly. The shortcut would be right to
    /// within a rounding error on this grid and wrong on any convention that
    /// changes.
    ///
    /// One accumulation pass, shared by
    /// [`residual_phase_variance`](Self::residual_phase_variance) and
    /// [`fit_gradient`](Self::fit_gradient) — they used to carry a copy each.
    fn fit_plane(&self, phase: &Array2<f64>) -> Result<(f64, f64, f64, f64)> {
        if phase.dim() != (self.grid.n, self.grid.n) {
            bail!(
                "aperture: phase array is {:?}, expected {:?}",
                phase.dim(),
                (self.grid.n, self.grid.n)
            );
        }
        let (mut n, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sp, mut sxp, mut syp) = (0.0f64, 0.0f64, 0.0f64);
        self.for_each(|x, y, idx| {
            let p = phase[idx];
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            sp += p;
            sxp += x * p;
            syp += y * p;
        });
        if n < 3.0 {
            bail!("aperture: {n} samples in the pupil, too few to fit a plane");
        }
        let m = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]];
        let rhs = [sp, sxp, syp];
        let det = det3(&m);
        if det.abs() <= f64::MIN_POSITIVE {
            bail!("aperture: the pupil's plane fit is singular");
        }
        let solve = |col: usize| {
            let mut mc = m;
            for (r, v) in rhs.iter().enumerate() {
                mc[r][col] = *v;
            }
            det3(&mc) / det
        };
        Ok((solve(0), solve(1), solve(2), n))
    }

    /// Least-squares phase gradient `(∂φ/∂x, ∂φ/∂y)` over the pupil (rad/m).
    fn fit_gradient(&self, phase: &Array2<f64>) -> Result<(f64, f64)> {
        let (_, b, c, _) = self.fit_plane(phase)?;
        Ok((b, c))
    }
}

/// Determinant of a 3×3.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn grid() -> Grid {
        Grid::new(128, 1e-3)
    }

    /// A unit-amplitude field carrying the phase `phase(x, y)`.
    fn flat_with_phase(g: Grid, phase: impl Fn(f64, f64) -> f64) -> Field {
        let u = Array2::from_shape_fn((g.n, g.n), |(iy, ix)| {
            Complex64::from_polar(1.0, phase(g.coord(ix), g.coord(iy)))
        });
        Field {
            grid: g,
            wavelength: 1e-6,
            u,
        }
    }

    #[test]
    fn rejects_apertures_it_cannot_integrate() {
        let g = grid();
        assert!(Aperture::new(g, -1.0).is_err(), "negative diameter");
        assert!(Aperture::new(g, f64::NAN).is_err(), "NaN diameter");
        assert!(
            Aperture::new(g, 2.0 * g.extent()).is_err(),
            "aperture larger than the grid"
        );
        assert!(
            Aperture::new(g, 4.0 * g.dx).is_err(),
            "4 samples across is not an integral"
        );
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();
        assert!((a.samples_across() - 64.0).abs() < 1e-12);
        // πr²/dx² samples, to within the pixellation of the rim.
        let expect = PI * 32.0 * 32.0;
        assert!((a.sample_count() as f64 / expect - 1.0).abs() < 0.01);
    }

    /// **S2 — estimator sanity.** A flat wavefront is the best case, and no
    /// wavefront beats it.
    #[test]
    fn flat_wavefront_is_unit_strehl_and_nothing_exceeds_it() {
        let g = grid();
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();

        let flat = flat_with_phase(g, |_, _| 0.0);
        let s = a.phase_only_strehl(&flat).unwrap();
        assert!((s - 1.0).abs() < 1e-12, "flat wavefront gave S = {s}");

        // A constant phase offset is not an aberration.
        let piston = flat_with_phase(g, |_, _| 1.234);
        let s = a.phase_only_strehl(&piston).unwrap();
        assert!((s - 1.0).abs() < 1e-12, "piston changed S: {s}");

        // Nothing exceeds 1 — the triangle inequality on the coherent sum.
        for amp in [0.3, 3.0] {
            for cycles in [0.0, 0.5, 2.0, 7.0] {
                let f = flat_with_phase(g, |x, _| amp * (2.0 * PI * cycles * x / (64.0 * g.dx)));
                let s = a.phase_only_strehl(&f).unwrap();
                assert!(
                    (0.0..=1.0 + 1e-12).contains(&s),
                    "S = {s} outside [0, 1] at {cycles} cycles"
                );
            }
        }
    }

    /// **S2 — tilt steers, it does not blur.** The sharpest statement of why the
    /// two reported quantities are different things: a pure tilt moves the spot
    /// while leaving the peak intensity untouched.
    #[test]
    fn pure_tilt_moves_the_spot_without_dimming_it() {
        let g = grid();
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();
        let lambda = 1e-6;
        let k = 2.0 * PI / lambda;
        let theta = 3e-6; // rad

        let phase = Array2::from_shape_fn((g.n, g.n), |(_, ix)| k * theta * g.coord(ix));
        let (tx, ty) = a.mean_tilt(&phase, lambda).unwrap();
        assert!((tx - theta).abs() < 1e-15, "θx = {tx}, expected {theta}");
        assert!(ty.abs() < 1e-15, "θy = {ty}, expected 0");

        // 10 m lens ⇒ 30 µm displacement.
        let (wx, wy) = a.focal_wander(&phase, lambda, 10.0).unwrap();
        assert!((wx - 3e-5).abs() < 1e-12, "wander_x = {wx}");
        assert!(wy.abs() < 1e-12);

        // And a pure tilt is fully removed by the tip/tilt projection.
        let v = a
            .residual_phase_variance(&phase, TiltRemoval::PistonTipTilt)
            .unwrap();
        assert!(v < 1e-18, "tip/tilt removal left {v} rad² of a pure tilt");
        // While piston-only removal sees all of it.
        let v0 = a
            .residual_phase_variance(&phase, TiltRemoval::PistonOnly)
            .unwrap();
        assert!(v0 > 1e-3, "piston-only removal hid the tilt: {v0} rad²");
    }

    /// **S1 — the estimator reduces to Maréchal as the aberration vanishes.**
    ///
    /// `S → exp(−σ_φ²)` is the weak-aberration limit of the exact integral, so
    /// the residual must fall as the aberration shrinks. Run on synthetic phase
    /// rather than turbulence, which is what isolates the estimator.
    #[test]
    fn strehl_approaches_marechal_in_the_weak_limit() {
        let g = grid();
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();
        let d = a.diameter();

        // Defocus-like, so the aberration is smooth and its variance is easy to
        // shrink continuously.
        let mut residuals = Vec::new();
        for amp in [0.8_f64, 0.4, 0.2, 0.1] {
            let shape = |x: f64, y: f64| (x * x + y * y) / (0.25 * d * d);
            let f = flat_with_phase(g, |x, y| amp * shape(x, y));
            let phase =
                Array2::from_shape_fn((g.n, g.n), |(iy, ix)| amp * shape(g.coord(ix), g.coord(iy)));
            let s = a.phase_only_strehl(&f).unwrap();
            let var = a
                .residual_phase_variance(&phase, TiltRemoval::PistonOnly)
                .unwrap();
            residuals.push((s - (-var).exp()).abs());
        }
        for w in residuals.windows(2) {
            assert!(
                w[1] < w[0],
                "Maréchal residual did not shrink with the aberration: {residuals:?}"
            );
        }
        assert!(
            *residuals.last().unwrap() < 1e-3,
            "weak-limit residual {:.3e} too large: {residuals:?}",
            residuals.last().unwrap()
        );
    }

    #[test]
    fn scintillation_is_not_counted_as_wavefront_error() {
        // The distinction the module note insists on: an amplitude-only
        // perturbation must leave the phase-only Strehl at 1 while genuinely
        // reducing the focal intensity against an unperturbed reference.
        let g = grid();
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();
        let flat = flat_with_phase(g, |_, _| 0.0);
        let speckled = {
            let mut f = flat_with_phase(g, |_, _| 0.0);
            f.u.indexed_iter_mut().for_each(|((iy, ix), u)| {
                let (x, y) = (g.coord(ix), g.coord(iy));
                *u *= 0.5 + 0.5 * (x / g.dx * 0.7).sin() * (y / g.dx * 0.7).cos();
            });
            f
        };
        let s = a.phase_only_strehl(&speckled).unwrap();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "amplitude-only perturbation read as wavefront error: S = {s}"
        );
        let ratio = a.focal_intensity_ratio(&speckled, &flat).unwrap();
        assert!(
            ratio < 0.9,
            "the amplitude perturbation should cost focal intensity, got {ratio}"
        );
    }

    /// The wrapped-phase trap, closed — with the contrast that makes it a real
    /// test rather than an assertion.
    ///
    /// At a tilt steep enough to wrap `arg(u)` several times across the pupil,
    /// the local-gradient route stays right while fitting a plane to the
    /// wrapped phase does not. Both are measured here; without the second the
    /// first would only be showing that a correct method is correct.
    #[test]
    fn field_tilt_survives_phase_wrapping_where_a_plane_fit_does_not() {
        let g = Grid::new(256, 5e-4);
        let a = Aperture::new(g, 128.0 * g.dx).unwrap();
        let lambda = 1e-6;
        let k = 2.0 * PI / lambda;

        // Accuracy first, at tilts of the size turbulence actually produces
        // (θ ~ λ/r₀ ~ 10⁻⁵ rad). The estimator is a central difference, so its
        // error is O((∂φ/∂x·dx)²/6) in the per-sample phase step — negligible
        // here, and the next block is where it starts to matter.
        for theta in [1e-7_f64, 1e-6, 1e-5] {
            let f = flat_with_phase(g, |x, _| k * theta * x);
            let (tx, ty) = a.mean_tilt_of_field(&f, lambda).unwrap();
            assert!(
                (tx / theta - 1.0).abs() < 5e-4,
                "θ = {theta:e}: got {tx:e}, off by {:.2e}",
                (tx / theta - 1.0).abs()
            );
            assert!(ty.abs() < 1e-9 * theta, "θy = {ty:e}");
        }

        // The residual above is not slop, it is the central difference's own
        // truncation: `(∂φ/∂x·dx)²/6`. Predicted 1.64e-4 at θ = 1e-5 on this
        // grid, and it must fall as dx² — halving dx at a fixed pupil takes a
        // factor 4 off it. Gating that is what distinguishes "second-order and
        // correct" from "close enough today".
        let theta = 1e-5_f64;
        let err_at = |n: usize, dx: f64| {
            let gg = Grid::new(n, dx);
            let aa = Aperture::new(gg, 0.064).unwrap();
            let ff = flat_with_phase(gg, |x, _| k * theta * x);
            let (t, _) = aa.mean_tilt_of_field(&ff, lambda).unwrap();
            (t / theta - 1.0).abs()
        };
        let coarse = err_at(256, 5e-4);
        let fine = err_at(512, 2.5e-4);
        let predicted = (k * theta * 5e-4_f64).powi(2) / 6.0;
        assert!(
            (coarse / predicted - 1.0).abs() < 0.05,
            "truncation {coarse:.3e} does not match the predicted {predicted:.3e}"
        );
        assert!(
            (coarse / fine / 4.0 - 1.0).abs() < 0.05,
            "error fell {:.2}x on halving dx, expected 4x (dx²)",
            coarse / fine
        );

        // Now wrap it. λ/D is one wrap across the pupil, so this is five.
        let theta = 5.0 * lambda / a.diameter();
        let f = flat_with_phase(g, |x, _| k * theta * x);
        let (tx, _) = a.mean_tilt_of_field(&f, lambda).unwrap();
        assert!(
            (tx / theta - 1.0).abs() < 0.02,
            "five wraps across the pupil broke the local gradient: {tx:e} vs {theta:e}"
        );

        // The contrast: the same tilt through a plane fit on the wrapped phase.
        // `arg` folds into (−π, π], so the fit sees a sawtooth, not a ramp.
        let wrapped = Array2::from_shape_fn((g.n, g.n), |idx| f.u[idx].arg());
        let (wx, _) = a.mean_tilt(&wrapped, lambda).unwrap();
        assert!(
            (wx / theta - 1.0).abs() > 0.5,
            "the plane fit on wrapped phase gave {wx:e} against {theta:e} — if it              is now accurate this contrast is not demonstrating anything and the              gate above proves nothing"
        );

        let (wx, _) = a.focal_wander_of_field(&f, lambda, 10.0).unwrap();
        assert!((wx / (10.0 * theta) - 1.0).abs() < 0.02, "wander {wx:e}");
    }

    #[test]
    fn degenerate_inputs_are_refused() {
        let g = grid();
        let a = Aperture::new(g, 64.0 * g.dx).unwrap();
        let dark = Field {
            grid: g,
            wavelength: 1e-6,
            u: Array2::zeros((g.n, g.n)),
        };
        assert!(a.phase_only_strehl(&dark).is_err(), "no power in the pupil");
        let flat = flat_with_phase(g, |_, _| 0.0);
        assert!(
            a.focal_intensity_ratio(&flat, &dark).is_err(),
            "dark reference"
        );
        let wrong = Array2::zeros((g.n / 2, g.n / 2));
        assert!(
            a.residual_phase_variance(&wrong, TiltRemoval::PistonOnly)
                .is_err(),
            "shape mismatch"
        );
        assert!(a.mean_tilt(&Array2::zeros((g.n, g.n)), -1.0).is_err());
        assert!(a.mean_tilt_of_field(&dark, 1e-6).is_err(), "dark pupil");
        assert!(a.mean_tilt_of_field(&flat, -1.0).is_err(), "bad wavelength");
        assert!(
            a.focal_wander_of_field(&flat, 1e-6, 0.0).is_err(),
            "zero focal length"
        );
    }
}
