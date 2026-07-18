//! Symmetric split-step beam propagator (the M1 substrate).
//!
//! Implements the numerical contract locked by the eng review:
//!
//! - **Symmetric (Strang) splitting** — half-step diffraction, full-step
//!   medium phase, half-step diffraction — for 2nd-order global accuracy.
//! - **Absorbing transverse boundary** — the FFT is periodic, so without a
//!   guard band any energy reaching the grid edge wraps around and re-enters
//!   the beam as a silent wrong answer. A super-Gaussian mask, exactly `1.0`
//!   in the interior (so contained beams conserve power to machine precision),
//!   absorbs in the outer guard band on every step.
//! - **Sampling-aware diffraction step** — angular-spectrum ("transfer
//!   function") below the critical distance `z_c = N·dx²/λ`, Fresnel
//!   impulse-response beyond it, chosen per step at runtime.
//! - **Runtime adequacy checks** at propagation start (beam containment in
//!   the guard interior, transverse resolution).

use anyhow::{Result, bail};
use ndarray::{Array2, Zip};
use ndrustfft::{FftHandler, ndfft, ndifft};
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::field::Field;
use crate::grid::Grid;
use crate::medium::Medium;

/// Fraction of the half-extent that is interior (mask exactly 1); the rest is
/// the absorbing guard band.
const GUARD_INTERIOR_FRAC: f64 = 0.8;
/// Super-Gaussian exponent of the absorber profile.
const GUARD_POWER: f64 = 4.0;
/// Absorber strength: mask value at the grid edge is `exp(-GUARD_STRENGTH)`.
const GUARD_STRENGTH: f64 = 20.0;
/// Maximum relative power allowed outside the guard interior at start.
const CONTAINMENT_TOL: f64 = 1e-9;
/// Minimum samples across a beam width (2σ) for the resolution check. Four
/// samples across 2σ keeps a Gaussian comfortably under the grid Nyquist
/// limit; genuinely under-sampled beams alias silently, which is the failure
/// this guards against.
const MIN_SAMPLES_PER_WIDTH: f64 = 4.0;

/// Which diffraction kernel a step used (selected by `dz` vs `z_c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffractionMethod {
    /// Angular-spectrum transfer function — exact for `dz ≤ z_c`.
    AngularSpectrum,
    /// Fresnel impulse-response — better sampled for `dz > z_c`.
    FresnelImpulseResponse,
}

/// Split-step propagator bound to one grid and wavelength.
///
/// Owns its FFT plans and scratch buffers so they are built once and reused
/// across all steps (and later, across Monte-Carlo realizations).
pub struct Propagator {
    grid: Grid,
    wavelength: f64,
    handler: FftHandler<f64>,
    scratch_a: Array2<Complex64>,
    scratch_b: Array2<Complex64>,
    boundary: Array2<f64>,
    apply_boundary: bool,
    guard_absorbed: f64,
    /// Cached diffraction transfer function, keyed by the dz it was built for.
    cached: Option<(f64, DiffractionMethod, Array2<Complex64>)>,
}

impl Propagator {
    /// Build a propagator for `grid` at vacuum wavelength `wavelength` (m).
    pub fn new(grid: Grid, wavelength: f64) -> Result<Self> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("wavelength must be positive and finite, got {wavelength}");
        }
        let n = grid.n;
        Ok(Self {
            grid,
            wavelength,
            handler: FftHandler::new(n),
            scratch_a: Array2::zeros((n, n)),
            scratch_b: Array2::zeros((n, n)),
            boundary: boundary_mask(grid),
            apply_boundary: true,
            guard_absorbed: 0.0,
            cached: None,
        })
    }

    /// Disable the absorbing boundary. **Test-only**: exists so the wraparound
    /// validation test can demonstrate the failure mode the boundary prevents.
    pub fn without_boundary(mut self) -> Self {
        self.apply_boundary = false;
        self
    }

    /// Total power absorbed by the guard band so far, accumulated over this
    /// propagator's lifetime (same units as [`Field::power`]).
    ///
    /// In a lossless medium this is exactly the run's power deficit; a
    /// non-negligible fraction means the beam reached the grid edge and the
    /// result is contaminated by the finite domain — enlarge the grid.
    pub fn guard_absorbed(&self) -> f64 {
        self.guard_absorbed
    }

    /// Critical distance `z_c = N·dx²/λ` separating the sampling regimes of
    /// the two diffraction kernels.
    pub fn critical_distance(&self) -> f64 {
        self.grid.n as f64 * self.grid.dx * self.grid.dx / self.wavelength
    }

    /// The kernel that will be used for a step of length `dz`.
    pub fn method_for(&self, dz: f64) -> DiffractionMethod {
        if dz <= self.critical_distance() {
            DiffractionMethod::AngularSpectrum
        } else {
            DiffractionMethod::FresnelImpulseResponse
        }
    }

    /// Runtime adequacy checks (contract: assert at run start).
    ///
    /// - the field lives on this propagator's grid;
    /// - the beam is contained in the guard interior (relative power outside
    ///   below [`CONTAINMENT_TOL`]) — otherwise the absorber eats real signal;
    /// - the beam is transversely resolved (≥ [`MIN_SAMPLES_PER_WIDTH`]
    ///   samples across its second-moment width).
    pub fn check_field(&self, field: &Field) -> Result<()> {
        if field.grid != self.grid {
            bail!(
                "field grid {:?} != propagator grid {:?}",
                field.grid,
                self.grid
            );
        }
        if field.wavelength != self.wavelength {
            bail!(
                "field wavelength {} != propagator wavelength {}",
                field.wavelength,
                self.wavelength
            );
        }
        // Containment: power-weighted, against the same separable interior the
        // mask uses.
        let half = self.grid.extent() / 2.0;
        let r0 = GUARD_INTERIOR_FRAC * half;
        let mut inside = 0.0;
        let mut total = 0.0;
        for ((iy, ix), u) in field.u.indexed_iter() {
            let p = u.norm_sqr();
            total += p;
            if self.grid.coord(ix).abs() <= r0 && self.grid.coord(iy).abs() <= r0 {
                inside += p;
            }
        }
        if total <= 0.0 {
            bail!("field carries no power");
        }
        let outside_frac = 1.0 - inside / total;
        if outside_frac > CONTAINMENT_TOL {
            bail!(
                "beam not contained in guard interior: relative power outside = {outside_frac:.3e} \
                 (tolerance {CONTAINMENT_TOL:.1e}); enlarge the grid or shrink the beam"
            );
        }
        // Resolution: second-moment width must span enough samples.
        let (wx, wy) = beam_width(field);
        let w_min = wx.min(wy);
        if w_min < MIN_SAMPLES_PER_WIDTH * self.grid.dx {
            bail!(
                "beam under-resolved: width {w_min:.3e} m < {MIN_SAMPLES_PER_WIDTH} samples \
                 at dx = {:.3e} m",
                self.grid.dx
            );
        }
        Ok(())
    }

    /// Assert the slab phase `k·δn·dz` is transversely resolved: the jump
    /// between adjacent samples must stay below π, or the diffraction FFT
    /// aliases the accumulating tilt. Checked only for nonlinear media, whose
    /// index is not pre-validated by [`check_field`](Self::check_field); the
    /// x-direction (along the wind, where the blooming integral varies most)
    /// bounds it.
    fn check_phase_sampling(&self, dn: &Array2<f64>, k: f64, dz: f64) -> Result<()> {
        let mut max_jump = 0.0_f64;
        for row in dn.rows() {
            for w in row.windows(2) {
                let jump = (k * (w[1] - w[0]) * dz).abs();
                if jump > max_jump {
                    max_jump = jump;
                }
            }
        }
        if max_jump >= PI {
            bail!(
                "blooming phase under-resolved: adjacent-sample jump {max_jump:.3} rad ≥ π; \
                 shrink dz or refine dx"
            );
        }
        Ok(())
    }

    /// Advance the field by `n_steps` slabs of thickness `dz` through
    /// `medium`, starting at slab `first_slab`. Calls `on_step(slab_index,
    /// field)` after each completed step.
    ///
    /// Runs [`check_field`](Self::check_field) once at the start.
    pub fn propagate(
        &mut self,
        field: &mut Field,
        medium: &dyn Medium,
        dz: f64,
        first_slab: usize,
        n_steps: usize,
        mut on_step: impl FnMut(usize, &Field),
    ) -> Result<()> {
        if !(dz > 0.0 && dz.is_finite()) {
            bail!("dz must be positive and finite, got {dz}");
        }
        // A finite-slab medium (a fixed screen stack) is only defined for so
        // many slabs; marching past it would index out of bounds. Reject the
        // whole run up front rather than panic mid-march.
        if let Some(max) = medium.slab_count()
            && first_slab + n_steps > max
        {
            bail!(
                "propagation of {n_steps} slabs from slab {first_slab} exceeds the medium's \
                 {max} defined slabs"
            );
        }
        self.check_field(field)?;
        for i in 0..n_steps {
            self.step(field, medium, first_slab + i, dz)?;
            on_step(first_slab + i, field);
        }
        Ok(())
    }

    /// One symmetric split step: diffract `dz/2`, apply the slab's medium
    /// phase (and Beer–Lambert amplitude decay, if the medium is lossy) over
    /// `dz`, diffract `dz/2`, then apply the absorbing boundary.
    pub fn step(
        &mut self,
        field: &mut Field,
        medium: &dyn Medium,
        z_slab: usize,
        dz: f64,
    ) -> Result<()> {
        self.diffract(field, dz / 2.0)?;

        // The field is now at the slab centre (one half-step of diffraction in).
        // A field-coupled medium (thermal blooming) forms its index from this
        // slab-centre intensity — the predictor step that keeps the symmetric
        // split second-order; linear media ignore the intensity.
        let dn = if medium.needs_intensity() {
            medium.index_response(z_slab, &field.intensity(), dz)?
        } else {
            medium.index_perturbation(z_slab)
        };
        if dn.dim() != field.u.dim() {
            bail!(
                "medium returned δn of shape {:?}, expected {:?}",
                dn.dim(),
                field.u.dim()
            );
        }
        let k = 2.0 * PI / self.wavelength;
        // Nonlinear media grow phase along the path; assert the per-slab phase
        // ramp stays resolved (adjacent-sample jump < π), or diffraction
        // aliases. Linear screens are pre-validated by check_field.
        if medium.needs_intensity() {
            self.check_phase_sampling(&dn, k, dz)?;
        }
        match medium.extinction(z_slab) {
            Some(alpha) => {
                if alpha.dim() != field.u.dim() {
                    bail!(
                        "medium returned α of shape {:?}, expected {:?}",
                        alpha.dim(),
                        field.u.dim()
                    );
                }
                // α is the power extinction coefficient, so the amplitude
                // decays at α/2: intensity goes as exp(−α·dz).
                Zip::from(&mut field.u)
                    .and(&dn)
                    .and(&alpha)
                    .for_each(|u, &d, &a| {
                        *u *= Complex64::from_polar((-0.5 * a * dz).exp(), k * d * dz);
                    });
            }
            None => {
                Zip::from(&mut field.u).and(&dn).for_each(|u, &d| {
                    *u *= Complex64::from_polar(1.0, k * d * dz);
                });
            }
        }

        self.diffract(field, dz / 2.0)?;

        if self.apply_boundary {
            let before = field.power();
            Zip::from(&mut field.u)
                .and(&self.boundary)
                .for_each(|u, &m| *u *= m);
            self.guard_absorbed += before - field.power();
        }
        Ok(())
    }

    /// Pure diffraction over distance `dz` (no medium, no boundary), with the
    /// kernel chosen by [`method_for`](Self::method_for).
    pub fn diffract(&mut self, field: &mut Field, dz: f64) -> Result<()> {
        let method = self.method_for(dz);
        self.ensure_kernel(dz, method);
        // FFT to spectrum (axis 0 then axis 1), multiply by the cached
        // transfer function, inverse FFT back.
        ndfft(&field.u, &mut self.scratch_a, &self.handler, 0);
        ndfft(&self.scratch_a, &mut self.scratch_b, &self.handler, 1);
        let h = &self.cached.as_ref().expect("kernel just ensured").2;
        Zip::from(&mut self.scratch_b)
            .and(h)
            .for_each(|s, &hv| *s *= hv);
        ndifft(&self.scratch_b, &mut self.scratch_a, &self.handler, 1);
        ndifft(&self.scratch_a, &mut field.u, &self.handler, 0);
        Ok(())
    }

    /// Build (or reuse) the frequency-domain transfer function for `dz`.
    fn ensure_kernel(&mut self, dz: f64, method: DiffractionMethod) {
        if let Some((cached_dz, cached_method, _)) = &self.cached
            && *cached_dz == dz
            && *cached_method == method
        {
            return;
        }
        let h = match method {
            DiffractionMethod::AngularSpectrum => {
                angular_spectrum_kernel(self.grid, self.wavelength, dz)
            }
            DiffractionMethod::FresnelImpulseResponse => fresnel_ir_kernel(
                self.grid,
                self.wavelength,
                dz,
                &self.handler,
                &mut self.scratch_a,
            ),
        };
        self.cached = Some((dz, method, h));
    }
}

/// Angular-spectrum transfer function `H(kx, ky) = exp(i·dz·√(k² − kx² − ky²))`,
/// with evanescent components (`kx² + ky² > k²`) decayed exponentially.
///
/// `|H| = 1` for all propagating components, so lossless propagation conserves
/// power to machine precision — the invariant the validation harness asserts.
fn angular_spectrum_kernel(grid: Grid, wavelength: f64, dz: f64) -> Array2<Complex64> {
    let n = grid.n;
    let k = 2.0 * PI / wavelength;
    let k2 = k * k;
    // FFT-ordered spatial angular frequencies: kx_i = 2π·i/(N·dx) with the
    // usual wraparound for i ≥ N/2.
    let kfreq: Vec<f64> = (0..n)
        .map(|i| {
            let i_signed = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * i_signed / (n as f64 * grid.dx)
        })
        .collect();
    Array2::from_shape_fn((n, n), |(iy, ix)| {
        let kt2 = kfreq[ix] * kfreq[ix] + kfreq[iy] * kfreq[iy];
        let kz2 = k2 - kt2;
        if kz2 >= 0.0 {
            Complex64::from_polar(1.0, dz * kz2.sqrt())
        } else {
            // evanescent: real exponential decay
            Complex64::new((-dz * (-kz2).sqrt()).exp(), 0.0)
        }
    })
}

/// Fresnel impulse-response transfer function: the FFT of the sampled spatial
/// kernel `h(x, y) = exp(i·k·dz)/(i·λ·dz) · exp(i·k·(x² + y²)/(2·dz))`,
/// scaled by `dx²` so the DFT convolution approximates the continuous one.
///
/// Better sampled than the angular-spectrum kernel when `dz > z_c`; does not
/// conserve power exactly, and the sampled chirp leaves a faint wide-field
/// halo (harmless to the beam, but visible to wing-sensitive diagnostics like
/// second moments). Prefer many angular-spectrum steps when both apply.
fn fresnel_ir_kernel(
    grid: Grid,
    wavelength: f64,
    dz: f64,
    handler: &FftHandler<f64>,
    scratch: &mut Array2<Complex64>,
) -> Array2<Complex64> {
    let n = grid.n;
    let k = 2.0 * PI / wavelength;
    let dx2 = grid.dx * grid.dx;
    let prefactor = Complex64::from_polar(1.0, k * dz) / Complex64::new(0.0, wavelength * dz);
    // Kernel on centred coordinates, then ifftshift (roll by n/2) so its peak
    // sits at index 0 for the circular convolution.
    let h_spatial = Array2::from_shape_fn((n, n), |(iy, ix)| {
        let x = grid.coord((ix + n / 2) % n);
        let y = grid.coord((iy + n / 2) % n);
        prefactor * Complex64::from_polar(1.0, k * (x * x + y * y) / (2.0 * dz)) * dx2
    });
    let mut h_freq = Array2::zeros((n, n));
    ndfft(&h_spatial, scratch, handler, 0);
    ndfft(scratch, &mut h_freq, handler, 1);
    h_freq
}

/// Separable super-Gaussian absorbing mask: exactly `1.0` inside
/// [`GUARD_INTERIOR_FRAC`] of the half-extent, decaying to
/// `exp(-GUARD_STRENGTH)` at the grid edge.
fn boundary_mask(grid: Grid) -> Array2<f64> {
    let half = grid.extent() / 2.0;
    let r0 = GUARD_INTERIOR_FRAC * half;
    let band = half - r0;
    let m1d: Vec<f64> = (0..grid.n)
        .map(|i| {
            let r = grid.coord(i).abs();
            if r <= r0 {
                1.0
            } else {
                (-GUARD_STRENGTH * (((r - r0) / band).powf(GUARD_POWER))).exp()
            }
        })
        .collect();
    Array2::from_shape_fn((grid.n, grid.n), |(iy, ix)| m1d[ix] * m1d[iy])
}

/// Second-moment beam widths `(wx, wy)` of the intensity distribution,
/// defined as `w = 2σ` so that a Gaussian beam's `w` is its `1/e²` radius.
///
/// # Panics
/// Panics on a field with no power (the moments are undefined).
pub fn beam_width(field: &Field) -> (f64, f64) {
    let inten = field.intensity();
    let g = field.grid;
    let mut total = 0.0;
    let (mut mx, mut my) = (0.0, 0.0);
    for ((iy, ix), &p) in inten.indexed_iter() {
        total += p;
        mx += p * g.coord(ix);
        my += p * g.coord(iy);
    }
    assert!(total > 0.0, "beam_width of a field with no power");
    let (cx, cy) = (mx / total, my / total);
    let (mut vx, mut vy) = (0.0, 0.0);
    for ((iy, ix), &p) in inten.indexed_iter() {
        let dxc = g.coord(ix) - cx;
        let dyc = g.coord(iy) - cy;
        vx += p * dxc * dxc;
        vy += p * dyc * dyc;
    }
    (2.0 * (vx / total).sqrt(), 2.0 * (vy / total).sqrt())
}

/// Intensity centroid `(x̄, ȳ)` in metres.
///
/// # Panics
/// Panics on a field with no power (the centroid is undefined).
pub fn centroid(field: &Field) -> (f64, f64) {
    let inten = field.intensity();
    let g = field.grid;
    let mut total = 0.0;
    let (mut mx, mut my) = (0.0, 0.0);
    for ((iy, ix), &p) in inten.indexed_iter() {
        total += p;
        mx += p * g.coord(ix);
        my += p * g.coord(iy);
    }
    assert!(total > 0.0, "centroid of a field with no power");
    (mx / total, my / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FFT round-trip must be the identity (checks ndrustfft normalization
    /// convention the propagator relies on).
    #[test]
    fn fft_roundtrip_is_identity() {
        let grid = Grid::new(64, 1e-3);
        let field = Field::gaussian(grid, 1e-6, 8e-3);
        let mut p = Propagator::new(grid, 1e-6).unwrap();
        let mut f = field.clone();
        // dz = 0 → H = 1 exactly, so diffract() is a pure FFT round-trip.
        p.diffract(&mut f, 0.0).unwrap();
        let max_err =
            f.u.iter()
                .zip(field.u.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
        assert!(max_err < 1e-12, "round-trip error {max_err}");
    }

    #[test]
    fn boundary_mask_is_one_in_interior() {
        let grid = Grid::new(128, 1e-3);
        let mask = boundary_mask(grid);
        // centre must be exactly 1 (bitwise), or contained beams lose power
        assert_eq!(mask[[64, 64]], 1.0);
        // and the edge must be strongly absorbing
        assert!(mask[[64, 0]] < 1e-8);
    }

    #[test]
    fn second_moment_width_matches_gaussian() {
        let grid = Grid::new(256, 1e-3);
        let w0 = 2e-2;
        let field = Field::gaussian(grid, 1e-6, w0);
        let (wx, wy) = beam_width(&field);
        assert!((wx - w0).abs() / w0 < 1e-6, "wx = {wx}, expected {w0}");
        assert!((wy - w0).abs() / w0 < 1e-6);
    }

    #[test]
    fn check_field_rejects_uncontained_beam() {
        let grid = Grid::new(64, 1e-3);
        // waist comparable to the grid: lots of power in the guard band
        let field = Field::gaussian(grid, 1e-6, 40e-3);
        let p = Propagator::new(grid, 1e-6).unwrap();
        assert!(p.check_field(&field).is_err());
    }

    #[test]
    fn check_field_rejects_underresolved_beam() {
        let grid = Grid::new(256, 1e-3);
        // width of ~3 samples: below the resolution floor
        let field = Field::gaussian(grid, 1e-6, 3e-3);
        let p = Propagator::new(grid, 1e-6).unwrap();
        assert!(p.check_field(&field).is_err());
    }

    #[test]
    fn method_selection_by_critical_distance() {
        let grid = Grid::new(512, 1e-3);
        let p = Propagator::new(grid, 1e-6).unwrap();
        let zc = p.critical_distance();
        assert!((zc - 512.0).abs() < 1e-9);
        assert_eq!(p.method_for(zc * 0.5), DiffractionMethod::AngularSpectrum);
        assert_eq!(
            p.method_for(zc * 2.0),
            DiffractionMethod::FresnelImpulseResponse
        );
    }
}
