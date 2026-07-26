//! Complex scalar optical field on a transverse grid, plus output helpers.

use std::path::Path;

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;

use crate::grid::Grid;

/// Conversion from the field's arbitrary `|u|²` units to physical W/m² (T4).
///
/// The propagator is scale-free: `Field` carries complex amplitude in whatever
/// units the launch condition happened to use, and every linear operation
/// preserves that arbitrariness. A model that contains an *absolute* intensity
/// — thermal blooming's `ΔT`, breakdown's threshold, the LSD deposition — has
/// to pin the scale to a physical beam power before it can say anything, and
/// it is always the same pinning:
///
/// ```text
/// I_phys(x, y) = (P_beam / P_field) · |u(x, y)|²
/// ```
///
/// where `P_field` is [`Field::power`] of the **launch** field. Fixing it at
/// launch is what makes the scale a constant of the run: propagation conserves
/// `P_field` (M1's energy gate), and any extinction the medium applies is
/// already carried by `|u|²` itself, so re-deriving the scale downstream would
/// double-count the loss.
///
/// Extracted from `ThermalBlooming`, which was its first consumer and computed
/// it inline; M6c's LSD driver is the second, needing W/m² from the
/// `propagate` callback both to test the breakdown trigger and to size the
/// deposition (`docs/M6C_SPEC.md` § Beam ↔ plasma coupling).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntensityScale {
    /// W/m² per unit of `|u|²`.
    per_unit: f64,
}

impl IntensityScale {
    /// Pin the scale from a physical beam power (W) and the launch field's
    /// power in its own units ([`Field::power`]).
    ///
    /// Errors on non-positive or non-finite inputs — a scale of zero, infinity,
    /// or NaN would silently disable whichever absolute-intensity model asked
    /// for it, which is exactly the failure the guard exists to prevent.
    pub fn from_beam_power(beam_power: f64, field_power: f64) -> Result<Self> {
        if !(beam_power > 0.0 && beam_power.is_finite()) {
            anyhow::bail!("beam power must be positive and finite, got {beam_power}");
        }
        if !(field_power > 0.0 && field_power.is_finite()) {
            anyhow::bail!("initial field power must be positive and finite, got {field_power}");
        }
        Ok(Self {
            per_unit: beam_power / field_power,
        })
    }

    /// The bare factor, W/m² per unit `|u|²`.
    pub fn per_unit_intensity(&self) -> f64 {
        self.per_unit
    }

    /// Convert one arbitrary-unit intensity sample to W/m².
    pub fn to_physical(&self, intensity: f64) -> f64 {
        self.per_unit * intensity
    }

    /// Convert a whole arbitrary-unit intensity map to W/m².
    pub fn map_to_physical(&self, intensity: &Array2<f64>) -> Array2<f64> {
        intensity.mapv(|i| self.per_unit * i)
    }
}

/// A monochromatic scalar optical field `u(x, y)` sampled on a [`Grid`].
///
/// The stored quantity is complex amplitude; intensity is `|u|²`. This is the
/// object the M1 propagator will advance in `z`; for M0 it only needs to be
/// constructible and writable to disk for inspection.
#[derive(Debug, Clone)]
pub struct Field {
    /// The transverse sampling grid.
    pub grid: Grid,
    /// Vacuum wavelength in metres.
    pub wavelength: f64,
    /// Complex amplitude, indexed `[iy, ix]`.
    pub u: Array2<Complex64>,
}

impl Field {
    /// A circular Gaussian beam of `1/e²`-intensity waist radius `w0` (metres),
    /// unit on-axis amplitude and flat phase, centred on the grid.
    ///
    /// This is the M1 validation input (its free-space evolution has a
    /// closed form), and for M0 it is simply a non-trivial field to render.
    ///
    /// # Panics
    /// Panics if `wavelength` or `w0` is not positive.
    pub fn gaussian(grid: Grid, wavelength: f64, w0: f64) -> Self {
        assert!(wavelength > 0.0, "wavelength must be positive");
        assert!(w0 > 0.0, "waist must be positive");
        let u = Array2::from_shape_fn((grid.n, grid.n), |(iy, ix)| {
            let x = grid.coord(ix);
            let y = grid.coord(iy);
            let r2 = x * x + y * y;
            // amplitude Gaussian: intensity ∝ exp(-2 r² / w0²)
            Complex64::new((-r2 / (w0 * w0)).exp(), 0.0)
        });
        Self {
            grid,
            wavelength,
            u,
        }
    }

    /// Intensity `|u|²` as a real array, indexed `[iy, ix]`.
    pub fn intensity(&self) -> Array2<f64> {
        self.u.mapv(|c| c.norm_sqr())
    }

    /// Total power `Σ |u|² · dx²` in the field's (arbitrary) amplitude units.
    ///
    /// Lossless propagation must conserve this; it is the invariant the M1
    /// energy-conservation test will assert on.
    pub fn power(&self) -> f64 {
        let dx2 = self.grid.dx * self.grid.dx;
        self.u.iter().map(|c| c.norm_sqr()).sum::<f64>() * dx2
    }

    /// Physical intensity `I(x, y)` in W/m² under `scale` (T4).
    ///
    /// The bridge between the scale-free propagator and any model that needs an
    /// absolute intensity — in particular the M6c driver, which reads this off
    /// the `propagate` callback each slab.
    pub fn physical_intensity(&self, scale: IntensityScale) -> Array2<f64> {
        scale.map_to_physical(&self.intensity())
    }

    /// Peak physical intensity over the transverse plane (W/m²).
    ///
    /// The quantity the breakdown trigger tests: `AirBreakdown` takes a peak
    /// intensity, and the driver has to hand it one in real units.
    pub fn peak_physical_intensity(&self, scale: IntensityScale) -> f64 {
        scale.to_physical(self.u.iter().map(|c| c.norm_sqr()).fold(0.0_f64, f64::max))
    }

    /// Write intensity to a NumPy `.npy` file (`float64`, shape `[n, n]`).
    ///
    /// The `.npy` path is the solver's file-output interface: all image
    /// rendering happens in Python/NumPy (`scripts/render.py`). Since M5 the
    /// same data is also reachable in-process via the PyO3 bindings.
    pub fn save_intensity_npy(&self, path: impl AsRef<Path>) -> Result<()> {
        ndarray_npy::write_npy(path, &self.intensity())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> Grid {
        Grid::new(64, 1e-3)
    }

    #[test]
    fn intensity_scale_maps_field_power_to_beam_power() {
        // The defining property (T4): integrating the scaled intensity over the
        // grid must return the physical beam power, whatever units the launch
        // field used.
        let grid = test_grid();
        let f = Field::gaussian(grid, 1e-6, 5e-3);
        let beam_power = 4.2e4;
        let scale = IntensityScale::from_beam_power(beam_power, f.power()).unwrap();
        let integrated = f.physical_intensity(scale).sum() * grid.dx * grid.dx;
        assert!(
            (integrated - beam_power).abs() / beam_power < 1e-12,
            "∫I dA = {integrated:.6e} W, expected {beam_power:.6e} W"
        );
    }

    #[test]
    fn intensity_scale_is_independent_of_field_units() {
        // Rescaling the launch amplitude must not move the physical intensity —
        // that is the whole point of pinning the scale to a beam power.
        let grid = test_grid();
        let mut f = Field::gaussian(grid, 1e-6, 5e-3);
        let a = IntensityScale::from_beam_power(1e4, f.power()).unwrap();
        let peak_a = f.peak_physical_intensity(a);

        f.u.mapv_inplace(|c| c * 137.0);
        let b = IntensityScale::from_beam_power(1e4, f.power()).unwrap();
        let peak_b = f.peak_physical_intensity(b);

        assert!(
            (peak_a - peak_b).abs() / peak_a < 1e-12,
            "peak intensity moved with the amplitude units: {peak_a:.6e} vs {peak_b:.6e}"
        );
    }

    #[test]
    fn intensity_scale_refuses_degenerate_inputs() {
        let f = Field::gaussian(test_grid(), 1e-6, 5e-3);
        for (beam, field_power) in [
            (0.0, f.power()),
            (-1.0, f.power()),
            (f64::INFINITY, f.power()),
            (1e4, 0.0),
            (1e4, f64::NAN),
        ] {
            assert!(
                IntensityScale::from_beam_power(beam, field_power).is_err(),
                "accepted degenerate scale ({beam}, {field_power})"
            );
        }
    }

    #[test]
    fn gaussian_peaks_at_centre() {
        let f = Field::gaussian(test_grid(), 1e-6, 5e-3);
        let peak = f.intensity()[[32, 32]];
        // on-axis amplitude is 1 → intensity 1, and it is the maximum
        assert!((peak - 1.0).abs() < 1e-12);
        let max = f.intensity().iter().copied().fold(0.0, f64::max);
        assert_eq!(peak, max);
    }

    #[test]
    fn power_is_positive_and_finite() {
        let f = Field::gaussian(test_grid(), 1e-6, 5e-3);
        let p = f.power();
        assert!(p > 0.0 && p.is_finite());
    }

    #[test]
    fn intensity_shape_matches_grid() {
        let f = Field::gaussian(test_grid(), 1e-6, 5e-3);
        assert_eq!(f.intensity().dim(), (64, 64));
    }
}
