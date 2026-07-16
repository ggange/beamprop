//! Complex scalar optical field on a transverse grid, plus output helpers.

use std::path::Path;

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;

use crate::grid::Grid;

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

    /// Write intensity to a NumPy `.npy` file (`float64`, shape `[n, n]`).
    ///
    /// The `.npy` path is the solver's output interface: analysis and all
    /// image rendering happen in Python/NumPy (`scripts/render.py`) until the
    /// PyO3 bindings arrive at M5.
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
