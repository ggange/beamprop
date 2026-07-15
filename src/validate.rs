//! First-class validation references (the V&V harness, T3).
//!
//! Analytic solutions the solver is checked against live here — not scattered
//! through test files — so every milestone's check cites the same trusted
//! reference implementations. Grown milestone by milestone: Gaussian-beam
//! free-space evolution (M1), Beer–Lambert decay (M2), turbulence structure
//! functions and Rytov statistics (M3).

use std::f64::consts::PI;

/// Analytic free-space evolution of a Gaussian beam (the M1 reference).
///
/// A beam with `1/e²` intensity waist radius `w0` at `z = 0` evolves as
/// `w(z) = w0·√(1 + (z/zR)²)` with Rayleigh range `zR = π·w0²/λ`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianBeam {
    /// Waist radius (`1/e²` intensity), metres.
    pub w0: f64,
    /// Vacuum wavelength, metres.
    pub wavelength: f64,
}

impl GaussianBeam {
    /// Rayleigh range `zR = π·w0²/λ` (m).
    pub fn rayleigh_range(&self) -> f64 {
        PI * self.w0 * self.w0 / self.wavelength
    }

    /// Beam radius `w(z)` (m).
    pub fn width_at(&self, z: f64) -> f64 {
        let zr = self.rayleigh_range();
        self.w0 * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Far-field half-angle divergence `θ = λ/(π·w0)` (rad).
    pub fn divergence(&self) -> f64 {
        self.wavelength / (PI * self.w0)
    }

    /// Total power of the unit-amplitude beam: `∫|u|² dA = π·w0²/2`.
    pub fn power(&self) -> f64 {
        PI * self.w0 * self.w0 / 2.0
    }
}

/// Observed order of accuracy from errors at step sizes `h` and `h/2`:
/// `p = log2(e(h) / e(h/2))`. The split-step contract requires `p ≈ 2`.
pub fn observed_order(err_h: f64, err_half: f64) -> f64 {
    (err_h / err_half).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_range_reference_value() {
        // w0 = 5 mm, λ = 1 µm → zR = π·25e-6/1e-6 = 25π m
        let b = GaussianBeam {
            w0: 5e-3,
            wavelength: 1e-6,
        };
        assert!((b.rayleigh_range() - 25.0 * PI).abs() < 1e-9);
    }

    #[test]
    fn width_doubles_at_sqrt3_rayleigh() {
        let b = GaussianBeam {
            w0: 5e-3,
            wavelength: 1e-6,
        };
        let z = 3f64.sqrt() * b.rayleigh_range();
        assert!((b.width_at(z) - 2.0 * b.w0).abs() / b.w0 < 1e-12);
    }

    #[test]
    fn gaussian_power_closed_form() {
        let b = GaussianBeam {
            w0: 5e-2,
            wavelength: 1e-6,
        };
        // matches the M0 CLI observation: π·0.05²/2 ≈ 0.003927
        assert!((b.power() - 0.003926990816987241).abs() < 1e-15);
    }

    #[test]
    fn observed_order_of_exact_second_order() {
        // e(h) = C·h² → p = 2 exactly
        let p = observed_order(4.0e-3, 1.0e-3);
        assert!((p - 2.0).abs() < 1e-12);
    }
}
