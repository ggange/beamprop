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

// ------------------------------------------------------------------------
// M3 references: Kolmogorov turbulence statistics (plane-wave forms).
// ------------------------------------------------------------------------

/// Fried parameter `r0 = (0.423·k²·Cn²·z)^(-3/5)` (m) of a uniform-`Cn²`
/// plane-wave path of length `z` (m), `cn2` in m^(-2/3).
pub fn fried_r0(cn2: f64, wavelength: f64, z: f64) -> f64 {
    let k = 2.0 * PI / wavelength;
    (0.423 * k * k * cn2 * z).powf(-3.0 / 5.0)
}

/// Plane-wave Rytov variance `σ_R² = 1.23·Cn²·k^(7/6)·z^(11/6)`.
///
/// In weak fluctuation (`σ_R² ≲ 0.3`) this equals the on-axis scintillation
/// index `σ_I² = ⟨I²⟩/⟨I⟩² − 1` — the M3 scintillation gate.
pub fn rytov_variance(cn2: f64, wavelength: f64, z: f64) -> f64 {
    let k = 2.0 * PI / wavelength;
    1.23 * cn2 * k.powf(7.0 / 6.0) * z.powf(11.0 / 6.0)
}

/// Kolmogorov phase structure function `D_φ(r) = 6.88·(r/r0)^(5/3)` (rad²) —
/// what generated phase screens must reproduce for `r` well inside the outer
/// scale.
pub fn kolmogorov_structure_function(r: f64, r0: f64) -> f64 {
    6.88 * (r / r0).powf(5.0 / 3.0)
}

impl GaussianBeam {
    /// Long-exposure (ensemble-averaged) beam radius after `z` metres of
    /// uniform-`Cn²` Kolmogorov turbulence, weak-fluctuation theory
    /// (Andrews & Phillips): `W_LT = W(z)·√(1 + 1.33·σ_R²·Λ^(5/6))` with
    /// `Λ = 2z/(k·W(z)²)`. Includes beam wander (long-term average).
    pub fn long_exposure_width(&self, z: f64, cn2: f64) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        let w = self.width_at(z);
        let lambda_param = 2.0 * z / (k * w * w);
        let t = 1.33 * rytov_variance(cn2, self.wavelength, z) * lambda_param.powf(5.0 / 6.0);
        w * (1.0 + t).sqrt()
    }
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

    #[test]
    fn fried_r0_and_rytov_scaling() {
        let (cn2, wl) = (1e-14, 1e-6);
        // r0 ∝ z^(-3/5): doubling the path shrinks r0 by 2^(3/5)
        let ratio = fried_r0(cn2, wl, 1000.0) / fried_r0(cn2, wl, 2000.0);
        assert!((ratio - 2f64.powf(0.6)).abs() < 1e-12);
        // σ_R² ∝ z^(11/6)
        let ratio = rytov_variance(cn2, wl, 2000.0) / rytov_variance(cn2, wl, 1000.0);
        assert!((ratio - 2f64.powf(11.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn structure_function_at_r0_is_6_88() {
        assert!((kolmogorov_structure_function(0.05, 0.05) - 6.88).abs() < 1e-12);
    }

    #[test]
    fn long_exposure_width_reduces_to_vacuum_without_turbulence() {
        let b = GaussianBeam {
            w0: 1e-2,
            wavelength: 1e-6,
        };
        let z = 1000.0;
        assert!((b.long_exposure_width(z, 0.0) - b.width_at(z)).abs() < 1e-15);
        assert!(b.long_exposure_width(z, 1e-14) > b.width_at(z));
    }
}
