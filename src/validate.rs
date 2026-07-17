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

// ------------------------------------------------------------------------
// M4 references: steady-state thermal blooming (crosswind, weak heating).
// ------------------------------------------------------------------------

/// Error function via Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5·10⁻⁷).
///
/// Used by the closed-form blooming reference; `f64::erf` is not in stable
/// std, and this accuracy is far below the 0.5 % B1 gate.
pub fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Parameters of a steady-state thermal-blooming case (SI units), collecting
/// the air state and beam that the closed-form references below need.
#[derive(Debug, Clone, Copy)]
pub struct BloomingCase {
    /// Absorbed-power coefficient α_abs (1/m).
    pub alpha_abs: f64,
    /// Total beam power P (W).
    pub power: f64,
    /// Gaussian 1/e² intensity radius w (m).
    pub w: f64,
    /// Crosswind speed v (m/s), along +x.
    pub wind: f64,
    /// Air density ρ (kg/m³).
    pub rho: f64,
    /// Isobaric specific heat c_p (J/(kg·K)).
    pub cp: f64,
    /// Background refractivity n₀ − 1 (dimensionless).
    pub n_minus_1: f64,
    /// Ambient temperature T₀ (K).
    pub t0: f64,
    /// Vacuum wavelength λ (m).
    pub wavelength: f64,
}

impl BloomingCase {
    /// On-axis peak intensity of the collimated Gaussian, `I₀ = 2P/(π·w²)`.
    pub fn peak_intensity(&self) -> f64 {
        2.0 * self.power / (PI * self.w * self.w)
    }

    /// Closed-form steady-state temperature rise (K) of a collimated Gaussian
    /// in uniform crosswind (M4_SPEC B1):
    /// `ΔT = (α·I₀·w/(ρ·c_p·v))·√(π/8)·exp(−2y²/w²)·(1 + erf(√2·x/w))`.
    pub fn delta_t_ref(&self, x: f64, y: f64) -> f64 {
        let i0 = self.peak_intensity();
        let amp = self.alpha_abs * i0 * self.w / (self.rho * self.cp * self.wind);
        amp * (PI / 8.0).sqrt()
            * (-2.0 * y * y / (self.w * self.w)).exp()
            * (1.0 + erf(std::f64::consts::SQRT_2 * x / self.w))
    }

    /// Closed-form accumulated blooming phase (rad) of a frozen field over a
    /// path `path_len` (m): `φ = −k·(n₀−1)/T₀·ΔT·L` (heated air thins).
    pub fn phase_ref(&self, x: f64, y: f64, path_len: f64) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        -k * self.n_minus_1 / self.t0 * self.delta_t_ref(x, y) * path_len
    }

    /// Peak phase distortion number (M4_SPEC convention):
    /// `N_φ = √(2/π)·k·(n₀−1)·α·P·L/(T₀·ρ·c_p·v·w)`.
    pub fn distortion_number(&self, path_len: f64) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        (2.0 / PI).sqrt() * k * self.n_minus_1 * self.alpha_abs * self.power * path_len
            / (self.t0 * self.rho * self.cp * self.wind * self.w)
    }

    /// Total beam power (W) that yields distortion number `n_phi` over
    /// `path_len` — the inverse of [`distortion_number`](Self::distortion_number),
    /// for setting up gates at a target blooming strength.
    pub fn power_for_distortion(&self, n_phi: f64, path_len: f64) -> f64 {
        let k = 2.0 * PI / self.wavelength;
        n_phi * self.t0 * self.rho * self.cp * self.wind * self.w
            / ((2.0 / PI).sqrt() * k * self.n_minus_1 * self.alpha_abs * path_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_reference_values() {
        assert!((erf(0.0)).abs() < 2e-7);
        assert!((erf(1.0) - 0.8427007929).abs() < 2e-7);
        assert!((erf(-1.0) + 0.8427007929).abs() < 2e-7);
        assert!((erf(2.0) - 0.9953222650).abs() < 2e-7);
        // saturates to ±1
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
    }

    fn sample_case() -> BloomingCase {
        BloomingCase {
            alpha_abs: 1e-4,
            power: 1e4,
            w: 1e-2,
            wind: 1.0,
            rho: 1.2,
            cp: 1005.0,
            n_minus_1: 2.7e-4,
            t0: 288.15,
            wavelength: 1e-6,
        }
    }

    #[test]
    fn delta_t_saturates_downwind_and_is_symmetric_in_y() {
        let c = sample_case();
        // Far upwind (x ≪ 0) the integral is ~0; far downwind it saturates to
        // twice the on-axis half value.
        let up = c.delta_t_ref(-5.0 * c.w, 0.0);
        let down = c.delta_t_ref(5.0 * c.w, 0.0);
        assert!(up < 1e-3 * down, "upwind {up} not ≪ downwind {down}");
        // y-symmetry
        assert!((c.delta_t_ref(0.0, c.w) - c.delta_t_ref(0.0, -c.w)).abs() < 1e-18);
        // heated → δn < 0 via phase sign
        assert!(c.phase_ref(5.0 * c.w, 0.0, 100.0) < 0.0);
    }

    #[test]
    fn distortion_number_inverts() {
        let c = sample_case();
        let l = 500.0;
        let p = c.power_for_distortion(3.0, l);
        let c2 = BloomingCase { power: p, ..c };
        assert!((c2.distortion_number(l) - 3.0).abs() < 1e-9);
    }

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
