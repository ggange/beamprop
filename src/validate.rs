//! First-class validation references (the V&V harness, T3).
//!
//! Analytic solutions the solver is checked against live here — not scattered
//! through test files — so every milestone's check cites the same trusted
//! reference implementations. Grown milestone by milestone: Gaussian-beam
//! free-space evolution (M1), Beer–Lambert decay (M2), turbulence structure
//! functions and Rytov statistics (M3), the exact Riemann solution (M6c).

use std::f64::consts::PI;

use anyhow::{Result, bail};

use crate::euler1d::{IdealGas, Primitive};

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

/// Published closed-form **collisional-cascade** breakdown threshold for air,
/// Thiyagarajan & Thompson 2012 (J. Appl. Phys. 111, 073302), their Eq. 4:
///
/// ```text
/// I_B(CC) = 1.44e6 · (p_atm² + 2.2e5·λ_µm⁻²)     W/cm²
/// ```
///
/// a MacDonald/Kroll–Watson microwave scaling extended to laser frequencies.
/// Arguments and return value are SI (Pa, m, W/m²), per project convention.
///
/// **Why this matters as a reference.** At 1064 nm the `λ⁻²` term is
/// `1.94×10⁵` while `p²` never exceeds 6.9 over 10–2000 Torr, so accepted
/// cascade theory predicts a threshold that is **flat in pressure** at this
/// wavelength. That makes it the apples-to-apples target for a cascade-only
/// kernel — the paper's *measured* curve is not, since it contains multiphoton
/// ionization (the authors correct for MPI before comparing to this formula,
/// and quote 88 % cascade / 12 % MPI at 760 Torr).
///
/// Not gospel: the authors need a 2.1× scaling factor (1.74× with dust
/// filtering) to reconcile it with their measurements.
pub fn tt2012_cascade_threshold(pressure: f64, wavelength: f64) -> f64 {
    const P_ATM: f64 = 101_325.0;
    let p_atm = pressure / P_ATM;
    let lambda_um = wavelength * 1e6;
    // W/cm² → W/m².
    1.44e6 * (p_atm * p_atm + 2.2e5 / (lambda_um * lambda_um)) * 1e4
}

/// Least-squares slope `n` of a power law `y ∝ x^n`, fitted to `(x, y)` pairs
/// in log-log space. The generic form behind any power-law trend gate (M6a's
/// threshold-vs-pressure branch is the first user).
///
/// The fitted exponent is only meaningful over the range actually sampled — a
/// curve that crosses between regimes has no single slope, so callers must
/// restrict the points to one branch before fitting.
///
/// Returns `None` for fewer than two points, or if all `x` coincide.
pub fn loglog_slope(curve: &[(f64, f64)]) -> Option<f64> {
    if curve.len() < 2 {
        return None;
    }
    let n = curve.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x_raw, y_raw) in curve {
        let x = x_raw.ln();
        let y = y_raw.ln();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    Some((n * sxy - sx * sy) / denom)
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

    /// Smith's (1977) whole-beam *geometrical-optics* distortion number `N_c`
    /// — the x-axis of his steady-state peak-irradiance curves. Unlike
    /// [`distortion_number`](Self::distortion_number) it carries **no
    /// wavenumber** (it is a ray-bending, not a phase, measure):
    ///
    /// ```text
    /// N_c = (−μ_T·I₀·α·z²)/(μ·ρ·c_p·v·a)·[2/(αz) − 2/(αz)²·(1 − e^(−αz))]
    /// ```
    ///
    /// with `−μ_T = (n₀−1)/T₀` (isobaric ∂n/∂T magnitude), `μ = n₀`, `I₀` the
    /// peak intensity `2P/(π·w²)`, and `a = w/√2` the `1/e` **amplitude** radius
    /// (Smith's convention; our `w` is the `1/e²` intensity radius). The bracket
    /// is the absorption-weighted path factor — it → 1 as `α·z → 0` and folds
    /// in beam attenuation over the path. Used only to place solver runs on
    /// Smith's published x-axis for the B3 quantitative gate.
    pub fn smith_distortion_number(&self, path_len: f64) -> f64 {
        let n0 = 1.0 + self.n_minus_1;
        let mu_t_neg = self.n_minus_1 / self.t0; // −μ_T = (n₀−1)/T₀ > 0
        let a = self.w / std::f64::consts::SQRT_2; // 1/e amplitude radius
        let i0 = self.peak_intensity();
        let az = self.alpha_abs * path_len;
        // Absorption path factor; → 1 as αz → 0 (2/(αz)·(1 − (1−e^-az)/(az))).
        let bracket = 2.0 / az - 2.0 / (az * az) * (1.0 - (-az).exp());
        mu_t_neg * i0 * self.alpha_abs * path_len * path_len
            / (n0 * self.rho * self.cp * self.wind * a)
            * bracket
    }

    /// Total beam power (W) giving Smith distortion number `n_c` over
    /// `path_len` — inverse of [`smith_distortion_number`](Self::smith_distortion_number)
    /// (which is linear in power through `I₀`), for placing runs at target `N_c`.
    pub fn power_for_smith_number(&self, n_c: f64, path_len: f64) -> f64 {
        let unit = BloomingCase {
            power: 1.0,
            ..*self
        };
        n_c / unit.smith_distortion_number(path_len)
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

// ------------------------------------------------------------------------
// M6c references: the exact solution of the 1-D Riemann problem.
// ------------------------------------------------------------------------

/// Exact solution of the Riemann problem for the 1-D Euler equations with an
/// ideal gas — the reference the Sod gate (G1) measures against.
///
/// Newton iteration on the star pressure `p*` (Toro §4.3): each non-linear wave
/// contributes `f_K(p)`, a shock branch above `p_K` and a rarefaction branch
/// below, and `f_L + f_R + Δu = 0` closes the system. The solution is then
/// sampled along rays `x/t`, which is all a self-similar solution needs.
///
/// It shares only the plain `(ρ, u, p)` struct with
/// [`euler1d`](crate::euler1d) — no flux, wave-speed, or EOS code in common —
/// so it stays an independent check on the solver rather than a second copy of
/// it (`docs/M6C_SPEC.md` gate decision 4).
#[derive(Debug, Clone, Copy)]
pub struct RiemannProblem {
    /// State on the `x < 0` side.
    pub left: Primitive,
    /// State on the `x > 0` side.
    pub right: Primitive,
    /// Ideal-gas EOS.
    pub gas: IdealGas,
}

/// Converged star-region state of a [`RiemannProblem`].
#[derive(Debug, Clone, Copy)]
pub struct StarState {
    /// Pressure between the two non-linear waves (Pa).
    pub p: f64,
    /// Velocity of the contact discontinuity (m/s).
    pub u: f64,
}

/// Sod's shock tube (Toro Test 1): the standard verification problem, and the
/// one whose exact solution every published Euler code is checked against.
pub const SOD_SHOCK_TUBE: RiemannProblem = RiemannProblem {
    left: Primitive {
        rho: 1.0,
        u: 0.0,
        p: 1.0,
    },
    right: Primitive {
        rho: 0.125,
        u: 0.0,
        p: 0.1,
    },
    gas: IdealGas { gamma: 1.4 },
};

impl RiemannProblem {
    /// Pressure function of one wave and its derivative: `(f_K, f_K')`.
    fn wave(&self, p: f64, side: Primitive) -> (f64, f64) {
        let g = self.gas.gamma;
        let c = self.gas.sound_speed(side.rho, side.p);
        if p > side.p {
            // Shock branch (Toro Eq. 4.7).
            let a = 2.0 / ((g + 1.0) * side.rho);
            let b = (g - 1.0) / (g + 1.0) * side.p;
            let root = (a / (b + p)).sqrt();
            (
                (p - side.p) * root,
                root * (1.0 - 0.5 * (p - side.p) / (b + p)),
            )
        } else {
            // Rarefaction branch (Toro Eq. 4.8).
            let ratio = p / side.p;
            (
                2.0 * c / (g - 1.0) * (ratio.powf(0.5 * (g - 1.0) / g) - 1.0),
                ratio.powf(-0.5 * (g + 1.0) / g) / (side.rho * c),
            )
        }
    }

    /// Solve for the star-region pressure and velocity.
    ///
    /// Errs if the two states cannot be joined by a solution at all — the
    /// pressure-positivity (vacuum-generation) condition
    /// `2(c_L + c_R)/(γ−1) > u_R − u_L` (Toro Eq. 4.40).
    pub fn star_state(&self) -> Result<StarState> {
        let g = self.gas.gamma;
        let (cl, cr) = (
            self.gas.sound_speed(self.left.rho, self.left.p),
            self.gas.sound_speed(self.right.rho, self.right.p),
        );
        let du = self.right.u - self.left.u;
        if 2.0 * (cl + cr) / (g - 1.0) <= du {
            bail!("riemann: initial data generates vacuum (Δu = {du:.6e} m/s too large)");
        }

        // Two-rarefaction initial guess, floored off zero.
        let exp = 0.5 * (g - 1.0) / g;
        let num = cl + cr - 0.5 * (g - 1.0) * du;
        let den = cl / self.left.p.powf(exp) + cr / self.right.p.powf(exp);
        let mut p = (num / den)
            .powf(1.0 / exp)
            .max(1e-12 * self.left.p.min(self.right.p));

        // Newton. The pressure function is smooth and monotone in p, so this
        // converges in a handful of iterations from the TR guess.
        for _ in 0..100 {
            let (fl, dfl) = self.wave(p, self.left);
            let (fr, dfr) = self.wave(p, self.right);
            let step = (fl + fr + du) / (dfl + dfr);
            let next = (p - step).max(1e-12 * p);
            let change = 2.0 * (next - p).abs() / (next + p);
            p = next;
            if change < 1e-14 {
                let (fl, _) = self.wave(p, self.left);
                let (fr, _) = self.wave(p, self.right);
                return Ok(StarState {
                    p,
                    u: 0.5 * (self.left.u + self.right.u) + 0.5 * (fr - fl),
                });
            }
        }
        bail!("riemann: star pressure did not converge (last p = {p:.6e} Pa)")
    }

    /// Sample the self-similar solution along the ray `x/t = speed` (m/s).
    ///
    /// The whole solution at time `t` is `sample(star, x/t)` — that is what
    /// makes it exact at any resolution, and why the gate can refine the mesh
    /// without re-deriving anything.
    pub fn sample(&self, star: StarState, speed: f64) -> Primitive {
        let g = self.gas.gamma;
        // Work in the frame of whichever side the ray falls on; the right side
        // is the left side with velocities mirrored.
        let (side, sign) = if speed <= star.u {
            (self.left, 1.0)
        } else {
            (self.right, -1.0)
        };
        let s = sign * speed;
        let (u_side, u_star) = (sign * side.u, sign * star.u);
        let c = self.gas.sound_speed(side.rho, side.p);
        let ratio = star.p / side.p;

        let w = if ratio > 1.0 {
            // Shock: constant either side of it.
            let s_shock = u_side - c * (0.5 * (g + 1.0) / g * ratio + 0.5 * (g - 1.0) / g).sqrt();
            if s <= s_shock {
                Primitive {
                    rho: side.rho,
                    u: u_side,
                    p: side.p,
                }
            } else {
                Primitive {
                    rho: side.rho * (ratio + (g - 1.0) / (g + 1.0))
                        / ((g - 1.0) / (g + 1.0) * ratio + 1.0),
                    u: u_star,
                    p: star.p,
                }
            }
        } else {
            // Rarefaction: constant outside the fan, self-similar inside it.
            let s_head = u_side - c;
            let c_star = c * ratio.powf(0.5 * (g - 1.0) / g);
            let s_tail = u_star - c_star;
            if s <= s_head {
                Primitive {
                    rho: side.rho,
                    u: u_side,
                    p: side.p,
                }
            } else if s >= s_tail {
                Primitive {
                    rho: side.rho * ratio.powf(1.0 / g),
                    u: u_star,
                    p: star.p,
                }
            } else {
                let f = 2.0 / (g + 1.0) + (g - 1.0) / ((g + 1.0) * c) * (u_side - s);
                Primitive {
                    rho: side.rho * f.powf(2.0 / (g - 1.0)),
                    u: 2.0 / (g + 1.0) * (c + 0.5 * (g - 1.0) * u_side + s),
                    p: side.p * f.powf(2.0 * g / (g - 1.0)),
                }
            }
        };
        Primitive { u: sign * w.u, ..w }
    }

    /// The exact state at `(x, t)` for a diaphragm at `x = 0`, `t > 0`.
    pub fn state_at(&self, x: f64, t: f64) -> Result<Primitive> {
        Ok(self.sample(self.star_state()?, x / t))
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
    fn smith_number_positive_linear_and_inverts() {
        let c = sample_case();
        let l = 500.0;
        // Positive (blooming) and linear in power.
        let n1 = c.smith_distortion_number(l);
        let n2 = BloomingCase {
            power: 2.0 * c.power,
            ..c
        }
        .smith_distortion_number(l);
        assert!(n1 > 0.0, "N_c = {n1}");
        assert!((n2 / n1 - 2.0).abs() < 1e-12, "N_c not linear in power");
        // Inverse recovers the target.
        let p = c.power_for_smith_number(2.5, l);
        let c2 = BloomingCase { power: p, ..c };
        assert!((c2.smith_distortion_number(l) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn smith_bracket_matches_thin_path_expansion() {
        // The absorption path factor expands as
        //   bracket = 1 − αz/3 + (αz)²/12 − …  (→ 1 as αz → 0).
        // Check N_c against the no-attenuation limit times this expansion at a
        // realistic αz (avoiding the catastrophic cancellation of very thin
        // paths, where 2/(αz)² amplifies the roundoff of 1 − e^(−αz)).
        let c = BloomingCase {
            alpha_abs: 1e-5,
            ..sample_case()
        }; // αz = 5e-3
        let l = 500.0;
        let az = c.alpha_abs * l;
        let n0 = 1.0 + c.n_minus_1;
        let a = c.w / std::f64::consts::SQRT_2;
        let limit = (c.n_minus_1 / c.t0) * c.peak_intensity() * c.alpha_abs * l * l
            / (n0 * c.rho * c.cp * c.wind * a);
        let expected = limit * (1.0 - az / 3.0 + az * az / 12.0);
        let rel = (c.smith_distortion_number(l) - expected).abs() / expected;
        assert!(rel < 1e-5, "N_c off thin-path expansion by {rel:.2e}");
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
    fn sod_star_state_matches_published_values() {
        // Toro Test 1, the tabulated exact solution: p* = 0.30313,
        // u* = 0.92745. Getting these right is the whole reference.
        let star = SOD_SHOCK_TUBE.star_state().unwrap();
        assert!((star.p - 0.30313).abs() < 1e-5, "p* = {}", star.p);
        assert!((star.u - 0.92745).abs() < 1e-5, "u* = {}", star.u);
    }

    #[test]
    fn riemann_solution_is_mirror_symmetric() {
        // Swapping the two sides and negating velocities must reflect the
        // solution — the sign handling in `sample` is the easy thing to get
        // wrong, and it is exactly what the right-running shock depends on.
        let p = SOD_SHOCK_TUBE;
        let mirrored = RiemannProblem {
            left: Primitive {
                u: -p.right.u,
                ..p.right
            },
            right: Primitive {
                u: -p.left.u,
                ..p.left
            },
            gas: p.gas,
        };
        for i in 0..21 {
            let s = -2.0 + 0.2 * i as f64;
            let a = p.state_at(s, 1.0).unwrap();
            let b = mirrored.state_at(-s, 1.0).unwrap();
            assert!((a.rho - b.rho).abs() < 1e-12, "ρ asymmetric at x/t = {s}");
            assert!((a.u + b.u).abs() < 1e-12, "u asymmetric at x/t = {s}");
            assert!((a.p - b.p).abs() < 1e-12, "p asymmetric at x/t = {s}");
        }
    }

    #[test]
    fn riemann_far_field_recovers_the_initial_states() {
        let p = SOD_SHOCK_TUBE;
        let far_left = p.state_at(-100.0, 1.0).unwrap();
        let far_right = p.state_at(100.0, 1.0).unwrap();
        assert!((far_left.rho - p.left.rho).abs() < 1e-14);
        assert!((far_left.p - p.left.p).abs() < 1e-14);
        assert!((far_right.rho - p.right.rho).abs() < 1e-14);
        assert!((far_right.p - p.right.p).abs() < 1e-14);
    }

    #[test]
    fn riemann_refuses_vacuum_generating_data() {
        // Two strong rarefactions pulling apart leave vacuum; there is no
        // star state to find, and the solver must say so rather than iterate.
        let p = RiemannProblem {
            left: Primitive {
                rho: 1.0,
                u: -20.0,
                p: 0.4,
            },
            right: Primitive {
                rho: 1.0,
                u: 20.0,
                p: 0.4,
            },
            gas: IdealGas { gamma: 1.4 },
        };
        let err = p.star_state().unwrap_err().to_string();
        assert!(err.contains("vacuum"), "unexpected error: {err}");
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
