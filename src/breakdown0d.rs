//! 0-D optical-breakdown threshold kernel (M6a).
//!
//! A point in the gas sees an intensity history `I(t)`; its electron density
//! `n_e` avalanches by inverse-bremsstrahlung cascade ionization, seeded by
//! multiphoton ionization and drained by attachment and diffusion:
//!
//! ```text
//! dn_e/dt = (ν_i(I,p) − ν_att(p) − ν_diff(p)) · n_e + S_mpi(I,p)
//! ```
//!
//! Breakdown is declared when `n_e` reaches the criterion density within the
//! pulse. The kernel is a **driver-callable pure function** (no propagator, no
//! `Medium`): anything that needs an ignition test calls the same
//! [`AirBreakdown`] per point, which is how M6c's LSD trigger uses it. The governing
//! model, constants, and — importantly — which checks are physics gates versus
//! integrator unit tests are pinned in `docs/M6A_SPEC.md`.
//!
//! Cascade ionization is driven by the **net** power — inverse-bremsstrahlung
//! heating minus the inelastic excitation losses the electron pays climbing to
//! `U_i`. Both scale `∝ p`, so their difference gives `I_thr(p)` a constant
//! high-pressure plateau on top of the `1/p` avalanche term.
//!
//! The mean electron energy is **not** an input: the default
//! [`CascadeModel::SelfConsistentClimb`] eliminates it by solving the climb
//! `dε/dt = heating − δ_eff·ν_m·ε` exactly, leaving `δ_eff` as the only free
//! constant. Growth is **logistic**, not exponential: ionization consumes the
//! neutrals it feeds on, so `n_e` saturates at full ionization instead of
//! running away.
//!
//! The absolute threshold *level* depends on several order-of-magnitude
//! constants (`U_i`, `D_e`, `Λ`, `n_bd`, `δ_eff`) and is deliberately **not**
//! validated; published thresholds scatter 3–10× across labs, and this model
//! sits 4.8–7.2× above T&T.
//!
//! **Where the comparison stands (paper obtained 2026-07-25).** The kernel
//! models the collisional cascade only. Thiyagarajan & Thompson's *measured*
//! curve is therefore the wrong target — it contains multiphoton ionization
//! too (they quote 88 % cascade / 12 % MPI at 760 Torr, MPI-dominant below
//! 100 Torr, and MPI-correct the data before comparing it to theory). The
//! apples-to-apples reference is their cascade closed form, Eq. 4, which at
//! 1064 nm is **flat in pressure**: the `λ⁻²` term is `1.94×10⁵` against a
//! `p²` term that never exceeds 6.9.
//!
//! Against that reference the kernel does reasonably — 4.1–5.1× high in level
//! for this model, 1.3–3.2× for [`CascadeModel::FixedMeanEnergy`], gated by
//! `tt2012_cascade_theory_reference`. Against the raw measurement it does not,
//! and cannot: no cascade-only model can produce the measured `n = 0.329` when
//! accepted cascade theory says `n ≈ 0`. That gate
//! (`tt2012_threshold_slope_matches_measurement`) is red on purpose.
//!
//! On the **wavelength** axis the agreement is structural rather than a level
//! comparison: both terms of `I_thr` carry `1/h ∝ ω²`, so the kernel predicts
//! `I_thr ∝ λ⁻²` — the same exponent as Eq. 4's dominant term, matched to
//! −2.000 with a ratio constant to 2×10⁻⁵ over 0.53–10.6 µm
//! (`tt2012_wavelength_scaling_matches_cascade_theory`). The plateau `L′/h` and
//! Eq. 4's `λ⁻²` coefficient are the same physical quantity — `ω²` times the
//! inelastic energy loss per collision — and agree to 1 % at the literature
//! centre. That is the branch's sharpest physical result.
//!
//! The pressure-slope **bracket** (`n = 0.095` here, `0.468` for the
//! fixed-`⟨ε⟩` limit, straddling 0.329) is a weaker claim than it looks: the
//! two variants differ only in the cascade cutoff energy, so it is a
//! one-parameter sensitivity, not two independent limits. See [`CascadeModel`].
//!
//! **The one prediction here with nothing to tune** is the plateau floor
//! ([`cascade_plateau_intensity`]). Heating and inelastic loss both scale
//! `∝ ν_m`, so the collision frequency cancels exactly out of `ε_∞`, and the
//! intensity below which no cascade runs at any pressure is
//! `δ_eff·U_i·m_e c ε₀ ω²/e²` — containing neither `k_m` nor `D_e`, the kernel's
//! two least defensible constants. For a monatomic gas `δ = 2m_e/M` is the
//! atomic mass and `U_i` is spectroscopy, so the floor is parameter-free. Tested
//! against Chylek's He/Ar/Xe data by
//! `chylek1990_noble_gas_plateau_floors_are_unequally_tight`: the ordering is
//! right, the spacing is wrong by up to 8.8×, and the headroom above the floor
//! is mass-ordered (1.85× / 7.8× / 13.2×) in a way a cascade-only model gives no
//! reason for.
//!
//! Analysis in `docs/M6A_SPEC.md`; external gates in `tests/validation.rs`;
//! what is verified vs validated vs pinned vs ungated is in
//! `docs/MODELS.md` § Claims ledger.

use std::f64::consts::PI;

use anyhow::{Result, bail};

// --- Physical constants (SI; docs/M6A_SPEC.md) --------------------------------

/// Elementary charge (C).
const E_CHARGE: f64 = 1.602_176_634e-19;
/// Electron mass (kg).
const M_E: f64 = 9.109_383_701_5e-31;
/// Speed of light (m/s).
const C_LIGHT: f64 = 299_792_458.0;
/// Vacuum permittivity (F/m).
const EPS0: f64 = 8.854_187_812_8e-12;
/// Reduced Planck constant (J·s).
const HBAR: f64 = 1.054_571_817e-34;
/// Reference pressure, 1 atm (Pa).
const P_REF: f64 = 101_325.0;

/// Largest growth exponent `β·dt` the integrator evaluates; `exp` overflows
/// above ~709. See [`AirBreakdown::advance`].
const MAX_EXPONENT: f64 = 700.0;

/// Below this `γ` the direct form of [`keldysh_tunnel_exponent`] loses precision
/// to cancellation, and the small-`γ` series is used instead.
const KELDYSH_SERIES_CUTOVER: f64 = 0.1;

/// Bisection bracket for [`AirBreakdown::threshold_intensity`] (W/m²).
const I_BRACKET_LO: f64 = 1e12;
const I_BRACKET_HI: f64 = 1e22;

/// How the inelastic energy loss enters the cascade rate.
///
/// The two variants are the parameter-light limits of the same energy balance,
/// and they **bracket** the measured threshold slope: 0.468 and 0.095 against
/// T&T's 0.329. Neither reproduces it.
///
/// Read the bracket narrowly. The two differ only in **where the cascade cuts
/// off** — [`Self::FixedMeanEnergy`] at `ε_∞ = ⟨ε⟩ = 3 eV`,
/// [`Self::SelfConsistentClimb`] at `ε_∞ = U_i = 12.06 eV` — so sweeping that
/// one energy walks continuously between them (`⟨ε⟩` = 5 eV gives `n` = 0.346,
/// straddling the measurement all by itself). It is therefore a *one-parameter
/// sensitivity*, not two independent physical limits, and not a bound: fixed
/// `⟨ε⟩` at `U_i` gives 0.192, inside the interval. See `docs/M6A_SPEC.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeModel {
    /// Loss evaluated at a fixed mean electron energy `⟨ε⟩`:
    /// `ν_i = max(0, heating − δ_eff·ν_m·⟨ε⟩)/U_i`.
    ///
    /// `⟨ε⟩` is a free constant, literature-bounded to ≈2–5 eV. Retained as
    /// the steeper end of the bracket (`n = 0.468`) and for the unit tests that
    /// need a cascade rate linear in intensity; not the default.
    ///
    /// Two properties argue for it despite that free constant. Its plateau
    /// `L′/h` at the literature centre is **1.01×** T&T Eq. 4's `λ⁻²`
    /// coefficient (`tt2012_wavelength_scaling_matches_cascade_theory`), and at
    /// threshold it runs at `ε_∞` = 3.7–9.3 eV — comfortably above its own
    /// cutoff, unlike the default.
    FixedMeanEnergy,
    /// `⟨ε⟩` eliminated by solving the climb exactly.
    ///
    /// An electron's energy obeys `dε/dt = heating − δ_eff·ν_m·ε`, a linear
    /// ODE with solution `ε(t) = ε_∞(1 − e^{−t/t_r})`, `ε_∞ = heating/(δ_eff·ν_m)`
    /// and `t_r = 1/(δ_eff·ν_m)`. Ionization happens when the climb reaches
    /// `U_i`, so
    ///
    /// ```text
    /// ν_i = δ_eff·ν_m / ln(ε_∞/(ε_∞ − U_i)),    zero if ε_∞ ≤ U_i
    /// ```
    ///
    /// No free `⟨ε⟩` at all, and still exactly `∝ p` (the scaling the external
    /// `E_eff` gate confirms). **The default**, chosen for parameter parsimony
    /// — *not* for agreement: it gives `n = 0.095` against a measured 0.329,
    /// which is the flatter side of the bracket and misses by 3.5×.
    ///
    /// Its idealization is the flip side of its parsimony, and it is the likely
    /// reason it is too flat: putting every electron on the *mean* trajectory
    /// makes the threshold artificially sharp at `ε_∞ = U_i`, where the climb
    /// time diverges logarithmically. A real energy distribution has a tail
    /// that ionizes earlier, which would soften the plateau and steepen the
    /// slope back toward the data. That is the open question M6a hands forward.
    ///
    /// **It is also evaluated close to that divergence.** At threshold `ε_∞` is
    /// 14.95 eV at 300 Torr, 12.49 at 760 and 12.16 at 2000 — the last within
    /// 0.8 % of `U_i`, i.e. the threshold there is set by a logarithm a hair
    /// from its pole, which is exactly where a single-mean-energy treatment is
    /// least defensible. Its near-flatness is substantially an artifact of that
    /// near-hard cutoff, and the parsimony argument for this default should be
    /// read against it. Nothing currently gates that margin.
    SelfConsistentClimb,
    /// The climb resolved as a **distribution** rather than a trajectory.
    ///
    /// [`Self::SelfConsistentClimb`]'s defect is that it puts every electron on
    /// the mean path, so ionization switches on discontinuously at `ε_∞ = U_i`.
    /// That bifurcation *is* the model's threshold plateau — and the model is
    /// evaluated on top of it (`ε_∞/U_i` = 1.032 at 760 Torr, 1.011 at 1500).
    ///
    /// An electron does not climb smoothly. It absorbs inverse-bremsstrahlung
    /// quanta of size `ħω`, so its energy performs a random walk *with* drift —
    /// an Ornstein–Uhlenbeck process in energy space:
    ///
    /// ```text
    /// dε = δ_eff·ν_m·(ε_∞ − ε)·dt + √(2·D_ε)·dW,    D_ε = ½·P_heat·ħω
    /// ```
    ///
    /// The drift is exactly the other variant's `dε/dt`. The diffusion is photon
    /// shot noise — absorption events of size `ħω` arriving at rate `P_heat/ħω`
    /// — and it introduces **no new constant**: `P_heat` is
    /// [`AirBreakdown::heating_power`] and `ħω` is the laser. This variant is
    /// therefore not an extra knob, it is the removal of an idealization.
    ///
    /// Ionization becomes a first-passage problem, evaluated in closed form by
    /// [`first_passage_ionization_rate`]. Three consequences, all gated:
    ///
    /// - **The bifurcation is gone.** `ν_i > 0` below `ε_∞ = U_i` and continuous
    ///   through it: the tail ionizes while the mean is still short.
    /// - **It reduces exactly.** As `D_ε → 0` the rate returns
    ///   `SelfConsistentClimb`'s closed form, which is what makes this a
    ///   generalization rather than a different model.
    /// - **It carries a wavelength dependence with the opposite sign to the
    ///   cascade's `λ⁻²`.** `D_ε ∝ ħω`, so a shorter wavelength takes bigger
    ///   energy steps and reaches `U_i` more easily — the direction the measured
    ///   532/1064 ratio of ≈0.80 demands and the cascade gets backwards.
    ///
    /// **Not the default.** It lands as a variant so every published M6a number
    /// stays put and what it does to them is measured, not asserted; see
    /// `docs/M6A_SPEC.md` § Distribution-resolved cascade for the outcome.
    DistributionResolved,
}

/// Quadrature points for [`first_passage_ionization_rate`].
///
/// The scheme is 2nd order and the integrand is smooth, so the error falls by 4×
/// per doubling: 1.2e-4 relative at this value, 3e-5 at 1024. Pinned here rather
/// than exposed, and checked by `first_passage_quadrature_is_converged`. It is
/// **not** free to raise: this runs inside the threshold bisection's inner loop.
const FIRST_PASSAGE_STEPS: usize = 512;

/// Mean first-passage rate for the energy random walk to reach `u_ion` (s⁻¹).
///
/// The electron's energy obeys the Ornstein–Uhlenbeck process
/// `dε = ν_relax·(ε_∞ − ε)dt + √(2 D_ε)dW` with `ν_relax = δ_eff·ν_m` and
/// `D_ε = ½·ν_relax·ε_∞·ħω` (photon shot noise; see
/// [`CascadeModel::DistributionResolved`]). With a reflecting boundary at
/// `ε = 0` and an absorbing one at `U_i`, Siegert's formula gives the mean
/// first-passage time in closed form:
///
/// ```text
/// T = (1/D_ε)·∫₀^{U_i} dy e^{+φ(y)} ∫₀^y dz e^{−φ(z)},    φ(ε) ≡ Φ(ε)/D_ε
/// φ(ε) = (ε² − 2·ε_∞·ε)/(ε_∞·ħω)
/// ```
///
/// and `ν_i = 1/T`. A quadrature, not a PDE solve: no grid in time, no stability
/// condition, and an exact analytic limit to check against.
///
/// **Conditioning — the part that has to be got right.** The double integrand
/// `e^{φ(y)−φ(z)}` is bounded and well behaved, but the obvious `O(N)`
/// evaluation — accumulate `∫e^{−φ}` and multiply by `e^{+φ}` — splits that
/// bounded product into two factors that individually reach `10^±274` for a
/// small `ħω`. It loses every significant digit long before it overflows. (It
/// was written that way first, and the `D_ε → 0` reduction gate is what caught
/// it: the rate came back 3 % *above* the deterministic limit it must approach
/// from below.)
///
/// So the sum is carried as a recurrence in the **differences** instead. Writing
/// `S_k = e^{φ(y_k)}·∫₀^{y_k} e^{−φ}dz` and `r_k = e^{φ(y_k)−φ(y_{k−1})}`,
///
/// ```text
/// S_k = r_k·S_{k−1} + ½·h·(r_k + 1)
/// ```
///
/// which is algebraically identical, still `O(N)` and one `exp` per point, and
/// never forms an unbounded intermediate: the exponent of `r_k` is one step of
/// `dφ/dε`, bounded by `2·U_i/(N·ħω)` ≈ 2.4 in the worst case this model
/// reaches.
///
/// One genuine divergence survives, and it is physics rather than arithmetic:
/// for `ε_∞ < U_i` the passage time grows as `exp[(U_i − ε_∞)²/(ε_∞·ħω)]` and
/// runs away as `ε_∞ → 0`. That is checked up front from the closed form and
/// saturated to `ν_i = 0`.
pub fn first_passage_ionization_rate(
    nu_relax: f64,
    eps_inf: f64,
    u_ion: f64,
    photon_energy: f64,
) -> f64 {
    if !(nu_relax > 0.0 && eps_inf > 0.0 && u_ion > 0.0 && photon_energy > 0.0) {
        return 0.0;
    }
    if !(nu_relax.is_finite() && eps_inf.is_finite() && u_ion.is_finite()) {
        return 0.0;
    }
    let scale = eps_inf * photon_energy;
    let phi = |e: f64| (e * e - 2.0 * eps_inf * e) / scale;
    // The only real divergence: `φ` is a parabola with its minimum at `ε_∞`, so
    // when that minimum falls inside [0, U_i] the barrier is genuinely uphill.
    if phi(u_ion) - phi(eps_inf.min(u_ion)) > MAX_EXPONENT {
        return 0.0;
    }

    let total = first_passage_integral(eps_inf, u_ion, photon_energy, FIRST_PASSAGE_STEPS);
    let d_eps = 0.5 * nu_relax * eps_inf * photon_energy;
    let t = total / d_eps;
    if t > 0.0 && t.is_finite() {
        1.0 / t
    } else {
        0.0
    }
}

/// The Siegert double integral `∫₀^{U_i} dy e^{φ(y)} ∫₀^y dz e^{−φ(z)}` on `n`
/// intervals, by the difference recurrence described on
/// [`first_passage_ionization_rate`]. Split out so the convergence gate can
/// refine `n` against the code path the model actually runs.
fn first_passage_integral(eps_inf: f64, u_ion: f64, photon_energy: f64, n: usize) -> f64 {
    let scale = eps_inf * photon_energy;
    let phi = |e: f64| (e * e - 2.0 * eps_inf * e) / scale;
    let h = u_ion / n as f64;
    let mut s = 0.0; // S_k = e^{φ(y_k)}·∫₀^{y_k} e^{−φ} dz
    let mut total = 0.0; // outer trapezoid over S
    let mut prev_phi = phi(0.0);
    for k in 1..=n {
        let cur_phi = phi(k as f64 * h);
        let r = (cur_phi - prev_phi).exp();
        prev_phi = cur_phi;
        s = r * s + 0.5 * h * (r + 1.0);
        // S_0 = 0, so the k = 0 endpoint contributes nothing.
        let w = if k == n { 0.5 } else { 1.0 };
        total += s * w * h;
        if !total.is_finite() {
            return f64::INFINITY;
        }
    }
    total
}

/// Keldysh adiabaticity parameter `γ = ω√(2 m U_i)/(e E₀)`, from intensity.
///
/// `E₀` is the field **amplitude**, `I = ½ε₀cE₀²` — Keldysh's field is the
/// amplitude of `E₀cos ωt`, not the RMS value the T&T gates use for `E_B`.
/// Mixing the two here would be a flat `√2` in `γ`, which the `λ⁻²`-scale
/// conclusions would not survive.
///
/// `γ ≫ 1` is the multiphoton regime, `γ ≪ 1` the tunnelling regime. For
/// ns-pulse air breakdown at these intensities `γ ≈ 17` at 1064 nm and ≈ 34 at
/// 532 nm, so both wavelengths sit deep in the multiphoton branch.
pub fn keldysh_gamma(intensity: f64, omega: f64, u_ion: f64) -> f64 {
    let e_field = (2.0 * intensity / (EPS0 * C_LIGHT)).sqrt();
    omega * (2.0 * M_E * u_ion).sqrt() / (E_CHARGE * e_field)
}

/// The `γ`-dependent factor in Keldysh's ionization exponent,
///
/// ```text
/// f(γ) = (1 + 1/(2γ²))·asinh(γ) − √(1+γ²)/(2γ)
/// ```
///
/// so that `W ∝ exp[−(2U_i/ħω)·f(γ)]`. This single expression carries **both**
/// limits, which is what makes it worth using instead of a bare `σ_K I^K`:
///
/// - `γ → ∞`: `f → ln(2γ) − ½`, giving `W ∝ I^(U_i/ħω)` — a multiphoton power
///   law whose exponent is fixed by the gas and the wavelength, with nothing to
///   tune.
/// - `γ → 0`: `f → ⅔γ`, giving the standard tunnelling exponent
///   `exp[−4√(2m)·U_i^{3/2}/(3ħeE)]`.
///
/// Both limits are gated in `tests/validation.rs`.
///
/// The two terms each diverge as `1/(2γ)` when `γ → 0` and cancel, so below
/// [`KELDYSH_SERIES_CUTOVER`] the series `f = ⅔γ − γ³/15 + O(γ⁵)` is used
/// instead; `keldysh_exponent_series_matches_direct_form` pins the join.
pub fn keldysh_tunnel_exponent(gamma: f64) -> f64 {
    if gamma <= 0.0 || !gamma.is_finite() {
        return f64::NAN;
    }
    if gamma < KELDYSH_SERIES_CUTOVER {
        return gamma * (2.0 / 3.0 - gamma * gamma / 15.0);
    }
    let root = (1.0 + gamma * gamma).sqrt();
    (1.0 + 1.0 / (2.0 * gamma * gamma)) * gamma.asinh() - root / (2.0 * gamma)
}

/// Per-neutral Keldysh photoionization rate (s⁻¹).
///
/// ```text
/// W = prefactor · ω · exp[−(2U_i/ħω)·f(γ)]
/// ```
///
/// **On the prefactor, and how far it can be trusted.** Keldysh's exponent is
/// derived and carries no adjustable content. His *prefactor* is an order-unity
/// function of `γ` and `U_i/ħω` whose form differs between the atomic and
/// solid-state versions of the theory and between later re-derivations (PPT adds
/// Coulomb corrections that can move it by orders of magnitude), so it is
/// exposed here as an explicit multiplier rather than faked to a precise value.
///
/// A threshold is set by `W·τ ~ 1` with `W ∝ I^K`, `K = U_i/ħω = 5.2–10.3`, so a
/// prefactor wrong by `x` moves the threshold by only `x^(1/K)`. That suppression
/// is real but it is **not** enough to make the wavelength *ratio*
/// prefactor-free, because `K` differs between the two wavelengths: once MPI
/// dominates at both, the ratio scales as `x^(1/10.35 − 1/5.175) = x^(−0.097)`,
/// so three decades of prefactor still move it 1.9×.
///
/// *(An earlier revision of this comment claimed the ratio was
/// prefactor-insensitive and that the `K`-th root "crushed" the uncertainty.
/// That was wrong, and measuring it is what showed so — the ratio runs 3.99 →
/// 0.48 across a prefactor sweep of 10¹⁵. The `1/K` argument only applies once
/// MPI dominates at both wavelengths, which is not the regime the transition
/// happens in.)*
///
/// This is *not* an MPI cross-section from a table, and it is not the
/// T&T-calibrated `σ_K` of [`AirBreakdown::with_tt2012_mpi`], which is anchored
/// to a number 37× below the same paper's own measurement. It is a
/// first-principles rate with one soft multiplier — and the finding it delivers
/// is negative: see
/// `keldysh_mpi_does_not_close_the_wavelength_gap`.
pub fn keldysh_rate(intensity: f64, omega: f64, u_ion: f64, prefactor: f64) -> f64 {
    if intensity <= 0.0 || prefactor <= 0.0 {
        return 0.0;
    }
    let gamma = keldysh_gamma(intensity, omega, u_ion);
    let s = 2.0 * u_ion / (HBAR * omega) * keldysh_tunnel_exponent(gamma);
    if !s.is_finite() {
        return 0.0;
    }
    prefactor * omega * (-s.min(MAX_EXPONENT)).exp()
}

/// A monatomic gas, carrying only the two constants that are **exactly known**
/// for one: the ionization potential, and the elastic fractional energy loss per
/// collision `δ = 2m_e/M`.
///
/// This type exists because those two constants are enough to state the
/// cascade's **plateau floor** ([`cascade_plateau_intensity`]) — and because
/// they are all this repository has a source for. The transport constants a full
/// threshold curve also needs (`k_m`, `D_e`) come from momentum-transfer cross
/// sections, which for Ar and Xe swing two orders of magnitude across the
/// Ramsauer minimum and the resonance peak; no citable table for them has been
/// landed here, so [`Gas::from_monatomic`] makes the caller supply them rather
/// than shipping a guess dressed as a constant.
///
/// **Why `δ` is not free here, and why that matters.** In air `δ_eff` is a
/// lumped fitted-range constant (0.01–0.05) that sets the plateau level, and its
/// freedom is one of M6a's open weaknesses. A monatomic gas has no vibrational
/// modes and no low-lying electronic ones, so below the first excitation
/// threshold the only energy loss is elastic recoil, `δ = 2m_e/M` — fixed by the
/// atomic mass to five figures, with nothing to choose. That removes the knob,
/// which is what makes the noble gases a sharper test of the cascade model than
/// air can be.
///
/// **And why it is a lower bound, not the answer.** The last leg of every climb
/// here is above the gas's first excitation threshold — 19.82 of 24.59 eV in He,
/// 11.55 of 15.76 in Ar, 8.32 of 12.13 in Xe — and in that leg inelastic
/// excitation dominates elastic recoil by orders of magnitude. So `δ_elastic` is
/// a **lower bound** on the true `δ_eff`, [`cascade_plateau_intensity`] built
/// from it is a **lower bound** on the real plateau, and a measured threshold
/// that already sits close to that bound is in trouble. Quantifying the
/// correction needs an energy-resolved treatment of the climb, which is exactly
/// the distribution-resolved cascade M6a hands forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonatomicGas {
    name: &'static str,
    /// Ionization potential (J).
    u_ion: f64,
    /// Elastic fractional energy loss per collision, `δ = 2m_e/M`.
    delta_elastic: f64,
    /// First excitation threshold (J) — above this, `delta_elastic` understates
    /// the true loss badly. Carried so the bound's direction can be gated.
    u_excite: f64,
}

/// Unified atomic mass unit (kg), CODATA.
const AMU: f64 = 1.660_539_066_60e-27;

impl MonatomicGas {
    /// Build from the ionization potential (eV), first excitation threshold
    /// (eV), and atomic mass (u). `δ = 2m_e/M` is computed, never supplied.
    const fn new(name: &'static str, u_ion_ev: f64, u_excite_ev: f64, mass_u: f64) -> Self {
        Self {
            name,
            u_ion: u_ion_ev * E_CHARGE,
            delta_elastic: 2.0 * M_E / (mass_u * AMU),
            u_excite: u_excite_ev * E_CHARGE,
        }
    }

    /// Helium. `U_i` = 24.587 eV, first excitation (2³S) 19.82 eV, `M` = 4.0026 u
    /// ⇒ `δ` = 2.741e-4. Chylek's Fig. 2 filled squares; `K` = 11 at 532 nm.
    pub const HELIUM: Self = Self::new("He", 24.587, 19.82, 4.002_602);
    /// Argon. `U_i` = 15.760 eV, first excitation (³P₂) 11.55 eV, `M` = 39.948 u
    /// ⇒ `δ` = 2.747e-5. Chylek's Fig. 2 filled triangles; `K` = 7 at 532 nm.
    pub const ARGON: Self = Self::new("Ar", 15.760, 11.55, 39.948);
    /// Xenon. `U_i` = 12.130 eV, first excitation (³P₂) 8.32 eV, `M` = 131.293 u
    /// ⇒ `δ` = 8.357e-6. Chylek's Fig. 2 saltire crosses; `K` = 6 at 532 nm.
    pub const XENON: Self = Self::new("Xe", 12.130, 8.32, 131.293);

    /// Chemical symbol, for gate failure messages.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Ionization potential (J).
    pub fn ionization_energy(&self) -> f64 {
        self.u_ion
    }

    /// Elastic fractional energy loss per collision, `δ = 2m_e/M` — a **lower
    /// bound** on the effective `δ_eff`; see the type docs.
    pub fn elastic_loss_fraction(&self) -> f64 {
        self.delta_elastic
    }

    /// First excitation threshold (J), above which [`Self::elastic_loss_fraction`]
    /// understates the loss.
    pub fn excitation_energy(&self) -> f64 {
        self.u_excite
    }

    /// Fraction of the climb to `U_i` that lies above the first excitation
    /// threshold — i.e. how much of the ascent the elastic bound does not cover.
    pub fn inelastic_climb_fraction(&self) -> f64 {
        (self.u_ion - self.u_excite) / self.u_ion
    }
}

/// The cascade's **plateau intensity** (W/m²): the intensity below which
/// [`CascadeModel::SelfConsistentClimb`] cannot ionize at all, at any pressure
/// and for any pulse length.
///
/// ```text
/// I_plateau = δ_eff · U_i · m_e·c·ε₀·ω² / e²
/// ```
///
/// **This is the sharpest thing the kernel says, because `k_m` and `D_e` are not
/// in it.** The equilibrium energy is
/// `ε_∞ = (e²I/(m_e c ε₀))·ν_m/((ν_m²+ω²)·δ_eff·ν_m)`, and in the optical regime
/// `ν_m ≪ ω` the collision frequency cancels **exactly** — heating and the
/// energy-loss rate both scale `∝ ν_m`. Ionization requires `ε_∞ > U_i`, so the
/// floor depends only on `δ_eff`, `U_i` and the wavelength. Every transport
/// constant in the model, including the two least defensible ones, drops out.
///
/// For a monatomic gas, where `δ_eff` is fixed by the atomic mass rather than
/// fitted, that makes the floor a **parameter-free prediction** — which is what
/// `chylek1990_noble_gas_plateau_floors_are_unequally_tight` tests against
/// measurement.
///
/// A cascade-only threshold can never fall below this. A measurement that does
/// falsifies the model for that gas outright, and a measurement that merely sits
/// close to it leaves the model no room for the losses it also has to overcome.
pub fn cascade_plateau_intensity(omega: f64, u_ion: f64, delta_eff: f64) -> f64 {
    delta_eff * u_ion * M_E * C_LIGHT * EPS0 * omega * omega / (E_CHARGE * E_CHARGE)
}

/// The gas-dependent constants of the avalanche balance, separated from the
/// laser and the focal geometry.
///
/// Everything here is a property of the **species**, not of the experiment:
/// ionization potential, momentum-transfer collision frequency, inelastic loss
/// fraction, free-electron diffusion, and the attachment chemistry. The rest of
/// [`AirBreakdown`] — `ω`, `Λ`, the focal volume, the breakdown criterion — is a
/// property of the bench.
///
/// The split exists so a second gas can be run through the *same* kernel. That
/// is not a generality flourish: air's threshold data confounds photon order
/// with wavelength (a shorter `λ` lowers `K` and raises the cascade term at the
/// same time), and the only way to separate them in this repo's data is to hold
/// `λ` fixed and change the gas. See [`Gas::helium`], [`Gas::argon`],
/// [`Gas::xenon`].
///
/// **Constants are set from each species' own literature before any gate runs.**
/// Tuning one to move a threshold onto a measured curve is curve-fitting and is
/// forbidden here for the same reason it is forbidden for `Λ` (D5).
#[derive(Debug, Clone, Copy)]
pub struct Gas {
    /// Effective ionization energy `U_i` (J).
    u_ion: f64,
    /// Electron-neutral collision-frequency slope `ν_m = k_m·p` (s⁻¹·Pa⁻¹).
    k_m: f64,
    /// Free-electron diffusion coefficient at `P_REF` (m²/s); scales as `1/p`.
    d_e_ref: f64,
    /// Fractional electron energy loss per collision, `δ_eff` (dimensionless).
    /// Literature ≈ 0.01–0.05 for air above ~1 eV; **never tuned to data**.
    delta_eff: f64,
    /// Mean electron energy `⟨ε⟩` (J), used only by
    /// [`CascadeModel::FixedMeanEnergy`]. Literature ≈ 2–5 eV.
    mean_energy: f64,
    /// Dissociative-attachment rate coefficient `e + O₂ → O⁻ + O` (m³/s).
    k_att_2body: f64,
    /// Three-body attachment rate coefficient `e + O₂ + M → O₂⁻ + M` (m⁶/s).
    /// Dominant channel in air at atmospheric density; scales as `p²`.
    k_att_3body: f64,
    /// Fraction of the gas, by number, that is the attaching species (O₂ in
    /// air). Zero for the noble gases, which have no attachment channel at all.
    f_att: f64,
}

impl Gas {
    /// Dry air — the M6a case. Every constant is the one the kernel shipped
    /// with; this constructor is a pure re-packaging and changes no number.
    pub fn dry_air() -> Self {
        Self {
            u_ion: 12.06 * E_CHARGE,
            k_m: 3.9e7,
            d_e_ref: 2.0e-1,
            // δ_eff = 0.02, ⟨ε⟩ = 3 eV — the CENTRE of the literature ranges
            // (δ_eff ≈ 0.01–0.05 for air above ~1 eV, ⟨ε⟩ ≈ 2–5 eV), not a
            // value chosen to improve agreement. The ~12× spread in these
            // ranges is why the external slope gate is an envelope test rather
            // than a point comparison. See docs/M6A_SPEC.md.
            delta_eff: 0.02,
            mean_energy: 3.0 * E_CHARGE,
            // Attachment from measured rate coefficients, not an s⁻¹Pa⁻¹ fudge
            // (Kossyi et al. 1992; Itikawa 2009). The three-body channel is the
            // dominant one at atmospheric density. Both are ~150x smaller than
            // the order-of-magnitude constant they replaced, which makes
            // attachment negligible against diffusion and the growth
            // requirement — see docs/M6A_SPEC.md.
            k_att_2body: 1.0e-17,
            k_att_3body: 1.0e-43,
            f_att: 0.21,
        }
    }

    /// A monatomic gas, with the transport constants supplied explicitly.
    ///
    /// `U_i` and `δ_eff` come from [`MonatomicGas`] and are exact. `k_m` and
    /// `d_e_ref` are **required arguments rather than tabulated defaults**, and
    /// that is deliberate: momentum-transfer cross sections for Ar and Xe swing
    /// two orders of magnitude between the Ramsauer minimum and the resonance
    /// peak, so a single `ν_m = k_m·p` is a far cruder approximation for them
    /// than it is for air, and no citable table has been landed here. Making the
    /// caller name them keeps an unsourced number from acquiring the authority
    /// of a constructor default.
    ///
    /// Attachment is identically zero — a noble gas has no attaching species,
    /// which is the other reason these gases isolate the cascade.
    ///
    /// Note that the plateau floor ([`cascade_plateau_intensity`]) does **not**
    /// depend on either supplied constant, so the parameter-free gate does not
    /// inherit this uncertainty.
    pub fn from_monatomic(gas: MonatomicGas, k_m: f64, d_e_ref: f64) -> Self {
        Self {
            u_ion: gas.u_ion,
            k_m,
            d_e_ref,
            delta_eff: gas.delta_elastic,
            // No FixedMeanEnergy ⟨ε⟩ is defensible for a gas whose loss is
            // elastic recoil; the self-consistent climb is the only variant
            // these species should be run under.
            mean_energy: gas.delta_elastic * gas.u_ion,
            k_att_2body: 0.0,
            k_att_3body: 0.0,
            f_att: 0.0,
        }
    }

    /// Effective ionization energy `U_i` (J).
    pub fn ionization_energy(&self) -> f64 {
        self.u_ion
    }

    /// Electron-neutral collision-frequency slope `k_m`, `ν_m = k_m·p`
    /// (s⁻¹·Pa⁻¹) — the constant `tt2012_collision_frequency_matches_literature`
    /// checks against T&T's measured `E_eff/E_B`.
    pub fn collision_frequency_slope(&self) -> f64 {
        self.k_m
    }

    /// Free-electron diffusion coefficient at [`P_REF`] (m²/s).
    pub fn diffusion_coefficient_ref(&self) -> f64 {
        self.d_e_ref
    }

    /// Fractional electron energy loss per collision, `δ_eff`.
    pub fn inelastic_loss_fraction(&self) -> f64 {
        self.delta_eff
    }

    /// Mean electron energy `⟨ε⟩` (J) — [`CascadeModel::FixedMeanEnergy`] only.
    pub fn mean_energy(&self) -> f64 {
        self.mean_energy
    }

    /// Free-electron diffusion coefficient this gas' `k_m` implies at `pressure`
    /// for an electron of energy `energy` (J), from kinetic theory:
    ///
    /// ```text
    /// D_e = v²/(3·ν_m) = 2ε/(3·m_e·k_m·p)
    /// ```
    ///
    /// **This is why `D_e,ref` is not an independent constant.** `k_m` is
    /// externally gated against T&T's measured `E_eff/E_B` ratio
    /// (`tt2012_collision_frequency_matches_literature`, 1.05× and flat over
    /// 46–1858 Torr), so once an electron energy is named, `D_e` follows. What
    /// had been a free number becomes a statement about **what energy the
    /// diffusing electrons are assumed to have** — a quantity a reader can
    /// judge, and one the model states elsewhere.
    ///
    /// The inverse reading is the useful one and is what
    /// `d_e_ref_implies_a_stated_electron_energy` gates: divide the shipped
    /// `D_e,ref` by this and ask what `ε` it corresponds to.
    pub fn kinetic_diffusion_coefficient(&self, pressure: f64, energy: f64) -> f64 {
        2.0 * energy / (3.0 * M_E * self.k_m * pressure)
    }

    /// The electron energy (J) that a given `d_e` at `pressure` implies, i.e.
    /// [`kinetic_diffusion_coefficient`](Self::kinetic_diffusion_coefficient)
    /// solved for `ε`.
    pub fn diffusion_implied_energy(&self, pressure: f64, d_e: f64) -> f64 {
        1.5 * M_E * self.k_m * pressure * d_e
    }
}

/// An optical-breakdown point model: one [`Gas`] at one wavelength in one focus.
///
/// Construct with [`AirBreakdown::air_1064nm`] for the pinned M6a case, or
/// [`AirBreakdown::new`] to vary the gas or the geometry. All rate methods are
/// pure functions of `(intensity, pressure)`; the struct holds only fixed gas
/// and laser parameters.
///
/// The name is historical — the kernel is no longer air-specific — and is kept
/// deliberately, because every gate in `tests/validation.rs` and every number in
/// `docs/MODELS.md` refers to it. Renaming it would churn the record for no
/// physics.
#[derive(Debug, Clone, Copy)]
pub struct AirBreakdown {
    /// Laser angular frequency `ω = 2πc/λ` (rad/s).
    omega: f64,
    /// The species: `U_i`, `k_m`, `D_e`, `δ_eff`, `⟨ε⟩`, attachment.
    gas: Gas,
    /// Which limit of the energy balance drives the cascade.
    cascade_model: CascadeModel,
    /// Pulse-integration half-window, in FWHMs. A converged model must give a
    /// threshold independent of this; see [`AirBreakdown::peak_ne`].
    window_half_widths: f64,
    /// Diffusion length `Λ` (m), from the focal geometry — **pinned, not fit**.
    lambda_diff: f64,
    /// Per-neutral multiphoton ionization rate at [`Self::mpi_i_ref`] (s⁻¹).
    /// Zero disables the MPI source entirely (the default).
    mpi_rate_ref: f64,
    /// Reference intensity for the MPI rate (W/m²).
    mpi_i_ref: f64,
    /// Number of photons `K` for multiphoton ionization.
    k_photons: i32,
    /// Prefactor for the Keldysh photoionization source; zero disables it.
    /// Mutually exclusive in practice with [`Self::mpi_rate_ref`] — both are
    /// multiphoton sources, and enabling both double-counts the same channel.
    keldysh_prefactor: f64,
    /// Breakdown criterion density `n_bd` (m⁻³).
    n_bd: f64,
    /// Neutral-density-per-pressure `N/p` at reference temperature (m⁻³·Pa⁻¹),
    /// i.e. `1/(k_B·T)`.
    n_over_p: f64,
    /// Initial seed density `n_e0` (m⁻³): one electron in the focal volume.
    n_seed: f64,
}

impl AirBreakdown {
    /// General constructor. `wavelength` (m), the [`Gas`], `lambda_diff`
    /// diffusion length (m, from geometry), `temperature` (K) for the neutral
    /// density, `focal_volume` (m³) for the single-electron seed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wavelength: f64,
        gas: Gas,
        lambda_diff: f64,
        temperature: f64,
        focal_volume: f64,
    ) -> Result<Self> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("wavelength must be positive and finite, got {wavelength}");
        }
        if !(lambda_diff > 0.0 && lambda_diff.is_finite()) {
            bail!("diffusion length must be positive and finite, got {lambda_diff}");
        }
        if !(temperature > 0.0 && temperature.is_finite()) {
            bail!("temperature must be positive and finite, got {temperature}");
        }
        if !(focal_volume > 0.0 && focal_volume.is_finite()) {
            bail!("focal volume must be positive and finite, got {focal_volume}");
        }
        if !(gas.u_ion > 0.0 && gas.u_ion.is_finite()) {
            bail!(
                "ionization energy must be positive and finite, got {} J",
                gas.u_ion
            );
        }
        const K_B: f64 = 1.380_649e-23;
        let omega = 2.0 * PI * C_LIGHT / wavelength;
        let photon_energy = HBAR * omega;
        // Photons to reach the ionization potential, ⌈U_ion/ħω⌉.
        let k_photons = (gas.u_ion / photon_energy).ceil() as i32;
        Ok(Self {
            omega,
            gas,
            cascade_model: CascadeModel::SelfConsistentClimb,
            window_half_widths: 2.0,
            lambda_diff,
            // MPI source off by default: the seed electron (n_e0 = 1/V_focal)
            // is the initial condition, and the avalanche multiplies it. The
            // continuous multiphoton source is the swappable term (docs Open
            // Question 2); a physical σ_K would be supplied when it is enabled.
            mpi_rate_ref: 0.0,
            mpi_i_ref: 1.0,
            k_photons,
            // Keldysh photoionization also off by default, so every pre-existing
            // gate keeps its published numbers. Enable with
            // `with_keldysh_mpi`.
            keldysh_prefactor: 0.0,
            n_bd: 1.0e23,
            n_over_p: 1.0 / (K_B * temperature),
            n_seed: 1.0 / focal_volume,
        })
    }

    /// The pinned M6a case: 1064 nm in dry air at 288 K, with the focal
    /// geometry of Thiyagarajan & Thompson 2012 taken from **their Eq. 5**.
    ///
    /// The focus is **divergence-limited**, not diffraction-limited: a 1 cm
    /// beam with 1 mrad divergence through an `f` = 4 cm lens gives
    /// `r₀ = f·α/2 = 20 µm` and a depth of focus `l₀ = 0.414·(α/d)·f² = 66 µm`.
    /// (The Rayleigh range of a *diffraction-limited* 20 µm waist would be
    /// 1.2 mm — 36× longer — so assuming a diffraction-limited filament here
    /// is wrong in the other direction.) The diffusion length of that cylinder
    /// is their Eq. 5,
    ///
    /// ```text
    /// (1/Λ)² = (π/l₀)² + (2.405/r₀)²   →   Λ = 7.74 µm
    /// ```
    ///
    /// matching the `Λ = 8 µm` the paper states. An earlier version modelled
    /// the focus as a *sphere* with `Λ = r₀/π = 6.37 µm`, which overstated
    /// `ν_diff` by 1.48×.
    pub fn air_1064nm() -> Self {
        // Unwrap: 1064 nm is a known-good literal.
        Self::dry_air_tt2012_focus(1064e-9).unwrap()
    }

    /// Dry air in T&T's focal geometry at an arbitrary wavelength.
    ///
    /// The geometry is deliberately held **fixed** as `wavelength` varies, and
    /// that is physical here rather than a convenience: T&T's focus is
    /// *divergence*-limited (`r₀ = f·α/2`, `l₀ = 0.414·(α/d)·f²`), so both the
    /// spot and the depth of focus are set by the beam's 1 mrad divergence and
    /// the `f` = 4 cm lens — not by diffraction. `Λ` and the focal volume are
    /// therefore wavelength-independent, and the only `λ`-dependence left in
    /// the threshold is the one the rate model puts there.
    ///
    /// That is what makes `tt2012_wavelength_scaling_matches_cascade_theory`
    /// meaningful: it varies `λ` over 20× with every other input frozen.
    pub fn dry_air_tt2012_focus(wavelength: f64) -> Result<Self> {
        let r_focus = 20e-6_f64; // f·α/2, f = 4 cm, α = 1 mrad
        let l_axial = 0.414 * (1e-3 / 1e-2) * 0.04_f64.powi(2); // = 66.2 µm
        let lambda_diff = 1.0 / ((PI / l_axial).powi(2) + (2.405 / r_focus).powi(2)).sqrt();
        let focal_volume = PI * r_focus * r_focus * l_axial;
        Self::new(wavelength, Gas::dry_air(), lambda_diff, 288.0, focal_volume)
    }

    /// The gas this model is running.
    pub fn gas(&self) -> Gas {
        self.gas
    }

    /// Rebuild with a different inelastic-loss parameterisation, from the two
    /// physical quantities it lumps: the fractional energy loss per collision
    /// `δ_eff` and the mean electron energy `⟨ε⟩` (eV).
    ///
    /// Exists so the external gate can sweep the **literature ranges**
    /// (`δ_eff ≈ 0.01–0.05`, `⟨ε⟩ ≈ 2–5 eV`) and test that the measurement
    /// falls inside the resulting envelope. It is not a tuning knob: the
    /// default sits at the centre of those ranges and the gate never selects a
    /// value that improves agreement.
    pub fn with_inelastic_loss(mut self, delta_eff: f64, mean_energy_ev: f64) -> Self {
        self.gas.delta_eff = delta_eff;
        self.gas.mean_energy = mean_energy_ev * E_CHARGE;
        self
    }

    /// Rebuild with a different free-electron diffusion coefficient at
    /// [`P_REF`] (m²/s).
    ///
    /// Exists for the same reason [`Self::with_inelastic_loss`] does, and under
    /// the same rule: it is there so a gate can **sweep** `D_e` across the band
    /// kinetic theory allows and pin how far the threshold slope moves, not so a
    /// value can be selected. The project's 2026-07-24 sensitivity audit
    /// flagged `D_e` as the constant with the largest unmeasured leverage on the
    /// result; `d_e_sensitivity_is_pinned_across_the_kinetic_band` is what turns
    /// that word into a number.
    pub fn with_diffusion_coefficient(mut self, d_e_ref: f64) -> Self {
        self.gas.d_e_ref = d_e_ref;
        self
    }

    /// Enable the multiphoton source, calibrated to Thiyagarajan & Thompson's
    /// own MPI estimate (their Sec. II A): `I_B(MPI) = 4.42×10⁹ W/cm²` with
    /// `S = 14` photons, from `U_i = 15.6 eV` for air at `ħω = 1.165 eV`.
    ///
    /// The calibration reads their number as its own definition — at
    /// `I = I_B(MPI)`, multiphoton ionization alone reaches the breakdown
    /// criterion within one pulse — giving `W_ref = n_bd/(N·τ)`.
    ///
    /// `K = 14` here rather than the 11 that `U_i = 12.06 eV` (O₂) implies,
    /// because the calibration must be self-consistent with the photon count
    /// the paper used. The cascade keeps 12.06 eV, the lowest ionization
    /// channel available to a collisional electron.
    ///
    /// **Off by default, and see `tt2012_mpi_calibration_undershoots_the_data`
    /// before switching it on**: their MPI threshold sits 45× *below* their own
    /// measured threshold, so a rate anchored to it predicts breakdown far too
    /// early.
    pub fn with_tt2012_mpi(mut self, fwhm: f64, pressure: f64) -> Self {
        const I_B_MPI: f64 = 4.42e9 * 1e4; // W/cm² → W/m²
        self.k_photons = 14;
        self.mpi_i_ref = I_B_MPI;
        self.mpi_rate_ref = self.n_bd / (self.n_over_p * pressure * fwhm);
        self
    }

    /// Enable Keldysh photoionization as the multiphoton source, with an
    /// explicit order-unity `prefactor` (see [`keldysh_rate`] for why the
    /// prefactor does not carry the conclusions).
    ///
    /// This is the channel M6a's open question names. Unlike
    /// [`Self::with_tt2012_mpi`], nothing here is anchored to a measured
    /// threshold: the rate's intensity dependence and its wavelength dependence
    /// both come out of the theory, which is the only way a multiphoton term can
    /// be added without making the `λ` comparison circular.
    pub fn with_keldysh_mpi(mut self, prefactor: f64) -> Self {
        self.keldysh_prefactor = prefactor;
        // The two multiphoton paths model the same physics; keep one live.
        self.mpi_rate_ref = 0.0;
        self
    }

    /// Override the initial electron density `n_e0` (m⁻³).
    ///
    /// The default is one electron in the focal volume, `1/V_focal`, which for
    /// T&T's geometry is `1.2×10¹³ m⁻³`. **That is not a physical background
    /// density**: cosmic-ray ionization maintains ~`10⁹–10¹⁰ m⁻³` in the lower
    /// atmosphere, so an `8.3×10⁻¹⁴ m³` focus contains ~`10⁻⁴` free electrons —
    /// it essentially never has one. Assuming a seed is present is a ~10⁴
    /// overestimate, and it is not a harmless one: handing the cascade a free
    /// electron removes the *seed-production* step, which is where almost all of
    /// the wavelength dependence of real breakdown lives.
    ///
    /// Set this small and enable [`Self::with_keldysh_mpi`] to let multiphoton
    /// ionization create the seed instead, which is the physically ordered
    /// calculation. See `docs/M6A_SPEC.md` § "Seeding".
    pub fn with_seed_density(mut self, n_seed: f64) -> Self {
        self.n_seed = n_seed;
        self
    }

    /// Select which limit of the energy balance drives the cascade — see
    /// [`CascadeModel`]. The default is [`CascadeModel::SelfConsistentClimb`].
    pub fn with_cascade_model(mut self, cascade_model: CascadeModel) -> Self {
        self.cascade_model = cascade_model;
        self
    }

    /// Set the pulse-integration half-window in FWHMs (default 2.0, at which
    /// the Gaussian is `1.5×10⁻⁵` of peak).
    ///
    /// Exists so `threshold_is_window_independent` can vary it. Results must
    /// not depend on it; if they do, the model is being propped up by its
    /// integration bounds rather than by its physics.
    pub fn with_window(mut self, window_half_widths: f64) -> Self {
        self.window_half_widths = window_half_widths;
        self
    }

    /// Equilibrium electron energy `ε_∞ = heating/(δ_eff·ν_m)` (J) — the energy
    /// at which inelastic losses balance inverse-bremsstrahlung heating.
    ///
    /// Independent of pressure to the `(ν_m/ω)²` correction, since heating and
    /// loss both scale `∝ p`. [`CascadeModel::SelfConsistentClimb`] ionizes only
    /// where this exceeds `U_i`, which is what fixes its threshold plateau.
    pub fn equilibrium_energy(&self, intensity: f64, pressure: f64) -> f64 {
        self.heating_power(intensity, pressure) / (self.gas.delta_eff * self.gas.k_m * pressure)
    }

    /// Initial seed density `n_e0` (m⁻³): one electron in the focal volume.
    pub fn seed_density(&self) -> f64 {
        self.n_seed
    }

    /// Breakdown criterion density `n_bd` (m⁻³) — the level `n_e` must reach
    /// within the pulse for breakdown to be declared.
    pub fn criterion_density(&self) -> f64 {
        self.n_bd
    }

    /// Neutral number density `N = p/(k_B·T)` (m⁻³) — the ceiling on `n_e`,
    /// since ionization consumes the neutrals it feeds on.
    pub fn neutral_density(&self, pressure: f64) -> f64 {
        self.n_over_p * pressure
    }

    /// Inverse-bremsstrahlung heating power absorbed per electron (W).
    ///
    /// `P = (e²·I)/(m_e·c·ε₀) · ν_m/(ν_m²+ω²)` with `ν_m = k_m·p`. In the
    /// optical regime `ν_m ≪ ω` this is `∝ I·p`.
    pub fn heating_power(&self, intensity: f64, pressure: f64) -> f64 {
        let nu_m = self.gas.k_m * pressure;
        let p_abs = E_CHARGE * E_CHARGE * intensity / (M_E * C_LIGHT * EPS0);
        p_abs * nu_m / (nu_m * nu_m + self.omega * self.omega)
    }

    /// Inelastic energy-loss power per electron (W), `L = L′·p`.
    ///
    /// Vibrational and electronic excitation of N₂/O₂ drain the electron on its
    /// climb to `U_i`. Scales `∝ p` because it goes as the collision frequency.
    pub fn inelastic_loss_power(&self, pressure: f64) -> f64 {
        self.gas.delta_eff * self.gas.k_m * pressure * self.gas.mean_energy
    }

    /// Cascade (avalanche) ionization frequency `ν_i(I, p)` (s⁻¹).
    ///
    /// Driven by the **net** power — heating minus inelastic loss — since an
    /// electron that cannot outrun its excitation losses never reaches `U_i`:
    ///
    /// ```text
    /// ν_i = max(0, heating(I,p) − L(p)) / U_i
    /// ```
    ///
    /// Both terms scale `∝ p`, so `ν_i ∝ p` at fixed `I` (which is what the
    /// external `E_eff` gate confirms), while the `I`-dependence is *affine*,
    /// not linear: there is a finite intensity below which no cascade runs at
    /// all. That offset is what gives `I_thr(p)` its high-pressure plateau.
    pub fn cascade_rate(&self, intensity: f64, pressure: f64) -> f64 {
        match self.cascade_model {
            CascadeModel::FixedMeanEnergy => {
                let net =
                    self.heating_power(intensity, pressure) - self.inelastic_loss_power(pressure);
                net.max(0.0) / self.gas.u_ion
            }
            CascadeModel::SelfConsistentClimb => {
                let eps_inf = self.equilibrium_energy(intensity, pressure);
                if eps_inf <= self.gas.u_ion {
                    // Losses cap the electron below the ionization potential:
                    // the climb never completes, however long the pulse.
                    return 0.0;
                }
                // t_climb = t_r·ln(ε_∞/(ε_∞ − U_i)), t_r = 1/(δ_eff·ν_m).
                let nu_relax = self.gas.delta_eff * self.gas.k_m * pressure;
                nu_relax / (eps_inf / (eps_inf - self.gas.u_ion)).ln()
            }
            CascadeModel::DistributionResolved => {
                // The same drift, plus the photon shot noise the mean trajectory
                // discards. No cutoff: the tail ionizes below ε_∞ = U_i.
                let nu_relax = self.gas.delta_eff * self.gas.k_m * pressure;
                let eps_inf = self.equilibrium_energy(intensity, pressure);
                first_passage_ionization_rate(
                    nu_relax,
                    eps_inf,
                    self.gas.u_ion,
                    self.photon_energy(),
                )
            }
        }
    }

    /// Photon energy `ħω` (J) at this model's wavelength.
    pub fn photon_energy(&self) -> f64 {
        HBAR * self.omega
    }

    /// Total loss frequency `ν_att + ν_diff` (s⁻¹).
    ///
    /// Attachment has two channels, both from measured rate coefficients:
    /// dissociative `k₂·n_O₂ ∝ p` and three-body `k₃·n_O₂·n ∝ p²`. Diffusion
    /// goes as `1/p` (free-electron `D_e ∝ 1/p` over the fixed diffusion
    /// length `Λ`).
    ///
    /// At 760 Torr the balance is attachment 6.7e7, diffusion 3.3e9 — i.e.
    /// **attachment is negligible here**, and the threshold is set by diffusion
    /// and by the finite-pulse growth requirement.
    ///
    /// Within attachment the *two-body* channel leads at this density (5.4e7
    /// against 1.4e7); three-body overtakes it only above `n = k₂/k₃ = 10²⁶
    /// m⁻³`, about 4 atm. Since three-body is the `∝ p²` channel, that
    /// crossover is what puts the model's positive-slope branch near 10⁴ Torr.
    ///
    /// Like the other rate methods this is a bare pure function and assumes
    /// `pressure > 0`; `p = 0` gives an infinite diffusion loss and `p < 0` a
    /// sign-flipped one. The `Result`-returning entry point
    /// [`threshold_intensity`](Self::threshold_intensity) is where pressure is
    /// validated.
    pub fn loss_rate(&self, pressure: f64) -> f64 {
        let n_neutral = self.n_over_p * pressure;
        let n_o2 = self.gas.f_att * n_neutral;
        let nu_att = self.gas.k_att_2body * n_o2 + self.gas.k_att_3body * n_o2 * n_neutral;
        let nu_diff = self.diffusion_coefficient(pressure) / (self.lambda_diff * self.lambda_diff);
        nu_att + nu_diff
    }

    /// Free-electron diffusion coefficient at `pressure` (m²/s),
    /// `D_e = D_e,ref·(p_ref/p)`.
    ///
    /// Exposed as its own method so the `D_e` gates test the path
    /// [`loss_rate`](Self::loss_rate) actually takes, rather than a
    /// re-derivation of it in the test file.
    pub fn diffusion_coefficient(&self, pressure: f64) -> f64 {
        self.gas.d_e_ref * (P_REF / pressure)
    }

    /// Diffusion length `Λ` (m) — from the focal geometry, **pinned, not fit**.
    pub fn diffusion_length(&self) -> f64 {
        self.lambda_diff
    }

    /// This model's cascade plateau floor (W/m²) — see
    /// [`cascade_plateau_intensity`]. Below it
    /// [`CascadeModel::SelfConsistentClimb`] gives `ν_i = 0` at every pressure,
    /// so no cascade-only threshold can lie beneath it.
    pub fn plateau_intensity(&self) -> f64 {
        cascade_plateau_intensity(self.omega, self.gas.u_ion, self.gas.delta_eff)
    }

    /// Multiphoton seed rate `S_mpi` (m⁻³·s⁻¹), written about a reference
    /// intensity rather than as a bare cross-section:
    ///
    /// ```text
    /// S_mpi = N · W_ref · (I/I_ref)^K
    /// ```
    ///
    /// The `σ_K·I^K` form is unusable here: at `K = 14` and SI intensities it
    /// overflows `f64` well inside the bisection bracket (`I^14 > 10³⁰⁸` for
    /// `I > 10²²`), while `σ_K` itself underflows to ~`10⁻¹⁸⁶`. The ratio form
    /// keeps every intermediate in range.
    pub fn mpi_source(&self, intensity: f64, pressure: f64) -> f64 {
        let n_neutral = self.n_over_p * pressure;
        if self.keldysh_prefactor > 0.0 {
            return n_neutral
                * keldysh_rate(
                    intensity,
                    self.omega,
                    self.gas.u_ion,
                    self.keldysh_prefactor,
                );
        }
        if self.mpi_rate_ref == 0.0 {
            return 0.0;
        }
        n_neutral * self.mpi_rate_ref * (intensity / self.mpi_i_ref).powi(self.k_photons)
    }

    /// Advance `n_e` by one time-slice `dt` at constant intensity, using the
    /// exact exponential solution of the linear rate ODE:
    /// `n_e' = n_e·e^{β dt} + S·expm1(β dt)/β` with `β = ν_i − ν_loss`,
    /// `S = S_mpi`. `expm1` keeps the `β → 0` limit (`n_e + S·dt`) accurate.
    /// The growth exponent is saturated at [`MAX_EXPONENT`] so a violently
    /// over-threshold pulse stays monotone rather than overflowing to `NaN`.
    pub fn advance(&self, n_e: f64, intensity: f64, pressure: f64, dt: f64) -> f64 {
        let nu_i = self.cascade_rate(intensity, pressure);
        let beta = nu_i - self.loss_rate(pressure);
        let s = self.mpi_source(intensity, pressure);
        // Ionization consumes neutrals, so the cascade cannot outrun the gas it
        // feeds on: `b = ν_i/N` makes the growth logistic and caps `n_e` at full
        // ionization. Without it the equation is linear and `n_e` runs away —
        // it reached 10^40 m⁻³ in the shipped `breakdown` case, 10^14 times the
        // neutral density and 10^13 times critical density, i.e. pure
        // extrapolation dressed as a result.
        let n_neutral = self.n_over_p * pressure;
        let b = if n_neutral > 0.0 {
            nu_i / n_neutral
        } else {
            0.0
        };

        let bdt = beta * dt;

        if s == 0.0 {
            if n_e <= 0.0 {
                return 0.0;
            }
            // Exact solution of `dn/dt = β·n − b·n²` (logistic/Bernoulli),
            // written as
            //     n' = n / (e^{−β dt} + b·n·(1 − e^{−β dt})/β)
            // rather than the textbook `β n e^{βdt} / (β + b n (e^{βdt} − 1))`.
            // Two reasons, both load-bearing. The textbook form overflows to
            // `inf/inf = NaN` far above threshold. And `e^{−β dt}` *underflows*
            // harmlessly to 0 for a strong cascade, leaving `n' → β/b` — the
            // saturation density — so no exponent clamp is needed on the
            // growing side. Clamping there would corrupt the saturation limit
            // itself (it made `peak_ne` non-monotonic in intensity: 0.1·N at
            // 10²⁴ W/m² between 1.0·N at 10²⁰ and 10³⁰).
            // Still exact, still step-size independent.
            let neg = (-bdt).min(MAX_EXPONENT); // only β < 0 can overflow `exp`
            let em = neg.exp();
            // (1 − e^{−β dt})/β, with the removable singularity at β → 0.
            let factor = if bdt.abs() < 1e-12 {
                dt
            } else {
                -neg.exp_m1() / beta
            };
            let denom = em + b * n_e * factor;
            return if denom > 0.0 && denom.is_finite() {
                n_e / denom
            } else {
                n_e * bdt.min(MAX_EXPONENT).exp()
            };
        }

        // With the multiphoton source the equation is Riccati and has no
        // elementary closed form. Fall back to the exact *linear* step with the
        // depletion factor frozen at the slice start — first-order in `dt`, and
        // documented as such (`σ_K = 0` by default, so the exact path above is
        // the one every gate exercises).
        let rate = beta - b * n_e;
        let d = (rate * dt).clamp(-MAX_EXPONENT, MAX_EXPONENT);
        // expm1(x)/x → 1 as x → 0; guard the exact-zero and tiny cases.
        let source_factor = if d.abs() < 1e-12 {
            dt
        } else {
            d.exp_m1() / rate
        };
        let next = n_e * d.exp() + s * source_factor;
        // Full ionization is the ceiling on this path too.
        if n_neutral > 0.0 {
            next.min(n_neutral)
        } else {
            next
        }
    }

    /// Peak electron density reached over a Gaussian temporal pulse of peak
    /// intensity `i_peak` (W/m²) and full-width-half-maximum `fwhm` (s) at
    /// pressure `p`, sampled with `n_steps` slices over
    /// `[−w·fwhm, +w·fwhm]`, `w` = [`window_half_widths`](Self::with_window).
    ///
    /// `n_e` is floored at the seed density throughout, and that floor is
    /// **physics, not hygiene**. Without it the model is not window-independent:
    /// losses grind the seed down during the quiet arm before the pulse
    /// arrives — at 760 Torr, `ν_loss ≈ 3.4×10⁹ s⁻¹` decays it by `e^-41` over
    /// 12 ns — so the avalanche must first climb back out of a hole that the
    /// arbitrary choice of `w` dug. That is meaningless as physics (`e^-41`
    /// times one electron is `10⁻¹⁸` of an electron in a volume that holds
    /// either one or none) and it made the threshold depend on `w`, which the
    /// `threshold_is_window_independent` gate now forbids. The floor states the
    /// modelling assumption plainly: one seed electron is available in the
    /// focal volume when the pulse arrives.
    pub fn peak_ne(&self, i_peak: f64, fwhm: f64, pressure: f64, n_steps: usize) -> f64 {
        let half = self.window_half_widths * fwhm;
        let dt = 2.0 * half / n_steps as f64;
        // I(t) = I_pk·exp(−4 ln2 (t/fwhm)²); t centred on the pulse peak.
        let c = 4.0 * std::f64::consts::LN_2 / (fwhm * fwhm);
        let mut n_e = self.n_seed;
        let mut peak = n_e;
        for step in 0..n_steps {
            let t = -half + (step as f64 + 0.5) * dt;
            let intensity = i_peak * (-c * t * t).exp();
            n_e = self.advance(n_e, intensity, pressure, dt).max(self.n_seed);
            if n_e > peak {
                peak = n_e;
            }
        }
        peak
    }

    /// Whether a pulse of peak intensity `i_peak` breaks the gas down at
    /// pressure `p` (peak `n_e` reaches the criterion density `n_bd`).
    pub fn breaks_down(&self, i_peak: f64, fwhm: f64, pressure: f64, n_steps: usize) -> bool {
        self.peak_ne(i_peak, fwhm, pressure, n_steps) >= self.n_bd
    }

    /// Breakdown threshold peak intensity `I_thr(p)` (W/m²) for a `fwhm` pulse,
    /// by bisection in log-intensity.
    ///
    /// The bracket is **fixed** at [`I_BRACKET_LO`] … [`I_BRACKET_HI`]
    /// (1e12 … 1e22 W/m²) — it is not widened. Errors, rather than looping or
    /// returning an endpoint, if the bracket does not straddle threshold; at
    /// low pressure the diffusion branch climbs towards the ceiling (≈3e21 W/m²
    /// at 10 Torr for the pinned focus), so a change to `D_e,ref` or `Λ` can
    /// push points out of range.
    pub fn threshold_intensity(&self, fwhm: f64, pressure: f64, n_steps: usize) -> Result<f64> {
        if !(pressure > 0.0 && pressure.is_finite()) {
            bail!("pressure must be positive and finite, got {pressure}");
        }
        let mut lo = I_BRACKET_LO;
        let mut hi = I_BRACKET_HI;
        if self.breaks_down(lo, fwhm, pressure, n_steps) {
            bail!("threshold below bracket floor {lo:.1e} W/m² at p = {pressure:.1} Pa");
        }
        if !self.breaks_down(hi, fwhm, pressure, n_steps) {
            bail!("threshold above bracket ceiling {hi:.1e} W/m² at p = {pressure:.1} Pa");
        }
        // Bisect in log-intensity: threshold spans orders of magnitude.
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.breaks_down(mid, fwhm, pressure, n_steps) {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi / lo < 1.0 + 1e-6 {
                break;
            }
        }
        Ok((lo * hi).sqrt())
    }

    /// Threshold curve `I_thr(p)` over `n` log-spaced pressures in
    /// `[p_min, p_max]` (Pa). Returns `(pressure, threshold)` pairs; a pressure
    /// whose threshold escapes the bracket is skipped with its error dropped
    /// (the sweep is a survey, not a gate).
    pub fn pressure_sweep(
        &self,
        p_min: f64,
        p_max: f64,
        n: usize,
        fwhm: f64,
        n_steps: usize,
    ) -> Vec<(f64, f64)> {
        (0..n)
            .filter_map(|i| {
                let frac = i as f64 / (n - 1).max(1) as f64;
                let p = p_min * (p_max / p_min).powf(frac);
                self.threshold_intensity(fwhm, p, n_steps)
                    .ok()
                    .map(|it| (p, it))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::loglog_slope;

    const TORR: f64 = 133.322_368_4; // Pa per Torr

    /// Pinned pressure range of the M6a slope gate (Pa) — see
    /// `high_pressure_threshold_slope_lies_between_analytic_limits` and
    /// docs/M6A_SPEC.md. Brackets the T&T measurement range; the fitted
    /// exponent is only defined relative to it.
    const GATE_P_LO: f64 = 300.0 * TORR;
    const GATE_P_HI: f64 = 2000.0 * TORR;

    fn model() -> AirBreakdown {
        AirBreakdown::air_1064nm()
    }

    // --- Integrator unit tests (NOT physics validation — closed-form limits of
    // the rate ODE, docs/M6A_SPEC.md § Integrator unit tests). ----------------

    #[test]
    fn pure_cascade_is_exponential_growth() {
        // ν_i only: a hand-made model with no losses, no MPI. Compare the
        // stepped solution to n_e0·exp(ν_i·t).
        let m = model();
        let p = 760.0 * TORR;
        let intensity = 1e15; // W/m²; keeps ν_i·t modest (no exp overflow)
        let nu_i = m.cascade_rate(intensity, p);
        // Suppress losses and source for this limit by evaluating advance with
        // a model whose loss and MPI are zero: build a bespoke one.
        let bare = AirBreakdown {
            gas: Gas {
                k_att_2body: 0.0,
                k_att_3body: 0.0,
                d_e_ref: 0.0,
                ..m.gas
            },
            mpi_rate_ref: 0.0,
            ..m
        };
        let dt = 1e-11;
        let steps = 50;
        let mut n_e = 1.0;
        for _ in 0..steps {
            n_e = bare.advance(n_e, intensity, p, dt);
        }
        let analytic = (nu_i * dt * steps as f64).exp();
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "cascade {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn pure_loss_is_exponential_decay() {
        let m = model();
        let p = 760.0 * TORR;
        // No cascade (zero intensity), no MPI: pure loss decay.
        let loss = m.loss_rate(p);
        let bare = AirBreakdown {
            mpi_rate_ref: 0.0,
            ..m
        };
        let dt = 1e-10;
        let steps = 30;
        let mut n_e = 1e20;
        for _ in 0..steps {
            n_e = bare.advance(n_e, 0.0, p, dt);
        }
        let analytic = 1e20 * (-loss * dt * steps as f64).exp();
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "loss {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn mpi_only_seeding_is_linear() {
        // β = 0 (no cascade, no loss), constant source S: n_e(t) = n_e0 + S·t.
        let m = model();
        let p = 760.0 * TORR;
        let s = 1e30; // m⁻³ s⁻¹
        let balanced = AirBreakdown {
            gas: Gas {
                k_att_2body: 0.0,
                k_att_3body: 0.0,
                d_e_ref: 0.0,
                ..m.gas
            },
            ..m
        };
        // Force β=0 by giving cascade zero intensity and injecting S via a
        // constant source: emulate with advance at I=0 but nonzero mpi by
        // choosing the reference so that S_mpi(I0) = s exactly.
        let i0: f64 = 1e17;
        let src = AirBreakdown {
            mpi_rate_ref: s / (balanced.n_over_p * p),
            mpi_i_ref: i0,
            ..balanced
        };
        // At I0 the cascade is nonzero, so also null it to isolate β=0: use a
        // separate check via advance math with cascade suppressed.
        let src0 = AirBreakdown {
            gas: Gas {
                u_ion: f64::INFINITY, // ν_i → 0
                ..src.gas
            },
            ..src
        };
        let dt = 1e-11;
        let steps = 40;
        let mut n_e = 0.0;
        for _ in 0..steps {
            n_e = src0.advance(n_e, i0, p, dt);
        }
        let analytic = s * dt * steps as f64;
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "mpi {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn balance_point_is_linear_from_seed() {
        // β=0 with a nonzero seed: n_e(t) = n_e0 + S·t.
        let m = model();
        let p = 760.0 * TORR;
        let n0 = 1e18;
        let s = 5e29;
        let i0: f64 = 1e17;
        let bal = AirBreakdown {
            gas: Gas {
                u_ion: f64::INFINITY, // no cascade → β = −loss; cancel loss too
                k_att_2body: 0.0,
                k_att_3body: 0.0,
                d_e_ref: 0.0,
                ..m.gas
            },
            mpi_rate_ref: s / (m.n_over_p * p),
            mpi_i_ref: i0,
            ..m
        };
        let dt = 1e-11;
        let steps = 25;
        let mut n_e = n0;
        for _ in 0..steps {
            n_e = bal.advance(n_e, i0, p, dt);
        }
        let analytic = n0 + s * dt * steps as f64;
        assert!(
            (n_e / analytic - 1.0).abs() < 1e-9,
            "balance {n_e:e} vs analytic {analytic:e}"
        );
    }

    #[test]
    fn slice_refinement_is_consistent() {
        // The exact per-slice solution must be step-size independent for
        // constant intensity: one step of dt equals m steps of dt/m.
        let m = model();
        let p = 500.0 * TORR;
        let intensity = 2e17;
        let dt = 2e-10;
        let coarse = m.advance(1e15, intensity, p, dt);
        let mut fine = 1e15;
        let sub = 64;
        for _ in 0..sub {
            fine = m.advance(fine, intensity, p, dt / sub as f64);
        }
        assert!(
            (coarse / fine - 1.0).abs() < 1e-9,
            "coarse {coarse:e} vs fine {fine:e}"
        );
    }

    // --- Model behaviour + physics-gate observable ---------------------------

    #[test]
    fn cascade_is_affine_in_intensity_and_linear_in_pressure() {
        // FixedMeanEnergy only: SelfConsistentClimb is nonlinear in I by design.
        // Both heating and inelastic loss carry the same factor p, so at fixed
        // intensity ν_i ∝ p EXACTLY — which is the scaling the external E_eff
        // gate confirms, and it survives the inelastic term untouched.
        let m = model().with_cascade_model(CascadeModel::FixedMeanEnergy);
        let p = 300.0 * TORR;
        let i = 1e17;
        // Linear only to the (ν_m/ω)² correction of the IB Lorentzian, ~1e-6.
        assert!((m.cascade_rate(i, 2.0 * p) / m.cascade_rate(i, p) - 2.0).abs() < 1e-4);

        // In intensity the rate is AFFINE, not linear: equal intensity steps
        // give equal rate steps, but the line does not pass through the origin.
        let (a, b, c) = (
            m.cascade_rate(i, p),
            m.cascade_rate(2.0 * i, p),
            m.cascade_rate(3.0 * i, p),
        );
        assert!(((b - a) / (c - b) - 1.0).abs() < 1e-12, "not affine in I");
        // ν_i = k·(I − I₀) with I₀ > 0, so doubling I more than doubles ν_i.
        // Exactly 2.0 would mean the inelastic offset had gone missing.
        assert!(
            b / a > 2.0,
            "ν_i/I is linear through the origin ({:.6}) — inelastic offset missing",
            b / a
        );
    }

    #[test]
    fn cascade_shuts_off_below_the_inelastic_loss() {
        // The new physics in one assertion: there is a finite intensity below
        // which heating cannot outrun excitation losses and no cascade runs at
        // all. This offset is what gives I_thr(p) a high-pressure plateau; the
        // model could not produce one without it.
        let m = model().with_cascade_model(CascadeModel::FixedMeanEnergy);
        let p = P_REF;
        let i_cut = m.inelastic_loss_power(p) / (m.heating_power(1.0, p));
        assert!(i_cut > 0.0 && i_cut.is_finite());
        assert_eq!(m.cascade_rate(0.5 * i_cut, p), 0.0);
        assert!(m.cascade_rate(2.0 * i_cut, p) > 0.0);
        // At the cut the two powers balance by construction.
        let net = m.heating_power(i_cut, p) - m.inelastic_loss_power(p);
        assert!(net.abs() < 1e-12 * m.inelastic_loss_power(p));
    }

    #[test]
    fn threshold_brackets_and_breaks_down_above() {
        let m = model();
        let p = 760.0 * TORR;
        let fwhm = 6e-9;
        let it = m.threshold_intensity(fwhm, p, 400).unwrap();
        assert!(it.is_finite() && it > 0.0);
        // Just above threshold breaks down; just below does not.
        assert!(m.breaks_down(it * 1.2, fwhm, p, 400));
        assert!(!m.breaks_down(it * 0.8, fwhm, p, 400));
    }

    #[test]
    fn inelastic_loss_envelope_brackets_the_slope() {
        // MODEL-CONSISTENCY GATE over the LITERATURE RANGE of the one lumped
        // constant the inelastic term adds (δ_eff ≈ 0.01–0.05, ⟨ε⟩ ≈ 2–5 eV,
        // ~12× in L′). The model's slope is an envelope, not a point, and this
        // pins that envelope so it cannot drift silently:
        //
        //   δ=0.01 ⟨ε⟩=2 → n=0.785     δ=0.05 ⟨ε⟩=5 → n=0.187
        //
        // Measured n = 0.329 is inside THIS envelope — but note this is the
        // FixedMeanEnergy variant, not the default, and note also that the
        // measurement is NOT a cascade-only observable (it contains MPI). The
        // default SelfConsistentClimb envelope is [0.023, 0.231]. The
        // apples-to-apples target is T&T's own cascade closed form, gated by
        // tt2012_cascade_theory_reference in tests/validation.rs.
        let mut ns = Vec::new();
        for delta in [0.01, 0.02, 0.05] {
            for ev in [2.0, 3.0, 5.0] {
                let m = AirBreakdown::air_1064nm()
                    .with_cascade_model(CascadeModel::FixedMeanEnergy)
                    .with_inelastic_loss(delta, ev);
                let c = m.pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400);
                assert_eq!(c.len(), 8, "sweep lost points at δ={delta}, ⟨ε⟩={ev}");
                ns.push(-loglog_slope(&c).unwrap());
            }
        }
        let lo = ns.iter().cloned().fold(f64::MAX, f64::min);
        let hi = ns.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (0.17..=0.21).contains(&lo) && (0.76..=0.81).contains(&hi),
            "literature envelope moved: n ∈ [{lo:.3}, {hi:.3}], expected ≈[0.187, 0.785]"
        );
        // The whole envelope must sit below the no-inelastic-loss floor of 1.74
        // — that improvement is the point of the term.
        assert!(
            hi < 1.5,
            "envelope top {hi:.3} did not improve on n = 1.737"
        );
    }

    #[test]
    fn self_consistent_climb_eliminates_the_mean_energy() {
        // The whole point of the self-consistent variant: ⟨ε⟩ is not an input.
        // Solving dε/dt = heating − δ_eff·ν_m·ε removes it, leaving only δ_eff.
        // If ⟨ε⟩ ever leaks back into this path, this fails.
        let sweep = |ev: f64| {
            AirBreakdown::air_1064nm()
                .with_inelastic_loss(0.02, ev)
                .with_cascade_model(CascadeModel::SelfConsistentClimb)
                .pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400)
        };
        let (a, b) = (sweep(2.0), sweep(5.0));
        assert_eq!(a.len(), 8);
        let (sa, sb) = (loglog_slope(&a).unwrap(), loglog_slope(&b).unwrap());
        assert!(
            (sa - sb).abs() < 1e-9,
            "⟨ε⟩ still affects the self-consistent path: {sa:.6} vs {sb:.6}"
        );
        // And it must keep ν_i ∝ p, which the external E_eff gate confirms.
        let m = AirBreakdown::air_1064nm().with_cascade_model(CascadeModel::SelfConsistentClimb);
        let (p, i) = (500.0 * TORR, 2e16);
        let ratio = m.cascade_rate(i, 2.0 * p) / m.cascade_rate(i, p);
        assert!(
            (ratio - 2.0).abs() < 1e-3,
            "self-consistent ν_i is no longer ∝ p: {ratio:.6}"
        );
    }

    #[test]
    fn the_two_cascade_models_bracket_the_measurement() {
        // THE DURABLE CLAIM of M6a, and the only slope statement that survived
        // the 2026-07-25 audit. The two limits of the same energy balance sit
        // either side of T&T's n = 0.329: FixedMeanEnergy (⟨ε⟩ free,
        // literature-bounded) is steeper at 0.468; SelfConsistentClimb (⟨ε⟩
        // eliminated) is flatter at 0.095. Neither reproduces the measurement;
        // the truth lies between, which is what a mean-energy treatment of a
        // process governed by the tail of the energy distribution should give.
        //
        // Recording the bracket keeps either from being quietly "improved" into
        // the other's territory — and note the bracket is what remained TRUE
        // when the seed-window artifact was removed and both endpoints moved
        // (0.800 → 0.551 → 0.468 and 0.356 → 0.127 → 0.095 as the seed-window
        // artifact and then the focal geometry were corrected).
        let slope_of = |model| {
            let c = AirBreakdown::air_1064nm()
                .with_cascade_model(model)
                .pressure_sweep(GATE_P_LO, GATE_P_HI, 8, 6e-9, 400);
            assert_eq!(c.len(), 8);
            -loglog_slope(&c).unwrap()
        };
        let fixed = slope_of(CascadeModel::FixedMeanEnergy);
        let selfc = slope_of(CascadeModel::SelfConsistentClimb);
        const MEASURED: f64 = 0.329;
        assert!(
            (0.45..=0.49).contains(&fixed),
            "FixedMeanEnergy slope moved: {fixed:.3}, expected ≈0.468"
        );
        assert!(
            (0.08..=0.11).contains(&selfc),
            "SelfConsistentClimb slope moved: {selfc:.3}, expected ≈0.095"
        );
        assert!(
            selfc < MEASURED && MEASURED < fixed,
            "the models no longer bracket the measurement: \
             {selfc:.3} .. {MEASURED} .. {fixed:.3}"
        );
    }

    #[test]
    fn high_pressure_threshold_slope_lies_between_analytic_limits() {
        // PHYSICS GATE, model-consistency form. Bracketed by two CLOSED-FORM
        // limits rather than a fitted band. Solving the avalanche criterion —
        // cascade must outrun losses and clear the n_seed → n_bd growth
        // requirement G = ln(n_bd/n_seed)/τ within the pulse — gives
        //
        //     I_thr(p) = [ν_att(p) + ν_diff(p) + G] / (A·p),   A ≡ ν_i/(I·p)
        //
        // With attachment from measured rate coefficients it is negligible here
        // (6.7e7 vs 3.3e9 for diffusion at 760 Torr), leaving two terms whose
        // exponents are exact:
        //
        //   * growth-limited, losses → 0 : I_thr = G/(A·p)        ∝ p^-1
        //   * diffusion-limited          : I_thr = ν_diff/(A·p)   ∝ p^-2
        //     (ν_diff = D_e,ref·(P_REF/p)/Λ² ∝ 1/p)
        //
        // Those two bracketed the model to n ∈ [1, 2] BEFORE the inelastic-loss
        // term existed (observed then: n = 1.737). The term adds a third,
        // pressure-independent contribution — a genuine plateau — which drags
        // the slope below that old floor, to n = 0.095 with the default
        // SelfConsistentClimb (0.468 with FixedMeanEnergy). So this test now
        // gates the *combined* form:
        //
        //     I_thr(p) = L′/h + U_i·(ν_diff + ν_att + G)/(h·p)
        //
        // whose exponent runs from 0 (plateau-dominated) to 2 (diffusion), and
        // the assertion is that the default sits in the plateau-influenced
        // regime n ∈ (0, 1) rather than the old loss-only regime n ≥ 1.
        //
        // Neither bracket is universal: they hold only while attachment is
        // negligible. Three-body attachment is ∝ p², contributing I_thr ∝ +p,
        // so far above this window the model leaves them entirely — the slope
        // turns positive above ~10^4 Torr. That is why the range is pinned.
        //
        // This is the model's own consistency, NOT agreement with experiment:
        // T&T measure n = 0.33, outside this interval entirely. That gap is the
        // real M6a finding and is gated separately in tests/validation.rs.
        let m = model();
        let n_points = 8;
        let curve = m.pressure_sweep(GATE_P_LO, GATE_P_HI, n_points, 6e-9, 400);
        assert_eq!(
            curve.len(),
            n_points,
            "gate sweep lost points to the bisection bracket: {curve:?}"
        );
        let slope = loglog_slope(&curve).unwrap();
        assert!(
            (-1.0..0.0).contains(&slope),
            "high-p threshold slope {slope:.3} over {:.0}–{:.0} Torr; with the inelastic-loss \
             plateau the model must sit in n∈(0,1), below the loss-only floor of n=1",
            GATE_P_LO / TORR,
            GATE_P_HI / TORR
        );
    }

    #[test]
    fn attachment_is_negligible_against_diffusion_at_one_atmosphere() {
        // Guards the finding that motivated switching to measured attachment
        // coefficients: the order-of-magnitude K_a·p it replaced was ~150x too
        // large and was the only thing flattening the modelled slope towards
        // the data. If a future edit reinstates a dominant attachment term,
        // this fails and points at docs/M6A_SPEC.md rather than letting the
        // slope quietly drift back into apparent agreement.
        let m = model();
        let p = P_REF;
        let n = m.n_over_p * p;
        let n_o2 = m.gas.f_att * n;
        let nu_att = m.gas.k_att_2body * n_o2 + m.gas.k_att_3body * n_o2 * n;
        let nu_diff = m.gas.d_e_ref * (P_REF / p) / (m.lambda_diff * m.lambda_diff);
        assert!(
            nu_att < 0.05 * nu_diff,
            "attachment {nu_att:e} is not negligible against diffusion {nu_diff:e}"
        );
        // Both channels are kept because both matter at this density, but the
        // *two-body* one leads here — 5.4e7 against 1.4e7. Three-body overtakes
        // it only above n = k₂/k₃ = 1e26 m⁻³ (≈4 atm), which is why the model's
        // ∝ p² branch, and with it the positive-slope regime, sits near 1e4
        // Torr rather than in the gate window. (An earlier comment here claimed
        // three-body was dominant at 1 atm; it is not, and nothing was gated on
        // it — this assertion only ever demanded it be non-negligible.)
        let two_body = m.gas.k_att_2body * n_o2;
        let three_body = m.gas.k_att_3body * n_o2 * n;
        assert!(three_body > 0.1 * two_body && three_body < two_body);
    }

    #[test]
    fn tt2012_mpi_calibration_undershoots_the_data() {
        // Why the multiphoton source stays OFF by default, recorded as a
        // measurement rather than an opinion.
        //
        // T&T give an MPI threshold of 4.42e9 W/cm² at 760 Torr (their
        // Sec. II A) and separately measure breakdown at 2.06e11 — a factor 47
        // higher. So MPI *alone*, read as a breakdown criterion, should have
        // ignited the gas at 4.42e9, and it demonstrably did not. Anchoring a
        // rate coefficient to that number inherits the problem: the threshold
        // collapses to ≈5.5e9 W/cm², ~37× below what the same paper measures,
        // and contradicts its own accounting of 88 % cascade / 12 % MPI at
        // 760 Torr.
        //
        // The conclusion is about the paper's MPI estimate, not about MPI: it
        // is an order-of-magnitude significance indicator (Nelson's flux-density
        // criterion, whose constant C the paper never states), not a rate. A
        // real σ_K from multiphoton cross-section data is the open item.
        let p = 760.0 * TORR;
        let cascade = model();
        let with_mpi = cascade.with_tt2012_mpi(6e-9, p);
        let i_cascade = cascade.threshold_intensity(6e-9, p, 400).unwrap();
        let i_mpi = with_mpi.threshold_intensity(6e-9, p, 400).unwrap();
        const MEASURED: f64 = 2.06e11 * 1e4; // W/m²

        assert!(
            i_mpi < i_cascade / 100.0,
            "MPI no longer dominates the cascade: {i_mpi:.3e} vs {i_cascade:.3e}"
        );
        assert!(
            i_mpi < MEASURED / 20.0,
            "T&T-calibrated MPI threshold {i_mpi:.3e} is no longer far below the \
             measured {MEASURED:.3e}; recheck the calibration before enabling it"
        );
        // And the default must keep it off.
        assert_eq!(model().mpi_source(1e16, p), 0.0);
    }

    #[test]
    fn threshold_is_window_independent() {
        // NUMERICAL-HYGIENE GATE, added 2026-07-25 after this exact defect
        // shipped. The pulse at |t| = 2·FWHM is 1.5e-5 of peak, so widening the
        // integration window must not move the threshold. It used to: losses
        // ground the seed down during the quiet arm (e^-60 at 760 Torr over
        // 12 ns), so the avalanche had to climb out of a hole whose depth was
        // set by an arbitrary integration bound. The threshold rose 11% from
        // w = 2 to w = 4 and never converged.
        //
        // Worse, that decay is pressure-dependent, so it manufactured SLOPE:
        // removing it moved the default model from n = 0.356 to n = 0.127
        // against a measured 0.329 (and the later focal-geometry correction took
        // it to its present 0.095). The external slope gate had been green on
        // the strength of a bookkeeping artifact.
        let base = model();
        let at = |w: f64| {
            base.with_window(w)
                .threshold_intensity(6e-9, 760.0 * TORR, 400)
                .unwrap()
        };
        let widths = [1.0, 1.5, 2.0, 3.0, 4.0];
        let vals: Vec<f64> = widths.iter().map(|&w| at(w)).collect();
        let lo = vals.iter().cloned().fold(f64::MAX, f64::min);
        let hi = vals.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            hi / lo < 1.01,
            "threshold depends on the integration window: {lo:.4e}..{hi:.4e} \
             over w ∈ {widths:?} ({:.1}% spread)",
            100.0 * (hi / lo - 1.0)
        );
    }

    #[test]
    fn cascade_saturates_at_full_ionization() {
        // The gas cannot be more than fully ionized. Before the logistic term
        // the rate equation was linear in n_e and ran to 1e40 m^-3 in the
        // shipped `breakdown` case — 1e14 times the neutral density, 1e13 times
        // critical density at 1064 nm — with the visible ceiling being a
        // plotting clamp rather than any physics.
        let m = model();
        let p = P_REF;
        let n_neutral = m.neutral_density(p);
        let mut prev = 0.0;
        for i in [1e17, 1e18, 1e20, 1e22, 1e24, 1e26, 1e30] {
            let ne = m.peak_ne(i, 6e-9, p, 400);
            assert!(ne.is_finite(), "peak n_e not finite at I = {i:e}");
            assert!(
                ne <= n_neutral * 1.001,
                "n_e = {ne:e} exceeds full ionization N = {n_neutral:e} at I = {i:e}"
            );
            // Monotone in intensity: the clamp used to break this, giving 0.1·N
            // at 1e24 between 1.0·N at 1e20 and 1e30.
            assert!(
                ne >= prev * 0.999,
                "peak n_e non-monotonic in intensity at I = {i:e}: {ne:e} < {prev:e}"
            );
            prev = ne;
        }
    }

    #[test]
    fn far_above_threshold_stays_broken_down() {
        // Monotonicity guard: the growth exponent saturates instead of
        // overflowing, so an absurdly over-threshold pulse must still report
        // breakdown. Unsaturated, exp(β·dt) → inf made n_e NaN and every
        // `n_e > peak` comparison false, so this returned *false*.
        let m = model();
        let p = 760.0 * TORR;
        for i_peak in [1e20, 1e22, 1e24, 1e30] {
            let n_e = m.peak_ne(i_peak, 6e-9, p, 400);
            assert!(!n_e.is_nan(), "peak n_e is NaN at I = {i_peak:e}");
            assert!(
                m.breaks_down(i_peak, 6e-9, p, 400),
                "no breakdown reported at I = {i_peak:e} (peak n_e = {n_e:e})"
            );
        }
    }

    #[test]
    fn threshold_out_of_bracket_errors_cleanly() {
        // A pressure so low that diffusion loss pushes threshold past the
        // bracket ceiling must error, not loop.
        let m = model();
        let fwhm = 6e-9;
        // 0.01 Torr: enormous diffusion loss.
        let r = m.threshold_intensity(fwhm, 0.01 * TORR, 400);
        assert!(r.is_err());
    }

    // --- T11: the distribution-resolved cascade -----------------------------

    #[test]
    fn first_passage_reduces_to_the_mean_energy_climb() {
        // T11-V1, THE gate that makes DistributionResolved a generalization
        // rather than a different model, plus the reason the generalization
        // matters. The first-passage rate is the same drift as
        // SelfConsistentClimb plus photon shot noise D_eps = 1/2 P_heat hbar_w.
        // Shrink hbar_w and the noise must vanish, returning the closed form
        // nu_i = delta_eff*nu_m/ln(e_inf/(e_inf - U_i)).
        //
        // Every point is GRID-CONVERGED before it is used. As hbar_w falls the
        // integrand sharpens (exponent ~ 2 U_i/hbar_w), so a reduction gate on a
        // fixed grid measures its own resolution rather than the limit: at the
        // shipped N = 512 this sweep returns 1.031 at hbar_w = 0.02 eV -- 3 % on
        // the WRONG side of the limit it is meant to approach. That happened
        // here, twice, which is why the self-check below is not optional. The
        // shipped N is gated separately, at the photon energies that occur.
        const FINE: usize = 1 << 17;
        let u_ion = 12.06 * E_CHARGE;
        let nu_relax = 7.9e10; // delta_eff*nu_m at 1 atm; any positive value works
        let rate = |eps_inf: f64, hw: f64| {
            let coarse = first_passage_integral(eps_inf, u_ion, hw, FINE / 2);
            let fine = first_passage_integral(eps_inf, u_ion, hw, FINE);
            assert!(
                (coarse / fine - 1.0).abs() < 1e-4,
                "the reduction sweep is not grid-converged at hbar_w = {:.4} eV \
                 ({:.3e} vs {:.3e}); raise FINE or stop the sweep higher",
                hw / E_CHARGE,
                coarse,
                fine
            );
            0.5 * nu_relax * eps_inf * hw / fine
        };

        // Away from the cutoff the limit must actually be reached.
        for x in [1.24_f64, 2.0, 5.0] {
            let eps_inf = x * u_ion;
            let deterministic = nu_relax / (eps_inf / (eps_inf - u_ion)).ln();
            let mut prev = 0.0;
            for hw_ev in [1.166_f64, 0.5, 0.2, 0.05, 0.02] {
                let ratio = deterministic / rate(eps_inf, hw_ev * E_CHARGE);
                assert!(
                    ratio < 1.0,
                    "at eps_inf/U_i = {x}, hbar_w = {hw_ev} eV the diffusive rate is \
                     SLOWER than the deterministic climb (ratio {ratio:.4}) -- noise \
                     should only ever help an electron over the barrier"
                );
                assert!(
                    ratio > prev,
                    "convergence is not monotone at eps_inf/U_i = {x}: {prev:.5} then {ratio:.5}"
                );
                prev = ratio;
            }
            assert!(
                prev > 0.99,
                "at eps_inf/U_i = {x} the rate is still {:.4}x off the closed form at \
                 hbar_w = 0.02 eV; the D_eps -> 0 limit is not being recovered",
                1.0 / prev
            );
        }

        // NEAR the cutoff it must NOT, and that is the physical content rather
        // than a shortfall of the gate. As eps_inf -> U_i the deterministic
        // climb time diverges logarithmically, so noise dominates it for any
        // finite photon energy and the ratio stays well below 1 however far
        // hbar_w is reduced. That divergence is precisely the bifurcation this
        // variant exists to remove -- and the model sits on top of it, at
        // eps_inf/U_i = 1.032 at 760 Torr.
        let near = 1.02 * u_ion;
        let det_near = nu_relax / (near / (near - u_ion)).ln();
        let r_coarse = det_near / rate(near, 1.166 * E_CHARGE);
        let r_fine = det_near / rate(near, 0.02 * E_CHARGE);
        assert!(
            r_fine > r_coarse && r_fine < 0.95,
            "near the cutoff (eps_inf/U_i = 1.02) the ratio runs {r_coarse:.4} -> \
             {r_fine:.4} over a 58x reduction in hbar_w; it is supposed to rise but \
             stay far short of 1, because the deterministic limit it is approaching \
             is itself divergent there"
        );
    }

    #[test]
    fn first_passage_quadrature_is_converged() {
        // T11-V2. The scheme is 2nd order on a smooth integrand, so the error
        // falls 4x per doubling. This pins both the order and the adequacy of
        // the shipped FIRST_PASSAGE_STEPS, which is NOT free to raise: it runs
        // inside the threshold bisection's inner loop.
        let u_ion = 12.06 * E_CHARGE;
        let hw = 1.166 * E_CHARGE;
        let eps_inf = 1.24 * u_ion;
        let at = |n: usize| first_passage_integral(eps_inf, u_ion, hw, n);
        let reference = at(32_768);
        let e_coarse = (at(128) / reference - 1.0).abs();
        let e_fine = (at(256) / reference - 1.0).abs();
        let order = crate::validate::observed_order(e_coarse, e_fine);
        assert!(
            (1.8..=2.2).contains(&order),
            "first-passage quadrature observed order {order:.3}, expected ~2 \
             (errors {e_coarse:.3e} then {e_fine:.3e})"
        );
        // The shipped N, at BOTH photon energies this model is ever run at and
        // across the ε_∞/U_i range the threshold bisection sweeps through. V1
        // relies on this: it refines the grid for the deep D_ε → 0 limit on the
        // grounds that the shipped N is adequate where the physics actually is,
        // and this is where that is checked rather than assumed.
        for hw_ev in [1.166_f64, 2.331] {
            for x in [1.02_f64, 1.05, 1.24, 2.0, 10.0] {
                let ei = x * u_ion;
                let hw_j = hw_ev * E_CHARGE;
                let reference = first_passage_integral(ei, u_ion, hw_j, 32_768);
                let shipped = (first_passage_integral(ei, u_ion, hw_j, FIRST_PASSAGE_STEPS)
                    / reference
                    - 1.0)
                    .abs();
                assert!(
                    shipped < 2e-4,
                    "at the shipped N = {FIRST_PASSAGE_STEPS} the quadrature is \
                     {shipped:.2e} from converged at ħω = {hw_ev} eV, ε_∞/U_i = {x}; \
                     the threshold bisection resolves to 1e-6"
                );
            }
        }
    }

    #[test]
    fn distribution_resolved_has_no_bifurcation() {
        // T11-V3 — the defect this variant exists to remove, in one test.
        //
        // SelfConsistentClimb switches ionization on discontinuously at
        // ε_∞ = U_i, and the model is evaluated on top of that step
        // (ε_∞/U_i = 1.032 at 760 Torr). A real distribution has a tail that
        // ionizes while the mean is still short of U_i, so the rate must be
        // positive below the old cutoff and continuous through it.
        let m = model().with_cascade_model(CascadeModel::DistributionResolved);
        let old = model().with_cascade_model(CascadeModel::SelfConsistentClimb);
        let p = 760.0 * TORR;
        let u_ion = m.gas.u_ion;

        // Intensity that puts ε_∞ exactly at U_i, then either side of it.
        let i_at_cut = u_ion / m.equilibrium_energy(1.0, p);
        assert!(
            old.cascade_rate(i_at_cut * 0.99, p) == 0.0
                && old.cascade_rate(i_at_cut * 1.01, p) > 0.0,
            "the old model no longer has the step this gate is contrasted against"
        );

        // Below the old cutoff the new rate is positive and rising.
        let mut prev = 0.0;
        for f in [0.5, 0.7, 0.9, 0.99] {
            let r = m.cascade_rate(i_at_cut * f, p);
            assert!(
                r > 0.0,
                "no ionization at ε_∞ = {f}·U_i — the bifurcation is still there"
            );
            assert!(r > prev, "rate not monotone in intensity below the cutoff");
            prev = r;
        }

        // And continuous across it: the old model jumps from 0, the new one does
        // not. Compare a tight bracket around the cut.
        let lo = m.cascade_rate(i_at_cut * 0.999, p);
        let hi = m.cascade_rate(i_at_cut * 1.001, p);
        assert!(
            (hi / lo - 1.0).abs() < 0.02,
            "rate jumps {:.3}× across ε_∞ = U_i ({lo:.4e} → {hi:.4e}); it should be \
             smooth there now",
            hi / lo
        );
    }

    #[test]
    fn first_passage_rate_depends_only_on_two_dimensionless_groups() {
        // T11-V4 — no new constant entered. Non-dimensionalising the OU process
        // by τ = δ_eff·ν_m·t leaves ν_i/(δ_eff·ν_m) a function of ε_∞/U_i and
        // ħω/U_i alone. So the rate must be blind to k_m and D_e at fixed groups,
        // and must scale exactly linearly in ν_relax — which is what keeps the
        // ν_i ∝ p scaling the external E_eff gate confirms.
        let u_ion = 12.06 * E_CHARGE;
        let hw = 1.166 * E_CHARGE;
        let eps_inf = 1.3 * u_ion;
        let base = first_passage_ionization_rate(1.0, eps_inf, u_ion, hw);
        for k in [1e-3, 1.0, 7.9e10, 1e14] {
            let r = first_passage_ionization_rate(k, eps_inf, u_ion, hw);
            assert!(
                (r / (k * base) - 1.0).abs() < 1e-12,
                "rate is not linear in ν_relax at k = {k:e}: {r:e} vs {:e}",
                k * base
            );
        }
        // Same groups, different absolute energies: identical dimensionless rate.
        let scaled = first_passage_ionization_rate(1.0, 2.0 * eps_inf, 2.0 * u_ion, 2.0 * hw);
        assert!(
            (scaled / base - 1.0).abs() < 1e-12,
            "the rate is not a function of the dimensionless groups alone: \
             {scaled:e} vs {base:e}"
        );
        // ν_i ∝ p survives, through the model rather than the free function.
        let m = model().with_cascade_model(CascadeModel::DistributionResolved);
        let (p, i) = (500.0 * TORR, 2e16);
        let ratio = m.cascade_rate(i, 2.0 * p) / m.cascade_rate(i, p);
        assert!(
            (ratio - 2.0).abs() < 1e-3,
            "distribution-resolved ν_i is no longer ∝ p: {ratio:.6}"
        );
    }

    #[test]
    fn photon_count_is_eleven_at_1064nm() {
        // ⌈12.06 eV / 1.166 eV⌉ = 11.
        assert_eq!(model().k_photons, 11);
    }
}
