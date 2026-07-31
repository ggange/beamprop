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
//! (`tt2012_threshold_slope_matches_measurement`) was red on purpose until the
//! cascade closure changed; see the distribution-resolved note below.
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
//! The pressure-slope **bracket** (`n = 0.095` for the mean-trajectory closure,
//! `0.468` for the fixed-`⟨ε⟩` limit, straddling 0.329) is a weaker claim than
//! it looks, and it is superseded by the distribution-resolved default below:
//! the
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
/// Hartree energy (J) — the atomic unit of energy (CODATA 2018).
const HARTREE: f64 = 4.359_744_722_207_1e-18;
/// Atomic unit of electric field (V/m) — `E_h/(e·a₀)` (CODATA 2018).
const F_ATOMIC: f64 = 5.142_206_747_63e11;

/// Largest growth exponent `β·dt` the integrator evaluates; `exp` overflows
/// above ~709. See [`AirBreakdown::advance`].
const MAX_EXPONENT: f64 = 700.0;

/// Below this `γ` the direct form of [`keldysh_tunnel_exponent`] loses precision
/// to cancellation, and the small-`γ` series is used instead.
const KELDYSH_SERIES_CUTOVER: f64 = 0.1;

/// Above this `x` the Maclaurin series for [`dawson`] loses to cancellation and
/// the asymptotic series is used instead. Both are accurate to ~1.3×10⁻⁷ at the
/// join, which is where the joint error is smallest — the series degrades fast
/// above it (4.5×10⁻¹ by `x` = 6) and the asymptotic series degrades below it.
const DAWSON_SERIES_CUTOVER: f64 = 4.0;

/// How deep into the exponential tail the PPT above-threshold sum is carried:
/// terms are kept until `α(γ)·(n−ν)` exceeds this, i.e. to `e⁻⁴⁰` ≈ 4×10⁻¹⁸.
///
/// The term count is **derived from this, not fixed**, because the required
/// depth runs from ~10 terms at `γ` = 10 to tens of thousands as `γ → 0`:
/// `α = 2[asinh γ − γ/√(1+γ²)] → ⅔γ³`, so the sum stops converging in the
/// tunnelling limit. A fixed 64 terms — which this originally shipped as — is
/// converged to 1.7×10⁻⁷ at `γ` = 0.8 and *29 %* wrong at `γ` = 0.2, with the
/// error masquerading as physics. See [`ppt_ati_terms`].
const PPT_ATI_LOG_DEPTH: f64 = 40.0;

/// Cap on the PPT above-threshold sum, so the cost stays bounded as `α → 0`.
/// At this cap the sum is converged down to `γ` ≈ [`PPT_TUNNELLING_CUTOVER`],
/// which is where the ADK closed form takes over.
const PPT_ATI_MAX_TERMS: usize = 1 << 16;

/// Below this `γ` the PPT above-threshold factor is replaced by its tunnelling
/// limit `A₀ → 1`, i.e. the rate becomes ADK.
///
/// Not a convenience: the sum's decay constant vanishes as `⅔γ³`, so reaching
/// the limit numerically would need ~10⁵ terms at `γ` = 0.1 and ~10⁸ by
/// `γ` = 0.01. The closed form is exact there and free. The join costs 0.6 %
/// (converged `A₀` = 0.9941 at `γ` = 0.1 against the limit's 1), which
/// `ppt_tunnelling_branch_joins_the_sum` gates.
const PPT_TUNNELLING_CUTOVER: f64 = 0.1;

/// Cosmic-ray and radon ion-pair production rate in air at the surface
/// (m⁻³·s⁻¹).
///
/// ≈ 10 ion pairs cm⁻³ s⁻¹, a standard atmospheric-electricity value (AFRL
/// *Handbook of Geophysics and the Space Environment*, ch. 20 "Atmospheric
/// Electricity", Sagalyn & Burke, 1985). It is a **band, not a point**: ≈2 over
/// ocean, where cosmic rays are the only source, rising to ≈10 over land where
/// radon dominates.
///
/// That range would matter if the answer depended on it. It does not — once
/// multiphoton ionization supplies the seed, the threshold is insensitive to
/// this over *decades*, which `ionization_background_is_not_load_bearing`
/// gates. That insensitivity is the reason it is defensible to introduce this
/// constant at all: it replaces an assumption the model was known to be making
/// wrongly, and it does not become a new one.
pub const ION_PAIR_PRODUCTION: f64 = 1.0e7;

/// Effective residual charge `Z_eff` for molecular O₂ in PPT.
///
/// **Published, not fitted**: Talebpour, Chien and Chin, *J. Phys. B* 32, 1229
/// (1999), fit PPT to measured O₂ ionization at 800 nm and report
/// `Z_eff` = 0.53. It is below 1 because the departing electron in a molecule
/// does not see a bare unit charge, and it is the single molecular parameter
/// [`ppt_rate`] needs. Supplying it from the literature rather than from this
/// project's own threshold data is what keeps the resulting rate an independent
/// prediction — see `docs/M6A_SPEC.md` § "PPT for molecular O₂".
pub const Z_EFF_O2: f64 = 0.53;

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
    /// `E_eff` gate confirms). **Was the default until 2026-07-30**, chosen for
    /// parameter parsimony — *not* for agreement: it gives `n = 0.095` against a
    /// measured 0.329, which is the flatter side of the bracket and misses by
    /// 3.5×. Retained because a great many gate numbers are stated against it,
    /// and because it is the `D_ε → 0` limit that
    /// [`Self::DistributionResolved`] is verified to reduce to.
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
    /// **The default since 2026-07-30.** It landed first as a variant, so that
    /// what it did to every published M6a number was measured rather than
    /// asserted; it was promoted once those numbers were in. The promotion
    /// retired M6a's long-standing red gate — `tt2012_threshold_slope_matches_measurement`
    /// had been `#[ignore]`d and failing since 2026-07-25 and now passes, with
    /// no tolerance moved and no constant touched. See `docs/M6A_SPEC.md`
    /// § Distribution-resolved cascade.
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

/// Dawson's integral `Φ(x) = e^{−x²}∫₀ˣ e^{y²} dy`.
///
/// Needed by [`ppt_rate`]'s above-threshold sum. Two series with a documented
/// join at [`DAWSON_SERIES_CUTOVER`], in the same pattern as
/// [`keldysh_tunnel_exponent`]:
///
/// - `x ≤ 4`: the Maclaurin series `Σ (−2)ⁿ x^{2n+1}/(2n+1)!!`, exact term
///   recurrence, accurate to 1.6×10⁻⁹ over the range.
/// - `x > 4`: the asymptotic series `(1/2x)·Σ (2n−1)!!/(2x²)ⁿ`, truncated at
///   its smallest term (it is divergent), accurate to 1.3×10⁻⁷ at the join and
///   rapidly better above it.
///
/// Odd, so negative arguments reflect. `ppt_rate` only ever calls it with
/// `x = √(β(n−ν)) ∈ [0, ~4]`, but the asymptotic branch is kept so the function
/// is correct as published rather than correct only where it happens to be used.
pub fn dawson(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_nan() { f64::NAN } else { 0.0 };
    }
    if x < 0.0 {
        return -dawson(-x);
    }
    if x <= DAWSON_SERIES_CUTOVER {
        let mut term = x;
        let mut sum = x;
        for n in 1..500 {
            term *= -2.0 * x * x / (2 * n + 1) as f64;
            sum += term;
            if term.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        return sum;
    }
    let y = 2.0 * x * x;
    let mut term = 0.5 / x;
    let mut sum = term;
    let mut prev = term.abs();
    for n in 1..60 {
        term *= (2 * n - 1) as f64 / y;
        // Divergent series: stop at the smallest term, which is where the
        // truncation error is minimised.
        if term.abs() > prev {
            break;
        }
        sum += term;
        prev = term.abs();
    }
    sum
}

/// `ln Γ(x)` for `x > 0`, Lanczos approximation (g = 7, n = 9).
///
/// Private because [`ppt_rate`] is its only caller, and it is only ever
/// evaluated at `n*` ≈ 0.56 and `n*+1` — but it is gated where the closed forms
/// are exact (integers, `Γ(½) = √π`, the reflection formula) rather than only
/// where it is used.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π/sin(πx).
        return (PI / (PI * x).sin()).abs().ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut series = COEF[0];
    for (i, c) in COEF.iter().enumerate().skip(1) {
        series += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// Decay constant of the PPT above-threshold sum,
/// `α(γ) = 2[asinh γ − γ/√(1+γ²)]`.
///
/// Its behaviour is what forces the rest of the design: `α → 2 ln 2γ − 2` for
/// `γ ≫ 1` (fast decay, a handful of terms) but `α → ⅔γ³` for `γ ≪ 1`, so the
/// number of terms needed diverges as the third power of `1/γ`.
fn ppt_ati_decay(gamma: f64) -> f64 {
    2.0 * (gamma.asinh() - gamma / (1.0 + gamma * gamma).sqrt())
}

/// Terms the above-threshold sum needs at this `γ` to reach
/// [`PPT_ATI_LOG_DEPTH`], capped at [`PPT_ATI_MAX_TERMS`].
///
/// Derived rather than fixed, and *cheaper* than the fixed 64 it replaces
/// wherever the kernel actually runs: 10 terms at `γ` = 10, 5 at `γ` = 40.
fn ppt_ati_terms(gamma: f64) -> usize {
    let alpha = ppt_ati_decay(gamma);
    // Non-positive or non-finite α means the decay has vanished entirely; ask
    // for the cap and let the caller's tunnelling branch take over.
    if alpha <= 0.0 || !alpha.is_finite() {
        return PPT_ATI_MAX_TERMS;
    }
    ((PPT_ATI_LOG_DEPTH / alpha).ceil() as usize).clamp(1, PPT_ATI_MAX_TERMS)
}

/// PPT's above-threshold-ionization factor
///
/// ```text
/// A₀(ω,γ) = (4/√(3π))·(γ²/(1+γ²))·Σ_{n≥⌈ν⌉} e^{−α(n−ν)}·Φ(√(β(n−ν)))
/// ```
///
/// with `α = 2[asinh γ − γ/√(1+γ²)]` and `β = 2γ/√(1+γ²)`. Split out of
/// [`ppt_rate`] with `terms` explicit so `ppt_ati_sum_is_converged` can vary the
/// truncation, which is the only way to show [`PPT_ATI_TERMS`] is a converged
/// choice rather than a tuned one.
///
/// The sum runs over channels absorbing `n ≥ ⌈ν⌉` photons — the `⌈·⌉` is a real
/// channel closing, so `A₀` has genuine kinks as `ν` crosses an integer. They
/// are physics (the ATI channel-closing structure), not numerical noise.
fn ppt_ati_sum(gamma: f64, nu: f64, terms: usize) -> f64 {
    let root = (1.0 + gamma * gamma).sqrt();
    let alpha = ppt_ati_decay(gamma);
    let beta = 2.0 * gamma / root;
    let n0 = nu.ceil();
    let mut sum = 0.0;
    for k in 0..terms {
        let d = n0 + k as f64 - nu;
        let e = -alpha * d;
        if e < -MAX_EXPONENT {
            break;
        }
        sum += e.exp() * dawson((beta * d).sqrt());
    }
    4.0 / (3.0 * PI).sqrt() * (gamma * gamma / (1.0 + gamma * gamma)) * sum
}

/// Per-neutral PPT (Perelomov–Popov–Terent'ev) photoionization rate (s⁻¹) for
/// linear polarization and an `l = m = 0` ground state.
///
/// ```text
/// W = |C_n*|²·√(6/π)·U_i·(2F₀/(F√(1+γ²)))^{2n*−3/2}·A₀(ω,γ)·exp[−(2U_i/ħω)f(γ)]
/// ```
///
/// in atomic units, with `κ = √(2U_i)`, `F₀ = κ³`, `n* = Z_eff/κ`, and
///
/// ```text
/// |C_n*|² = 2^{2n*}/(n*·Γ(n*+1)·Γ(n*))
/// A₀(ω,γ) = (4/√(3π))·(γ²/(1+γ²))·Σ_{n≥⌈ν⌉} e^{−α(n−ν)}·Φ(√(β(n−ν)))
/// ν = (U_i/ħω)(1 + 1/2γ²)      α = 2[asinh γ − γ/√(1+γ²)]      β = 2γ/√(1+γ²)
/// ```
///
/// **Why this and not [`keldysh_rate`] with a multiplier.** Keldysh's prefactor
/// is an order-unity function left explicit there because no derivation pins it;
/// PPT's is *fully specified* once `Z_eff` is given, which is why this function
/// takes no prefactor argument. That is the entire point of the change: it turns
/// the soft multiplier into a prediction. `Z_eff` for O₂ is published
/// ([`Z_EFF_O2`]).
///
/// **The exponent is shared, deliberately.** PPT's `(2F₀/3F)·g(γ)` is
/// algebraically identical to `(2U_i/ħω)·f(γ)`, so
/// [`keldysh_tunnel_exponent`] is reused rather than re-derived, and every gate
/// that bounds the Keldysh exponent bounds this rate too
/// (`ppt_and_keldysh_share_the_same_exponent`). What PPT adds is entirely
/// prefactor: the Coulomb factor and the above-threshold sum `A₀`.
///
/// **`ν` is the ponderomotively shifted photon order**, `(U_i/ħω)(1+1/2γ²)`,
/// not `⌈U_i/ħω⌉` — the intensity-dependent Stark shift of the continuum edge
/// is in the theory, and it is why the fitted power law softens below `K` as
/// `γ` falls (`ppt_multiphoton_order_matches_nu`).
///
/// Validated in absolute magnitude against a measured cross-section, not
/// against threshold data: `ppt_rate_matches_the_measured_o2_cross_section`.
pub fn ppt_rate(intensity: f64, omega: f64, u_ion: f64, z_eff: f64) -> f64 {
    if !(intensity > 0.0 && omega > 0.0 && u_ion > 0.0 && z_eff > 0.0) {
        return 0.0;
    }
    // Atomic units.
    let ip = u_ion / HARTREE;
    let kappa = (2.0 * ip).sqrt();
    let f0 = kappa * kappa * kappa;
    let n_star = z_eff / kappa;
    let omega_au = omega * HBAR / HARTREE;
    let field = (2.0 * intensity / (EPS0 * C_LIGHT)).sqrt() / F_ATOMIC;
    let gamma = omega_au * kappa / field;
    if !(gamma > 0.0 && gamma.is_finite()) {
        return 0.0;
    }
    let root = (1.0 + gamma * gamma).sqrt();

    // Coulomb normalisation |C_n*|², via ln Γ to stay in range for any n*.
    let ln_c2 =
        2.0 * n_star * 2.0f64.ln() - n_star.ln() - ln_gamma(n_star + 1.0) - ln_gamma(n_star);
    let coulomb = (2.0 * n_star - 1.5) * (2.0 * f0 / (field * root)).ln();

    // Above-threshold factor. Below the cutover the sum's decay constant has
    // collapsed and its tunnelling limit A₀ → 1 is used instead, which turns
    // the whole expression into ADK — gated both ways by
    // `ppt_reduces_to_adk_in_the_tunnelling_limit` and
    // `ppt_tunnelling_branch_joins_the_sum`.
    let a0 = if gamma < PPT_TUNNELLING_CUTOVER {
        1.0
    } else {
        let nu = ip / omega_au * (1.0 + 1.0 / (2.0 * gamma * gamma));
        ppt_ati_sum(gamma, nu, ppt_ati_terms(gamma))
    };
    if a0 <= 0.0 {
        return 0.0;
    }

    let exponent = 2.0 * ip / omega_au * keldysh_tunnel_exponent(gamma);
    let ln_w = ln_c2 + 0.5 * (6.0 / PI).ln() + ip.ln() + coulomb + a0.ln() - exponent;
    if ln_w.is_nan() {
        return 0.0;
    }
    // Saturate rather than return zero. An overflowing rate means "faster than
    // f64 can say", and reporting that as *no ionization* would be a silent
    // sign error in the one direction that matters — the same reason
    // `keldysh_rate` clamps its exponent instead of bailing.
    // Atomic unit of rate is E_h/ħ.
    ln_w.min(MAX_EXPONENT).exp() * HARTREE / HBAR
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

    /// Electron attachment frequency `ν_att` (s⁻¹) at `pressure` (Pa) and
    /// neutral density `n_neutral` (m⁻³): two-body dissociative attachment plus
    /// three-body attachment to O₂,
    ///
    /// ```text
    /// ν_att = k₂·n_O₂ + k₃·n_O₂·N,      n_O₂ = f_att·N
    /// ```
    ///
    /// Split out of [`AirBreakdown::loss_rate`] so that the **loss** term and
    /// the **background seed** — which is the equilibrium between attachment
    /// and cosmic-ray production, [`AirBreakdown::background_electron_density`]
    /// — are computed from one expression and cannot drift apart. Same
    /// discipline as [`Focus`] bundling the three length scales.
    pub fn attachment_rate(&self, n_neutral: f64) -> f64 {
        let n_o2 = self.f_att * n_neutral;
        self.k_att_2body * n_o2 + self.k_att_3body * n_o2 * n_neutral
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

    /// Mean electron speed `v̄` (m/s), from the two constants that already fix
    /// it: `D_e = v̄²/(3·ν_m)` with `ν_m = k_m·p` gives
    /// `v̄ = √(3·D_e,ref·p_ref·k_m)`, and the pressure cancels exactly.
    ///
    /// For dry air this is 1.540e6 m/s — an electron of 6.740 eV, the same
    /// energy `d_e_ref_implies_a_stated_electron_energy` reads out of `D_e`.
    /// Nothing new is asserted here; it is a rearrangement of constants the
    /// model already carries, which is what lets [`AirBreakdown::escape_rate`]
    /// fix the collisionless limit without adding a parameter.
    pub fn mean_speed(&self) -> f64 {
        (3.0 * self.d_e_ref * P_REF * self.k_m).sqrt()
    }

    /// The electron energy (J) that a given `d_e` at `pressure` implies, i.e.
    /// [`kinetic_diffusion_coefficient`](Self::kinetic_diffusion_coefficient)
    /// solved for `ε`.
    pub fn diffusion_implied_energy(&self, pressure: f64, d_e: f64) -> f64 {
        1.5 * M_E * self.k_m * pressure * d_e
    }
}

/// The focal region's length scales — all three derived from **one** geometry.
///
/// The kernel needs three different lengths from the same little volume, and
/// they are genuinely different quantities rather than one number reused:
///
/// - **`Λ`**, the diffusion length. Not a distance — it is the eigenvalue of the
///   lowest diffusion mode, `ν = D_e/Λ²`, and for T&T's focus it is 7.74 µm.
/// - **`ℓ`**, the Cauchy mean chord `4V/S`. This *is* a distance: the mean
///   straight-line path an isotropically-directed particle travels before
///   leaving a convex body. It is a theorem, not a model choice, and for the
///   same focus it is **30.7 µm — 4.0× `Λ`**. It sets the collisionless escape
///   time `ℓ/v̄`.
/// - **`V`**, the volume, for the single-electron seed density.
///
/// Bundling them prevents the mistake this type was introduced to fix: `Λ` had
/// been doing duty as the ballistic transit distance in
/// [`AirBreakdown::escape_rate`], which is dimensionally fine and physically
/// wrong, and understated the free-molecular correction by 4×.
///
/// Both constructors take the geometry, never the derived lengths, so the three
/// cannot drift apart.
#[derive(Debug, Clone, Copy)]
pub struct Focus {
    lambda_diff: f64,
    mean_chord: f64,
    volume: f64,
}

impl Focus {
    /// A cylindrical focus of radius `r` and length `l`.
    ///
    /// `Λ` is T&T's Eq. 5, `(1/Λ)² = (π/l)² + (2.405/r)²` — the 2.405 is the
    /// first zero of `J₀`, i.e. the radial diffusion eigenvalue. The mean chord
    /// is `4V/S` with `V = πr²l` and `S = 2πr² + 2πrl`.
    pub fn cylinder(r: f64, l: f64) -> Self {
        let volume = PI * r * r * l;
        let surface = 2.0 * PI * r * r + 2.0 * PI * r * l;
        Self {
            lambda_diff: 1.0 / ((PI / l).powi(2) + (2.405 / r).powi(2)).sqrt(),
            mean_chord: 4.0 * volume / surface,
            volume,
        }
    }

    /// A spherical focus of radius `r`: `Λ = r/π`, mean chord `4V/S = 4r/3`.
    pub fn sphere(r: f64) -> Self {
        Self {
            lambda_diff: r / PI,
            mean_chord: 4.0 * r / 3.0,
            volume: 4.0 / 3.0 * PI * r * r * r,
        }
    }

    /// Diffusion length `Λ` (m).
    pub fn diffusion_length(&self) -> f64 {
        self.lambda_diff
    }

    /// Cauchy mean chord `4V/S` (m) — the free-molecular escape distance.
    pub fn mean_chord(&self) -> f64 {
        self.mean_chord
    }

    /// Focal volume (m³).
    pub fn volume(&self) -> f64 {
        self.volume
    }

    fn is_finite_positive(&self) -> bool {
        [self.lambda_diff, self.mean_chord, self.volume]
            .iter()
            .all(|x| *x > 0.0 && x.is_finite())
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
    /// The focal geometry — **pinned, not fit**.
    focus: Focus,
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
    /// Effective residual charge for the PPT photoionization source; zero
    /// disables it. Mutually exclusive with the other two multiphoton paths for
    /// the same reason. See [`Z_EFF_O2`].
    ppt_z_eff: f64,
    /// Breakdown criterion density `n_bd` (m⁻³).
    n_bd: f64,
    /// Neutral-density-per-pressure `N/p` at reference temperature (m⁻³·Pa⁻¹),
    /// i.e. `1/(k_B·T)`.
    n_over_p: f64,
    /// Explicit override for the initial electron density (m⁻³). `None` means
    /// use the derived background,
    /// [`AirBreakdown::background_electron_density`].
    n_seed: Option<f64>,
}

impl AirBreakdown {
    /// General constructor. `wavelength` (m), the [`Gas`], `lambda_diff`
    /// diffusion length (m, from geometry), `temperature` (K) for the neutral
    /// density, `focal_volume` (m³) for the single-electron seed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(wavelength: f64, gas: Gas, focus: Focus, temperature: f64) -> Result<Self> {
        if !(wavelength > 0.0 && wavelength.is_finite()) {
            bail!("wavelength must be positive and finite, got {wavelength}");
        }
        if !focus.is_finite_positive() {
            bail!("focal geometry must be positive and finite, got {focus:?}");
        }
        if !(temperature > 0.0 && temperature.is_finite()) {
            bail!("temperature must be positive and finite, got {temperature}");
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
            cascade_model: CascadeModel::DistributionResolved,
            window_half_widths: 2.0,
            focus,
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
            // PPT photoionization is ON by default: with a physical ambient
            // electron density (see `background_electron_density`) the focus
            // holds ~10⁻¹⁵ electrons, so seed *production* is not optional —
            // without it nothing would ever break down. The other two
            // multiphoton paths stay off.
            ppt_z_eff: Z_EFF_O2,
            n_bd: 1.0e23,
            n_over_p: 1.0 / (K_B * temperature),
            n_seed: None,
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
        Self::new(
            wavelength,
            Gas::dry_air(),
            Focus::cylinder(r_focus, l_axial),
            288.0,
        )
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
        // The multiphoton paths model the same channel; keep one live. Without
        // this, enabling the calibrated σ_K would silently do nothing, because
        // `mpi_source` tests the PPT path first and PPT is now the default.
        self.keldysh_prefactor = 0.0;
        self.ppt_z_eff = 0.0;
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
        // The multiphoton paths model the same physics; keep one live.
        self.mpi_rate_ref = 0.0;
        self.ppt_z_eff = 0.0;
        self
    }

    /// Disable every multiphoton source.
    ///
    /// Exists for the gates that **isolate** the cascade or the loss terms:
    /// once photoionization is on by default, a gate that means to measure a
    /// cascade closure has to say so, or it measures the closure *and* the
    /// seeding together and quietly stops testing what it is named for.
    /// Typically paired with [`Self::with_seed_density`].
    pub fn without_mpi(mut self) -> Self {
        self.ppt_z_eff = 0.0;
        self.keldysh_prefactor = 0.0;
        self.mpi_rate_ref = 0.0;
        self
    }

    /// Enable PPT photoionization as the multiphoton source, with the effective
    /// residual charge `z_eff` ([`Z_EFF_O2`] for air).
    ///
    /// Unlike [`Self::with_keldysh_mpi`] this takes **no free prefactor** —
    /// PPT's is determined by `z_eff`, and `z_eff` is published. That makes the
    /// resulting source an independent prediction, and it is validated in
    /// absolute magnitude against a measured cross-section
    /// (`ppt_rate_matches_the_measured_o2_cross_section`).
    ///
    /// Off by default, like the other two, so every pre-existing gate keeps its
    /// published numbers.
    pub fn with_ppt_mpi(mut self, z_eff: f64) -> Self {
        self.ppt_z_eff = z_eff;
        self.keldysh_prefactor = 0.0;
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
        self.n_seed = Some(n_seed);
        self
    }

    /// Select which limit of the energy balance drives the cascade — see
    /// [`CascadeModel`]. The default is [`CascadeModel::DistributionResolved`],
    /// promoted from `SelfConsistentClimb` once it was measured to fix the
    /// high-pressure branch.
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

    /// Initial electron density `n_e0` (m⁻³) at `pressure` — the explicit
    /// override if one was set with [`Self::with_seed_density`], otherwise
    /// [`Self::background_electron_density`].
    pub fn seed_density(&self, pressure: f64) -> f64 {
        self.n_seed
            .unwrap_or_else(|| self.background_electron_density(pressure))
    }

    /// Ambient free-electron density (m⁻³) — the steady state between
    /// cosmic-ray/radon ionization and attachment,
    ///
    /// ```text
    /// n_e0(p) = q(p)/ν_att(p),      q(p) = q_ref·(p/p_ref)
    /// ```
    ///
    /// `q` scales with gas density because ionizing radiation deposits per
    /// molecule; `ν_att` is the **same** expression [`Self::loss_rate`] uses,
    /// via [`Gas::attachment_rate`], so the two cannot drift apart.
    ///
    /// **This is tiny, and that is the physics.** At 1 atm `ν_att` = 1.9×10⁸
    /// s⁻¹, giving `n_e0` ≈ 0.05 m⁻³ — about 4×10⁻¹⁵ electrons in a 8.3×10⁻¹⁴ m³
    /// focus. Air is electronegative: a free electron attaches to O₂ in ~5 ns,
    /// so the lower atmosphere holds essentially **no** free electrons, and a
    /// tightly focused pulse cannot expect to find one waiting.
    ///
    /// An earlier version of this kernel assumed `n_e0 = 1/V_focal`, and its own
    /// documentation defended that as "~10⁴ above the cosmic-ray background of
    /// 10⁹–10¹⁰ m⁻³". That comparison was against the **ion** density; free
    /// electrons are ~10 orders scarcer, so the assumption was wrong by ~14
    /// orders, not 4. The seed has to be *produced* by the pulse, which is what
    /// the multiphoton source is for.
    ///
    /// Gases with no attachment channel (the noble gases, [`Gas::from_monatomic`])
    /// have no such equilibrium — nothing removes an electron but diffusion —
    /// so this returns 0 for them and the caller must supply a seed explicitly.
    pub fn background_electron_density(&self, pressure: f64) -> f64 {
        let n_neutral = self.n_over_p * pressure;
        let nu_att = self.gas.attachment_rate(n_neutral);
        if nu_att <= 0.0 {
            return 0.0;
        }
        ION_PAIR_PRODUCTION * (pressure / P_REF) / nu_att
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
        let nu_att = self.gas.attachment_rate(self.n_over_p * pressure);
        let nu_diff = self.escape_rate(pressure);
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
        self.focus.lambda_diff
    }

    /// The focal geometry: `Λ`, the Cauchy mean chord, and the volume.
    pub fn focus(&self) -> Focus {
        self.focus
    }

    /// Knudsen number `Kn = λ_mfp/Λ` — the ratio of the electron mean free path
    /// to the focal region it has to escape.
    ///
    /// **This is the validity test for the diffusion loss, and the model used to
    /// fail it badly.** `ν_diff = D_e/Λ²` is a continuum random-walk result: it
    /// assumes an electron collides many times while crossing the region, i.e.
    /// `Kn ≪ 1`. In T&T's focus that holds at atmospheric pressure (`Kn` = 0.05
    /// at 760 Torr) and fails completely at the bottom of Chylek's range —
    /// **`Kn` = 3.83 at 10 Torr**, where the mean free path is nearly four times
    /// the diffusion length and the electron simply flies out without colliding
    /// at all. Applying `D_e/Λ²` there overstates the loss by 2.3×, and because
    /// the overstatement is pressure-dependent it manufactures slope. See
    /// [`Self::escape_rate`].
    pub fn knudsen_number(&self, pressure: f64) -> f64 {
        self.gas.mean_speed() / (self.gas.k_m * pressure * self.focus.mean_chord)
    }

    /// Free-electron escape rate from the focal volume (s⁻¹), valid at **any**
    /// Knudsen number.
    ///
    /// An electron leaves by random walk, but it cannot cross the region faster
    /// than it can physically travel across it. So the escape *time* is the
    /// diffusive time plus the ballistic transit time, and the rate is their
    /// harmonic sum:
    ///
    /// ```text
    /// ν_esc = 1/(τ_diff + τ_ballistic),   τ_diff = Λ²/D_e,   τ_ballistic = Λ/v̄
    /// ```
    ///
    /// - `Kn ≪ 1`: `τ_diff` dominates and this is exactly the old `D_e/Λ²`.
    /// - `Kn ≫ 1`: `τ_ballistic` dominates and the loss **saturates** at `v̄/Λ`,
    ///   independent of pressure — which is the physics, because a collisionless
    ///   electron's escape time does not care how thin the gas is.
    ///
    /// **No new constant.** `v̄` is not supplied: `D_e = v̄²/(3ν_m)` already ties
    /// it to two quantities the model has, so
    /// `v̄ = √(3·D_e,ref·p_ref·K_m)` = 1.540e6 m/s — which is 6.740 eV, the same
    /// energy `d_e_ref_implies_a_stated_electron_energy` reads out of `D_e`. The
    /// correction is therefore forced by constants already gated, not chosen.
    pub fn escape_rate(&self, pressure: f64) -> f64 {
        let d_e = self.diffusion_coefficient(pressure);
        let tau_diff = self.focus.lambda_diff * self.focus.lambda_diff / d_e;
        let tau_ballistic = self.focus.mean_chord / self.gas.mean_speed();
        1.0 / (tau_diff + tau_ballistic)
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
        if self.ppt_z_eff > 0.0 {
            return n_neutral * ppt_rate(intensity, self.omega, self.gas.u_ion, self.ppt_z_eff);
        }
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
        // An EXPLICIT seed (`with_seed_density`) is a modelling assumption —
        // "this many electrons are available" — and so acts as a floor, which
        // is what keeps a source-free run independent of the integration
        // window. The DERIVED background is a physical initial condition and is
        // free to deplete: nothing holds it up, and nothing needs to, because
        // the multiphoton source refills it. That distinction is the whole
        // change; `seed_floor_applies_only_to_an_explicit_seed` gates it.
        let floor = self.n_seed;
        let mut n_e = self.seed_density(pressure);
        let mut peak = n_e;
        for step in 0..n_steps {
            let t = -half + (step as f64 + 0.5) * dt;
            let intensity = i_peak * (-c * t * t).exp();
            n_e = self.advance(n_e, intensity, pressure, dt);
            if let Some(floor) = floor {
                n_e = n_e.max(floor);
            }
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
            ppt_z_eff: 0.0,
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
            ppt_z_eff: 0.0,
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
            ppt_z_eff: 0.0,
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
            ppt_z_eff: 0.0,
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
            (0.16..=0.20).contains(&lo) && (0.71..=0.78).contains(&hi),
            "literature envelope moved: n ∈ [{lo:.3}, {hi:.3}], expected ≈[0.176, 0.745]"
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
            (0.42..=0.46).contains(&fixed),
            "FixedMeanEnergy slope moved: {fixed:.3}, expected ≈0.440"
        );
        assert!(
            (0.075..=0.098).contains(&selfc),
            "SelfConsistentClimb slope moved: {selfc:.3}, expected ≈0.086"
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
        let nu_diff = m.escape_rate(p);
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
        // And the default must not be *this* source. It is no longer "no
        // source at all" — the default multiphoton channel is now PPT, whose
        // magnitude is validated against a measured cross-section — so the
        // claim to gate is that the two differ, by a lot, in the direction that
        // matters.
        let default_source = model().mpi_source(1e16, p);
        let tt_source = with_mpi.mpi_source(1e16, p);
        assert!(
            default_source > 0.0 && tt_source > default_source * 100.0,
            "the default MPI source is {default_source:.3e} and the \
             T&T-calibrated one {tt_source:.3e}; the calibrated one is supposed \
             to be the wildly over-strong one"
        );
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
        // WHY IT HOLDS HAS CHANGED, and the new reason is stronger. It used to
        // hold because the seed was clamped at a floor, which patched the
        // symptom. Now the seed is *produced* by the pulse rather than assumed
        // present, so there is no initial condition left to decay: the quiet
        // arm contributes nothing at any w, and the residual spread across
        // w ∈ [1,4] is 5e-5 rather than the 1 % this gate tolerates. If a floor
        // is ever reintroduced for the derived seed, this gate would keep
        // passing while meaning much less — `seed_floor_applies_only_to_an_
        // explicit_seed` is what guards that.
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

    // --- Free-molecular escape (the Knudsen correction) ---------------------

    #[test]
    fn the_diffusion_approximation_is_invalid_at_low_pressure() {
        // THE DIAGNOSTIC, and the reason escape_rate exists. `ν = D_e/Λ²` is a
        // continuum random-walk result: it assumes the electron collides many
        // times while crossing the region. Measured Knudsen numbers
        // `Kn = λ_mfp/ℓ` in T&T's focus, against Chylek's pressure range:
        //
        //     760 Torr  Kn = 0.013     100 Torr  Kn = 0.10
        //     300 Torr  Kn = 0.034      10 Torr  Kn = 0.96
        //
        // At the bottom of the range the mean free path is comparable to the
        // whole escape distance — and against the DIFFUSION length Λ, which is
        // 4× smaller, it is 3.8×. The old kernel applied `D_e/Λ²` there anyway.
        // That is not a small error and it is not a symmetric one: it overstates
        // the loss more at lower pressure, so it manufactures slope, which is
        // exactly where the model was worst.
        let m = model();
        let hi = m.knudsen_number(760.0 * TORR);
        let lo = m.knudsen_number(10.0 * TORR);
        assert!(
            hi < 0.05,
            "Kn = {hi:.4} at 760 Torr; the continuum limit is supposed to be safe here"
        );
        assert!(
            lo > 0.5,
            "Kn = {lo:.4} at 10 Torr; this gate exists because the diffusion \
             approximation FAILS there — if it no longer does, escape_rate's \
             justification has gone"
        );
        // Kn ∝ 1/p exactly.
        assert!(
            (lo / hi / 76.0 - 1.0).abs() < 1e-9,
            "Kn is not ∝ 1/p: {lo:.4} at 10 Torr vs {hi:.4} at 760"
        );
    }

    #[test]
    fn escape_rate_recovers_both_limits() {
        // The correction must change nothing where the old formula was valid,
        // and must saturate where it was not.
        let m = model();
        let l_chord = m.focus().mean_chord();
        let v_bar = m.gas.mean_speed();

        // Continuum limit: at high pressure this converges to D_e/Λ².
        //
        // It converges from below, and not as fast as one might assume — the
        // correction is still 6.3 % at 760 Torr and 2.4 % at 2000, which is why
        // landing it moved the high-pressure gate numbers by a few percent
        // rather than not at all. It is a genuine correction everywhere; it is
        // merely *dominant* only at low pressure.
        let mut prev = 0.0;
        for (p_torr, want) in [(760.0, 0.9375), (2000.0, 0.9757), (10_000.0, 0.9951)] {
            let p = p_torr * TORR;
            let continuum =
                m.diffusion_coefficient(p) / (m.diffusion_length() * m.diffusion_length());
            let ratio = m.escape_rate(p) / continuum;
            assert!(
                ratio < 1.0 && (ratio - want).abs() < 0.01,
                "at {p_torr} Torr escape_rate is {ratio:.4}× the continuum D_e/Λ², \
                 expected ≈{want}"
            );
            assert!(
                ratio > prev,
                "convergence to the continuum limit is not monotone"
            );
            prev = ratio;
        }

        // Free-molecular limit: the rate saturates at v̄/ℓ and stops caring
        // about pressure. This is the physics — a collisionless electron's
        // escape time does not depend on how thin the gas is.
        let ceiling = v_bar / l_chord;
        let very_low = m.escape_rate(0.001 * TORR);
        assert!(
            (very_low / ceiling - 1.0).abs() < 1e-3,
            "escape rate at 0.001 Torr is {very_low:.4e}, expected the ballistic \
             ceiling v̄/ℓ = {ceiling:.4e}"
        );
        // And it is monotone, never exceeding the ceiling anywhere.
        let mut prev = f64::MAX;
        for p_torr in [0.1, 1.0, 10.0, 100.0, 760.0] {
            let r = m.escape_rate(p_torr * TORR);
            assert!(r <= ceiling * (1.0 + 1e-12), "escape rate exceeded v̄/ℓ");
            assert!(r < prev, "escape rate is not monotone in 1/p");
            prev = r;
        }
    }

    #[test]
    fn the_escape_correction_adds_no_constant() {
        // v̄ is not supplied. D_e = v̄²/(3ν_m) with ν_m = k_m·p already fixes it,
        // and the pressure cancels: v̄ = √(3·D_e,ref·p_ref·k_m). So the
        // correction is forced by two constants that are already gated —
        // K_m externally (tt2012_collision_frequency_matches_literature) and
        // D_e for consistency (d_e_ref_implies_a_stated_electron_energy).
        let gas = model().gas();
        let v_bar = gas.mean_speed();

        // The round trip: v̄ must reproduce D_e at any pressure.
        for p_torr in [10.0, 760.0] {
            let p = p_torr * TORR;
            let d_from_v = v_bar * v_bar / (3.0 * gas.k_m * p);
            let d_direct = gas.d_e_ref * (P_REF / p);
            assert!(
                (d_from_v / d_direct - 1.0).abs() < 1e-12,
                "v̄ does not reproduce D_e at {p_torr} Torr: {d_from_v:e} vs {d_direct:e}"
            );
        }
        // And it is the SAME electron energy the D_e gate reads out: 6.740 eV.
        let ev = 0.5 * M_E * v_bar * v_bar / E_CHARGE;
        assert!(
            (ev - 6.740).abs() < 0.01,
            "the escape correction's v̄ is {ev:.3} eV; \
             d_e_ref_implies_a_stated_electron_energy reads 6.740 eV out of the \
             same constant, and these must not drift apart"
        );
    }

    #[test]
    fn focus_geometry_separates_its_three_length_scales() {
        // Λ is a diffusion EIGENVALUE, the Cauchy chord 4V/S is a DISTANCE, and
        // they differ by 4× for T&T's focus. Using Λ as the ballistic transit
        // distance — which the first cut of escape_rate did — is dimensionally
        // fine and physically wrong, and understates the correction fourfold.
        let f = model().focus();
        assert!(
            (f.diffusion_length() * 1e6 - 7.74).abs() < 0.01,
            "Λ = {:.3} µm, expected 7.74 (T&T Eq. 5)",
            f.diffusion_length() * 1e6
        );
        assert!(
            (f.mean_chord() * 1e6 - 30.72).abs() < 0.05,
            "Cauchy mean chord = {:.3} µm, expected 30.72 (4V/S for the focal cylinder)",
            f.mean_chord() * 1e6
        );
        assert!(
            (f.mean_chord() / f.diffusion_length() - 3.97).abs() < 0.02,
            "chord/Λ = {:.3}, expected 3.97 — if these converged, one of the two \
             is no longer being computed from the geometry",
            f.mean_chord() / f.diffusion_length()
        );
        // A sphere, checked against its own closed forms: Λ = r/π, 4V/S = 4r/3.
        let sph = Focus::sphere(3e-5);
        assert!((sph.diffusion_length() - 3e-5 / PI).abs() < 1e-18);
        assert!((sph.mean_chord() - 4.0 * 3e-5 / 3.0).abs() < 1e-18);
    }

    #[test]
    fn photon_count_is_eleven_at_1064nm() {
        // ⌈12.06 eV / 1.166 eV⌉ = 11.
        assert_eq!(model().k_photons, 11);
    }

    // --- Special functions used by `ppt_rate` (NOT physics validation — these
    // are closed-form checks on two standard functions). ----------------------

    /// T12-V1: Dawson's integral against the values that pin it, and against
    /// the ODE it satisfies.
    ///
    /// `Φ` is the one non-elementary function in [`ppt_rate`], and an error in
    /// it would move the above-threshold sum without moving the exponent — i.e.
    /// it would look exactly like a physics result. Hence four independent
    /// checks rather than one table lookup.
    #[test]
    fn dawson_matches_its_closed_forms() {
        // Odd, and zero at the origin.
        assert_eq!(dawson(0.0), 0.0);
        assert_eq!(dawson(-1.5), -dawson(1.5));

        // The maximum: Φ(0.9241388730…) = 0.5410442246…, a standard reference
        // value. It is a stationary point, so it also checks the series near
        // where cancellation starts to matter.
        let x_max = 0.924_138_873_015_5;
        assert!(
            (dawson(x_max) - 0.541_044_224_635_5).abs() < 1e-12,
            "Φ(x_max) = {:.15e}",
            dawson(x_max)
        );

        // Small-x: Φ = x − 2x³/3 + 4x⁵/15 − … The residual against the
        // two-term expansion must be the *next term*, not merely small — a
        // tolerance loose enough to swallow it would pass on a wrong series.
        // The tolerance is the size of the *following* term, `(8/105)x⁷`, which
        // is `0.286·x²` of the one being checked; anything below `1e-4` has no
        // f64 resolution left at this cancellation depth, so the sweep starts
        // at `1e-2`.
        for x in [1e-2f64, 3e-2, 1e-1] {
            let truncated = x - 2.0 * x * x * x / 3.0;
            let residual = dawson(x) - truncated;
            let next_term = 4.0 * x.powi(5) / 15.0;
            assert!(
                (residual / next_term - 1.0).abs() < x * x,
                "small-x at {x}: residual {residual:.6e} vs next term {next_term:.6e}"
            );
        }

        // Large-x asymptote Φ → 1/(2x), approached from above.
        for x in [20.0, 50.0, 200.0] {
            let ratio = dawson(x) / (0.5 / x);
            assert!(
                ratio > 1.0 && ratio < 1.0 + 2.0 / (x * x),
                "asymptote at {x}: ratio {ratio}"
            );
        }

        // The defining ODE Φ' = 1 − 2xΦ, by central difference. This is the
        // check that does not depend on any tabulated number, and it brackets
        // the series/asymptotic join at x = 4 from both sides.
        //
        // `h` is deliberately 1e-3, not smaller: differencing amplifies Φ's own
        // ~2e-10 evaluation error by 1/2h, so a "more accurate" h = 1e-5 makes
        // this check *worse* (1e-5 of noise) rather than better. At 1e-3 the
        // amplified noise (~1e-7) and the difference truncation (~1.7e-7) are
        // comparable, which is the sweet spot. The join itself is not straddled
        // — `dawson_series_join_is_continuous` owns that.
        let h = 1e-3;
        for &x in &[0.3, 1.0, 2.5, 3.9, 4.1, 6.0, 12.0] {
            let numeric = (dawson(x + h) - dawson(x - h)) / (2.0 * h);
            let exact = 1.0 - 2.0 * x * dawson(x);
            assert!(
                (numeric - exact).abs() < 1e-5,
                "ODE at x = {x}: dΦ/dx = {numeric:.9e} vs 1 − 2xΦ = {exact:.9e}"
            );
        }
    }

    /// T12-V1b: the two Dawson branches agree across their join, so
    /// [`DAWSON_SERIES_CUTOVER`] is a documented seam and not a discontinuity.
    /// Same role as `keldysh_exponent_series_matches_direct_form`.
    #[test]
    fn dawson_series_join_is_continuous() {
        let c = DAWSON_SERIES_CUTOVER;
        let below = dawson(c - 1e-9);
        let above = dawson(c + 1e-9);
        let rel = (above - below).abs() / below;
        assert!(
            rel < 3e-7,
            "Dawson branches disagree by {rel:.2e} at the x = {c} join \
             ({below:.15e} vs {above:.15e})"
        );
    }

    /// T12-V2: `ln_gamma` where Γ is known exactly.
    #[test]
    fn ln_gamma_matches_its_closed_forms() {
        // Γ(n) = (n−1)!
        let mut factorial = 1.0f64;
        for n in 1..=15u32 {
            let exact = factorial.ln();
            assert!(
                (ln_gamma(n as f64) - exact).abs() < 1e-12 * exact.abs().max(1.0),
                "ln Γ({n}) = {} vs {exact}",
                ln_gamma(n as f64)
            );
            factorial *= n as f64;
        }
        // Γ(½) = √π, and the half-integer ladder Γ(x+1) = xΓ(x).
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-13);
        for &x in &[0.25, 0.5, 0.563, 1.5, 2.5, 7.5] {
            let step = ln_gamma(x + 1.0) - ln_gamma(x) - x.ln();
            assert!(step.abs() < 1e-12, "recurrence at {x}: residual {step:.3e}");
        }
        // Reflection: Γ(x)Γ(1−x) = π/sin(πx) — exercises the x < ½ branch.
        for &x in &[0.1, 0.3, 0.45] {
            let lhs = ln_gamma(x) + ln_gamma(1.0 - x);
            let rhs = (PI / (PI * x).sin()).ln();
            assert!(
                (lhs - rhs).abs() < 1e-12,
                "reflection at {x}: {lhs:.15e} vs {rhs:.15e}"
            );
        }
    }

    /// T12-V6: [`ppt_ati_terms`] returns a **converged** truncation at every
    /// `γ` the rate is evaluated at, down to the tunnelling cutover.
    ///
    /// This gate originally swept only `γ ≥ 1` against a fixed 64 terms, and
    /// that hid a real defect: `α(γ) → ⅔γ³`, so the sum stops converging in the
    /// tunnelling limit and a fixed truncation silently returns a number set by
    /// the loop bound rather than by the physics — 29 % low at `γ` = 0.2, 78 %
    /// low at `γ` = 0.1. The sweep now runs down to
    /// [`PPT_TUNNELLING_CUTOVER`], which is the whole point.
    #[test]
    fn ppt_ati_sum_is_converged() {
        const REFERENCE: usize = 1 << 21;
        for &nu in &[10.35, 5.18, 7.5] {
            for &gamma in &[PPT_TUNNELLING_CUTOVER, 0.2, 0.5, 1.0, 3.0, 10.0, 40.0] {
                let converged = ppt_ati_sum(gamma, nu, REFERENCE);
                let shipped = ppt_ati_sum(gamma, nu, ppt_ati_terms(gamma));
                assert!(
                    converged > 0.0,
                    "reference sum vanished at γ = {gamma}, ν = {nu}"
                );
                let rel = (shipped / converged - 1.0).abs();
                assert!(
                    rel < 1e-6,
                    "A₀ at γ = {gamma}, ν = {nu} moves {rel:.3e} between \
                     {} adaptive terms and {REFERENCE}",
                    ppt_ati_terms(gamma)
                );
            }
        }
        // The adaptive count must be doing real work in both directions: cheap
        // where the decay is fast, deep where it is not. A gate that passed
        // because every truncation agreed would be worthless.
        assert!(
            ppt_ati_terms(10.0) < 32,
            "γ = 10 asks for {} terms — the adaptive count is not exploiting \
             fast decay",
            ppt_ati_terms(10.0)
        );
        assert!(
            ppt_ati_terms(PPT_TUNNELLING_CUTOVER) > 10_000,
            "γ = {PPT_TUNNELLING_CUTOVER} asks for only {} terms — that is the \
             corner where a fixed truncation was wrong",
            ppt_ati_terms(PPT_TUNNELLING_CUTOVER)
        );
        let short = ppt_ati_sum(1.0, 7.5, 4);
        let full = ppt_ati_sum(1.0, 7.5, REFERENCE);
        assert!(
            short / full < 0.95,
            "a 4-term sum is already {:.4} of converged at γ = 1",
            short / full
        );
    }

    /// **T12-V3: `ppt_rate` reduces to the ADK closed form as `γ → 0`.**
    ///
    /// This gate was specified in the plan and then not written, which is how
    /// the truncation defect above survived to be committed — it is precisely
    /// the check that exercises the corner a converged-at-γ≥1 sweep never
    /// reaches. Restored, and it now anchors the tunnelling branch:
    ///
    /// ```text
    /// W_ADK = |C_n*|²·√(6/π)·U_i·(2F₀/F)^{2n*−3/2}·exp(−2F₀/3F)
    /// ```
    ///
    /// the standard cycle-averaged linear-polarization result. Everything in
    /// it is closed form; there is nothing to tune.
    #[test]
    fn ppt_reduces_to_adk_in_the_tunnelling_limit() {
        let u_ion = 12.06 * E_CHARGE;
        let omega = 2.0 * PI * C_LIGHT / 1064e-9;
        let ip = u_ion / HARTREE;
        let kappa = (2.0 * ip).sqrt();
        let f0 = kappa * kappa * kappa;
        let n_star = Z_EFF_O2 / kappa;
        let ln_c2 =
            2.0 * n_star * 2.0f64.ln() - n_star.ln() - ln_gamma(n_star + 1.0) - ln_gamma(n_star);

        // Sweep γ downward; the ratio to ADK must converge to 1, and the
        // residual must fall like γ² (the leading correction, from √(1+γ²) in
        // the Coulomb factor and the ⅔γ − γ³/15 exponent).
        let mut previous = f64::MAX;
        for &gamma in &[3e-2, 1e-2, 3e-3, 1e-3] {
            // Invert γ = ω_au·κ/F for the field, then the intensity.
            let field = omega * HBAR / HARTREE * kappa / gamma;
            let intensity = 0.5 * EPS0 * C_LIGHT * (field * F_ATOMIC).powi(2);
            let adk = (ln_c2
                + 0.5 * (6.0 / PI).ln()
                + ip.ln()
                + (2.0 * n_star - 1.5) * (2.0 * f0 / field).ln()
                - 2.0 * f0 / (3.0 * field))
                .exp()
                * HARTREE
                / HBAR;
            let ratio = ppt_rate(intensity, omega, u_ion, Z_EFF_O2) / adk;
            let residual = (ratio - 1.0).abs();
            assert!(
                residual < 5.0 * gamma * gamma,
                "γ = {gamma}: PPT/ADK = {ratio:.9}, residual {residual:.3e} \
                 exceeds the O(γ²) it should be"
            );
            assert!(
                residual < previous,
                "γ = {gamma}: residual {residual:.3e} did not fall below the \
                 previous {previous:.3e} — the limit is not being approached"
            );
            previous = residual;
        }
    }

    /// T12-V3b: the ATI sum and its tunnelling limit agree across the join, so
    /// [`PPT_TUNNELLING_CUTOVER`] is a documented seam and not a step.
    ///
    /// 0.6 % — the converged `A₀` is 0.9941 just above the cutover against the
    /// limit's exactly 1. Small because the cutover was *chosen* where the two
    /// meet, which is the only honest way to place it.
    #[test]
    fn ppt_tunnelling_branch_joins_the_sum() {
        let g = PPT_TUNNELLING_CUTOVER;
        for &nu in &[10.35, 5.18] {
            let summed = ppt_ati_sum(g, nu, 1 << 21);
            assert!(
                (summed - 1.0).abs() < 0.01,
                "at the γ = {g} cutover the converged A₀ is {summed:.6}, not \
                 within 1 % of its tunnelling limit 1 — move the cutover down \
                 rather than widening this tolerance"
            );
        }
    }

    // --- Seed production (S-V1 … S-V4). -------------------------------------

    /// S-V1: the ambient seed is the attachment equilibrium, against the very
    /// same `ν_att` the loss term uses.
    ///
    /// The two are computed from one expression ([`Gas::attachment_rate`]) so
    /// they cannot drift; this gate is what makes that structural guarantee
    /// checkable rather than merely intended.
    #[test]
    fn background_electron_density_is_the_attachment_equilibrium() {
        let m = model();
        for torr in [10.0f64, 100.0, 760.0, 2000.0] {
            let p = torr * TORR;
            let n_neutral = m.neutral_density(p);
            let nu_att = m.gas.attachment_rate(n_neutral);
            let q = ION_PAIR_PRODUCTION * (p / P_REF);
            let expected = q / nu_att;
            let got = m.background_electron_density(p);
            assert!(
                (got / expected - 1.0).abs() < 1e-12,
                "at {torr} Torr the background is {got:.6e} but q/ν_att is \
                 {expected:.6e}"
            );
            // And ν_att must be the one `loss_rate` actually uses: the loss
            // rate is attachment plus escape, so the difference is escape.
            let residual = m.loss_rate(p) - nu_att - m.escape_rate(p);
            assert!(
                residual.abs() < 1e-6 * m.loss_rate(p),
                "at {torr} Torr the seed's ν_att is not the loss term's: \
                 residual {residual:.3e}"
            );
        }
    }

    /// **S-V2: the focus essentially never holds a free electron.**
    ///
    /// This is the claim the whole change exists to make, and it is also a
    /// correction. The kernel previously assumed `n_e0 = 1/V_focal` and
    /// defended it as "~10⁴ above the cosmic-ray background of 10⁹–10¹⁰ m⁻³".
    /// That comparison was against the **ion** density; air is
    /// electronegative, so free electrons attach to O₂ in ~5 ns and the free
    /// electron background is ~10 orders scarcer. The old assumption was wrong
    /// by ~14 orders, not 4.
    #[test]
    fn the_focus_holds_essentially_no_free_electrons() {
        let m = model();
        let p = 760.0 * TORR;
        let n_e0 = m.background_electron_density(p);
        // ~0.149 m⁻³ at 1 atm.
        assert!(
            (0.05..=0.5).contains(&n_e0),
            "ambient free-electron density is {n_e0:.4e} m⁻³, expected ≈0.15"
        );
        let in_focus = n_e0 * m.focus().volume();
        assert!(
            in_focus < 1e-12,
            "the focus holds {in_focus:.3e} free electrons; the claim is that it \
             holds essentially none, so the seed must be produced by the pulse"
        );
        // And that this is far below what the old assumption asserted.
        let old_assumption = 1.0 / m.focus().volume();
        assert!(
            old_assumption / n_e0 > 1e12,
            "the retired seed was only {:.2e}× the physical background; the \
             record says ~14 orders",
            old_assumption / n_e0
        );
    }

    /// **S-V3: the ionization background is not load-bearing.**
    ///
    /// The change introduces one new constant, [`ION_PAIR_PRODUCTION`], and the
    /// defence of it is not its provenance but its irrelevance. Sweeping the
    /// seed over **twelve decades** either side of the derived value leaves the
    /// threshold *bit-identical*; it takes ~10 orders before it moves 0.04 %.
    ///
    /// The same sweep is why the retired assumption mattered: `1/V_focal` =
    /// 1.2×10¹³ m⁻³ sits in the range where the seed *does* move the answer
    /// (2.6 % at 10¹²). The old constant was load-bearing and wrong; the new one
    /// is neither.
    #[test]
    fn ionization_background_is_not_load_bearing() {
        let m = model();
        let p = 760.0 * TORR;
        let base = m.threshold_intensity(6e-9, p, 400).expect("threshold");
        for seed in [1e-6f64, 1e-3, 1.0, 1e3, 1e6] {
            let moved = m
                .with_seed_density(seed)
                .threshold_intensity(6e-9, p, 400)
                .expect("threshold");
            assert!(
                (moved / base - 1.0).abs() < 1e-6,
                "seed {seed:.0e} m⁻³ moves the threshold to {moved:.6e} from \
                 {base:.6e}; over this range it must not move at all, or the \
                 ion-pair production rate is a fitted constant in disguise"
            );
        }
        // The sensitivity is real, just very far away — this half stops the
        // gate above from passing on a threshold that ignores the seed entirely.
        let far = m
            .with_seed_density(1e14)
            .threshold_intensity(6e-9, p, 400)
            .expect("threshold");
        assert!(
            far < 0.97 * base,
            "a seed of 10¹⁴ m⁻³ gives {far:.4e} against {base:.4e}; if even that \
             changes nothing, the seed is not entering the calculation"
        );
    }

    /// S-V4: the floor applies to an **explicit** seed only.
    ///
    /// An explicit seed is a modelling assumption — "this many electrons are
    /// available" — and holding it as a floor is what keeps a source-free run
    /// independent of the integration window. The derived background is a
    /// physical initial condition and must be free to deplete, because nothing
    /// holds it up. Re-clamping it would silently restore the behaviour this
    /// change removed, so the distinction is gated.
    #[test]
    fn seed_floor_applies_only_to_an_explicit_seed() {
        let p = 760.0 * TORR;
        // Two runs identical in every way except how the same starting density
        // is declared. With no multiphoton source, losses grind the population
        // down through the quiet arm; the floored run climbs from the full
        // seed at the peak, the unfloored one from whatever survived.
        let bare = model().without_mpi();
        let n0 = bare.seed_density(p);
        let pinned = bare.with_seed_density(n0);
        assert_eq!(
            n0,
            pinned.seed_density(p),
            "the two runs must start from the same density"
        );

        // 8e15 W/m² is just under the source-free threshold, so the floored
        // run avalanches five orders while the free one never recovers from the
        // quiet arm at all.
        let free = bare.peak_ne(8e15, 6e-9, p, 400);
        let floored = pinned.peak_ne(8e15, 6e-9, p, 400);
        assert!(
            floored > free * 1000.0,
            "floored peak {floored:.4e} vs free peak {free:.4e} from the same \
             seed {n0:.3e}: the explicit seed is supposed to act as a floor and \
             the derived one is not, so these must differ"
        );
        // The floored run cannot fall below its seed at any point, which is the
        // property that makes it window-independent.
        assert!(
            floored >= n0,
            "floored peak {floored:.4e} fell below its own seed {n0:.3e}"
        );
    }

    /// T12-V6b: the rate is **monotonic** in intensity across the whole
    /// bisection bracket.
    ///
    /// `threshold_intensity` bisects on this function, so a rate that falls as
    /// intensity rises is not merely inelegant — it can put the bracket on the
    /// wrong root. The un-converged truncation did exactly that: 40 of 200
    /// sampled steps decreased, with `W(10²²) < W(10²⁰)` at 532 nm.
    #[test]
    fn ppt_rate_is_monotonic_across_the_bisection_bracket() {
        let u_ion = 12.06 * E_CHARGE;
        for &nm in &[1064e-9, 532e-9] {
            let omega = 2.0 * PI * C_LIGHT / nm;
            let mut previous = 0.0;
            for k in 0..=200 {
                let i = I_BRACKET_LO * 10f64.powf(k as f64 * 10.0 / 200.0);
                let w = ppt_rate(i, omega, u_ion, Z_EFF_O2);
                assert!(
                    w >= previous,
                    "λ = {:.0} nm: rate fell from {previous:.4e} to {w:.4e} at \
                     I = {i:.3e} W/m², inside the bisection bracket",
                    nm * 1e9
                );
                previous = w;
            }
        }
    }
}
