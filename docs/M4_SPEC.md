# M4 pre-spec — thermal blooming (the M3.5 gate)

Written **before** any M4 code, per the plan's M3.5 gate: pin the fluid model,
the beam↔medium coupling and its convergence test, the stability bounds, and a
reproducible anchor benchmark. If any of this proves wrong during
implementation, amend this document first, then the code.

Conventions follow [MODELS.md](MODELS.md): SI units, `k = 2π/λ`,
intensity `I = |u|²` (W/m²). New symbols: wind speed `v` (m/s, transverse,
along `+x` by convention), air density `ρ` (kg/m³), isobaric specific heat
`c_p` (J/(kg·K)), thermal conductivity `κ_t` (W/(m·K)), ambient temperature
`T0` (K), absorption coefficient `α_abs` (1/m — the *absorbed* fraction of
extinction, not scattering).

## Gate decisions (recorded)

1. **Physics target: pure thermal blooming.** CW steady-state, local
   thermodynamic equilibrium, small temperature perturbation (`ΔT/T0 ≪ 1`).
   No laser-induced plasma, no breakdown, no kinetic cooling. Consequence:
   air properties are a static function of `(T, p)` → **offline tabulation,
   no runtime FFI** (no Mutation++ linking, no LGPL obligation in the core
   or the M5 wheels). The `cxx` escalation path stays documented in the plan
   but is not exercised.
2. **Property table pinned** (§ Property closure below): dry air, `T ∈
   [200, 400] K` × `p ∈ [40, 110] kPa`, 2 K × 2.5 kPa grid, bilinear
   interpolation; properties `ρ`, `c_p`, `κ_t`, and `(n−1)` at reference
   conditions. Frozen as data in the repo with its generation script.
3. **FFI surface: none** (follows from 1).

## Fluid model

### Governing equation

Energy conservation for air heated by absorption, forced convection by wind:

```text
ρ·c_p·(∂T/∂t + v·∂T/∂x) = κ_t·∇²T + α_abs·I(x, y, z)
```

M4 v1 solves the **steady state** (`∂T/∂t = 0`), consistent with the CW
scope. Two further approximations, each with a validity check asserted at
run start:

- **Isobaric response.** Pressure equilibrates at the sound speed, far
  faster than heating; density responds at constant pressure:
  `Δρ/ρ0 = −ΔT/T0`. Valid when the heating time is long against the
  acoustic transit time `w/c_s` — always true for CW at our scales.
  (Transient/slewed regimes where acoustics matter are out of scope with
  the rest of time-domain physics.)
- **Convection-dominated transport.** Conduction is dropped when the Péclet
  number `Pe = ρ·c_p·v·w / κ_t ≫ 1` (for `w = 5 cm`, `v = 1 m/s`:
  `Pe ≈ 2×10³`). The solver asserts `Pe > 100` and refuses stagnant-air
  cases in v1 rather than silently mis-modelling them; the conduction-only
  (`v = 0`) closure is a documented later extension.

With both, the steady state is an exact upwind integral, slab-local in `z`:

```text
ΔT(x, y, z) = (α_abs / (ρ·c_p·v)) · ∫_{−∞}^{x} I(x', y, z) dx'
```

### Index closure

Isobaric density perturbation and Gladstone–Dale (`n − 1 ∝ ρ`) give

```text
δn(x, y, z) = −(n0 − 1)·ΔT(x, y, z)/T0,   (dn/dT)_p = −(n0 − 1)/T0
```

Air heats → thins → `δn < 0`: a negative lens that also bends the beam
**into the wind** (the upwind side stays cool and dense). Both the defocus
and the upwind bend are benchmark observables below.

### Property closure (the tabulation decision)

`ρ`, `c_p`, `κ_t`, `(n0 − 1)` enter as ambient values at `(T0, p0)` — for
`ΔT/T0 ≪ 1` they are constants per run, looked up once, not per grid point.
The table exists so the lookup is principled and so the machinery is in
place if a larger-ΔT regime is ever admitted:

- **Grid:** dry air, `T ∈ [200, 400] K` step 2 K × `p ∈ [40, 110] kPa`
  step 2.5 kPa (101 × 29 points); bilinear interpolation. All four
  properties vary smoothly here; interpolation error ≪ 0.1 %.
- **Source:** `scripts/make_air_table.py` generates the table (see its
  docstring for the exact reproduction command; it needs a Python with the
  Mutation++ bindings importable — currently the local build on
  Python 3.10 — and `MPP_DATA_DIRECTORY` set). Per property:
  - `ρ`, `c_p`: **Mutation++** (N₂/O₂/Ar equilibrium mixture, real dry-air
    composition), **cross-checked** against an independent analytic model
    (ideal gas + NASA-9 polynomials). **Gate: agreement < 0.5 % over the
    whole grid or nothing is written.** (Measured at generation:
    `Δρ ≤ 1.2×10⁻⁵`, `Δc_p ≤ 0.15 %`.)
  - `κ_t`: **single-source** Sutherland-form correlation, spot-checked
    < 2 % against NIST reference points (Lemmon et al. 2004). Rationale,
    measured 2026-07-17: Mutation++ transport targets high-T plasma
    regimes and deviates +4 % to +25 % from NIST for 200–400 K air, so it
    is not used for `κ_t`. Acceptable because `κ_t` only feeds the Péclet
    validity assert, never the blooming physics.
  - `(n − 1)`: Ciddor (1996) standard dry-air dispersion at
    `λ_ref = 1 µm`, density-scaled (Gladstone–Dale) from the tabulated
    `ρ` — inherits the `ρ` gate. Runs at other wavelengths rescale by the
    Ciddor dispersion ratio (an M4 implementation item, recorded in the
    sidecar).
- **Frozen:** committed as `data/air_properties.npy` + sidecar
  `data/air_properties.json` (grid axes, composition, gate results, and
  full provenance: script, beamprop git state, Python version, Mutation++
  package version + local-build git describe + data directory). **The
  `.npy` is canonical: the Rust solver and CI only read it and never
  regenerate it** — the script ships for auditability and reproduction,
  not as a build step. The table is input data, bit-stable across
  platforms — this preserves the reproducibility property the Rust choice
  was made for. No Mutation++ code links into the solver.

## Beam ↔ medium coupling

The steady-state `ΔT` at slab `z` depends only on the intensity **at that
slab**, and the paraxial march is one-way in `z` — so the coupling needs
**no global outer iteration**: compute `δn` from the local field inside the
existing split-step march, through the same `Medium` trait turbulence uses.

Naive slab-local evaluation costs an order of accuracy (the medium operator
would use the field at the slab entrance, not the centre). To keep the
propagator's verified 2nd order:

- **Predictor–corrector per slab:** predict `u* = D(dz/2)·u`, evaluate
  `δn` from `|u*|²` (the slab-centre field), then apply
  `M(dz) = exp(i·k·δn·dz − α_ext·dz/2)` and the trailing `D(dz/2)`. One
  extra `ΔT` quadrature per slab, no extra FFTs.
- **Convergence test (gates M4, extends T3):** on a bloomed case, refine
  `dz` → observed order ≈ 2 on receiver-plane intensity; refine `dx` →
  consistent limit. A first-order coupling shows slope ≈ 1 here — this test
  is what catches it.

The upwind quadrature `∫ I dx'` is a cumulative sum along wind rows — exact
for the discretized field, `O(N²)` per slab, negligible next to the FFTs.

**✅ IMPLEMENTATION RECORD (2026-07-17).** The predictor–corrector fell out
of the existing propagator structure: `step()` already applies the medium
after the leading `D(dz/2)`, so the field at the medium call *is* the
slab-centre field — realized as `Medium::index_response(z_slab, intensity,
dz)` with `needs_intensity()`, no extra FFTs. One subtlety the convergence
gate caught exactly as designed: the handed-over intensity predates the
slab's own Beer–Lambert decay, and using it raw makes the heating a
rectangle rule in *absorbed power* — measured order dropped cleanly to
1.00. The fix is the half-slab factor `I_mid = |u*|²·e^(−α_abs·dz/2)` in
`ThermalBlooming::index_response`; measured order after: **2.000 / 2.000**.
The order gate itself uses **self-convergence** (Cauchy differences between
successive resolutions), not a fixed fine reference, whose finite fineness
was found to corrupt the order estimate at the finer test points.

## Stability and resolution bounds

- **Steady-state advection:** the upwind integral is exact quadrature — no
  CFL condition in v1. **Recorded for the transient extension:** an
  explicit time-marched advection step is bounded by `Δt ≤ Δx/v`; an
  operator-split semi-Lagrangian step avoids it. Decide there, not now.
- **Phase-gradient sampling (extends M1's runtime asserts):** blooming adds
  a smooth but growing phase; alias-free propagation needs the per-slab
  bending resolved: assert `max|∇⊥(δn)|·k·dz < π/dx` each slab, abort loudly
  on violation (same policy as the M1 sampling asserts).
- **Downwind saturation vs the grid:** `ΔT` does not decay downwind — it
  saturates (erf profile). The phase screen therefore reaches the grid edge
  by construction. This is benign for the *field* (the guard band absorbs
  the beam before the edge) but the run must assert the beam stays inside
  the guard band as blooming bends it upwind — wraparound of a *bent* beam
  is the new silent-failure mode here.
- **Numerical stability test (the "blow-up" gate):** a strong-blooming run
  (distortion number `N_φ ≈ 20`, see below) must stay finite, conserve
  power to the M1 invariant minus absorption, and agree with itself under
  `dz` refinement. Note: the *phase-compensation instability* in the
  literature is an adaptive-optics feedback instability — no AO exists in
  this solver, so that specific instability cannot arise in M4; what is
  gated here is the numerical boundedness of the open-loop coupled march.

## Anchor benchmarks (the M4 gates)

Three tiers, from exact to trend. Tier 1 is the reproducible anchor the
M3.5 gate demands; it is closed-form, so the tolerance is tight and the
test is deterministic.

### B1 — closed-form single-slab blooming phase (tight)

For a collimated Gaussian `I = I0·exp(−2(x²+y²)/w²)`, `I0 = 2P/(π·w²)`, in
uniform crosswind, the steady-state temperature integral is closed-form:

```text
ΔT(x, y) = (α_abs·I0·w / (ρ·c_p·v)) · √(π/8) · exp(−2y²/w²) · (1 + erf(√2·x/w))
```

and the phase accumulated by a frozen (non-evolving) field over a path `L`:

```text
φ(x, y) = −k·(n0 − 1)/T0 · ΔT(x, y) · L
```

**Gate:** solver in frozen-field mode (diffraction off) reproduces this
screen to **< 0.5 % relative** everywhere the intensity exceeds `10⁻⁶·I0` —
it is pure quadrature + interpolation, so this is a correctness test, not a
physics tolerance. The erf-saturation shape (peak phase at the downwind
edge, Gaussian in `y`) is checked structurally, not just point-wise.
References: Smith (1977) §III; Manning, NASA/TM—2012-217634, Eq. (63) —
the same erf thin-screen form.

Define the **peak phase distortion number** from B1 (our exact, documented
convention — literature definitions differ by O(1) factors):

```text
N_φ ≡ |φ|max = √(2/π) · k·(n0 − 1)·α_abs·P·L / (T0·ρ·c_p·v·w)
```

`N_φ` is the run-summary blooming strength (reported in `_notes.md`
alongside the Fried parameter and Rytov variance), and the axis for B3.

### B2 — weak-blooming perturbation limit (tight, coupled)

With diffraction ON, first-order perturbation theory predicts the
receiver-plane on-axis intensity change is **linear in `N_φ`** as `P → 0`.
**Gate:** the fully coupled solver, run at a ladder of powers
(`N_φ = 0.05 … 0.4`), extrapolates to the analytic slope within **1 %**,
with the quadratic residual consistent with `O(N_φ²)`. This is the test
that the coupling (not just the medium formula) is right.

**✅ OPERATIONALIZED (2026-07-17):** the first-order reference is built
analytically — per-slab phase screens from the *undisturbed* Gaussian
(`w(z)` from the M1 closed form, power attenuated `e^(−α_abs·z)`) fed
through the closed-form ΔT of B1, applied as a linear `Medium`
(`TurbulentPath::from_screens`). The coupled deficit is normalized by the
Beer–Lambert transmission so the comparison isolates blooming from plain
absorption. Measured: coupled vs first-order agreement **0.008 %** at
`N_φ = 0.1`; back-reaction gap ratio (0.2 vs 0.1) **3.65** (theory 4).

### B3 — Gebhardt/Smith irradiance curve (magnitude + trend, published)

The classic steady-state result for a collimated Gaussian in crosswind:
relative peak irradiance `I_peak/I_peak,vac` vs distortion number falls
monotonically; the profile shifts **upwind** and develops the crescent
shape; for a *focused* beam, delivered peak irradiance vs power exhibits
the critical-power rollover (more power → less irradiance beyond it).
**Gate:** digitize the published curve (Smith 1977, whole-beam
steady-state figure — pin the exact figure number when the paper is in
hand at M4 start; Gebhardt 1976 as cross-check) onto our `N_φ` axis with
the conversion factor between conventions derived symbolically and stated
in the test, and require agreement within **±15 %** over `N_φ ∈ [1, 10]`
plus the three qualitative signatures (upwind shift, crescent, rollover).
This is the magnitude+trend anchor; the tight anchors are B1/B2.

**Fallback (pre-committed):** if the digitized Smith/Gebhardt curve cannot
be matched to ±15 % after the convention conversion is triple-checked, the
M4 gate becomes B1 + B2 + published-signature checks (qualitative) + a
cross-code comparison against a modern open wave-optics blooming result,
and this spec gets amended with the reason — decided now, not renegotiated
at the finish line.

**✅ STATUS (2026-07-17, resolved):** both parts are gated in
`tests/blooming.rs`. Qualitative: `b3_qualitative_signatures` (upwind peak
shift, downwind crescent, monotone irradiance rollover over `N_φ = 1, 3, 6`).
Quantitative: `b3_smith1977_curve_quantitative` reproduces Smith's whole-beam
`I_REL(N)` curve to **7.2 % max** over `N ∈ [0.5, 1.8]` (±15 % gate), rollover
minimum at `N ≈ 1` matched to 0.7 %.

Source figure and convention (as supplied): Smith (1977), the whole-beam
steady-state top panel `I_REL` vs `N` for `F₀ = k·a²/z₀ ∈ {∞, 20, 10, 5}`; we
target the **F₀ = 5** dash-dot branch, WebPlotDigitizer-traced into
`tests/data/smith1977_F5.csv` (13 points out to N ≈ 1.87). Two corrections to
the original plan, both material:

1. **Axis is Smith's `N_c`, not our `N_φ`.** The plan assumed a symbolic O(1)
   conversion from `N_φ`. In fact the abscissa of the forced-convection
   whole-beam curves is Smith's *geometrical-optics* distortion number `N_c`
   (no wavenumber; `a = w/√2` the 1/e amplitude radius):

   ```text
   N_c = −μ_T·I₀·α·z² / (μ·ρ·c_p·v·a) · [2/(αz) − 2/(αz)²·(1 − e^(−αz))]
   ```

   with `−μ_T = (n₀−1)/T₀`, `μ = n₀`, `I₀ = 2P/(πw²)`.
   `BloomingCase::smith_distortion_number` implements this **verbatim**,
   evaluating the full absorption bracket (0.9835 here; we do not use Smith's
   `αz ≪ 1` simplification), so the run sits on the exact published axis with
   **no conversion and no approximation**. (An earlier draft justified the axis
   via an effective-path brace `N = N_c·{(2/z²)∫Q⁻¹∫ e^(−αz'')/(ΩQ²)}` and a
   sub-Rayleigh `N ≈ N_c` argument — that brace is the generalization for
   *focused / slewed* beams from a **later** section of the paper and does not
   govern these collimated forced-convection curves; the `≲4 %` uncertainty it
   implied does not exist.) Matching `F₀` is done by geometry
   (`w₀ = √(2·F₀·z/k)`), not by rescaling.
2. **`I_REL` normalization.** Smith's `I_REL = I_bloomed/I_unbloomed` cancels
   the common Beer–Lambert loss (→ 1 at `N → 0`); our vacuum reference has no
   absorption, so the test divides the peak ratio by `e^(−αz)` — the same
   transmission normalization used in B2.

Honest residual: at the high-N end (`N = 1.8`) the wave solver shows a mild
diffractive recovery (`I_REL` rising 0.757 → 0.807) that Smith's flat `F₀ = 5`
curve does not. This is the 7.2 % worst point; the descent and rollover are
matched much more tightly (0.1 %, 0.7 %, 3.0 % at N = 0.5, 1.0, 1.5). The
fallback (B1+B2+signatures+cross-code) was **not** needed. The mechanism is
dissected below.

What the residual is *not*: it is not an axis-convention error (the abscissa is
Smith's exact `N_c`, § above — and *any* x-axis rescaling is power-independent,
so it shifts the curve uniformly and cannot produce a deviation that is ~0 % at
low N and grows to 7 % at high N; empirically the worst deviation only drops
below 7 % for an unphysical ≥15 % x-stretch, which then worsens the low-N fit).
Nor is it Smith reducing to geometrical optics (`F₀ = 5` is a *finite*-Fresnel
curve that already carries diffraction — only `F₀ = ∞` is the ray limit).

A power sweep through the M5 bindings (`scripts/sweep_blooming.py`) plus a
resolution study settle what it *is*.

**It is not numerical.** At both `F₀ = 5` and `F₀ = 20`, high-N `I_REL` is
converged to ≤ 0.04 % under halving `dx`, doubling the step count, and doubling
the domain, with guard-band absorption at machine zero (`~1e-13`); the receiver
peak sits at the upwind crescent edge (`x ≈ −2.5 cm` at `F₀ = 5, N = 1.8`;
`−4.1 cm` at `F₀ = 20`) at every resolution. No aliasing, no domain leakage.

**It is the off-axis crescent cusp.** As `N` grows the on-axis irradiance
collapses monotonically (`I_REL` on axis 0.90 → 0.19 over `N = 0.13 → 1.8`) —
the whole-beam defocus+tilt sweeping energy off the axis — while the *global*
peak migrates upwind and recovers (0.758 → 0.806). Both our `I_REL` and Smith's
report this global peak (Smith's ≈ 0.75 sits far above our on-axis 0.19), so the
comparison is like-for-like: the residual is a real disagreement about the cusp,
not a definitional mismatch. The cusp is a caustic-like re-concentration the
ray-folding thermal lens forms on the cool upwind side — resolving it is exactly
what a full-wave field march does and a reduced-order **whole-beam** steady-state
theory (Smith's) does only approximately.

**It tracks the strength of that caustic.** Re-running the solver on Smith's own
`F₀ = 10` and `F₀ = 20` branches localizes the disagreement: over `N ∈ [0.3,
1.15]` the solver matches the digitized curves to **1.1 % (F₀ = 5), 2.0 %
(F₀ = 10), 5.3 % (F₀ = 20)** — the gap grows with `F₀` and `N`, i.e. with the
strength of the caustic recovery, our full-wave peak recovering somewhat *less*
than the whole-beam prediction. That is the regime where a low-moment whole-beam
expansion is least faithful (strong, non-quadratic erf-shaped phase), so a
divergence there follows from the formulation difference, not a solver error. The
larger 7 % at `F₀ = 5, N ≈ 1.8` is additionally inflated by digitization of
Smith's flat `F₀ = 5` dash-dot tail, which does not recover at all — inconsistent
with the recovery that both his own `F₀ = 10/20` curves (minima at `N ≈ 0.73`,
`0.58`, then rising) and our solver exhibit, and with the monotonic-in-`F₀`
recovery trend. (A tried-and-rejected "rising effective Fresnel number" reading:
deviation from the `F₀ = 10/20` curves *grows* rather than shrinks with `N`, so
the beam is not simply behaving like a higher `F₀`.) The residual stays well
inside the ±15 % gate and never touches the descent or the rollover minimum,
which both formulations match to ~1 % — the observables B3 actually anchors.

### Stability gate

The strong-blooming boundedness test from § Stability above (`N_φ ≈ 20`,
finite, power-conserving, `dz`-refinement-consistent).

## Failure modes (new codepaths)

- **Slab-entrance coupling (1st order masquerading as 2nd):** caught by the
  convergence-slope test — the reason it gates.
- **Wind-axis sign error:** beam bends downwind instead of upwind — caught
  by B1's structural check and B3's upwind-shift signature.
- **Bent beam into the guard band:** silent wraparound — caught by the
  runtime beam-centroid assert.
- **Convention mismatch in B3:** a wrong factor between our axis and Smith's
  `N_c` reads as a uniform horizontal shift of the whole curve. Resolved by
  implementing Smith's forced-convection `N_c` verbatim (`smith_distortion_number`,
  full absorption bracket) rather than rescaling `N_φ` — the abscissa is exact,
  not a sub-Rayleigh approximation. Because any such factor is power-independent,
  it can only shift the curve uniformly and cannot mimic the N-growing high-N
  residual; the B1/B2 tight anchors and the near-perfect low-N fit disambiguate a
  real physics error from a slip.

## References

- D. C. Smith, *High-power laser propagation: Thermal blooming*,
  Proc. IEEE **65**, 1679–1714 (1977) — the review; steady-state crosswind
  theory and whole-beam irradiance curves (B3).
- F. G. Gebhardt, *High power laser propagation*, Appl. Opt. **15**,
  1479–1493 (1976) — scaling laws, distortion-number phenomenology (B3
  cross-check).
- J. A. Fleck, J. R. Morris, M. D. Feit, *Time-dependent propagation of
  high energy laser beams through the atmosphere*, Appl. Phys. **10**,
  129–160 (1976) — the coupled split-step method (already the M1
  propagator reference).
- R. M. Manning, *High Energy Laser Beam Propagation in the Atmosphere…*,
  NASA/TM—2012-217634 (2012) — forced-convection heat budget (Eq. (7)) and
  the closed-form erf thin-screen phase (Eq. (63)) behind B1.
- P. E. Ciddor, *Refractive index of air: new equations for the visible
  and near infrared*, Appl. Opt. **35**, 1566–1573 (1996) — `(n − 1)` for
  the property table.
- Mutation++ (VKI), <https://github.com/mutationpp/Mutationpp> — offline
  property-table generator (LGPL code never linked into the solver).
- E. W. Lemmon, R. T. Jacobsen, S. G. Penoncello, D. G. Friend,
  *Thermodynamic properties of air and mixtures of nitrogen, argon, and
  oxygen…*, J. Phys. Chem. Ref. Data **33**, 111 (2004) — NIST reference
  values gating `κ_t`.
- B. J. McBride, M. J. Zehe, S. Gordon, *NASA Glenn coefficients for
  calculating thermodynamic properties of individual species*,
  NASA/TP—2002-211556 (2002) — NASA-9 polynomials in the analytic
  cross-check.
