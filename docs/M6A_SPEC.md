# M6a pre-spec — 0-D optical-breakdown threshold kernel

Written **before** the M6a code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md)): pin the rate model, the constants, the integrator,
and — most importantly — *which* checks actually validate physics versus which
are integrator unit tests. If any of this proves wrong during implementation,
amend this document first, then the code.

Scope: the standalone 0-D rung of the M6 laser-induced gas-breakdown ladder.
No propagator changes; the kernel is a **driver-callable pure function** rather
than a `Medium`, so anything that needs an ignition test can call the same core
(M6c's LSD trigger is the first thing that does).

Conventions follow [MODELS.md](MODELS.md): SI units, `k = 2π/λ`, intensity
`I = |u|²` (W/m²). New symbols: electron number density `n_e` (m⁻³), pressure
`p` (Pa), laser angular frequency `ω = 2πc/λ` (rad/s), electron-neutral
momentum-transfer collision frequency `ν_m` (s⁻¹), effective ionization energy
`U_i` (J), diffusion length `Λ` (m).

## Physics target (recorded)

- **Regime: nanosecond-pulse breakdown in clean dry air at 1064 nm**, over
  ~10–2000 Torr. Cascade (avalanche) ionization dominant, multiphoton
  ionization as the electron seed. Local thermodynamic **non**-equilibrium is
  not modelled — this is a single-point electron-density rate balance, not a
  plasma-kinetics code.
- **0-D**: one point, one intensity history `I(t)`; no transport of `n_e`
  between points.
- **Single pulse**: no inter-pulse recombination/afterglow state.

## Rate model

Electron density at a point obeys the avalanche balance

```text
dn_e/dt = (ν_i(I,p) − ν_loss(p)) · n_e + S_mpi(I,p)
ν_loss = ν_att(p) + ν_diff(p)
```

### Cascade ionization `ν_i` (inverse-bremsstrahlung heating)

An electron gains energy from the optical field by inverse bremsstrahlung; the
ionization frequency is the energy-absorption rate divided by the effective
ionization energy `U_i` (Zel'dovich & Raizer; Kroll & Watson 1972; Morgan 1975):

```text
heating(I, p) = (e²·I) / (m_e·c·ε₀) · ν_m / (ν_m² + ω²)
ν_m(p)        = K_m · p      (electron-neutral collisions ∝ neutral density)
```

This is the **gross** heating. The ionization rate uses the *net* power, after
the inelastic losses of the next section:
`ν_i = max(0, heating − L)/U_i`.

The Lorentzian `ν_m/(ν_m²+ω²)` is the standard IB absorption factor. It has two
limits, and which limit we are in sets the sign structure the gates rest on:

- `ν_m ≪ ω` (low-collision / optical regime): `ν_i ∝ I·ν_m ∝ I·p`.
- `ν_m ≫ ω` (collision-limited): `ν_i ∝ I/ν_m ∝ I/p`.

At 1064 nm, `ω = 2πc/λ ≈ 1.77×10¹⁵ rad/s`, while `ν_m` at 1 atm is `~4×10¹² s⁻¹`
— so **`ν_m ≪ ω` across the entire 10–2000 Torr range**, and `ν_i ∝ I·p`
throughout. There is no IB (`ν_m = ω`) minimum in range; the low-pressure
threshold rise comes from diffusion loss, not from the Lorentzian.

### Inelastic energy loss `L` — why the cascade coefficient falls with pressure

*(Added 2026-07-24, before the code, per the rule at the top of this document.
This is the term the first external-gate failure demanded; the reasoning that
led here is under "External gates" below.)*

An electron cannot convert all its inverse-bremsstrahlung heating into
ionization: on the climb to `U_i` it loses energy to vibrational and electronic
excitation of N₂ and O₂. The cascade rate is driven by the **net** power:

```text
ν_i(I, p) = max(0, heating(I,p) − L(p)) / U_i
heating(I,p) = (e²·I)/(m_e·c·ε₀) · ν_m/(ν_m²+ω²) ≈ h·I·p
L(p) = δ_eff · ν_m(p) · ⟨ε⟩ ≡ L′·p          (∝ p, since ν_m ∝ p)
```

`δ_eff` is the standard fractional energy loss per collision and `⟨ε⟩` the mean
electron energy; only their product enters, as the single lumped constant
`L′ = δ_eff·K_m·⟨ε⟩` (W·Pa⁻¹ per electron).

**Why this is the right shape.** Both terms carry the same factor `p`, so

```text
I_thr(p) = L′/h + U_i·(ν_diff + ν_att + G)/(h·p)
```

— a **constant plus a 1/p term**. The constant is a genuine high-pressure
plateau, which is exactly the behaviour the data shows and which the previous
model could not produce at any parameter value (its floor was `n = 1`). Here
`n → 0` as `p → ∞` and `n → 1` at low `p`, so `n ∈ (0, 1)`.

**Two limits of the same balance.** `⟨ε⟩` above is an assumption, and it can
be removed. The electron energy obeys a linear ODE,

```text
dε/dt = heating − δ_eff·ν_m·ε      =>   ε(t) = ε_∞(1 − e^{−t/t_r})
ε_∞   = heating/(δ_eff·ν_m)         (pressure-independent: both terms ∝ p)
t_r   = 1/(δ_eff·ν_m)
```

Ionization occurs when the climb reaches `U_i`, so `t_climb = t_r·ln(ε_∞/(ε_∞ −
U_i))` and

```text
ν_i = δ_eff·ν_m / ln(ε_∞/(ε_∞ − U_i)),     zero if ε_∞ ≤ U_i
```

with **no `⟨ε⟩` anywhere** and still exactly `∝ p`. Both forms are implemented
as `CascadeModel::{FixedMeanEnergy, SelfConsistentClimb}`; the self-consistent
one is the **default**.

| model | free constants | slope `n` | vs measured 0.329 |
|---|---|---|---|
| no inelastic loss | — | 1.737 | 5.3× too steep |
| `FixedMeanEnergy` | `δ_eff`, `⟨ε⟩` | 0.468 | 1.4× steeper |
| **`SelfConsistentClimb`** (default) | `δ_eff` | **0.095** | 3.5× flatter |

*(Numbers corrected 2026-07-25. The values previously in this table — 0.800 and
0.356 — were inflated by the seed-window artifact described under Seeding
below, which supplied ~60 of the ~82 nats the threshold criterion demanded and,
being pressure-dependent, manufactured slope. The apparent 8 % agreement of the
self-consistent model was an artifact of that bug.)*

**What survives the correction is the bracket, not a match.** The two limits sit
either side of the measurement — 0.095 and 0.468 against 0.329 — and neither
reproduces it. That straddle is what a mean-energy treatment of a tail-driven
process should give, and it is the claim
`the_two_cascade_models_bracket_the_measurement` gates. It is also the claim
that *stayed true* while both endpoints moved.

**But the bracket is a one-parameter sensitivity, not two independent limits**
*(narrowed 2026-07-25)*. Both variants reduce to `ν_i → heating/U_i` at high
intensity; they differ only in **where the cascade cuts off** —
`FixedMeanEnergy` at `ε_∞ = ⟨ε⟩ = 3 eV`, `SelfConsistentClimb` at
`ε_∞ = U_i = 12.06 eV` — and that cutoff sets the plateau, hence the slope.
Sweeping that single energy walks continuously between the endpoints:

| `FixedMeanEnergy` `⟨ε⟩` | 3 eV | 4 eV | 5 eV | 6 eV | 8 eV | 12.06 eV |
|---|---|---|---|---|---|---|
| slope `n` | 0.468 | 0.397 | 0.346 | 0.309 | 0.255 | 0.192 |

So containing 0.329 is a statement about the uncertainty in one loosely-known
energy, not a physical straddle, and the interval is **not a bound**: fixing
`⟨ε⟩` at `U_i` gives 0.192, *inside* it. The wavelength gate below is the
external agreement that does not have this weakness, and should be read as the
milestone's headline in place of the bracket. Also note the sense of the
one-parameter walk: `⟨ε⟩` ≈ 5 eV reproduces the measurement on its own, which
is exactly why it must not be selected — see "The line on fixing it".

**Honesty about status.** `δ_eff` remains free within its asserted range
(0.01–0.05), which for the default gives `n ∈ [0.023, 0.231]` — **excluding**
the measurement, so `tt2012_threshold_slope_matches_measurement` is red. For
`FixedMeanEnergy` the same sweep over `(δ_eff, ⟨ε⟩)` gives `[0.187, 0.785]`,
which does contain it. The default is chosen for parameter parsimony, **not**
for agreement; switching it to the better-agreeing variant would be fitting.

Two further caveats, stated because they bound how much any of this means:

- **The `δ_eff` range 0.01–0.05 is asserted, not sourced.** It is a plausible
  band for the fractional energy loss per collision in air above ~1 eV, but no
  citation backs those endpoints, and `δ_eff` sets the threshold *level*
  linearly. Matching T&T's level would need `δ_eff ≈ 0.003`. There *is* an
  independent number available — Eq. 4's `λ⁻²` coefficient implies
  `δ_eff·⟨ε⟩ = 0.060 eV`, which is exactly the literature centre already in use
  — but adopting it as a pin would make the Eq. 4 level gates circular. See the
  wavelength-gate section for the trade.
- **`⟨ε⟩` is not fully eliminated even in the default.** A mean energy still
  enters through `D_e,ref = 0.2 m²/s` (an energy-dependent transport
  coefficient; `v̄·λ_mfp/3` at 3 eV gives ≈0.08 m²/s) and through holding
  `δ_eff` itself constant. `self_consistent_climb_eliminates_the_mean_energy`
  proves only that the `with_inelastic_loss` `⟨ε⟩` knob cannot reach that path.
  Given the audit below found `D_e` sensitivity **large**, the parsimony
  argument for the default is thinner than "one free constant" suggests.
- **The level is 4.8–7.2× high and drifts** across the window; the drift is the
  residual slope error in absolute clothing. Level stays ungated (3–10×
  inter-lab scatter).

**Sensitivity audit (2026-07-24, values superseded 2026-07-25)** — run after the
gate first went green, because a passing gate earns *more* scrutiny, not less.
It verified that an independent Python reimplementation (from these equations,
not the Rust) reproduced the Rust to four digits, that the result was stable
under discretization and T&T's ±1 ns pulse uncertainty, and that two ungated
constants moved the slope substantially:

| perturbation | effect on `n` |
|---|---|
| `n_bd` ×0.1 / ×10 | negligible |
| pulse FWHM 5 / 7 ns | ±2 % |
| `D_e` ×0.5 / ×2 | **large** — `ν_diff` and `G` are comparable, not sub-dominant |
| generation prefactor `ln 2` vs 1 | **~30 %** |

Both findings stand. The `D_e` sensitivity in particular withdraws any claim
that the high-pressure branch is "Λ-independent" — that described the
pre-plateau model. What the 2026-07-24 audit *missed* was the window artifact,
which a numerical-hygiene check (vary the integration bounds, demand
invariance) would have caught immediately; that check now exists as
`threshold_is_window_independent`.

### Diffusion loss `ν_diff` — Λ is pinned, never fit (D5)

```text
ν_diff(p) = D_e(p) / Λ²,   D_e(p) = D_e,ref · (p_ref / p)   (free-electron, D_e ∝ 1/p)
```

`Λ` is the fundamental diffusion length of the **focal geometry**, computed from
the spot, not tuned to the data.

*(Corrected 2026-07-25 from the paper itself.)* T&T's Eq. 5 gives it for their
geometry:

```text
(1/Λ)² = (π/l₀)² + (2.405/r₀)²,    r₀ = f·α/2 = 20 µm,
                                   l₀ = 0.414·(α/d)·f² = 66 µm
```

with `f` = 4 cm, `α` = 1 mrad divergence, `d` = 1 cm beam — giving
**Λ = 7.74 µm**, matching the `Λ = 8 µm` the paper states, and a focal volume
`π r₀² l₀ = 8.32×10⁻¹⁴ m³`.

Two earlier guesses were both wrong. Modelling the focus as a **sphere**
(`Λ = r₀/π = 6.37 µm`) overstated `ν_diff` by 1.48×. Modelling it as a
diffraction-limited **filament** (Rayleigh range 1.2 mm) was wrong the other
way: the beam is *divergence*-limited at 1 mrad, so the depth of focus is
66 µm, not 2.4 mm.

`Λ` remains **documented and asserted as an input**; tuning it to move the
threshold onto the measured curve is curve-fitting and is explicitly
forbidden.

### Attachment loss `ν_att` — two channels, from measured rate coefficients

```text
ν_att(p) = k₂·n_O₂ + k₃·n_O₂·n,     n = p/(k_B T),  n_O₂ = 0.21·n
```

with `k₂ = 1e-17 m³/s` (dissociative, `e + O₂ → O⁻ + O`) and
`k₃ = 1e-43 m⁶/s` (three-body, `e + O₂ + M → O₂⁻ + M`; Kossyi et al. 1992,
Itikawa 2009).

*(Corrected 2026-07-25.)* At 1 atm the **two-body** channel leads:
`5.4×10⁷ s⁻¹` against `1.4×10⁷` for three-body. Three-body overtakes it only
above `n = k₂/k₃ = 10²⁶ m⁻³` (≈4 atm). Earlier revisions of this document
asserted three-body was dominant at atmospheric density, which is wrong by 4×;
nothing was gated on it (the guard test only ever demanded three-body be
non-negligible, and now also demands it be sub-dominant). The `∝ p²` scaling of
the three-body channel is still what turns the threshold slope positive far
above the gate window — that crossover is just at ~4 atm, not at 1 atm.

*(Amended 2026-07-24. This replaced a single order-of-magnitude constant
`ν_att = K_a·p` with `K_a = 1e5 s⁻¹Pa⁻¹`. That value was ~150× too large:
it gave `1.0×10¹⁰ s⁻¹` at 1 atm where the measured coefficients give
`6.7×10⁷ s⁻¹`. The consequence matters — see the external gates below — because
the inflated constant was the only term flattening the modelled threshold slope
toward the measured one. With attachment correct it is **negligible against
diffusion** (`3.3×10⁹ s⁻¹` at 1 atm), and the modelled slope moves from −0.72
to −1.74, i.e. **away** from the data. The apparent near-agreement was an
artifact of a wrong constant.)*

### Multiphoton seed `S_mpi`

`S_mpi = σ_K · I^K · N`, with `K = ⌈U_ion/(ħω)⌉` photons (O₂ 12.06 eV at
ħω = 1.166 eV → `K = 11`) and `N` the neutral density. Tiny in absolute terms;
its role is to seed the first electron so the cascade has something to multiply.
`σ_K` is a documented, **swappable** coefficient (design-doc Open Question 2);
the gate is constructed so a `σ_K` swap does not move the slope.

### Breakdown criterion

Breakdown at a point is declared when `n_e` reaches the criterion density
`n_bd` within the pulse. `n_bd = 10²³ m⁻³` (onset of optically-significant
ionization; documented constant). The initial seed is one electron in the focal
volume, `n_e0 = 1/V_focal`.

### Seeding and the integration window (amended 2026-07-25)

`n_e` is **floored at the seed density** throughout the pulse integration, and
that floor is physics, not hygiene.

Without it the model was not window-independent. The integration runs over
`[−w·FWHM, +w·FWHM]`; during the quiet arm before the pulse, losses grind the
seed down — at 760 Torr `ν_loss ≈ 3.4×10⁹ s⁻¹` decays it by `e^-41` over 12 ns —
so the avalanche had to climb out of a hole whose depth was set by `w`. Of the
~64 nats of growth the criterion then demanded, ~41 came from the arbitrary
integration bound and only ~23 from the physical `ln(n_bd/n_seed)`.

That is meaningless as physics (`e^-41` of one electron is `10⁻¹⁸` of an
electron, in a volume that contains either one or none), and it had two
consequences:

- the threshold rose 11 % from `w = 2` to `w = 4` and never converged;
- because `ν_loss` is pressure-dependent, it **manufactured slope** — see the
  retraction under External gates.

The floor states the modelling assumption plainly: one seed electron is
available in the focal volume when the pulse arrives. `threshold_is_window_
independent` gates the resulting insensitivity (< 1 % over `w ∈ [1, 4]`).

Note this is an *assumption*, not a derivation. At 1 atm the ambient electron
density from background ionization (~10⁹ m⁻³) puts ~8×10⁻⁵ electrons in a
8.3×10⁻¹⁴ m³ focal volume, i.e. usually **none** — which is why ns breakdown at
tight focus is stochastic and why the physically correct seed is multiphoton
ionization during the pulse. Turning `S_mpi` on with a defensible `σ_K` is the
open item; until then the floor is the honest stand-in and is labelled as such.

## Constants (SI, documented)

| symbol | value | source / note |
|--------|-------|---------------|
| e | 1.602176634e-19 C | exact |
| m_e | 9.1093837015e-31 kg | CODATA |
| c | 299792458 m/s | exact |
| ε₀ | 8.8541878128e-12 F/m | CODATA |
| ħ | 1.054571817e-34 J·s | CODATA |
| U_i | 12.06 eV → 1.932e-18 J | effective ionization energy (O₂ IP); prefactor only, does **not** affect the slope gate |
| K_m | 3.9e7 s⁻¹·Pa⁻¹ | ν_m/p for air ≈ 5.3×10⁹ s⁻¹·Torr⁻¹ rescaled to Pa |
| D_e,ref | 2.0e-1 m²/s at p_ref | free-electron diffusion, order-of-magnitude; sets absolute low-p rise only |
| L′ | 3.75e-13 W·Pa⁻¹ | `δ_eff·K_m·⟨ε⟩` at the literature centre (δ_eff = 0.02, ⟨ε⟩ = 3 eV); sets the high-p plateau |
| k₂ | 1.0e-17 m³/s | dissociative attachment `e + O₂ → O⁻ + O` (Itikawa 2009) |
| k₃ | 1.0e-43 m⁶/s | three-body attachment `e + O₂ + M → O₂⁻ + M` (Kossyi 1992); sub-dominant below ≈4 atm, `∝ p²` |
| f_O₂ | 0.21 | O₂ number fraction of dry air |
| n_bd | 1.0e23 m⁻³ | breakdown criterion density |
| p_ref | 101325 Pa | 1 atm reference |

The absolute threshold *level* depends on `U_i, D_e,ref, k₂, k₃, n_bd, Λ` and is
**not gated** (inter-lab scatter is 3–10×). The *slope* is the gate; with
attachment negligible it is set by the competition between diffusion loss and
the finite-pulse growth requirement, and is bracketed analytically at
`n ∈ [1, 2]`.

## Integrator — exact exponential per constant-I slice

Over a time-slice `dt` where `I` (hence `β = ν_i − ν_loss` and `S = S_mpi`) is
constant, the linear ODE has the closed form

```text
n_e(t+dt) = n_e·exp(β·dt) + S·expm1(β·dt)/β        (β ≠ 0)
n_e(t+dt) = n_e + S·dt                              (β → 0, the expm1 limit)
```

**Amended 2026-07-25 — growth is logistic, not linear.** Ionization consumes
the neutrals it feeds on, so the rate equation carries a depletion term:

```text
dn_e/dt = ν_i·n_e·(1 − n_e/N) − ν_loss·n_e + S_mpi,     N = p/(k_B T)
```

With `S = 0` this is Bernoulli and still has an exact per-slice solution,
evaluated as

```text
n_e(t+dt) = n_e / ( e^{−β dt} + b·n_e·(1 − e^{−β dt})/β ),   b = ν_i/N
```

**not** the textbook `β n e^{βdt}/(β + b n (e^{βdt} − 1))`, which overflows to
`inf/inf = NaN` far above threshold. In this form `e^{−β dt}` underflows
harmlessly and `n_e → β/b`, the saturation density. No exponent clamp is
applied on the growing side — clamping there corrupts the saturation limit
itself and made `peak_ne` non-monotonic in intensity.

Without depletion the equation is linear and `n_e` ran away: the shipped
`breakdown` case reached `10⁴⁰ m⁻³`, `10¹⁴×` the neutral density and `10¹³×`
critical density at 1064 nm, with the apparent ceiling being a plotting clamp
rather than physics. Saturation does **not** move the threshold (`n_e/N ≈ 0.4 %`
at the criterion), so it is a pure correction to the post-breakdown regime.

`expm1` keeps the `β → 0` and small-`β·dt` cases accurate. The exponent is
saturated at `β·dt = 700` (just under the `f64` `exp` ceiling): far above
threshold the unsaturated `exp` overflows to `+inf`, and `S·expm1(β dt)/β` then
evaluates `0·inf = NaN` whenever the multiphoton source is off — which would
make a *stronger* pulse report no breakdown and silently break the bisection's
monotonicity assumption. Saturating is safe because any `β·dt` near 700 is
already many orders past `n_bd`. The ODE is **linear
in n_e per slice**, so this is exact for piecewise-constant `I` — no stiff
solver, no sub-stepping error. Pulse integration samples a Gaussian temporal
profile `I(t) = I_pk·exp(−4 ln2 (t/τ_FWHM)²)` over `[−2τ, 2τ]` and tracks
`max n_e`.

## Gates

### Physics gate

**Log-log slope of `I_thr(p)` over the pinned range 300–2000 Torr** — the
high-pressure branch only.

*(Amended twice during implementation, per the rule at the top of this document.
**2026-07-23:** the original text gated the slope over the full 10–2000 Torr
sweep against `I_thr ∝ p^-1`; both halves were wrong. **2026-07-24:** the band
was replaced by an analytic bracket after attachment was corrected — see below.)*

Solving the avalanche criterion — cascade must outrun losses *and* clear the
`n_seed → n_bd` growth requirement `G = ln(n_bd/n_seed)/τ` within the pulse —
gives

```text
I_thr(p) = L′/h + U_i·[ν_att(p) + ν_diff(p) + G] / (h·p)
```

**This closed form is a scaling tool, not the model.** It assumes the cascade
runs at its peak rate for the whole of `τ`, whereas the code integrates a
Gaussian pulse whose intensity is near peak for only part of it, so the code's
threshold is higher than the closed form. The offset is pressure-independent,
so the exponents below are unaffected — but never quote the closed form as a
level, and note that being further above the plateau makes the code's slope
*steeper* than the closed form predicts. This bit twice: for `FixedMeanEnergy`
the closed form said 0.521 and the code gives 0.468; for `SelfConsistentClimb`
it said 0.060 and the code gives 0.095. **Trust the code.**

Attachment is negligible at these densities (`6.7×10⁷` against `3.3×10⁹ s⁻¹`
for diffusion at 1 atm), leaving three terms whose exponents are **exact**:

```text
plateau (inelastic-limited):   I_thr → L′/h                     ∝ p^0
growth-limited:                I_thr = U_i·G/(h·p)              ∝ p^-1
diffusion-limited:             I_thr = U_i·ν_diff/(h·p)         ∝ p^-2
```

Any mixture falls between them, and the gate asserts **`n ∈ (0, 1)`** — the
plateau must be doing work, since without `L′` the model cannot get below
`n = 1` at all. Observed with the default `SelfConsistentClimb`: **`n = 0.095`**,
level at 760 Torr `1.18×10¹² W/cm²` (not gated). With `FixedMeanEnergy`:
`n = 0.468`, level `4.58×10¹¹`. Both envelopes are pinned separately —
`[0.023, 0.231]` and `[0.187, 0.785]` respectively.

**The bracket is not universal, and the qualifier is load-bearing.** It holds
only while attachment is negligible. Three-body attachment is `∝ p²` and
contributes `I_thr ∝ +p`, so far above this window the model does leave the
interval: measured on this code, the slope is `−0.81` over 2000–10000 Torr and
turns *positive* above ~10⁴ Torr. That is precisely why the gate range is
pinned rather than open-ended, and it is also the honest limit on the
"unreachable" claim below.

The range still matters and is pinned to **300–2000 Torr**, bracketing T&T's
measurement range. Fitting the full 10–2000 Torr sweep mixes in the low-p
diffusion branch that D5 excluded.

This gate is the model's **self-consistency**, not agreement with experiment —
T&T measure `n = 0.33`, outside the interval entirely. See the external gates.

The external comparison against the digitized Thiyagarajan & Thompson (J.
Appl. Phys. 111, 073302 (2012)) curves is no longer pending — it is
implemented; see the next section. Absolute agreement remains NOT gated
(inter-lab scatter). The slope check above stays as the internal
model-consistency layer beneath it.

### External gates — Thiyagarajan & Thompson 2012 (digitized 2026-07-24)

T&T **Fig. 4** plots **two** curves against pressure: the breakdown threshold
field `E_B` and the effective field `E_eff`. Both are digitized into
`tests/data/tt2012_*.csv`. Together with the paper's own closed forms (Eq. 4
cascade theory, Eq. 5 focal geometry) they yield **six** gates:

| gate | status |
|---|---|
| `tt2012_collision_frequency_matches_literature` | passes (5 %) |
| `tt2012_effective_field_rises_with_pressure` | passes |
| `tt2012_cascade_theory_reference` | passes |
| `tt2012_wavelength_scaling_matches_cascade_theory` | passes — the headline |
| `tt2012_level_ratio_is_bounded_within_scatter` | passes (regression pin) |
| `tt2012_threshold_slope_matches_measurement` | **RED**, `#[ignore]`d on purpose |

*(This paragraph previously claimed four gates, all passing and none
`#[ignore]`d — written before the slope gate was retracted on 2026-07-25 and
left deliberately red.)* The history below records what each stage passed and
failed, because the failures drove the model corrections.

The paper pins every experimental input the kernel takes — 1064 nm, 6 ± 1 ns
FWHM, **20 µm radius** focus, dry air, 10–2000 Torr — which is exactly what
`AirBreakdown::air_1064nm()` uses. No geometry is assumed and nothing is
fitted, so any disagreement is a statement about the rate model.

1. **Collision frequency `K_m` — PASSES to 5%.** By definition
   `E_eff/E_B = ν_m/√(ν_m²+ω²)`, so the ratio of the paper's own two curves
   measures `ν_m` using nothing from this crate except `ω = 2πc/λ`. This is
   the **independent, non-circular anchor** the fallback clause below demanded:
   `K_m` entered the kernel from Raizer-lineage literature and is here checked
   against measurement. Implied `K_m = 4.21×10⁷` vs the kernel's `3.90×10⁷
   s⁻¹Pa⁻¹`, and the ratio is *flat* at 1.05 ± 0.01 across 46–1858 Torr — a
   constant ratio over a 40× span is `ν_m ∝ p` and `ν_m ≪ ω` together, i.e. the
   entire IB Lorentzian branch. It also settles a convention the axis label
   does not state: **`E_B` is an RMS amplitude**, since reading it as a peak
   puts the ratio off by a flat √2.
2. **`E_eff` rises with pressure — PASSES.** Predicted `p^+0.642`, measured
   `p^+0.695`. The sign is the physics: `E_eff` only rises because `ν_m ≪ ω`
   makes the IB factor grow ∝ p faster than the threshold field falls.
3. **Threshold pressure-slope — RED, retracted 2026-07-25.** Measured
   `E_B ∝ p^-0.164` over 300–2000 Torr, i.e. `I_thr ∝ p^-0.329`. The default
   kernel gives `p^-0.095` and its `δ_eff` envelope `[0.023, 0.231]` excludes
   the measurement, so the gate is `#[ignore]`d rather than re-banded.

   This gate was **green before the 2026-07-25 audit**, reporting `n = 0.356`
   and an 8 % match. That was an artifact: the seed decayed by `e^-41` during
   the pre-pulse arm, so an arbitrary integration bound supplied ~41 of the ~64
   nats the criterion demanded, and its pressure-dependence manufactured slope.
   The three-stage narrative it supported is correspondingly corrected —
   1.737 → 0.468 → 0.095, not 1.737 → 0.800 → 0.356.

   **But the raw curve is the wrong target** — see "The comparison target"
   above. Against T&T's own cascade closed form the kernel is within 4.1–5.1×
   in level, and its flatness is what that theory predicts.

   | stage | slope `n` | vs measured |
   |---|---|---|
   | original (no inelastic loss) | 1.737 | 5.3× — and unreachable at *any* parameter value, since `n = 1` was the model's floor |
   | + inelastic loss at fixed `⟨ε⟩` | 0.468 | 1.4× |
   | + `⟨ε⟩` eliminated (self-consistent climb) | 0.095 | 3.5× the other way |

   The inelastic-loss term is still justified — it broke the hard `n = 1` floor
   — but it no longer *closes* the gap, and eliminating `⟨ε⟩` overshoots.

**What this does and does not settle.** Neither the shape nor the level is
reproduced. The level runs 4.8–7.2× above T&T and **drifts** by 1.48× across
the window — that drift is the residual slope error in absolute clothing, and
the earlier claim that the ratio had gone flat (6.4–7.4×, spread 1.16×) was
itself a product of the window artifact. Level remains ungated (3–10×
inter-lab scatter); the drift is pinned only as a regression check.

**The line on fixing it.** Re-pinning a constant from independent data is
legitimate; so is adding a term the physics demands. Tuning any constant until
the slope lands in a band is the curve-fitting the pinned-`Λ` rule forbids, and
the bracket above now makes such tuning provably futile as well as improper.

### The comparison target — corrected 2026-07-25 from the paper

The kernel models **collisional cascade only** (`σ_K = 0`). T&T's measured
curve is not a cascade-only observable: the paper attributes 88 % of the
ionization at 760 Torr to cascade and 12 % to multiphoton ionization, with MPI
*dominant* below 100 Torr, and it explicitly MPI-corrects the measured data
before comparing to cascade theory. Comparing a cascade-only kernel to the raw
curve was therefore a category error on our side.

The right reference is their **Eq. 4**, the cascade closed form:

```text
I_B(CC) = 1.44×10⁶ · (p_atm² + 2.2×10⁵·λ_µm⁻²)   W/cm²
```

At 1064 nm `λ⁻²` contributes `1.94×10⁵` while `p²` never exceeds 6.9 over
10–2000 Torr, so **accepted cascade theory is flat in pressure at this
wavelength** (fitted `n = −0.00002`), and reproduces the `2.8×10¹¹ W/cm²` the
paper quotes at 760 Torr. Implemented as
`validate::tt2012_cascade_threshold`, gated by
`tt2012_cascade_theory_reference`.

This reframes the headline. The kernel's flatness is *agreement with cascade
theory*, not disagreement with experiment; and the measured `n = 0.329` is
unreachable by **any** cascade-only model, this one included. Level against
Eq. 4: `SelfConsistentClimb` 4.1–5.1× high, `FixedMeanEnergy` 1.3–3.2×
(measurement/theory is 0.74 at 760 Torr). Eq. 4 is not gospel either — the
authors need a 2.1× scaling factor (1.74× with dust filtering) to reconcile it
with their own measurements.

### Wavelength scaling vs Eq. 4 — added 2026-07-25, the headline shape gate

Comparing two flat curves at one wavelength is a level comparison wearing a
shape's clothes. **Wavelength** is the axis where the kernel and Eq. 4 make a
non-trivial, identical prediction, and where nothing in the kernel is tunable:
`δ_eff·⟨ε⟩` sets the plateau's level and cannot produce a `λ` exponent.

Both terms of the kernel's threshold carry `1/h ∝ ω²`, since
`h = e²K_m/(m_e c ε₀ ω²)`:

```text
I_thr(p) = L′/h + U_i·(ν_diff + ν_att + G)/(h·p)      ⇒  I_thr ∝ ω² ∝ λ⁻²
```

with a pressure- and geometry-independent proportionality — while Eq. 4's
dominant term at these wavelengths is `2.2×10⁵·λ_µm⁻²`. Measured over
**0.53–10.6 µm** at 760 Torr:

| quantity | kernel | T&T Eq. 4 |
|---|---|---|
| `d(ln I_thr)/d(ln λ)` | **−2.000** | **−2.000** |
| ratio kernel/Eq. 4, `FixedMeanEnergy` | 1.6383 → 1.6384 (drift `2×10⁻⁵`) | — |
| ratio kernel/Eq. 4, `SelfConsistentClimb` | 4.2218 → 4.2219 | — |

The residual `1.5×10⁻⁴` in the exponent is the `(ν_m/ω)²` correction to the
Lorentzian at 10.6 µm, not a modelling difference. Holding the geometry fixed
across `λ` is physical rather than convenient: T&T's focus is
*divergence*-limited (`r₀ = f·α/2`, `l₀ = 0.414·(α/d)·f²`), so `Λ` and the focal
volume do not depend on wavelength — implemented as
`AirBreakdown::dry_air_tt2012_focus`.

Sharper still, the plateau `L′/h` and Eq. 4's `λ⁻²` coefficient are the **same
physical quantity** — `ω²` times the inelastic energy loss per collision:

```text
L′/h = δ_eff·⟨ε⟩ · m_e c ε₀ ω² / e² = 2.838×10¹⁵ W/m²   (δ_eff = 0.02, ⟨ε⟩ = 3 eV)
Eq. 4 λ⁻² term                      = 2.798×10¹⁵ W/m²
```

— agreeing to **1.01×** at the literature centre of a range chosen before this
comparison existed. Gated by
`tt2012_wavelength_scaling_matches_cascade_theory`.

**What it does and does not establish.** The `λ⁻²` is analytic in the kernel
(it is the `ν_m ≪ ω` limit of the IB Lorentzian), so the gate does not
independently *discover* the scaling. It establishes that the kernel shares
Eq. 4's wavelength structure **exactly rather than approximately**, and it fails
loudly if that limit is ever left — a `ν_m ≳ ω` regime, a wavelength-dependent
geometry, or an MPI/photon-count term leaking into the cascade path all break
the constant ratio. Set against a pressure axis where the kernel and the
measurement disagree, an exactly-shared exponent over 20× in `λ` is the most
defensible external statement M6a has.

**Not a pin, and the trade if it ever becomes one.** `δ_eff·⟨ε⟩` stays asserted
from its literature range. Inverting Eq. 4 for it — `δ_eff·⟨ε⟩ = 0.060 eV` —
would be legitimate re-pinning under the rule below, and would retire the "range
is asserted, not sourced" caveat; but it would make the level assertions in both
this gate and `tt2012_cascade_theory_reference` circular, so both level
assertions would have to be retired in the same change, leaving only the
exponent. That trade is available and not taken.

### Multiphoton ionization — implemented, calibrated, and left off

`S_mpi = N·W_ref·(I/I_ref)^K`, written about a reference intensity because the
bare `σ_K·I^K` form is unusable at `K = 14`: it overflows `f64` inside the
bisection bracket while `σ_K` itself underflows to ~`10⁻¹⁸⁶`.

`AirBreakdown::with_tt2012_mpi` calibrates it to the paper's own MPI estimate
(`I_B(MPI) = 4.42×10⁹ W/cm²` at 760 Torr, `S = 14` photons from `U_i = 15.6 eV`
for air), reading that number as its definition: at `I_B(MPI)`, MPI alone
reaches the breakdown criterion within one pulse.

**The result is why it stays off.** The threshold collapses to `5.5×10⁹ W/cm²`,
**37× below the 2.06×10¹¹ the same paper measures**, and contradicts its own
88 %-cascade accounting. Their MPI number is an order-of-magnitude significance
indicator — Nelson's flux-density criterion, whose constant `C` the paper never
states — not a rate anchor. A real `σ_K` from multiphoton cross-section data is
the open item; gated as `tt2012_mpi_calibration_undershoots_the_data`.

### Integrator unit tests (NOT physics validation — labeled as such)

Closed-form limits of the rate ODE, each exact:

1. **Pure cascade** (`S=0`, `ν_loss=0`, `ν_i>0`): `n_e(t) = n_e0·exp(ν_i t)`.
2. **Pure loss** (`ν_i=0`, `S=0`, `ν_loss>0`): `n_e(t) = n_e0·exp(−ν_loss t)`.
3. **MPI-only seeding** (`n_e0=0`, `β=0`, `S>0`): `n_e(t) = S·t`.
4. **Balance** (`β=0`, `S>0`, `n_e0>0`): `n_e(t) = n_e0 + S·t`.
5. **Slice consistency**: one exact step over `dt` equals `m` steps over `dt/m`.

### Fallback (independent, not circular — D5)

If the digitized T&T trend cannot be reproduced even as a slope, the fallback
anchor is an **independent** source (a second published dataset, or an
independent theoretical scaling **not** derived from the same coefficients the
kernel uses) — the old Raizer-analytic fallback was circular because the kernel
already uses Raizer coefficients. Documented, not hidden.

**The trigger fired, and the fallback is only half-discharged — recorded as an
open debt against M6a (2026-07-25).** The slope gate *was* the condition: it is
red and `#[ignore]`d, so by D5 an independent anchor is owed. What was put in
its place is T&T's **Eq. 4**, and that satisfies the clause in one half and not
the other:

- **The exponent half is clean and is what M6a should be judged on.** The
  `λ⁻²` agreement (−2.000 vs −2.000, ratio constant to `2×10⁻⁵` over a 20× span
  in `λ`, with a negative control that fails drift by 24×) is a shape statement
  on an axis where nothing in the kernel is tunable. It does not depend on the
  value of `δ_eff·⟨ε⟩` at all, so no choice of coefficient can manufacture it.
- **The level half does not meet the clause's own test.** Eq. 4 is not a second
  dataset — it is the same paper — and its `λ⁻²` coefficient implies
  `δ_eff·⟨ε⟩ = 0.060 eV`, which is *the same literature centre the kernel
  already uses*. The 1.01× agreement in `L′/h` is therefore a consistency check
  within one coefficient lineage, not independent corroboration. The section
  above is right that *pinning* `δ_eff·⟨ε⟩` from Eq. 4 would make the level
  gates circular; the point here is narrower and still stands without the pin —
  shared provenance already drains most of the evidential weight from the level
  agreement, which is why it is carried as a regression pin and never quoted as
  a validation.

**What closing this requires: a genuinely independent threshold dataset** — a
different group, different apparatus, ideally a different wavelength so the
`λ⁻²` prediction is tested against measurement rather than against another
theory. Until that exists, M6a's honest status is *one clean shape gate against
accepted theory, no external agreement with measurement*, and it should be
stated that way rather than as a closed rung. This debt does not block M6c,
which gates independently on Chapman–Jouguet velocity and consumes only the
threshold trigger — but it must not be allowed to go implicit as the ladder
climbs.

## NOT in scope (M6a)

- Any propagator coupling.
- Plasma back-reaction on the beam (absorption/refraction) — M6c.
- Recombination / afterglow / multi-pulse.
- Absolute-threshold agreement as a gate (only the slope/trend is gated).
