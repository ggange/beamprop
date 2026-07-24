# M6a pre-spec — 0-D optical-breakdown threshold kernel

Written **before** the M6a code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md)): pin the rate model, the constants, the integrator,
and — most importantly — *which* checks actually validate physics versus which
are integrator unit tests. If any of this proves wrong during implementation,
amend this document first, then the code.

Scope: the standalone 0-D rung of the M6 laser-induced gas-breakdown ladder
(design doc `giuseppe-main-design-20260723`, Eng Review Outcome 2026-07-23).
No propagator changes; the kernel is a **driver-callable pure function** so the
later M6a.2 (Monte-Carlo ignition) and M6c (LSD trigger) rungs call the same
core.

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
  between points (that is M6a.2's job, which calls this kernel per realization).
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

| model | free constants | slope `n` |
|---|---|---|
| no inelastic loss | — | 1.737 |
| `FixedMeanEnergy` | `δ_eff`, `⟨ε⟩` | 0.800 |
| **`SelfConsistentClimb`** | `δ_eff` | **0.356** |
| *measured (T&T)* | | *0.329* |

**Honesty about its status.** `δ_eff` remains free within its literature range
(0.01–0.05), which gives `n ∈ [0.154, 0.583]` — containing the measurement — so
the external gate is an **envelope** test plus a factor-1.5 regression pin on
the literature-central value, not a point comparison. `δ_eff = 0.02` was fixed
before the self-consistent model existed, so nothing was tuned; but see the
sensitivity audit below before reading the central-value agreement as sharp.
Widening any range to manufacture containment is forbidden by the same rule
that pins `Λ`.

**Sensitivity audit (2026-07-24)** — run after the gate first went green,
because a passing gate earns *more* scrutiny, not less. Verified first that an
independent Python reimplementation (from these equations, not the Rust)
reproduces `n = 0.3563` to four digits, and that the value is stable under
discretization (0.346–0.356) and T&T's ±1 ns pulse uncertainty (0.352–0.362).
Then each ungated constant was perturbed:

| perturbation | n |
|---|---|
| baseline (literature-central) | 0.356 |
| `n_bd` ×0.1 / ×10 | 0.353 / 0.360 |
| pulse FWHM 5 / 7 ns | 0.362 / 0.352 |
| `D_e` ×0.5 / ×2 | **0.213 / 0.569** |
| generation prefactor `ln 2` | **0.469** |

Two findings demote the headline. **(a)** The slope is *not* `Λ/D_e`-robust in
the current balance: `ν_diff = 4.9×10⁹` vs `G = 3.7×10⁹ s⁻¹` at 760 Torr —
comparable, not sub-dominant — so the order-of-magnitude `D_e` sweeps the slope
across most of the envelope. Any older text claiming the high-p branch is
"Λ-independent" described the pre-plateau model and is withdrawn. **(b)** The
derivation's `ν_i = 1/t_climb` carries an order-unity generation-model
ambiguity (`ln 2/t_climb` under discrete doubling is equally defensible), worth
30% on the slope by itself.

Conclusion: the model family's uncertainty band from its own ungated constants
is roughly `n ∈ [0.21, 0.57]`. The measured 0.329 sits comfortably inside, and
that containment — not the 8% central agreement, which is partly luck — is the
claim the gate makes.

Two things are deliberately *not* claimed. The absolute level is ~7× above T&T
— gated only as a *flat* offset inside the inter-lab scatter, which says the
shape is right and the normalization is not. And the self-consistent model is
not asserted to be more correct in general: putting every electron on the mean
trajectory makes its threshold artificially sharp at `ε_∞ = U_i`, and a real
energy distribution would soften it. The measurement sitting *between* the two
limits is the more trustworthy statement, and it is gated as such.

### Diffusion loss `ν_diff` — Λ is pinned, never fit (D5)

```text
ν_diff(p) = D_e(p) / Λ²,   D_e(p) = D_e,ref · (p_ref / p)   (free-electron, D_e ∝ 1/p)
```

`Λ` is the fundamental diffusion length of the **focal geometry**, computed from
the spot, not tuned to the data. For the T&T 20 µm-radius focus modelled as a
sphere of radius `r`, the fundamental mode gives `Λ = r/π`. This value is
**documented and asserted as an input**; tuning `Λ` to move the threshold
minimum onto the measured curve is curve-fitting and is explicitly forbidden.

### Attachment loss `ν_att` — two channels, from measured rate coefficients

```text
ν_att(p) = k₂·n_O₂ + k₃·n_O₂·n,     n = p/(k_B T),  n_O₂ = 0.21·n
```

with `k₂ = 1e-17 m³/s` (dissociative, `e + O₂ → O⁻ + O`) and
`k₃ = 1e-43 m⁶/s` (three-body, `e + O₂ + M → O₂⁻ + M`; Kossyi et al. 1992,
Itikawa 2009). Three-body is the dominant channel at atmospheric density, and
`∝ p²`.

*(Amended 2026-07-24. This replaced a single order-of-magnitude constant
`ν_att = K_a·p` with `K_a = 1e5 s⁻¹Pa⁻¹`. That value was ~150× too large:
it gave `1.0×10¹⁰ s⁻¹` at 1 atm where the measured coefficients give
`6.7×10⁷ s⁻¹`. The consequence matters — see the external gates below — because
the inflated constant was the only term flattening the modelled threshold slope
toward the measured one. With attachment correct it is **negligible against
diffusion** (`4.9×10⁹ s⁻¹` at 1 atm), and the modelled slope moves from −0.72
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
| k₃ | 1.0e-43 m⁶/s | three-body attachment `e + O₂ + M → O₂⁻ + M` (Kossyi 1992); dominant channel |
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
the closed form said 0.521 and the code gives 0.800; for `SelfConsistentClimb`
it said 0.060 and the code gives 0.356. **Trust the code.**

Attachment is negligible at these densities (`6.7×10⁷` against `4.9×10⁹ s⁻¹`
for diffusion at 1 atm), leaving three terms whose exponents are **exact**:

```text
plateau (inelastic-limited):   I_thr → L′/h                     ∝ p^0
growth-limited:                I_thr = U_i·G/(h·p)              ∝ p^-1
diffusion-limited:             I_thr = U_i·ν_diff/(h·p)         ∝ p^-2
```

Any mixture falls between them, and the gate asserts **`n ∈ (0, 1)`** — the
plateau must be doing work, since without `L′` the model cannot get below
`n = 1` at all. Observed with the default `SelfConsistentClimb`: **`n = 0.356`**,
level at 760 Torr `1.32×10¹² W/cm²` (not gated). With `FixedMeanEnergy`:
`n = 0.800`. Both literature envelopes are pinned separately —
`[0.154, 0.583]` and `[0.413, 1.139]` respectively.

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

The T&T figure plots **two** curves against pressure: the breakdown threshold
field `E_B` and the effective field `E_eff`. Both are digitized into
`tests/data/tt2012_*.csv`. They yield four gates — collision frequency,
`E_eff` sign, level-ratio flatness, and slope-envelope containment — and all
four now pass, none `#[ignore]`d. The history below records what each stage
passed and failed, because the failures drove the model corrections.

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
3. **Threshold pressure-slope — now PASSES as envelope containment.**
   Measured `E_B ∝ p^-0.164` over 300–2000 Torr, i.e. `I_thr ∝ p^-0.329`; the
   kernel gives `p^-0.356` at literature-central constants. The central-value
   agreement is not claimed as sharp — see the sensitivity audit above — but
   the route here took two model corrections and no tuning:

   | stage | slope `n` | vs measured |
   |---|---|---|
   | original (no inelastic loss) | 1.737 | 5.3× — and unreachable at *any* parameter value, since `n = 1` was the model's floor |
   | + inelastic loss at fixed `⟨ε⟩` | 0.800 | 2.4× |
   | + `⟨ε⟩` eliminated (self-consistent climb) | **0.356** | **1.08×** |

   Gated as an envelope over `δ_eff`'s literature range (`n ∈ [0.154, 0.583]`,
   containing the measurement) plus a factor-1.5 band on the central value.

**What this does and does not settle.** The *shape* of `I_thr(p)` is now
reproduced; the *level* is not, sitting ~7× above T&T. But that offset is
**flat** (6.4–7.4× across the window, spread 1.16×), which is the signature of
a correct shape with an uncertain normalization — `U_i`, `n_bd`, `Λ` and
`δ_eff` all scale it. Earlier models had *drifting* ratios (3.10× → 0.33×
without the loss term, 4.69× → 2.41× with fixed `⟨ε⟩`), and that drift was the
slope error showing up in absolute clothing. Level remains ungated.

**The line on fixing it.** Re-pinning a constant from independent data is
legitimate; so is adding a term the physics demands. Tuning any constant until
the slope lands in a band is the curve-fitting the pinned-`Λ` rule forbids, and
the bracket above now makes such tuning provably futile as well as improper.

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

## NOT in scope (M6a)

- Any propagator coupling (M6a.2/M6c).
- Plasma back-reaction on the beam (absorption/refraction) — M6c.
- Recombination / afterglow / multi-pulse.
- Absolute-threshold agreement as a gate (only the slope/trend is gated).
