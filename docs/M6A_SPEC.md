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
limits, and **which limit we are in sets the parameter-free gate below**:

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

**Honesty about its status.** This introduces one new constant, and the
literature ranges (`δ_eff ≈ 0.01–0.05` for air above ~1 eV, `⟨ε⟩ ≈ 2–5 eV`)
span ~12× in `L′`. So the model's prediction is an **envelope, not a point**:
across that range **the code** gives `n ∈ [0.413, 1.139]`, against a measured
`n = 0.329`. (A closed-form estimate suggests a wider `[0.19, 0.89]`; the code
is steeper because the Gaussian pulse raises the threshold above the analytic
full-FWHM value, pushing it further from the plateau. Trust the code.)

So the measurement sits **just below** the envelope's lower edge — 1.25× from
the most favourable literature value, against 5.3× before this term existed.
The gate is an envelope test and it is **still red**: the envelope is not
widened to swallow the measurement. That is weaker than a parameter-free
prediction and is labelled as such. **`L′` is set from the centre of the literature range and is never
adjusted to improve agreement**; the pinned-`Λ` rule applies to it verbatim.

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

### Physics gate (the real one, parameter-free)

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
threshold is higher (`6.4×10¹¹` vs `≈4.1×10¹¹ W/cm²` at 760 Torr). The offset
is pressure-independent, so the exponents below are unaffected — but never
quote the closed form as a level, and note that being further above the plateau
makes the code's slope *steeper* than the closed form predicts (0.800 vs 0.521).

Attachment is negligible at these densities (`6.7×10⁷` against `4.9×10⁹ s⁻¹`
for diffusion at 1 atm), leaving three terms whose exponents are **exact**:

```text
plateau (inelastic-limited):   I_thr → L′/h                     ∝ p^0
growth-limited:                I_thr = U_i·G/(h·p)              ∝ p^-1
diffusion-limited:             I_thr = U_i·ν_diff/(h·p)         ∝ p^-2
```

Any mixture falls between them, and the gate asserts **`n ∈ (0, 1)`** — the
plateau must be doing work, since without `L′` the model cannot get below
`n = 1` at all. Observed: `n = 0.800`, level at 760 Torr `6.4×10¹¹ W/cm²`
(not gated). The literature envelope of `L′` is pinned separately at
`n ∈ [0.413, 1.139]`.

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

**When the digitized Thiyagarajan & Thompson (J. Appl. Phys. 111, 073302
(2012)) curve is available**, the gate additionally asserts the model matches
the measured `I_thr ∝ p^-n` trend within a factor-2 band. Absolute agreement is
NOT gated (inter-lab scatter). That external comparison is **pending the CSV**
(design-doc T2, the user's homework — needs the paper figure); until then the
slope check above is a *model-consistency* check, explicitly labeled as
not-yet-external-validation.

### External gates — Thiyagarajan & Thompson 2012 (digitized 2026-07-24)

The T&T figure plots **two** curves against pressure: the breakdown threshold
field `E_B` and the effective field `E_eff`. Both are digitized into
`tests/data/tt2012_*.csv`. Together they give three checks, and the kernel
passes two.

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
3. **Threshold pressure-slope — still FAILS, but narrowed 4×.** Measured
   `E_B ∝ p^-0.164` over 300–2000 Torr, i.e. `I_thr ∝ p^-0.33`. The kernel gave
   `p^-1.74` before the inelastic-loss term and gives `p^-0.80` after it; the
   literature envelope of the lumped loss constant spans `n ∈ [0.413, 1.139]`,
   so the measurement now sits **1.25× below the envelope's edge** instead of
   5.3× away. Still committed **red** (`#[ignore]`d with its reason). The
   envelope is *not* widened to cover the measurement.

**A wrong diagnosis, recorded because the correction is the finding.** The gap
first looked like a factor of 2 caused by two-body vs three-body attachment.
Implementing attachment from measured rate coefficients showed the opposite:
real attachment is ~150× *smaller* than the constant it replaced, hence
negligible, and the corrected model moved from `p^-0.72` to `p^-1.74` — further
from the data. The near-agreement had been propped up by a wrong number.

**What is left is structural, and that is the interesting part.** With
attachment negligible the model is bracketed by its own closed forms:

```text
growth-limited (losses → 0):   I_thr = G/(A·p)       ∝ p^-1
diffusion-limited:             I_thr = ν_diff/(A·p)  ∝ p^-2
```

so `n ∈ [1, 2]` while attachment is negligible. The sharper statement is that
`n = 1` — the growth-limited `I_thr = G/(A·p)` — is the **flattest the model
can be** without a loss that grows faster than `p`: diffusion only steepens it.
The measured `n = 0.33` lies below that floor. Only a super-linear loss can
flatten past it, and three-body attachment at its *measured* coefficient does
not reach the window (that regime starts near 10⁴ Torr); forcing it would take
`k₃` ≈ 100× the measured value, which is fabrication rather than re-pinning. It
would take the cascade coefficient itself to fall with pressure, `A ∝ p^-0.67`,
which is what an effective ionization energy `U_i` growing with collision
frequency would produce (inelastic losses per ionization scale ∝ p, and the
literature's "effective" `U_i` is known to exceed the true ionization potential).

That term is now implemented (see *Inelastic energy loss* above) and closes
most of the gap; the residue is the open question M6a hands forward.

On the absolute level: converting `E_B` to intensity requires the RMS form
`I = ε₀·c·E_rms²` (**not** `½ε₀cE²`, which applies to a peak amplitude — the
`E_eff` ratio established `E_B` is RMS), giving `2.06×10¹¹ W/cm²` measured at
760 Torr against `6.4×10¹¹` modelled. Across the range the model now sits
uniformly *above* the data by a bounded factor:

| p (Torr) | 380 | 569 | 759 | 950 | 1420 | 1896 |
|---|---|---|---|---|---|---|
| model/measured, with `L` | 4.69 | 3.90 | 3.09 | 3.15 | 2.74 | 2.41 |
| model/measured, before `L` | 3.10 | 1.95 | 1.19 | 0.96 | 0.53 | 0.33 |

The earlier model *crossed* the data (a 9× swing that was the slope error in
absolute clothing); the current one is a roughly constant 2.4–4.7× offset,
inside the inter-lab scatter this spec declines to gate. The remaining downward
drift is the residual slope gap. The level is still not gated, and the slope
remains the observable that matters.

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
