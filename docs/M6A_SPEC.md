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
ν_i(I, p) = (e²·I) / (m_e·c·ε₀·U_i) · ν_m / (ν_m² + ω²)
ν_m(p)    = K_m · p          (electron-neutral collisions ∝ neutral density)
```

The Lorentzian `ν_m/(ν_m²+ω²)` is the standard IB absorption factor. It has two
limits, and **which limit we are in sets the parameter-free gate below**:

- `ν_m ≪ ω` (low-collision / optical regime): `ν_i ∝ I·ν_m ∝ I·p`.
- `ν_m ≫ ω` (collision-limited): `ν_i ∝ I/ν_m ∝ I/p`.

At 1064 nm, `ω = 2πc/λ ≈ 1.77×10¹⁵ rad/s`, while `ν_m` at 1 atm is `~4×10¹² s⁻¹`
— so **`ν_m ≪ ω` across the entire 10–2000 Torr range**, and `ν_i ∝ I·p`
throughout. There is no IB (`ν_m = ω`) minimum in range; the low-pressure
threshold rise comes from diffusion loss, not from the Lorentzian.

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
I_thr(p) = [ν_att(p) + ν_diff(p) + G] / (A·p),      A ≡ ν_i/(I·p)
```

With attachment from measured rate coefficients it is negligible at these
densities (`6.7×10⁷` against `4.9×10⁹ s⁻¹` for diffusion at 1 atm), leaving two
terms whose exponents are **exact**:

```text
growth-limited (losses → 0):   I_thr = G/(A·p)       ∝ p^-1
diffusion-limited:             I_thr = ν_diff/(A·p)  ∝ p^-2      (ν_diff ∝ 1/p)
```

Any mixture must fall strictly between them, so the gate asserts
**`n ∈ [1, 2]`** — a bracket forced by the model's structure, not a fitted band.
`Λ` moves where inside the interval the model lands, never outside it, which is
what makes this independent of the quantity D5 forbade tuning. Observed:
`n = 1.737`, level at 760 Torr `2.4×10¹¹ W/cm²` (not gated).

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
3. **Threshold pressure-slope — FAILS, structurally.** Measured
   `E_B ∝ p^-0.164` over 300–2000 Torr, i.e. `I_thr ∝ p^-0.33`; the kernel
   gives `p^-1.74`. Committed **red** (`#[ignore]`d with its reason, so the
   suite stays green while the gap stays named).

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

so `n ∈ [1, 2]` is **forced by the model's structure**. No choice of `Λ`,
`D_e`, `K_a` or `n_bd` can reach the measured `n = 0.33` — it is outside the
reachable interval, so this cannot be repaired by re-pinning any constant. It
would take the cascade coefficient itself to fall with pressure, `A ∝ p^-0.67`,
which is what an effective ionization energy `U_i` growing with collision
frequency would produce (inelastic losses per ionization scale ∝ p, and the
literature's "effective" `U_i` is known to exceed the true ionization potential).

That is new physics, not a constant to re-pin, and it is the open question M6a
hands forward. The absolute level is *not* the problem, and the attachment
correction in fact **improved** it: measured `1.0×10¹¹ W/cm²` (RMS) vs modelled
`2.4×10¹¹` is 2.4×, down from 6×, and well inside the inter-lab scatter this
spec declines to gate. Level and slope moved in opposite directions, which is
itself evidence the remaining gap is in the pressure *scaling* of the cascade
rather than in its magnitude.

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
