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

### Attachment loss `ν_att`

Two-body electron attachment to O₂, `ν_att = K_a · p` (∝ neutral density),
`K_a` from air-discharge data. Sub-dominant to cascade near threshold; included
for completeness and honesty about the loss balance.

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
| K_a | 1.0e5 s⁻¹·Pa⁻¹ | two-body O₂ attachment, air-discharge order |
| n_bd | 1.0e23 m⁻³ | breakdown criterion density |
| p_ref | 101325 Pa | 1 atm reference |

The absolute threshold *level* depends on `U_i, D_e,ref, K_a, n_bd, Λ` and is
**not gated** (inter-lab scatter is 3–10×). The *slope* depends only on the
`ν_i ∝ p` scaling and is the gate.

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

*(Amended 2026-07-23, during implementation, per the rule at the top of this
document. The original text gated the slope over the full 10–2000 Torr sweep
against `I_thr ∝ p^-1`. Both halves were wrong and the code now differs, so the
spec is corrected here rather than in the code.)*

Why the range is part of the gate, not a free knob. Solving the avalanche
criterion — cascade must outrun losses *and* clear the `n_seed → n_bd` growth
requirement `G = ln(n_bd/n_seed)/τ` within the pulse — gives

```text
I_thr(p) = K_a/A + (ν_diff(p) + G)/(A·p),      A ≡ ν_i/(I·p)
```

so the exponent `n` in `I_thr ∝ p^-n` is **not** a single number: `n → 1` where
the `1/p` terms dominate, and `n → 0` as `p → ∞` and `I_thr → K_a/A`. Fitting
over 10–2000 Torr therefore mixes branches and returns ≈ −2.3 (the low-p end is
the Λ-sensitive diffusion branch, which D5 excluded); fitting far above 2000
Torr would return something flatter than the band. The gate range is pinned to
**300–2000 Torr**, which brackets the T&T measurement range, and the gate is a
statement about the model *on that range*.

What is parameter-free about it: on this branch cascade and attachment both
scale ∝ p and diffusion is sub-dominant, so the fitted `n` does not depend on
`Λ` — the quantity D5 forbade tuning. Measured: `n = 0.716`, asserted band
`n ∈ [0.4, 1.0]`, level at 760 Torr `6.3×10¹¹ W/cm²` (not gated).

**When the digitized Thiyagarajan & Thompson (J. Appl. Phys. 111, 073302
(2012)) curve is available**, the gate additionally asserts the model matches
the measured `I_thr ∝ p^-n` trend within a factor-2 band. Absolute agreement is
NOT gated (inter-lab scatter). That external comparison is **pending the CSV**
(design-doc T2, the user's homework — needs the paper figure); until then the
slope check above is a *model-consistency* check, explicitly labeled as
not-yet-external-validation.

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
