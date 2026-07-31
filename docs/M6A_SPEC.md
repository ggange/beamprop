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
| **`SelfConsistentClimb`** (default until 2026-07-30) | `δ_eff` | **0.095** | 3.5× flatter |

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
| `D_e` ×0.5 / ×2 | **large** — `ν_diff` and `G` are comparable, not sub-dominant (quantified 2026-07-30, below) |
| generation prefactor `ln 2` vs 1 | **~30 %** |

Both findings stand. The `D_e` sensitivity in particular withdraws any claim
that the high-pressure branch is "Λ-independent" — that described the
pre-plateau model. What the 2026-07-24 audit *missed* was the window artifact,
which a numerical-hygiene check (vary the integration bounds, demand
invariance) would have caught immediately; that check now exists as
`threshold_is_window_independent`.

**`D_e` gated — 2026-07-30.** The audit's "large" was a word for two years of
this milestone's life, and a word cannot fail in CI. Two gates now replace it.

*`d_e_ref_implies_a_stated_electron_energy` (verification).* `D_e` is not
independent of `K_m`, which **is** externally gated. Kinetic theory gives
`D_e = v²/(3ν_m) = 2ε/(3·m_e·K_m·p)`, so naming an energy fixes `D_e` and naming
`D_e` fixes an energy. Read that way the shipped constant says

```text
D_e,ref = 0.2 m²/s   ⟺   ε = 6.740 eV   at p_ref
```

which is a defensible energy for an electron mid-climb — above the 2–5 eV swarm
band, below the `U_i` = 12.06 eV it is climbing to. `D_e` is no longer free; it
is a stated assumption about the diffusing population.

*The inconsistency this exposed, pinned not fixed.* `FixedMeanEnergy` evaluates
its inelastic loss at `⟨ε⟩ = 3 eV`. Diffusion and inelastic loss therefore assign
the same electrons **two energies differing by 2.25×**, in two terms of one
balance. The gate asserts that ratio so it cannot drift. It is deliberately not
repaired here: matching `D_e` to `⟨ε⟩` would take it to 0.089 m²/s and move every
published M6a number, which needs its own argument. (The default
`SelfConsistentClimb` has no `⟨ε⟩` and runs to `ε_∞` ≈ 12–15 eV at threshold, so
for the default variant 6.74 eV is the less strained of the two.)

*`d_e_sensitivity_is_pinned_across_the_kinetic_band` (the number).* Sweeping
`D_e` over the full band that formula admits — `ε` from 2 eV to `U_i`, a 6.0×
range — and refitting the 300–2000 Torr slope at each end:

| `ε` (eV) | `D_e,ref` (m²/s) | fitted `n` |
|---|---|---|
| 2.00 | 0.0593 | 0.0532 |
| 3.00 | 0.0890 | 0.0608 |
| 5.00 | 0.1484 | 0.0779 |
| **6.74 (shipped)** | **0.2000** | **0.0951** |
| 10.00 | 0.2967 | 0.1307 |
| 12.06 (`U_i`) | 0.3578 | 0.1545 |

**The conclusion is negative, and it is the useful part.** A 6.0× range in the
milestone's largest ungated constant moves the slope by 0.101, against a
shortfall of 0.234 to T&T's measured 0.329. Even the most diffusion-heavy
defensible choice leaves the model less than halfway there. `D_e` **cannot**
explain the slope gap, and the gate asserts that it cannot — closing off a
re-tuning route that nothing in the repo previously ruled out.

**The external-anchor debt, stated precisely.** No swarm-data gate is landed, and
the reason is stronger than "the table was not obtained". Swarm measurements
(Dutton 1975; Huxley & Crompton) sit at characteristic energies of 0.1–2 eV; the
cascade electron sits at 6.7 eV. Getting from one to the other runs back through
*this same kinetic formula*, whose only external input is `K_m` — already
validated. A swarm gate would re-validate `K_m` while appearing to validate
`D_e`. So the honest status is **verified as consistent, not validated as
correct**, and closing it needs a diffusion measurement at the cascade's own
energy. That is the D5 discipline applied to a second constant.

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

### Superseded 2026-07-31 — the seed is now produced, not assumed

Everything above describes the kernel as it stood until seed production landed.
It is kept because the reasoning was right and the fix it argued for is the one
that eventually happened; what follows is what replaced it.

**The stand-in was worse than this section admitted.** It compared `1/V_focal`
against a background of ~10⁹ m⁻³ and concluded the focus holds ~8×10⁻⁵
electrons. That figure is the **ion** density. Air is electronegative — the
kernel's own validated attachment gives `ν_att` = 6.7×10⁷ s⁻¹ at 1 atm, so a
free electron survives ~15 ns — and the free-*electron* background is
`q/ν_att` = **0.149 m⁻³**, i.e. **1.2×10⁻¹⁴** electrons in the focus. The
assumption was off by ~14 orders, not ~4, and this document was the source of
the smaller number.

**What landed.** `n_e0(p) = q(p)/ν_att(p)` with `q` ≈ 10 ion pairs cm⁻³ s⁻¹
(AFRL *Handbook of Geophysics*, ch. 20), PPT photoionization on by default, and
the floor removed. Three edits that are one change: any one alone is wrong. The
formulation and results are in `docs/MODELS.md` § "Seed production". Four things
belong here as milestone record.

**1. The new constant is not load-bearing, and that was the acceptance test.**
Introducing `q` would be a poor trade if the answer depended on it. It does not:
the threshold is *bit-identical* across twelve decades of seed density, and
takes ~10 orders to move 0.04 %. The retired `1/V_focal` sat where the seed
*does* matter (2.6 % at 10¹² m⁻³) — the old constant was load-bearing and wrong,
the new one is neither. This gate was written before the change was committed,
with the stated intent that a failure would stop it landing.

**2. It repaired the low-pressure branch.** 10–100 Torr goes 1.292 → **0.501**
against a measured 0.428: from 3.0× too steep to 1.17×. That branch was M6a's
worst residual for the milestone's entire life, and *both* source papers said
multiphoton ionization dominates there. The largest single improvement M6a has
had, from deleting an assumption rather than adding a term.

**3. The floor had to be kept for an explicit seed.** Removing it outright would
have moved every isolation gate's baseline, because those gates run without a
multiphoton source and their seed would then decay through the quiet arm — the
exact window-dependence this section was written about. The rule that resolves
it: an **explicit** seed is a modelling assumption and acts as a floor; the
**derived** background is a physical initial condition and is free to deplete.
That keeps `seeding_suppressed` gates numerically identical to what they
published, so the closure and loss-term comparisons are unaffected by this
change, which is what makes it reviewable.

**4. A gate caught a real bug in the change.** With PPT made default,
`with_tt2012_mpi` silently became a no-op, because `mpi_source` tests the PPT
path first and that builder never zeroed it. `tt2012_mpi_calibration_undershoots_
the_data` failed and said so. The builder now zeroes the other paths, as the
other two already did.

**What it costs.** The 300–786 Torr window slips 0.431 → 0.386 against 0.468,
and a mid-pressure bump survives at 100–300 Torr (0.857 vs 0.413) — the seeding
transition, where the kernel crosses from multiphoton-supplied to
cascade-supplied electrons too abruptly. That is now M6a's sharpest open
question, and it is a better one than the branch it replaced.

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
`n = 1` at all. Observed with `SelfConsistentClimb` (the default until
2026-07-30): **`n = 0.095`**,
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

### D5 discharged — 2026-07-30, in the negative

The dataset above now exists in the suite: **Chylek, Jarzembski, Srivastava &
Pinnick, Appl. Opt. 29, 2303 (1990)**, clean air at **532 nm**, digitized from
their Fig. 3 into `tests/data/chylek1990_air_threshold_vs_pressure.csv`.

It meets every condition the clause set, including the one that looked hardest:

| condition | status |
|---|---|
| different group | Oklahoma / UAH / USRA / Army ASL White Sands |
| different apparatus | own pressure chamber, 10 cm lens, 1–800 Torr |
| different wavelength | **532 nm**, exactly half T&T's 1064 nm |
| not Raizer-lineage | `α` is an empirical fit to their own data; their Sec. II uses only the bracketing `α = 1` (cascade) and `α = 1/k` (MPI) limits. No `K_m`, `δ_eff` or `⟨ε⟩` shared with the kernel |

The confound that usually makes cross-paper threshold comparisons worthless is
absent here by luck: Chylek's pulse is **6.5 ns** into a **16.5 µm** focal radius
against T&T's **6 ± 1 ns** into **20 µm** — it is the second harmonic of the same
laser family. Chylek's own Sec. II names pulse duration and focal spot as the
reason published `α` values contradict one another; with both matched, the
532/1064 comparison is a measurement of the wavelength scaling.

**Digitization provenance.** Traced programmatically, not by hand, by
`scripts/digitize_chylek1990.py`: axes calibrated by least squares on the
log-decade minor ticks (24 in `x`, residual 0.51 % of a decade; 17 in `y`,
0.38 %), markers located by normalised cross-correlation against a rhombus
template. Two checks, neither used to fit anything — the BULK WATER rule was held
out of the `y` calibration and lands at 4.61×10¹⁰ against the 4.7×10¹⁰ the paper
states (2.0 %), and the 49 recovered points give `α = 0.4450 ± 0.0049` against
the `0.45 ± 0.01` the paper's own caption prints (1.0 %). The first is gated as
`chylek1990_digitization_reproduces_the_published_slope`, so a bad re-trace
cannot land silently. Unlike the T&T curves, this trace is reproducible from the
PDF.

**The finding: it falsifies `λ⁻²` against measurement.**

```text
cascade / kernel:  I_th(532)/I_th(1064) = 3.99   (= λ⁻²; shorter λ costs more)
measured:          I_th(532)/I_th(1064) ≈ 0.80   (532 nm breaks down EASIER)
```

Wrong by ~5× and wrong in **sign**: cascade theory demands the 532 nm threshold
sit 4× above the 1064 nm one; the two experiments put it ~20 % below. At 532 nm
the multiphoton order falls from `K = 11` to `K = 6`, so the MPI channel the
kernel leaves OFF is strongest exactly where the measurement drops. Gated as
`chylek1990_tt2012_wavelength_ratio_falsifies_cascade_lambda_squared`.

**Second finding: "too flat" was window-specific.** Extending the pressure
comparison two decades below T&T's range shows the kernel is not a power law at
all — local exponents 1.951 (10–100 Torr), 1.047 (100–300), 0.170 (300–786),
against a measurement holding 0.41–0.47 throughout. The kernel is 4.6× too steep
at the bottom, 2.8× too flat at the top, and crosses the data near 250 Torr. The
`n = 0.095` vs `0.329` framing elsewhere in this document is fitted over
300–2000 Torr and is correct only there; the defect is curvature. Gated as
`chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not`.

### Keldysh photoionization — implemented, verified, and it does not fix the gap

Added 2026-07-30 as the first attempt on the open MPI question, because the
suspect was obvious: at 532 nm the photon order falls from `U_i/ħω` = 10.35 to
5.17, so multiphoton ionization is far stronger exactly where the measured
threshold drops.

**What is verified** (`breakdown0d::keldysh_rate`, `keldysh_tunnel_exponent`).
The rate is `W = A·ω·exp[−(2U_i/ħω)·f(γ)]` with
`f(γ) = (1 + 1/2γ²)·asinh γ − √(1+γ²)/2γ` and `γ = ω√(2mU_i)/(eE₀)`. One
expression carries both limits, and both are gated:

| limit | closed form | agreement |
|---|---|---|
| `γ ≫ 1` multiphoton | `W ∝ I^(U_i/ħω)` | exponent to 0.2 % at both λ |
| `γ ≪ 1` tunnelling | `S → 4√(2m)U_i^{3/2}/(3ħeE)` | 1 part in 10⁶ |
| series join at γ = 0.1 | `f = ⅔γ − γ³/15` | 3×10⁻⁶ |

`γ` is 16.6 at 1064 nm and 37.7 at 532 nm at the measured thresholds, so air ns
breakdown sits deep in the multiphoton branch. **The exponent has no free
constant**, which is the whole reason this is a test and not a fit; `E₀` is the
field *amplitude*, not the RMS value the `E_B` gates use, and mixing them would
be a flat `√2` in `γ`.

**What it delivers: a negative result.** Only the prefactor is soft (Keldysh's
atomic prefactor is an order-unity function whose form varies between
re-derivations), so it is exposed explicitly and swept:

| prefactor × ω | `I_th(532)/I_th(1064)` |
|---|---|
| 0 (cascade only) | 3.99 |
| **1 (order unity)** | **3.87** |
| 10³ | 1.84 |
| 10⁶ | 0.48 |
| **measured** | **0.80** |

An order-unity prefactor closes **3 %** of the gap. Closing it needs `~10⁵·ω` — an
ionization rate faster than the optical frequency, which is not a rate. Gated as
`keldysh_mpi_does_not_close_the_wavelength_gap`.

**A withdrawn claim of mine, recorded because it shaped the design.** I first
argued the wavelength ratio would be prefactor-*insensitive*, since a threshold
set by `W·τ ~ 1` with `W ∝ I^K` suppresses prefactor error as `x^(1/K)`. That is
wrong for a *ratio* between two different `K`: once MPI dominates at both
wavelengths the ratio scales as `x^(1/10.35 − 1/5.175) = x^(−0.097)`, so three
decades still move it 1.9×, and across the transition it runs 3.99 → 0.48.
Measuring it is what showed this. Any future prefactor must be justified to ~2
orders of magnitude; it cannot be waved through.

**The seed density is unphysical — a separate, latent defect.** `n_e0 = 1/V_focal`
= `1.2×10¹³ m⁻³` is ~10⁴ above the cosmic-ray background (`10⁹–10¹⁰ m⁻³`), which
puts ~`10⁻⁴` free electrons in the `8.3×10⁻¹⁴ m³` focus: it essentially never
holds one. Seed *production* should therefore be MPI's job, and that step carries
enormous wavelength leverage — at prefactor 1, MPI makes 295 electrons per pulse
in the focus at 532 nm against `2×10⁻⁸` at 1064 nm, ten orders of magnitude.
Measured, removing the seed moves the ratio only 3.99 → 3.85, because the
threshold is already 5.7–28× above measurement and MPI is abundant there at both
wavelengths. So the defect is masked by the level error and cannot be the
explanation — but it would matter the moment the level is fixed, and it is
therefore recorded rather than left implicit. Exposed as
`AirBreakdown::with_seed_density`; both the Keldysh source and the old
T&T-calibrated `σ_K` remain **off by default**, so every pre-existing gate keeps
its published numbers.

**Where this leaves the open question.** Narrower than "add MPI". Either a
PPT-corrected rate for molecular O₂ (Coulomb corrections can lift the prefactor
by orders of magnitude, and that is checkable against published `σ_K` rather than
against the threshold data), or a systematic in the two-paper comparison. It is
*not* a missing channel at these intensities, and the curvature defect
(`chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not`) is untouched by
any of this — that one points at the mean-energy closure, not at MPI.

**Consequence.** The `λ⁻²` gate stays — it is a true statement about the kernel's
internal structure and fails loudly if the IB Lorentzian limit is left. It may no
longer be described as external agreement with measurement, and it is no longer
M6a's headline. M6a's status, stated plainly: **verified against cascade theory,
falsified against air on both the pressure and the wavelength axis.** The debt is
closed in the sense D5 cares about — an independent anchor exists and has been
confronted — not in the sense of the model having been vindicated.

### General-gas kernel — 2026-07-30, and the one knob-free prediction

The kernel's gas-dependent constants (`U_i`, `K_m`, `D_e`, `δ_eff`, `⟨ε⟩`, the
attachment chemistry) are now a `Gas` value; the laser and the focal geometry
stay on `AirBreakdown`. `Gas::dry_air()` is a pure re-packaging — the `breakdown`
case's output is bit-identical across the change and no gate number moved. The
type name `AirBreakdown` is kept deliberately: every gate and every published
number refers to it, and renaming would churn the record for no physics.

**What this unlocks, and why it is sharper than air.** Three digitized Chylek
Fig. 2 curves (He / Ar / Xe) had been sitting in `tests/data/` with only their own
integrity gate, because nothing could consume them. They are the only data here
that tests the cascade with `δ_eff` **not free**. A monatomic gas has no
vibrational and no low-lying electronic modes, so below its first excitation
threshold the loss per collision is elastic recoil, `δ = 2m_e/M` — the atomic
mass, to five figures, with nothing to choose. In air `δ_eff` is a
literature-range constant that sets the plateau level and is one of M6a's
standing weaknesses. Here it is arithmetic.

**The prediction has no parameters at all.** Writing out the equilibrium energy,
`ε_∞ = (e²I/(m_e c ε₀))·ν_m/((ν_m²+ω²)·δ_eff·ν_m)`, the collision frequency
**cancels exactly** in the optical regime `ν_m ≪ ω` — heating and loss both go as
`ν_m`. Ionization requires `ε_∞ > U_i`, so the cascade has a hard floor

```text
I_plateau = δ_eff · U_i · m_e·c·ε₀·ω² / e²
```

containing no transport constant whatsoever. `D_e` and `K_m` — the kernel's two
least defensible numbers, and for the noble gases wholly unsourced — drop out.
`cascade_plateau_floor_is_independent_of_the_transport_constants` proves this by
perturbing `D_e` 100× either way and demanding the floor not move, and by
checking the integrated `cascade_rate` is identically zero just below the floor
at every pressure. So the comparison below inherits none of that uncertainty.

**The comparison** (`chylek1990_noble_gas_plateau_floors_are_unequally_tight`),
at Chylek's 532 nm:

| gas | `K` | `δ = 2m_e/M` | `U_i` (eV) | floor (W/cm²) | measured `I_th` | headroom |
|---|---|---|---|---|---|---|
| He | 11 | 2.741e-4 | 24.587 | 1.275e11 | 2.36e11 @ 672 Torr | **1.85×** |
| Ar | 7 | 2.747e-5 | 15.760 | 8.190e9 | 6.37e10 @ 838 Torr | 7.8× |
| Xe | 6 | 8.357e-6 | 12.130 | 1.918e9 | 2.53e10 @ 725 Torr | 13.2× |

*What passes.* No curve falls below its floor, so the elastic-`δ` cascade is not
outright falsified for any of the three, and the **ordering** He > Ar > Xe is
right, as `δ·U_i` demands.

*What fails, and is pinned.* The **spacing** is wrong and tracks the mass ratio.
Predicted floor ratios are He/Ar = 15.6 and Ar/Xe = 4.27; measured threshold
ratios are ≈2.5 and ≈3.0. Ar/Xe is nearly right; He/Ar is over by 6.3× and He/Xe
by 8.8×. There is no constant left to turn — that is what "parameter-free" costs
when the prediction misses.

*The sharper reading is the headroom.* 1.85× / 7.8× / 13.2×, monotone in atomic
mass. A cascade-only kernel offers no reason for that spread: every gas must
clear the same diffusion and finite-pulse growth requirement above its own floor,
and He is left with almost none. And `δ_elastic` is a **lower bound** — the top
of each climb runs above the first excitation threshold (19 % of the ascent in
He, 27 % in Ar, 31 % in Xe), where inelastic losses dwarf elastic recoil, so the
true floors are all higher than these. He, with 1.85×, is the gas that breaks
first. Computing that correction is precisely the distribution-resolved cascade
work M6a hands forward, and this gate is where it would show up.

**What is deliberately not landed.** Full noble-gas *threshold curves*. Those
need `K_m` and `D_e` per gas, which come from momentum-transfer cross sections —
and for Ar and Xe those swing two orders of magnitude between the Ramsauer
minimum and the resonance peak, so a single `ν_m = K_m·p` is a far cruder
approximation there than in air. No citable table was obtained, so
`Gas::from_monatomic` takes both as **required arguments** rather than shipping a
guess with the authority of a constructor default. Chylek's focal geometry is
short of a derivation too: the paper gives a 10 cm lens and a ~33 µm focal
diameter but not the beam diameter, so the divergence-limited depth of focus —
and with it `Λ` and the focal volume — cannot be reconstructed the way T&T's Eq. 5
was. Both are recorded as debts rather than filled with plausible numbers.

### Distribution-resolved cascade — 2026-07-30, and it splits the failures in two

`SelfConsistentClimb` puts every electron on the **mean** energy trajectory, so
ionization switches on discontinuously at `ε_∞ = U_i`. That bifurcation *is* the
threshold plateau, and the model is evaluated sitting on top of it — `ε_∞/U_i` is
1.032 at 760 Torr and 1.011 at 1500, i.e. the answer is set by a logarithm a hair
from its pole. A real energy distribution has no bifurcation; it has a tail that
ionizes while the mean is still short.

**The formulation.** An electron does not climb smoothly — it absorbs
inverse-bremsstrahlung quanta of size `ħω`, so its energy is a random walk *with*
drift, an Ornstein–Uhlenbeck process in energy space:

```text
dε = δ_eff·ν_m·(ε_∞ − ε)·dt + √(2·D_ε)·dW,     D_ε = ½·P_heat·ħω
```

The drift is exactly the old variant's `dε/dt`. The diffusion is photon shot
noise — events of size `ħω` at rate `P_heat/ħω` — and it adds **no new
constant**: `P_heat` is already in the model and `ħω` is the laser. Ionization
becomes first passage to `U_i` with a reflecting wall at 0, evaluated by
Siegert's formula (`breakdown0d::first_passage_ionization_rate`). A quadrature,
not a PDE solve: no time grid, no stability condition, and an exact analytic
limit to check against.

**Landed as a variant, then promoted — 2026-07-30.** It went in as a variant
first so that its effect on every published number was *measured* rather than
argued, and was made the default once those numbers were in hand. Flipping it
moved five gates and one M6c claim; all are itemised below.

**It retired M6a's red gate.** `tt2012_threshold_slope_matches_measurement` had
been `#[ignore]`d and failing since 2026-07-25 — deliberately left red rather
than re-banded to pass. It now passes, with **no tolerance moved and no constant
touched**: the measured 0.329 sits inside the `δ_eff` envelope `[0.183, 0.407]`
where the old closure's `[0.023, 0.231]` excluded it. That history is why the
gate now carries a warning to read it carefully — the same test passed once
before, in 2026-07-25, for a bad reason (an integration bound was setting the
answer). The difference is that this pass comes from removing an idealization
while every constant stayed put. The suite has no `#[ignore]`d gates left.

**Verified** — `first_passage_reduces_to_the_mean_energy_climb` (as `D_ε → 0` the
rate returns the old closed form, ratio 0.821 → 0.990 as `ħω` falls from 1.166 to
0.05 eV), `first_passage_quadrature_is_converged` (2nd order; the shipped
N = 512 within 2e-4 at both physical photon energies),
`distribution_resolved_has_no_bifurcation`, and
`first_passage_rate_depends_only_on_two_dimensionless_groups` (the rate is a
function of `ε_∞/U_i` and `ħω/U_i` alone — the arithmetic proof that no constant
was added, and it preserves `ν_i ∝ p`).

**The result, and it is sharply split.**

*The high-pressure branch is essentially fixed* (T11-P1). At the untouched
literature centre `δ_eff` = 0.02:

| window | mean-trajectory | distribution-resolved | measured |
|---|---|---|---|
| T&T 300–2000 Torr, 1064 nm | 0.0951 | **0.2793** | 0.329 |
| Chylek 300–786 Torr, 532 nm | 0.1717 | **0.4665** | 0.468 |

The more careful statement is the envelope over the single free constant
(`δ_eff` ∈ 0.01–0.05): `[0.023, 0.231]` **excludes** the measured 0.329 and
becomes `[0.183, 0.407]`, which **contains** it; on Chylek's window `[0.039,
0.414]` excluding 0.468 becomes `[0.307, 0.657]` containing it. The measurement
moves from outside the model's literature envelope to inside it with nothing
tuned. (This is *not* compared against `FixedMeanEnergy`, whose envelope already
contains 0.329 — but with **two** free constants, which is why that was never a
strong claim. Here the count of free constants is held at one and only the
closure changes.)

*The low-pressure branch is untouched* (T11-P2): 1.952 → 1.954 against a measured
0.428. That is the useful half. It **localises the two failures to two different
mechanisms** — the high-pressure branch was the cascade cutoff, the low-pressure
branch is diffusion loss `ν_diff = D_e/Λ²`, which no cascade closure can reach.
Combined with `d_e_sensitivity_is_pinned_across_the_kinetic_band` (the whole
defensible `D_e` band supplies 0.101 of slope against a 0.234 shortfall), the
remaining low-pressure curvature is now attributable to neither the mean-energy
idealization nor `D_e`.

*The wavelength gap does not close* (T11-P3), but the sign is finally right.
`D_ε ∝ ħω`, so a shorter wavelength takes bigger energy steps and reaches `U_i`
more easily — the opposite sign to the cascade's `λ⁻²`. The ratio moves
4.00 → 3.39 against a measured ≈0.80: a 15 % move against a 5× gap. Two
mechanisms with the correct sign have now been tried on this axis — Keldysh MPI
and photon shot noise — and neither is within an order of magnitude.

*The hard plateau floor is gone* (T11-P4), which changes how the noble-gas result
may be read. `cascade_plateau_intensity` is a hard lower bound **for the
mean-trajectory closure only**; with the distribution resolved the threshold
slides underneath it — 1.09× the floor at 300 Torr, 0.75 at 760, 0.63 at 2000. So
"He has only 1.85× of headroom above a floor it cannot cross" no longer holds:
it can cross it. What survives from
`chylek1990_noble_gas_plateau_floors_are_unequally_tight` is the **spacing**
failure (He/Ar predicted 15.6 against a measured ≈2.5), which is a statement
about `δ·U_i` and is untouched. Whether the noble-gas thresholds are actually
reproduced still needs the unsourced per-gas `K_m` and `D_e`.

**What promotion cost, itemised.** Five gates were re-pinned against measured
numbers, none by widening a tolerance to accommodate a failure:

| gate | before → after |
|---|---|
| `tt2012_threshold_slope_matches_measurement` | red/`#[ignore]`d → **green** |
| `tt2012_level_ratio_is_bounded_within_scatter` | drift 1.48× → 1.20×, level 3.90–4.69× high |
| `chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not` | spread 11.5× → 4.19×; the high-pressure window now *agrees* (0.466 vs 0.468), so the gate asserts a **one-sided** failure |
| `chylek1990_tt2012_wavelength_ratio_falsifies_cascade_lambda_squared` | 3.99 → 3.39, overshoot 4.99× → 4.24× |
| `keldysh_mpi_does_not_close_the_wavelength_gap` | baseline 3.99 → 3.39; order-unity 3.87 → 2.89 |

And one **M6c claim needed refining rather than retracting**. `run_lsd`'s docs
call M6a's threshold an *intensity floor* — flat in pulse length, which under the
old closure it was to 3.5 %, because no pulse length could buy past the hard
`ε_∞ = U_i` cutoff. With no cutoff, a longer pulse does buy something:

```text
6 ns   8.510e15        600 ns  6.886e15
60 ns  7.204e15        6 µs    6.814e15
                       1 ms    6.797e15
```

It buys 1.25×, and then stops — the threshold *converges* to a genuine floor of
6.797e15 W/m² by ~10 µs. That is the physically expected shape; real breakdown
thresholds do depend on pulse duration, and the perfectly flat one was an
artifact of the bifurcation. M6c's two-stage argument is untouched, because what
would break it is a threshold falling *without limit* with pulse length — a
fluence criterion — and `the_sustaining_drive_is_far_below_the_breakdown_threshold`
now asserts both the bounded fall and the asymptote.

**Two implementation notes worth keeping**, because both were caught by gates
rather than by reading. The obvious `O(N)` evaluation of the Siegert integral —
accumulate `∫e^{−φ}` and multiply by `e^{+φ}` — splits a bounded product into
factors reaching `10^±274`, and loses every digit before it overflows; the sum is
carried as a recurrence in the *differences* instead. And the reduction gate must
refine its own grid: at the shipped N it returned a rate 3 % on the **wrong side**
of the limit it was meant to approach, so a fixed-grid reduction gate measures
its own resolution. Both failures looked like physics and were not.

### Free-molecular escape — 2026-07-30, and the low-pressure branch half-fixed

After the distribution-resolved closure landed, the low-pressure branch was the
milestone's one remaining structural failure, and it was well cornered: 4.6× too
steep against Chylek's clean power law, shown immune to the cascade closure
(`distribution_resolved_does_not_fix_the_low_pressure_branch`, 1.952 → 1.954) and
unreachable by any defensible `D_e` (`d_e_sensitivity_is_pinned_across_the_kinetic_band`,
the whole 6.0× band buys 0.078 of slope). Neither of the two obvious knobs.

**The defect was the loss term's validity, not its value.** `ν = D_e/Λ²` is a
continuum random-walk result: it assumes the electron collides many times while
crossing the focus. The Knudsen number `Kn = λ_mfp/ℓ` says it does not —

| Torr | 760 | 300 | 100 | 10 |
|---|---|---|---|---|
| `Kn` | 0.013 | 0.034 | 0.10 | **0.96** |

— and against the *diffusion length* `Λ`, which is 4× smaller than the escape
distance, the mean free path at 10 Torr is **3.8×** it. The kernel was applying a
continuum formula deep in the collisionless regime, across the entire window
where it was worst, and because the overstatement is pressure-dependent it was
manufacturing slope.

**The correction.** An electron cannot cross the region faster than it can travel
across it, so the escape *time* is the diffusive time plus the ballistic transit
time:

```text
ν_esc = 1/(τ_diff + τ_ballistic),   τ_diff = Λ²/D_e,   τ_ballistic = ℓ/v̄
```

`Kn ≪ 1` recovers `D_e/Λ²` exactly; `Kn ≫ 1` saturates at `v̄/ℓ`, independent of
pressure, which is the physics — a collisionless electron's escape time does not
care how thin the gas is.

**It introduces no constant.** `v̄` is not supplied: `D_e = v̄²/(3ν_m)` with
`ν_m = K_m·p` already fixes it and the pressure cancels, giving
`v̄ = √(3·D_e,ref·p_ref·K_m)` = 1.540e6 m/s — an electron of **6.740 eV**, the
same energy `d_e_ref_implies_a_stated_electron_energy` reads out of `D_e`. Both
inputs are already gated. The correction is forced, not chosen.

**A modelling error caught on the way, worth recording.** The first cut used `Λ`
as the ballistic transit distance. That is dimensionally fine and physically
wrong: `Λ` is a diffusion *eigenvalue* (`ν = D_e/Λ²`), not a distance. The
free-molecular escape distance is the **Cauchy mean chord** `ℓ = 4V/S` — the mean
straight-line path of an isotropically-directed particle leaving a convex body, a
theorem rather than a model choice. For T&T's focal cylinder that is **30.72 µm
against `Λ` = 7.74 µm, a factor 4.0**, so the first version understated the
correction fourfold (low-pressure slope 1.636 instead of 1.293). The three
lengths — `Λ`, `ℓ`, `V` — now come from one geometry via `breakdown0d::Focus`,
so they cannot drift apart, and `focus_geometry_separates_its_three_length_scales`
pins all three.

**Result — about half the error, and the rest is stated plainly.**

```text
10–100 Torr    before  1.954      after  1.293      measured  0.428
```

The correction is **mandatory** — the old formula was being used outside its
domain of validity — but it is **not sufficient**, and
`free_molecular_escape_flattens_the_low_pressure_branch` asserts the residual
2.6× as loudly as the improvement. What remains is very likely the multiphoton
channel: both source papers state MPI *dominates* below 100 Torr (T&T quote 88 %
cascade / 12 % MPI at 760 Torr), and a cascade-plus-loss model has no mechanism
left that could flatten this branch further. That is now M6a's sharpest open
question, and it is the same channel the wavelength axis has been pointing at
since D5.

**It is not purely a low-pressure correction**, which is why the high-pressure
numbers moved a few percent when it landed: the loss rate is still 6.3 % below
the continuum value at 760 Torr and 2.4 % at 2000. Chylek's 300–786 Torr window
went 0.4665 → 0.4313 against a measured 0.468, and the T&T window 0.2793 →
0.2636 against 0.329 — both still inside the `δ_eff` literature envelope, so the
red gate stays green (envelope now `[0.174, 0.382]`, centre 0.264).

### PPT for molecular O₂ — 2026-07-31, and the MPI question closes

The Keldysh section above left the open item in two branches: *either* the
prefactor for molecular O₂ is orders above unity (PPT Coulomb corrections), *or*
the two-paper comparison carries a systematic. Both are now settled, and it is
the second one.

**What made this doable now was an anchor, not a formula.** T12 was written
requiring a prefactor "justified to ~2 orders", which is why it sat open. Two
published numbers do better:

- `σ₈ = (3.3 ± 0.3)×10⁻¹³⁰ W⁻⁸m¹⁶s⁻¹`, the **absolute** eight-photon ionization
  cross-section of O₂ at 800 nm, from counting the electrons directly by Rayleigh
  microwave scattering against calibrated dielectric scatterers (Sci. Rep. **8**,
  2874 (2018)). `K` = 8 at 800 nm sits *between* the kernel's `K` = 11 at 1064 nm
  and `K` = 6 at 532 nm, so using it is interpolation.
- `Z_eff` = 0.53 for O₂ (Talebpour, Chien and Chin, J. Phys. B **32**, 1229
  (1999)), the single molecular parameter PPT needs — published, so it enters as
  a cited constant rather than a fit. Their own rate point, 3×10⁹ s⁻¹ at
  3×10¹³ W/cm², sits 7× below `σ₈·I⁸`, which bounds the truth to about an order
  of magnitude before any theory is involved.

The formulation, the numbers and the gate list are in `docs/MODELS.md` §
"PPT photoionization for molecular O₂". Three things belong here as milestone
record.

**1. The first absolute validation of a rate in this milestone.** Every previous
MPI attempt compared a breakdown threshold to a breakdown threshold, which cannot
separate the ionization rate from the cascade that follows it. PPT's derived
prefactor reproduces `σ₈` to **1.99×**, nothing fitted, and on the high side —
the direction the source paper reports for purely theoretical predictions. That
closes the prefactor branch by measurement rather than by argument.

**2. The expectation that motivated the work was wrong, and measuring it is what
showed so.** The scoping estimate said the Coulomb correction would lift the
prefactor by ~10³–10⁴, because for an atom with `Z` = 1 the exponent `2n*` is
≈ 2.1. For O₂ at `Z_eff` = 0.53, `n*` = 0.563 and the exponent `2n* − 3/2` is
**negative**: the correction is order-unity. The λ ratio moves 3.349 → 2.947,
16 % of the gap — essentially where an order-unity Keldysh prefactor already
put it. Recorded because it was the whole hypothesis.

**3. A structural result a gate found by failing.** `ppt_multiphoton_order_*`
was written asserting the ponderomotively shifted order `ν = (U_i/ħω)(1+1/2γ²)`
and measured 10.998 at 1064 nm where `ν` = 10.34. The above-threshold sum's
leading term carries `e^{−α(γ)(⌈ν⌉−ν)}` with `dα/d ln I = −1`, which contributes
exactly `⌈ν⌉ − ν`: PPT returns the **integer** photon order. That is a
requirement, not a curiosity — you cannot absorb 10.34 photons — and the bare
Keldysh exponential's fractional order is an artifact of dropping the sum. The
gate was renamed to assert what it measured.

**What it leaves.** The seeding calculation (`N_seed = W·N·V·τ` at each paper's
own measured threshold, with no model threshold anywhere in it) puts the 532 nm
measurement **at** its multiphoton seeding threshold, 0.83×, and the 1064 nm
measurement 5.7× **below** its own. The two anchors are on opposite sides of that
transition, so their threshold ratio is not one mechanism's wavelength scaling.
M6a's remaining λ discrepancy is therefore part cascade closure and part
comparing two different experiments — and that second part is now a measured
statement rather than the hand-wave "or a systematic" it replaces.

**Four defects found by reviewing the change after it landed — recorded because
one of them is a process failure, not a coding one.**

1. **A planned gate was dropped silently.** The spec for this work listed an ADK
   reduction gate (`γ → 0`, `A₀ → 1`, compare against the closed form). It was
   not written and its absence was not stated. That is the failure mode this
   project's whole structure exists to prevent, and it is what let (2) survive
   to be committed.
2. **The above-threshold sum was truncated at a fixed 64 terms.** Its decay
   constant `α = 2[asinh γ − γ/√(1+γ²)]` goes as `⅔γ³`, so the sum stops
   converging in the tunnelling limit and a fixed truncation returns a number
   set by the loop bound: 29 % low at `γ` = 0.2, 78 % low at `γ` = 0.1. Worse,
   it made the rate **fall** with intensity over 40 of 200 sampled points inside
   the bracket `threshold_intensity` bisects on. Fixed by deriving the term count
   from `α` (which is also *cheaper* — 10 terms at `γ` = 10 against the old 64)
   with an ADK branch below `γ` = 0.1, and gated four ways: the restored ADK
   reduction, the branch join, convergence down to the cutover, and monotonicity
   across the whole bracket. **No published number moved**: every result here
   sits at `γ` = 8–54, and re-running at 64× the truncation reproduces the
   wavelength ratio and the σ₈ anchor to every printed digit.
3. **The overflow guard failed toward zero.** `ln_w > MAX_EXPONENT` returned
   `0.0` — "faster than `f64` can express" reported as *no ionization*. Now
   saturates, as `keldysh_rate` already did.
4. **The seeding gate transcribed its inputs.** It carried the two measured
   thresholds as literals rather than reading them from the digitized CSVs, and
   called a 785.7 Torr point "~760 Torr" while using 1 atm for both rows'
   neutral density. It now reads each curve's point nearest 1 atm and uses each
   point's own pressure; `N_seed` at 532 nm moves 3.05 → 3.15 and nothing else
   changes.

**Status of the MPI question, plainly.** Three channels have been tried against
this data: T&T's own calibrated `σ_K` (37× below their own measurement), Keldysh
(3–18 % of the gap), and PPT (16 %, with the rate itself now validated
absolutely). None closes it, and the PPT result is the one that cannot be
answered with "your prefactor was wrong".

## NOT in scope (M6a)

- **Molecular structure beyond `Z_eff`** — no interference or alignment terms in
  the PPT rate. `Z_eff` = 0.53 is the published effective-charge summary of
  exactly that physics; inventing more would be unanchored.
- Any propagator coupling.
- Plasma back-reaction on the beam (absorption/refraction) — M6c.
- Recombination / afterglow / multi-pulse.
- Absolute-threshold agreement as a gate (only the slope/trend is gated).
