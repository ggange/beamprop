# M6c pre-spec — laser-supported detonation wave (1-D Euler + laser deposition)

Written **before** any M6c code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md), [M6A_SPEC.md](M6A_SPEC.md)): pin the gas-dynamic
model, the deposition closure, the coupling cadence, the property table, and —
most importantly — *which* checks validate physics versus which are solver
verification. M6a's hard lesson (D5) was that a gate anchored to the same
source the model is built from proves nothing; that lesson is applied here
before the first line of code, not after. If any of this proves wrong during
implementation, amend this document first, then the code.

Scope: the deep rung of the M6 ladder. The beam ignites a spark at the
M6a threshold; the resulting absorption wave runs **back up the beam** toward
the laser as a detonation. The propagator sees the plasma column as **pure
Beer–Lambert absorption** — no Drude index (D7).

Conventions follow [MODELS.md](MODELS.md): SI units, `f64`, intensity
`I = |u|²` (W/m²). New symbols: density `ρ` (kg/m³), velocity `u` (m/s),
pressure `p` (Pa), total energy density `E` (J/m³), specific heat ratio `γ`,
sound speed `c` (m/s), LSD front speed `D` (m/s), absorbed intensity at the
front `S` (W/m²), plasma absorption coefficient `α_pl` (1/m), mean charge `Z̄`.

## Gate decisions (recorded)

1. **Physics target: the LSD (laser-supported detonation) regime only.**
   Planar 1-D, local thermodynamic equilibrium, single ionized species set from
   equilibrium chemistry. Not modelled: LSC (combustion) and LSR (radiation)
   regimes, non-LTE / two-temperature plasma, radiation transport. The solver
   asserts it is in the LSD regime (below) and refuses the others rather than
   silently mis-modelling them — the M4 Péclet precedent.
2. **Plasma couples to the beam through absorption only** (D7). The plasma
   column is a **read-only** `Medium` supplying `extinction(z_slab)`
   (`src/medium.rs:73`, already in the trait). No `index_perturbation`, no
   Drude `δn`. This sidesteps the near-critical failure a Drude column would
   hit in a paraxial envelope, and never touches `check_phase_sampling`.
3. **Coupled state lives in an outside driver** (D3). The driver owns the
   plasma column and the hydro state, reuses one `Propagator` across time
   slices, and records intensity through the existing
   `propagate(..., on_step)` callback (`src/propagate.rs:216`). No propagator
   or `Medium` trait changes.
4. **The Riemann solver is laser-agnostic.** The HLLC/Euler core is a separate
   module with no laser physics in it, so its verification gates (G1, G2) test
   a general-purpose solver against standard closed forms. Laser deposition,
   ignition, and the plasma column live one layer up. Deliberate: it keeps the
   verification gates uncontaminated by the model under test.
5. **Raizer's LSD velocity is the closed-form anchor, and it is a
   VERIFICATION gate — not the physics gate.** See § "The circularity, stated
   up front". The physics gate is the parameter-free scaling exponent (G4);
   absolute velocity vs measurement is documented and **ungated** (G7).
6. **Property closure: a separate plasma-range table** (D8), generated offline,
   frozen in the repo. `data/air_properties.npy` (M4) is left **byte-identical**
   — its green gate must not be perturbable by anything M6c does.
7. **Deferrals are recorded in this document's "NOT in scope"** (D9). The repo
   has no `TODOS.md`; SPEC "NOT in scope" is the project convention.

## The circularity, stated up front

Raizer's LSD velocity is not an independent check on a Chapman–Jouguet
comparison — **it is the CJ construction**, with the chemical heat release
replaced by laser deposition at the front:

```text
q = S / (ρ₀·D)                     energy released per unit mass swept up
D² = 2(γ² − 1)·q                   strong-detonation CJ velocity
⇒  D = [2(γ² − 1)·S / ρ₀]^(1/3)
```

So "gate the LSD velocity against C-J theory" and "anchor to Raizer" are the
**same closed form**, not two. A solver that
deposits energy at the front and is then asked to reproduce that expression is
being checked against its own derivation — the M6a Raizer-vs-Raizer trap that
D5 caught, in new clothes.

That does not make the check worthless; it makes it a **verification** gate,
and a good one: it tests that the HLLC solver, the Strang-split source term,
and the EOS together reproduce a nontrivial analytic self-similar solution to
within a tolerance that tightens under grid refinement. It just cannot be
labelled physics validation, and M6c must not ship claiming it is. M6a shipped
with exactly one clean shape gate and an honest statement of what was missing;
M6c should do better, and the way it does better is G4.

## Gas-dynamic model

### Governing equations

Conservative 1-D Euler with a volumetric laser source, `x` along the beam axis
(front propagating toward the laser, `−x`):

```text
∂U/∂t + ∂F(U)/∂x = Ṡ

U = (ρ, ρu, E)ᵀ
F = (ρu, ρu² + p, (E + p)u)ᵀ
Ṡ = (0, 0, α_pl(x)·I(x))ᵀ
E = ρ·e(ρ, p) + ½ρu²
```

The source is the **absorbed** laser power density; the beam is attenuated
through the column by Beer–Lambert marching from the laser side, so energy is
conserved between the two halves by construction (gated: G5).

**Amended (step 4): the attenuation is `dI/dx = −α_pl·I`, not `+`.** The beam
travels in `+x` (the laser sits beyond `x_min`; the front runs back up the beam
toward it, in `−x`), and an absorbing medium marched in the direction of travel
decays. The `+` written above would amplify the beam, and contradicts every
other statement in this document — Beer–Lambert, "the **absorbed** laser power
density", and the closed energy budget of G5. A sign slip, corrected here rather
than silently in the code.

**Also amended (step 4): the deposition is discretised conservatively, not as a
midpoint sample of `α·I`.** Each cell takes the intensity it actually removes
from the beam,

```text
I_{k+1} = I_k·exp(−α_k·dx),     q_k = (I_k − I_{k+1})/dx
```

so `Σ q_k·dx` is *identically* `I_in − I_out` with no quadrature error, and
reduces to `α·I` for small `α·dx`. This is what lets G5 close to round-off
(measured 2×10⁻¹⁶) instead of to a discretization tolerance.

### Equation of state

Two modes, deliberately:

- **Verification mode: ideal gas, constant `γ`.** Used for G1–G3, where the
  closed forms assume it. No table, no interpolation, nothing that can drift.
- **Production mode: table EOS** (§ Property closure), `e(ρ, p)` and an
  effective `γ_eff(T, p)` from equilibrium air at plasma temperatures.

Worth pinning now, because it bounds an argument made later: the difference is
**not** where the experimental discrepancy lives. `γ = 1.4` gives
`2(γ²−1) = 1.92`; a dissociated/ionized `γ_eff ≈ 1.2` gives `0.88`. Through the
cube root that is a **0.77×** shift in `D` — ~25%, not the ~2× gap to
measurement. The EOS is a real correction and a wrong reason to expect the gap
to close.

### The CJ state, and what it implies

For dry air at STP (`ρ₀ = 1.225 kg/m³`, `γ = 1.4`) and `S = 10¹¹ W/m²`
(10⁷ W/cm², a representative LSD drive):

```text
D    = [1.92 · 10¹¹ / 1.225]^(1/3)  = 5.4 km/s
p₁   = ρ₀D²/(γ+1)                   = 1.5×10⁷ Pa   ≈ 150 bar
ρ₁/ρ₀ = (γ+1)/γ                     = 1.71
u₁   = D/(γ+1)                      = 2.2 km/s
c₁   = γD/(γ+1)                     = 3.1 km/s     (CJ: D − u₁ = c₁ ✓)
```

Two consequences worth recording:

- **The post-front temperature justifies the table range.** Taking undissociated
  air, `T₁ = p₁/(ρ₁R) ≈ 2.5×10⁴ K`; real dissociation and ionization lower the
  mean molecular weight and so the actual `T₁` (plausibly 1.5–2×10⁴ K). Either
  way D8's "~30,000 K" upper bound is the right ceiling, and it is set by the CJ
  state rather than picked.
- **The timescales are cheap, but not for the propagator.** A 1 cm domain at
  10 µm resolution gives `dt_CFL ≈ 1.5 ns` and ~1200 hydro steps for a full
  crossing. That is nothing. Re-solving the split-step propagation 1200 times is
  **not** nothing — hence the sub-cycling knob below.

### Validity checks asserted at run start

In the M4 Péclet spirit — refuse, don't mis-model:

- **Optically thin enough to have a front.** The absorption length
  `1/α_pl` behind the front must be resolved by ≥ N cells and be short against
  the domain. Too thin → unresolved deposition; too thick → the deposition is
  volumetric and the detonation analogy fails (that is the LSC regime, out of
  scope).
- **Strong-detonation ordering.** `p₁ ≫ p₀` and `D ≫ c₀`; the closed form above
  assumes both.
- **Beam not extinguished before the front.** If the column absorbs essentially
  all the beam upstream, the front starves — a real physical effect, but one
  the 1-D model handles badly. Report loudly.

## Numerics

- **HLLC** approximate Riemann solver, Davis/Einfeldt wave-speed estimates.
- **MUSCL-Hancock** reconstruction with a minmod limiter → 2nd order in smooth
  flow, TVD across the shock. (Gated by G2.)
- **Strang splitting** between the hydro update and the source term, matching
  the propagator's existing symmetric-splitting discipline (`src/propagate.rs`)
  — and for the same reason: it is what keeps the coupled scheme 2nd order.
- **CFL ≤ 0.8**, asserted per step from the current wave speeds, not fixed.
- **Positivity guard**: `ρ > 0` and `p > 0` checked after every update; a
  violation **bails loudly** with the cell and step (per the plan's failure-mode
  table — never clamp silently, the M6a `n_e` runaway lesson).

## Beam ↔ plasma coupling

```text
per hydro step dt:
  1. propagate(field, PlasmaColumn(α from hydro state), …, on_step)
       on_step records the on-axis intensity profile I(z)
  2. if not yet ignited: AirBreakdown::breaks_down(I_peak, …) at the focal plane
  3. Euler step: advance dt with source α_pl(x)·I(x)
  4. recompute α_pl(x) from the new (ρ, T) via the plasma table
```

- `PlasmaColumn` is a new read-only `Medium`: `extinction(z_slab) -> Option<Array2<f64>>`,
  filled from the current hydro state.

  **Amended (step 4): `index_perturbation` returns zeros, not an unreachable
  stub.** The `ThermalBlooming` precedent does not carry: that medium sets
  `needs_intensity() = true`, so the propagator routes around
  `index_perturbation` and never calls it. `PlasmaColumn` is *linear*, so the
  propagator does call it, and an `unreachable!` there would panic on the first
  slab. D7's "absorption only" is expressed correctly as `δn ≡ 0`, which is what
  zeros are.
- **`physical_intensity_scale` (T4/D4) is required here.** The driver must turn
  `on_step`'s `|u|²` into W/m² both to test the M6a trigger and to size the
  deposition. The extraction was scheduled for whichever model first needed an
  absolute intensity outside blooming; **M6c is that consumer**, so it lands with
  the driver, still under its CRITICAL guard: M4's blooming gate byte-for-byte
  green.
- **Sub-cycling.** Re-solving the propagation every hydro step is the dominant
  cost. The driver takes `propagation_every: usize`; the spec requires a
  convergence check that the front trajectory is insensitive to it (halving it
  moves `D` by less than the G3 tolerance), and the default is whatever that
  check supports — not a guess.

### Inherited limitation: the ignition trigger carries M6a's ungated level

The velocity gates are independent of M6a. The **ignition time and position**
are not: they come from `AirBreakdown`'s absolute threshold, which is M6a's
explicitly ungated quantity (3.90–4.69× above the measured T&T curve, inside the
3–10× inter-lab scatter). So M6c can pass every velocity gate while lighting
the spark at the wrong intensity. This is a stated limitation, not a blocker —
`D` depends on the absorbed `S` at the front, not on where ignition happened —
but it must be written in `MODELS.md` when M6c lands, and it is a second reason
the M6a D5 debt is worth paying.

## Property closure (D8)

- `data/plasma_properties.npy` + `scripts/make_plasma_table.py`, generated
  offline from Mutation++ (VKI) with equilibrium ionization, `T` up to
  ~30,000 K. Properties: `e(ρ,p)` (or `h`, `c_p`), `γ_eff`, `Z̄`, and the
  inputs to `α_pl` (inverse bremsstrahlung: `α_IB ∝ n_e·n_i·Z̄ / (ω²·T^½)`
  with the stimulated-emission correction `1 − exp(−ħω/kT)`).

  **Landed (step 3), with two amendments this document owes the reader.**
  Shipped as `(4, 597, 33)` over `[ln ρ, e, γ_eff, ln n_e]`, uniform in `T`,
  uniform in `log₁₀ p`. (a) `n_i` is **not stored**: with only singly charged
  ions, quasi-neutrality makes it identical to `n_e` (6.2×10⁻¹¹, gated), so a
  stored copy would be redundant. (b) `Z̄` is **not stored and is not a
  prediction** — it is identically 1 because the RRHO database has no doubly
  ionized N or O. Real air doubly ionizes above ~20,000–25,000 K, so above
  `SECOND_IONIZATION_K` = 20,000 K the table understates `n_e` and hence
  `α_IB`, and is a singly-ionized approximation. That ceiling was not
  anticipated when this spec was written; it is a property of the available
  thermodynamic data, not of the mixture choice, and it bounds where the LSD
  absorption coefficient can be trusted.
- Same discipline as `airprops.rs`: shape/dtype asserted at load, uniform axes,
  bilinear interpolation, generation script committed, no runtime FFI, no LGPL
  in the build or the M5 wheels.
- **`data/air_properties.npy` is not touched.** Byte-identical, gated (G0).
- Mutation++ is trusted for the physics (D8); the gate is on the **tabulation**
  — frozen table vs direct evaluation at sampled `(T, p)`, especially across the
  ionization onset where interpolation is worst (G6).

## Gates

Following M6a's discipline, each gate is labelled by what it actually
establishes. Verification = "the code solves the equations I wrote down".
Validation = "the equations describe the world".

### G0 — M4 regression (CRITICAL, verification)

After the T4 extraction and anything else M6c touches, M4's blooming gate is
**byte-for-byte green** and `data/air_properties.npy` is unchanged. Non-
negotiable; it is the guard D4 attaches to.

### G1 — Sod shock tube vs the exact Riemann solution (verification)

Standard initial data, exact solution from an iterative Riemann solver in
`validate.rs`. Assert L1 error below a pinned bound and — the part that makes
it a real test — that the error **falls under refinement**. Laser physics off.

### G2 — observed convergence order on smooth flow (verification)

Isentropic advection or a smooth acoustic pulse; `validate::observed_order`
(already used by M1 and M4) must give ≈ 2 with MUSCL-Hancock. Laser physics off.

### G3 — Raizer LSD velocity, closed form (VERIFICATION, not validation)

Constant-`γ` ideal gas, planar, thin absorption layer, strong detonation — the
regime where the closed form's assumptions hold. The measured front speed must
match `D = [2(γ²−1)S/ρ₀]^(1/3)` within a pinned tolerance. **Labelled in the
code and in `MODELS.md` as solver verification**, for the reason given in
§ "The circularity". This is the M6c analogue of M6a's "closed-form sub-gates
are integrator unit tests".

**Amended (step 4): the refinement knob is the absorption length, not the
grid.** This document originally required the residual to shrink under *grid*
refinement. Measurement says that is the wrong parameter. At a fixed absorption
length `1/α = 50 µm`, refining `dx` from 10 µm to 5 µm moves `D` by 5×10⁻⁵ —
the answer is already grid-converged, and the residual that remains is not
discretization error. It is set by the *physical* thickness of the deposition
layer, which is the closed form's own "thin absorption layer" assumption.
Halving `1/α` from 400 µm to 50 µm walks the residual
`−8.3 % → −2.7 % → −0.4 % → +0.19 %`, monotonically and by at least 2.3× per
halving.

**Corrected during the step-5 review: the residual is a relaxation transient,
not a permanent thick-layer deficit.** This section first explained the sign by
saying a thick deposition zone releases energy *behind the sonic plane* where it
cannot support the front — implying a steady-state deficit. That contradicts the
theory the gate checks against: a Chapman–Jouguet velocity depends on the total
heat release, not on the reaction-zone length. Measurement agrees with the
theory, not with the explanation. Held at `1/α = 400 µm` and given longer to
settle (0.15 → 0.30 → 0.50 of the domain), the residual runs
`−8.3 % → −3.7 % → −1.5 %` with no sign of plateauing. A thicker deposition zone
relaxes onto the self-sustaining speed *more slowly*; at a fixed settle it sits
further from it; given long enough they all arrive at the same CJ speed. G3b is
therefore a statement about convergence at a fixed settle, which is what it
measures and now what it says. So G3 gates three things —
the level at a thin layer, grid-convergence at fixed `1/α` (which shows the
residual is physical), and convergence in `1/α` (which is the assumption
actually being relaxed). Gating grid refinement alone would have been gating the
knob that does not move the answer.

**Amended again (review before step 5): G3c, seed-independence.** A seeded
detonation starts overdriven and relaxes onto the CJ speed *slowly*, so the
settle time is not a free choice — and at the 1.0 µs originally used, the
answer moved by 5.3×10⁻³ between a 1× and a 2× CJ-pressure seed, against G3's
own 1 % tolerance. The seed was therefore an unexamined parameter sitting
directly under the headline number. `LSD_SETTLE` is now 1.8 µs, where the same
spread is 1.1×10⁻³, and `lsd_front_speed_is_seed_independent` gates it below
2×10⁻³ rather than leaving it to be assumed.

### G2b — the hydro↔source coupling is 2nd order (verification)

**Added (review before step 5).** This document and `MODELS.md` both claim
Strang splitting keeps the *coupled* scheme 2nd order. M1 gates that claim for
the propagator's split-step and M4 gates it for the blooming coupling; M6c
asserted it and gated nothing — G2 runs the homogeneous solver with the source
off. That was a gap in exactly the place this project's discipline exists to
cover.

Refinement is `dx` and `dt` **together at fixed CFL**, which is the limit the
claim is stated in: MUSCL-Hancock is 2nd order there, and degenerates to forward
Euler in time (1st order) if `dt` is driven to zero at fixed `dx`, so refining
`dt` alone measures the wrong thing. Measured: Strang **1.99 / 2.03 / 1.99**;
the deliberate 1st-order contrast (folding the source into the update)
**0.88 / 1.02 / 1.07**. The contrast is part of the gate — without it, a
measurement that could not resolve 1st from 2nd order would pass silently.

### G4 — parameter-free scaling exponent (THE PHYSICS GATE)

`D ∝ S^(1/3)` and `D ∝ ρ₀^(−1/3)`, fitted over **≥ one decade** of each, with
the exponents within a tight band of `±1/3`.

This is the gate that carries M6c, and it is the exact analogue of what M6a
learned to gate on. Every quantity that is uncertain about the absolute level —
`γ_eff`, the absorbed fraction, radial relief, radiation losses — enters as a
**coefficient**. None of them can produce a `1/3` exponent. So the exponent is
parameter-free in the D5 sense, it is what the model genuinely predicts, and
unlike the level it is directly comparable to measurement, where LSD velocities
are reported to follow the same one-third power. A gate on the exponent is a
statement about the world; a gate on the level would be a statement about the
coefficient soup.

Run in both EOS modes; the exponent must be EOS-independent (that is the point).

**Landed (step 5).** `lsd_velocity_follows_the_parameter_free_one_third_scaling`,
over 1.52 decades of `S` and 1.50 decades of `ρ₀`, exponents gated inside
`±0.01`. Measured at `γ = 1.4`: **+0.33190** and **−0.33020**.

Three amendments this step owes the reader.

1. **The EOS-independence leg is done by moving `γ`, not by running the table
   EOS.** The table EOS is not wired into the hydro — that is this document's
   "production mode", and it is not landed — so the honest available
   demonstration is to vary the coefficient directly. `2(γ²−1)` runs 0.88 → 3.56
   from `γ = 1.2` to `5/3`, moving the *level* of `D` by a factor 1.59, while
   the fitted exponents move by 0.001 (`S`) and 0.002 (`ρ₀`). That is the D5
   argument as a measurement rather than an assertion. It is **not** a
   demonstration that a real equilibrium EOS leaves the exponent alone: a
   `γ_eff` that varies with local state is not the same thing as a different
   constant `γ`, and the gate does not claim it is. Closing that properly needs
   the production EOS.

2. **The density sweep holds ambient *temperature* fixed** (`p₀` scaled with
   `ρ₀`), not ambient pressure. At fixed `p₀`, sweeping `ρ₀` over a decade moves
   the ambient specific internal energy by the same decade, which makes any
   fixed ignition threshold meaningless at one end — at the thin end the
   *undisturbed* gas crosses it and the whole column absorbs. Fixing the
   temperature keeps `e₀` and `c₀` constant and is the physically natural
   reading of "change the ambient density" anyway.

3. **The ignition threshold had to be re-expressed as a multiple of ambient**
   `e₀`, and the margin is not generous everywhere. G3 sits at one point where a
   10× threshold faces an 85× post-shock state. The sweep corners are far
   tighter: at `γ = 1.2`, `ρ₀ = 12.25` the post-shock state is only 11×
   ambient, so a 10× threshold stops *enabling* the front and starts
   *controlling* it — the fitted density exponent goes to **−0.459**, a 38 %
   error, which is the gate catching a modelling error rather than a bug. At 5×
   the margin is restored; the evidence that the threshold is out of the loop is
   that 3× and 5× agree on the exponents to better than 1e-3.

Recorded separately: `lsd_velocity_level_tracks_the_eos_coefficient` checks that
the *level* follows `(0.88/1.92)^(1/3) = 0.772` when `γ` moves 1.4 → 1.2
(measured 0.7722). The solver tracks the coefficient exactly where the
coefficient is knowable — which is the sharpest statement of why agreement on
the level is not evidence about the physics, and why G7 is ungated.

### G5 — energy budget closure (verification)

Laser energy absorbed = Δ(internal + kinetic) in the domain + flux through the
boundaries, to ~1e-10 relative. M4's closed power budget, one dimension down.

### G6 — plasma-table tabulation consistency (D8)

Frozen table vs direct Mutation++ at sampled `(T, p)`, within a stated
tolerance, with the ionization onset explicitly among the samples.

### G8 — the plasma column shields the beam as Beer–Lambert (verification)

**Added at step 6, not in the original gate list.** D7 says the propagator sees
the plasma column as pure absorption with `δn ≡ 0`; until step 6 that was
checked only on `PlasmaColumn`'s `Medium` methods in isolation, and no field had
ever been marched through one. The demonstration run needed that path, so it is
now gated on it: a real beam through a real column built from G3's settled hydro
state, against `exp(−τ)` with `τ = Σ α_k·dx` — the M2 `beer_lambert_matches_closed_form`
precedent, with the absorber coming from gas dynamics instead of a constant.

The reference is exact and beam-independent because the column is transversely
uniform. `δn ≡ 0` is asserted at every slab rather than assumed, since a Drude
index appearing there is precisely the near-critical failure D7 avoids. The gate
also runs two slab resolutions, because `PlasmaColumn::from_column_resampled` is
what makes marching a 2500-cell hydro state through an FFT propagator affordable
and a binning that lost optical depth would surface here.

Measured on G3's own column (`τ = 339`): agreement to **1.7×10⁻¹³ at 500 slabs
and 8.4×10⁻¹⁴ at 100**, across 500 successive amplitude multiplications against
a single exponential. The transmission itself is 4.9×10⁻¹⁴⁸ — an established LSD
plasma is not a partial shield, it is a shutter.

### G7 — absolute velocity vs measurement: DOCUMENTED, UNGATED

The 1-D planar solver is **expected to land high** against measured LSD
velocities, and the spec says so in advance rather than discovering it:

- A planar 1-D code has **no radial relief**. If the dominant reason measured
  LSD speeds fall below the CJ prediction is lateral rarefaction of a
  finite-diameter beam, a 1-D model structurally cannot reproduce it — it is the
  one effect the geometry has removed by assumption.
- Radiation losses and incomplete absorption push the same direction and are
  also out of scope (§ NOT in scope).

**Amended (M6d): the first bullet is no longer available as an excuse.** Radial
relief is modelled, and its size is measured and pinned:
`δ = 1 − D/D_wide` = **0.230** at `R_b·α` = 3.2 and 0.305 at 1.6, i.e. a
finite-diameter beam costs roughly a quarter of the front speed. That is a real
effect and it is **not the whole ~2× gap** — so relief is part of the answer,
not the answer. The remaining candidates are the ones already named here
(radiation losses, incomplete absorption) plus the production EOS and, new with
M6d, the assumption that the beam travels in straight pencils and is not
refracted by the plasma it creates.

M6d also found something that changes how any future G7 comparison must be
made: the modelled front is **transversely unstable**, so a single run's speed
carries the instability's signature and a comparison against a 1-D calculation
would attribute that to relief. See [M6D_SPEC.md](M6D_SPEC.md) § G14.

**G7 therefore remains ungated for exactly one reason: there is no anchored
measured dataset.** That is open question 1 below, inherited by M6d and still
unpaid.

So the honest claim M6c can make is: **the 1-D model agrees with CJ/Raizer where
that theory applies, and the gap to experiment is in the predicted direction and
of the predicted order, for reasons the model has explicitly excluded.** That is
"expect and explain the ~½·v_CJ result", sharpened: the
discrepancy is a *prediction of the omissions*, not a tuning failure. Gating it
would be gating the absence of 2-D physics.

**Open item (see below): the experimental comparison needs a named, pinned
source** — a digitized dataset with the figure number recorded in the
provenance header at digitization time, exactly as `tests/data/tt2012_*.csv`
does. Until it exists, G7 is a comparison against a literature-quoted band and
is labelled as such.

## Failure modes (new codepaths)

| Failure | Detection | Response | Silent? |
|---|---|---|---|
| Shock instability / negative `ρ` or `p` | positivity check every step | bail with cell + step | no (loud) |
| CFL violation | wave speeds recomputed per step | bail | no |
| Absorption layer unresolved | cells-per-`1/α_pl` at run start | bail with the resolution needed | no |
| Beam extinguished before the front | transmitted fraction at the front | bail — outside LSD scope | no |
| Ignition never triggers | driver step budget exhausted | clean report, not a hang | no |
| Table queried outside its `(T,p)` range | range check in the interpolator | `Err` (the `airprops` precedent) | no |
| Sub-cycling too coarse | front-trajectory convergence check | documented default from the check | no |

## NOT in scope (M6c)

- **Full-Drude plasma shielding.** Not modelled, and the reason is recorded
  here so it is not rediscovered: the `δn` clamp such a model needs fires at `n_e > 0.1·n_crit ≈ 2×10²⁴ m⁻³`, which is *inside* the breakdown
  density range M6a targets (~10²³–10²⁴ m⁻³). A paraxial split-step envelope
  cannot carry a near-critical plasma — the phase-per-slab and the sampling
  criterion both fail — so a full-Drude treatment is not merely unimplemented
  here, it is unreachable from this solver. It needs a non-paraxial method,
  which is a different solver, not a `Medium`.
- **Radial / quasi-1-D expansion.** Planar first. This is also precisely why
  G7 is ungated and expected high. **Retired (M6d).** Axisymmetric `(r, x)`
  hydro landed in `src/euler2d.rs`, and the relief deficit is measured and
  pinned at 23 % of the front speed — see [M6D_SPEC.md](M6D_SPEC.md). The bullet
  stays here, struck through rather than deleted, so the record of what M6c did
  not do remains readable.
- **Runtime Mutation++ FFI.** Offline tabulation only (D8/P3). Re-opened only if
  the frozen LTE table provably fails.
- **Non-LTE / two-temperature plasma**, and finite-rate ionization kinetics.
  M6a's rate equation handles the ignition; past ignition this is equilibrium.
- **Radiation transport** (radiative losses, radiation-driven front propagation).
  Its absence is one of the stated reasons G7 is expected high.
- **LSC and LSR regimes.** LSD only; the regime is asserted, not assumed.
- **Recombination / afterglow / multi-pulse.**
- **2-D/3-D hydro.** The propagator is 3-D; the hydro is not, by construction.
  **Narrowed to 3-D by M6d**, which made the hydro axisymmetric. 3-D remains out
  of scope: axisymmetry assumes no azimuthal structure.

## Open questions

1. **Which measured LSD dataset anchors G7.** *Inherited by M6d and still
   open — and now the **only** thing keeping G7 ungated.* The M6c counterpart of
   M6a's digitized anchor, and the one input this spec cannot supply itself. Needs
   the same treatment
   `tests/data/tt2012_*.csv` got: a named paper, the figure number pinned in the
   provenance header at digitization time, and the setup quoted so the solver's
   inputs are fixed by the source rather than chosen. Until then G7 compares
   against a literature-quoted band.
2. **Constant-`γ` vs table EOS for G3.** *Resolved (step 4): constant-`γ`, as
   spec'd.* G3 runs the `GreyThreshold` absorption model too — a fixed `α` above
   an internal-energy threshold — so that the residual it measures is the
   solver's and carries nothing from the property table's interpolation.
3. **Sub-cycling default** — set by the convergence check, not chosen.
   *Resolved (step 6): the knob is **not** landed, and the reason is structural
   rather than a deferral of effort.* `propagation_every` exists to amortise
   re-solving the 3-D propagation against a plasma state that has moved. In this
   geometry the plasma state barely feeds back: everything upstream of the front
   is cold transparent air, so the front sees the full incident `S` no matter
   what the column behind it is doing, and the column's effect on the beam is
   *entirely* downstream of the front, where nothing remains to be driven. The
   coupling is one-way, and the convergence check the spec asked for would
   therefore measure nothing — it would pass at any `propagation_every` for a
   reason that has nothing to do with the sub-cycling being adequate.

   What the propagator genuinely adds is **shielding**, and step 6 lands that as
   a gate instead: G8 marches a real beam through `PlasmaColumn` and holds it to
   `exp(−τ)` (see Gates). The one thing that *would* make the loop two-way is
   diffraction — a front climbing out of a converging beam's focus sees a
   falling drive and decelerates — and that is a quasi-1-D approximation this
   document has never specified and nothing gates. It belongs to a later
   milestone with its own gate, not smuggled in under a performance knob.
4. **Where the front is initialized.** *Resolved (steps 4 and 6): both, as
   anticipated.* `SeededIgnition` is implemented and is what G3 and G4 use, so
   the velocity gates carry nothing from M6a's ungated absolute threshold. The
   `AirBreakdown`-triggered path landed with the `lsd` CLI case in step 6,
   along with the inherited-limitation note.

   **What that produced was not anticipated, and is now the case's headline.**
   Putting both models in one run forces the question "does the beam that drives
   the detonation also light it?", and the answer the two give together is *no,
   by five orders of magnitude*. M6a's threshold in air at 1 atm saturates at
   ≈1.14×10¹⁶ W/m² and **does not fall with pulse length** — 6 ns and 1 ms give
   1.18×10¹⁶ and 1.14×10¹⁶ — because below it the inelastic losses exceed the
   inverse-bremsstrahlung heating and the net cascade rate is negative. It is an
   intensity floor, not a fluence one, and widening the focus does not move it
   either (4 % over a 500× range of spot radius). The sustaining LSD drive is
   ~10¹¹ W/m². So the wave must be *initiated* by something far brighter than
   what *sustains* it — which is the known experimental situation, where LSD
   waves in clean air are started on a target, on an aerosol, or by a separate
   spike. M6a's ungated absolute level does not touch the conclusion: the gap is
   10⁵ against a ~7× uncertainty.

   **Amended (M6a, 2026-07-30/31): the numbers above are the ones this milestone
   landed with, and the shape of the claim has since changed.** The
   distribution-resolved closure removed the hard `ε_∞ = U_i` cutoff, so a longer
   pulse now buys *something* rather than nothing, and seed production raised the
   short-pulse end. The threshold is therefore **asymptotic rather than flat**:
   8.815×10¹⁵ at 6 ns falling to 6.745×10¹⁵ by 1 ms, a bounded 1.31× fall that is
   flat to 1 % over the last two decades. The focus figure is now 6 % over a 500×
   range, and the level uncertainty is 3.90–4.69× rather than ~7×. **The
   two-stage argument is untouched**, which is the only thing this step depended
   on: a bounded fall to a floor is still an intensity criterion, and the drive
   still sits ~10⁵ below it. What would break it is a threshold falling without
   limit, and the gate below asserts that it does not. Current numbers live in
   `docs/MODELS.md` § "The `lsd` demonstration run". Pinned by
   `the_sustaining_drive_is_far_below_the_breakdown_threshold`.
5. **How hot the run is allowed to get** — *new, and answered (step 4).* The
   `Z̄ ≡ 1` ceiling recorded under "Property closure" bounds where `α_IB` can be
   trusted, and the CJ state behind a strong LSD front (1.5–2.5×10⁴ K)
   legitimately crosses it. Rather than extend the thermodynamic database —
   which would mean hand-entering N⁺⁺/O⁺⁺ spectroscopic data that no shipped
   Mutation++ database contains, and which would then itself need a validation
   gate against published equilibrium-air composition — the ceiling is
   *enforced*: `IonizationCeiling::Refuse` (the default) bails naming the
   temperature, `::Flag` proceeds and records it on the run. An unquantified
   bias becomes an explicit boundary. Revisit only if a gate is found to need
   the range, which G3 and G4 do not: they run constant-`γ` with a grey
   absorber and never query the table.

## Implementation order

1. `src/euler1d.rs` — HLLC + MUSCL-Hancock, no laser physics. Gates G1, G2.
2. **T4** — extract `physical_intensity_scale`, refactor blooming to it. Gate G0.
3. Plasma table + `src/plasmaprops.rs`. Gate G6.
4. `src/lsd.rs` — deposition, `PlasmaColumn` `Medium`, the driver. Gates G3, G5.
   **Landed**, with the four amendments marked above (the Beer–Lambert sign, the
   conservative deposition discretisation, G3's refinement knob, and
   `index_perturbation`). `LsdColumn` is the 1-D coupled driver — hydro,
   Beer–Lambert attenuation, Strang-split deposition — which is what the closed
   forms are written for; `PlasmaColumn` exposes its `α(x)` to the propagator.
   The propagator↔hydro outer loop is not here (see open question 3).
5. Scaling sweep. Gate **G4**. **Landed**, with the three amendments recorded
   under G4 (the EOS-independence leg is a `γ` sweep, the density sweep holds
   ambient temperature fixed, and the ignition threshold is a multiple of
   ambient `e₀`).
6. `lsd` CLI case + `scripts/render_lsd.py`; `MODELS.md` updated in the same
   change (equation, site, gate numbers, references), including the inherited
   M6a level limitation and the G3-is-verification labelling. **Landed**, with
   three things this document owes the reader.

   (a) **G8 is new, and was not in the pre-spec.** D7's central claim — the
   propagator sees the plasma as pure absorption and nothing else — was carried
   by `PlasmaColumn`'s unit tests, which exercise its `Medium` methods in
   isolation; no field had ever been marched through one. Step 6 needed that
   path for the demonstration, found it ungated, and gated it.

   (b) **The sub-cycling knob is not landed, on purpose** — see open question 3,
   which step 6 answers by showing the coupling is one-way in this geometry.

   (c) **The demonstration runs the grey closure, and reports what the
   production one would say.** Evaluated at the run's own post-front state, the
   inverse-bremsstrahlung closure gives `α ≈ 6.8 1/m` at 1064 nm — the whole
   2.5 cm column is 0.17 optical depths, so there would be no front at all and
   `check_regime` would correctly refuse it as volumetric — against
   `α ≈ 1.1×10³ 1/m` at 10.6 µm, an absorption length of 0.92 mm that is 92
   cells on the demo grid and 3.7 % of the domain. Free-free absorption falls
   steeply toward short wavelengths, so **this is the model reproducing why LSD
   experiments are done with CO₂ lasers.** What blocks running that closure
   coupled is cost and it is specific: `PlasmaTable::temperature` bisects ~45
   times per cell per deposition call, and the driver deposits three times per
   step. That is a faster table inversion, not a finer grid, and it is a
   separate change with its own gate.

## References

- Yu. P. Raizer, *Laser-Induced Discharge Phenomena*, Consultants Bureau (1977)
  — LSD wave theory, the detonation analogy, and the velocity closed form.
- Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991) — plasma absorption,
  inverse bremsstrahlung.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer (2009) — HLLC, MUSCL-Hancock, the Sod exact solution.
- G. Strang, SIAM J. Numer. Anal. **5**, 506 (1968) — the operator splitting
  already used by the propagator.
- Mutation++ (VKI) — equilibrium thermochemistry for the plasma-range table
  (D8: trusted source, tabulation gated).
- `docs/M6A_SPEC.md` — the ignition trigger and the D5 discipline this spec
  inherits.
