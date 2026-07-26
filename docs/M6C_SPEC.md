# M6c pre-spec — laser-supported detonation wave (1-D Euler + laser deposition)

Written **before** any M6c code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md), [M6A_SPEC.md](M6A_SPEC.md)): pin the gas-dynamic
model, the deposition closure, the coupling cadence, the property table, and —
most importantly — *which* checks validate physics versus which are solver
verification. M6a's hard lesson (D5) was that a gate anchored to the same
source the model is built from proves nothing; that lesson is applied here
before the first line of code, not after. If any of this proves wrong during
implementation, amend this document first, then the code.

Scope: the deep rung of the M6 ladder (design doc `giuseppe-main-design-20260723`,
Eng Review Outcome 2026-07-23, decision **D7**). The beam ignites a spark at the
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
   Drude `δn`. This sidesteps the near-critical failure that broke M6b as
   specified, and never touches `check_phase_sampling`.
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

So "gate the LSD velocity against C-J theory", as the design doc phrased it,
and "anchor to Raizer" are the **same closed form**, not two. A solver that
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
through the column by Beer–Lambert, `dI/dx = +α_pl·I` marching from the laser
side, so energy is conserved between the two halves by construction (gated:
G5).

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
  filled from the current hydro state. `index_perturbation` is the unreachable
  stub, exactly as `ThermalBlooming` does it today.
- **`physical_intensity_scale` (T4/D4) is required here.** The driver must turn
  `on_step`'s `|u|²` into W/m² both to test the M6a trigger and to size the
  deposition. D4 scheduled this extraction for "when M6a.2 first needs it",
  written before the D7 reroute; post-reroute **M6c is the first consumer**, so
  T4 lands with the driver, still under its CRITICAL guard: M4's blooming gate
  byte-for-byte green.
- **Sub-cycling.** Re-solving the propagation every hydro step is the dominant
  cost. The driver takes `propagation_every: usize`; the spec requires a
  convergence check that the front trajectory is insensitive to it (halving it
  moves `D` by less than the G3 tolerance), and the default is whatever that
  check supports — not a guess.

### Inherited limitation: the ignition trigger carries M6a's ungated level

The velocity gates are independent of M6a. The **ignition time and position**
are not: they come from `AirBreakdown`'s absolute threshold, which is M6a's
explicitly ungated quantity (4.8–7.0× above the measured T&T curve, inside the
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
match `D = [2(γ²−1)S/ρ₀]^(1/3)` within a pinned tolerance, and the residual must
shrink under grid refinement. **Labelled in the code and in `MODELS.md` as
solver verification**, for the reason given in § "The circularity". This is the
M6c analogue of M6a's "closed-form sub-gates are integrator unit tests".

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

### G5 — energy budget closure (verification)

Laser energy absorbed = Δ(internal + kinetic) in the domain + flux through the
boundaries, to ~1e-10 relative. M4's closed power budget, one dimension down.

### G6 — plasma-table tabulation consistency (D8)

Frozen table vs direct Mutation++ at sampled `(T, p)`, within a stated
tolerance, with the ionization onset explicitly among the samples.

### G7 — absolute velocity vs measurement: DOCUMENTED, UNGATED

The 1-D planar solver is **expected to land high** against measured LSD
velocities, and the spec says so in advance rather than discovering it:

- A planar 1-D code has **no radial relief**. If the dominant reason measured
  LSD speeds fall below the CJ prediction is lateral rarefaction of a
  finite-diameter beam, a 1-D model structurally cannot reproduce it — it is the
  one effect the geometry has removed by assumption.
- Radiation losses and incomplete absorption push the same direction and are
  also out of scope (§ NOT in scope).

So the honest claim M6c can make is: **the 1-D model agrees with CJ/Raizer where
that theory applies, and the gap to experiment is in the predicted direction and
of the predicted order, for reasons the model has explicitly excluded.** That is
"expect and explain the ~½·v_CJ result" as the design doc asked, sharpened: the
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

- **Full-Drude M6b plasma shielding.** Deferred, and this is its OSS-repo home
  (D7, D9). The reason, recorded so it is not rediscovered: M6b's own `δn` clamp
  fires at `n_e > 0.1·n_crit ≈ 2×10²⁴ m⁻³`, which is *inside* the breakdown
  density range M6a targets (~10²³–10²⁴ m⁻³). A paraxial split-step envelope
  cannot carry a near-critical plasma — the phase-per-slab and the sampling
  criterion both fail — so full-Drude M6b is broken **as specified**, not merely
  unimplemented. Reviving it needs a non-paraxial treatment, which is a
  different solver, not a `Medium`.
- **Radial / quasi-1-D expansion.** Planar first (design-doc open question 4).
  This is also precisely why G7 is ungated and expected high.
- **Runtime Mutation++ FFI.** Offline tabulation only (D8/P3). Re-opened only if
  the frozen LTE table provably fails.
- **Non-LTE / two-temperature plasma**, and finite-rate ionization kinetics.
  M6a's rate equation handles the ignition; past ignition this is equilibrium.
- **Radiation transport** (radiative losses, radiation-driven front propagation).
  Its absence is one of the stated reasons G7 is expected high.
- **LSC and LSR regimes.** LSD only; the regime is asserted, not assumed.
- **Recombination / afterglow / multi-pulse.**
- **2-D/3-D hydro.** The propagator is 3-D; the hydro is not, by construction.

## Open questions

1. **Which measured LSD dataset anchors G7.** The T2-equivalent for M6c, and
   the one input this spec cannot supply itself. Needs the same treatment
   `tests/data/tt2012_*.csv` got: a named paper, the figure number pinned in the
   provenance header at digitization time, and the setup quoted so the solver's
   inputs are fixed by the source rather than chosen. Until then G7 compares
   against a literature-quoted band.
2. **Constant-`γ` vs table EOS for G3.** Spec'd as constant-`γ` above (the
   closed form assumes it); revisit if the table mode turns out to be the only
   configuration anyone runs.
3. **Sub-cycling default** — set by the convergence check, not chosen.
4. **Where the front is initialized.** Ignite from the M6a trigger at the focal
   plane (physical, inherits the ungated level) or from a seeded hot spot
   (clean, decouples the gates from M6a). Probably both: seeded for G3/G4,
   triggered for the demonstration run.

## Implementation order

1. `src/euler1d.rs` — HLLC + MUSCL-Hancock, no laser physics. Gates G1, G2.
2. **T4** — extract `physical_intensity_scale`, refactor blooming to it. Gate G0.
3. Plasma table + `src/plasmaprops.rs`. Gate G6.
4. `src/lsd.rs` — deposition, `PlasmaColumn` `Medium`, the driver. Gates G3, G5.
5. Scaling sweep. Gate **G4**.
6. `lsd` CLI case + `scripts/render_lsd.py`; `MODELS.md` updated in the same
   change (equation, site, gate numbers, references), including the inherited
   M6a level limitation and the G3-is-verification labelling.

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
