# M6d pre-spec — axisymmetric gas dynamics and radial relief (2-D Euler)

Written **before** any M6d code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md), [M6A_SPEC.md](M6A_SPEC.md), [M6C_SPEC.md](M6C_SPEC.md)):
pin the geometry, the discretisation, the axis condition, the beam model, and —
most importantly — *which* checks are verification, which are a pinned
measurement, and which would be validation if the data existed. If any of this
proves wrong during implementation, amend this document first, then the code.

Scope: M6c's hydro is planar 1-D. This milestone makes it axisymmetric `(r, x)`
so that a **finite-diameter beam** can relieve laterally, and then measures what
that does to the front speed.

The reason is specific and is written down in M6c. Its G7 — absolute LSD
velocity against measurement — is ungated, and [M6C_SPEC.md](M6C_SPEC.md)
§ G7 justifies that with one omission:

> A planar 1-D code has **no radial relief**. […] it is the one effect the
> geometry has removed by assumption.

M6c's "NOT in scope" says the same thing twice — "Radial / quasi-1-D expansion"
and "2-D/3-D hydro". **M6d retires exactly those two bullets.** After it, G7 is
still ungated, but for one reason only: there is no anchored measured dataset.
That is a different and much smaller claim, and it is the deliverable.

Conventions follow [MODELS.md](MODELS.md): SI units, `f64`. Symbols inherited
from M6c: `ρ`, `p`, `E`, `γ`, `c`, `D`, `S`, `α_pl`. New symbols: radius `r` (m),
radial velocity `u_r` (m/s), axial velocity `u_x` (m/s), beam radius `R_b` (m),
domain radius `R_dom` (m), on-axis front speed `D_2D` (m/s), **relief deficit**
`δ ≡ 1 − D_2D/D_wide` — amended from `D_1D` once the transverse instability was
found; see § Results — and front curvature `κ` (1/m).

## Gate decisions (recorded)

1. **Axisymmetric `(r, x)`, no swirl** (`u_θ ≡ 0`, `∂/∂θ ≡ 0`), discretised as
   an **area-weighted finite volume on annular cells** — *not* as Cartesian
   fluxes plus a `p/r` source. This is the decision the whole milestone rests on
   and § "The axis is not a special case" says why.
2. **A new `src/euler2d.rs`; `src/euler1d.rs` gains visibility changes and
   nothing else.** `hllc_flux` becomes a free `pub(crate)` function; `flux`,
   `minmod` and `is_positive` become `pub(crate)`. No change to `Euler1d`'s
   state, update loop, guards or public API. **CRITICAL guard:** every M6c gate
   number unchanged, and the `lsd` CLI artifacts byte-identical — verified by
   `shasum` against hashes recorded before the refactor, the recipe M6a's `Gas`
   split used.
3. **Strang, with the radial sweep outside**: `R(dt/2) → X(dt) → R(dt/2)`. Not
   an aesthetic choice — it is what makes the planar radially-uniform limit an
   *exact identity* in `R`, so `X(dt)` is bit-for-bit `Euler1d::step(dt)` and
   G9 can assert equality rather than a tolerance. Sweep order is **not**
   alternated between steps; that is 2nd order only on average and would make
   G11 read noise.
4. **`euler2d.rs` is laser-agnostic.** M6c gate decision 4, carried forward
   verbatim: the Riemann core has no laser physics in it, so its verification
   gates test a general-purpose solver against standard closed forms. Laser
   deposition lives one layer up, in `src/lsd2d.rs`.
5. **The beam is a bundle of independent parallel pencils** — one Beer–Lambert
   march per ring, no refraction, no diffraction. Making the beam bend in the
   radially structured plasma would make the coupling two-way, which
   [M6C_SPEC.md](M6C_SPEC.md) open question 3 already reserves for a later
   milestone with its own gate. It is not smuggled in here.
6. **The headline is verification plus one pinned measurement, not a dataset
   comparison.** M6d does not depend on acquiring a paper. G7 stays ungated;
   what changes is its justification. The measured-dataset debt is inherited and
   restated in § Open questions.
7. **Deferrals are recorded in this document's "NOT in scope"** (M6c D9). The
   repo has no `TODOS.md`.

## The circularity, stated up front

M6c had to say this about Raizer, and the same discipline applies here to two
new things.

**What is still circular.** Raizer's `D = [2(γ²−1)S/ρ₀]^(1/3)` is the
Chapman–Jouguet construction the deposition model is built from
([M6C_SPEC.md](M6C_SPEC.md) § "The circularity"). Nothing in M6d changes that,
so G14 — the wide-beam limit reproducing the 1-D column — is **verification**,
and so is any comparison of `D_2D` against `D_CJ` in the wide-beam limit.

**What is genuinely not circular, and is new.**

- **Sedov–Taylor.** A self-similar blast solution the model is *not* built from:
  no laser, no deposition closure, no CJ construction, and it shares only
  `Primitive`/`IdealGas` with the solver, per the independence rule at
  `src/validate.rs` § "M6c references". It is the repo's **first
  multidimensional verification anchor** — there is currently none — and it
  exercises both sweeps, the geometric source and the axis at once.
- **The relief deficit `δ`.** No closed form in this model predicts it. It is
  a measurement of the solver, pinned so its size cannot drift.

**One warning, recorded now so it cannot be forgotten later.** Condensed-phase
detonation theory has a *diameter effect*: `D` falls roughly as `D_CJ(1 − a/R)`,
with a failure radius below which the wave dies (Eyring; Wood–Kirkwood). It is
tempting to quote that as an anchor for `δ`. It is **not** one unless the
coefficient `a` comes from somewhere other than this solver. If `a` is fitted
here, the `1/R` law is a *description* of the measurement, not evidence for it.
This document therefore records the fit in a doc comment and **does not** put it
in `validate.rs`. Promoting it is § Open questions item 4, and it would need its
own honesty about what was fitted.

## Gas-dynamic model

### Governing equations

Axisymmetric Euler with a volumetric laser source, `x` along the beam axis (the
front propagating toward the laser, `−x`, as in M6c):

```text
∂U/∂t + (1/r)·∂(r·F_r)/∂r + ∂F_x/∂x = Ṡ_geom + Ṡ_laser

U   = (ρ, ρu_r, ρu_x, E)ᵀ
F_r = (ρu_r, ρu_r² + p, ρu_r u_x, (E + p)u_r)ᵀ
F_x = (ρu_x, ρu_r u_x, ρu_x² + p, (E + p)u_x)ᵀ
Ṡ_geom  = (0, p/r, 0, 0)ᵀ
Ṡ_laser = (0, 0, 0, q(r, x))ᵀ
E       = p/(γ−1) + ½ρ(u_r² + u_x²)
```

`Ṡ_geom` is the only non-conservative term. It is not an extra physical effect:
it is what is left over when the divergence `(1/r)∂(r·)/∂r` is written as a
plain derivative, because the *pressure* part of the radial momentum flux is not
divided by `r`.

### The axis is not a special case

The `1/r` above is why axisymmetric codes are reputed to be delicate. It is an
artefact of writing the scheme in the differential form. On annular
finite-volume cells it disappears.

Ring `j` spans `[r_{j−1/2}, r_{j+1/2}]`; interfaces carry area `A ∝ r`, cells
carry volume `V ∝ (r_+² − r_−²)/2`. The radial update is

```text
U_j ← U_j − (dt/V_j)·(A_{j+1/2}·F_{j+1/2} − A_{j−1/2}·F_{j−1/2})
          + dt·(0, p_j·(A_{j+1/2} − A_{j−1/2})/V_j, 0, 0)ᵀ
```

**The geometric source is written as `p_j·(A_{j+1/2} − A_{j−1/2})/V_j` — the same
expression as the flux term — and never as `p_j/r_j`.** Three consequences, each
of which turns a classic hazard into a structural non-issue:

1. **The axis interface has zero area.** `A_{1/2} = 0`, so nothing crosses
   `r = 0` by construction. The "`1/r` blows up on the axis" failure cannot
   arise, because there is no `1/r` anywhere in the code.
2. **The source is exactly the volume average of `1/r`.** Analytically
   `(A_+ − A_−)/V = 2/(r_+ + r_−) = 1/r_j` with `r_j` the arithmetic cell
   centre — finite at `r_1 = Δr/2`. The singularity is removed by the algebra,
   not by a floor or an epsilon.
3. **Well-balanced bit-exactly.** For a radially uniform state the pressure flux
   difference and the source are the same floating-point expression with
   opposite signs, so they cancel to *bit* precision, not to round-off. A
   radially uniform state is therefore an exact fixed point of the radial
   operator. This is what lets G9, G13(i) and G14 assert equality instead of a
   tolerance, and it is the single most valuable implementation detail in the
   milestone.

Mass, axial momentum and total energy are exactly conserved in the discrete
`r dr dx` measure. **Radial momentum is deliberately not conserved** — the
geometric source is real, and an implementation that conserved it would be
wrong. G12 asserts both halves of that sentence.

### The axis boundary, and the failure it invites

`r = 0` is a reflective boundary with **odd parity on `ρu_r`** and even parity on
`ρ`, `ρu_x`, `E`.

Getting this wrong — even parity on `u_r` — produces *wall heating*: a thin,
artificially hot and under-dense column on the axis that grows with time and
looks exactly like a physical result. It matters here more than in a generic
code, because the on-axis ring is precisely where this milestone's headline
number is measured. Two of the three usual causes (sampling `1/r` at a cell
centre; a source inconsistent with the face areas) are removed by the
formulation above. The parity one is removed by the ghost fill and **gated
directly**, with a deliberate even-parity contrast so the gate is non-vacuous
(G13).

### What relief is expected to do

The controlling dimensionless group is `R_b·α_pl` — the beam radius in units of
the deposition length. Relief acts by letting the shocked, heated gas expand
sideways out of the driving region before it has finished pushing the front, so
it bites when the transverse relief time is comparable to the time the gas
spends in the absorption zone.

At M6c's CJ state (`S = 10¹¹ W/m²`, `ρ₀ = 1.225 kg/m³`, `γ = 1.4`): `D` = 5.4
km/s, post-front sound speed `c₁ = γD/(γ+1)` = 3.1 km/s. With M6c's grey
`α_pl = 2×10⁴ 1/m` the absorption length is 50 µm, crossed by the front in
≈9 ns, while a rarefaction crosses a beam of radius `R_b` in `R_b/c₁`. Those are
comparable at `R_b ≈ 30 µm`, i.e. `R_b·α_pl ≈ O(1)`.

Three predictions follow, and G15 tests them as predictions rather than fitting
them:

- `D_2D < D_1D` always, and `δ` decreasing in `R_b`.
- `δ → 0` as `R_b·α_pl → ∞` (G14 is that limit, taken to its extreme).
- There is a **failure radius** below which the wave does not sustain at all.

It also follows that the front is **curved** — leading on the axis, lagging at
the beam edge — and that curvature *is* relief made visible. It is a diagnostic
and a figure, not a gate.

**A practical consequence worth recording, because it sets the cost.** `R_b`
must be within a couple of absorption lengths for relief to be measurable at
all, so `R_b` is tens of microns and the relief time `R_b/c₁` is tens of
nanoseconds — two orders below M6c's `LSD_SETTLE` of 1.8 µs. Relief equilibrates
long before the front speed has settled from the seed, which means the 2-D runs
need a *short* column, not a long one. That is what keeps them affordable.

### Equation of state

Verification mode only for every gate in this document: ideal gas at constant
`γ`, as M6c's G1–G3 use. The production table EOS is not exercised by M6d —
see § NOT in scope, and M6c's own note that the EOS difference is a ~25 % shift
in `D` and a wrong reason to expect the experimental gap to close.

### Validity checks asserted at run start

In the M4 Péclet spirit — refuse, don't mis-model. `src/lsd2d.rs` inherits every
check in `LsdColumn::check_regime` (`src/lsd.rs`) and adds three:

- **The beam is radially resolved.** At least `MIN_CELLS_PER_BEAM_RADIUS` cells
  across `R_b`. Below that the deposition profile is a staircase and `δ` would
  be a mesh artefact rather than a measurement.
- **The domain is wide enough.** `R_dom` at least a stated multiple of `R_b`,
  *and* the outermost ring undisturbed. Otherwise the outer wall is doing the
  relief and the number means nothing.
- **The front is still in the domain**, on the axis and at the beam edge. A
  curved front can leave through `x_min` at the axis while the edge is still
  inside, and the on-axis measurement would silently be reading the boundary.

## Numerics

- **HLLC** and **MUSCL-Hancock with minmod**, reused from `src/euler1d.rs`
  unmodified (see below). CFL ≤ 0.8, asserted per step from the current wave
  speeds in **both** directions:
  `dt = CFL / max( max(|u_r|+c)/Δr , max(|u_x|+c)/Δx )`, refused rather than
  quietly reduced.
- **Positivity guard after every stage: bails, never clamps** — M6c's rule,
  M6a's `n_e`-runaway lesson. The bail names the cell `(j, i)`, its physical
  `(r, x)`, the step, **and the stage**: `r-sweep`, `x-sweep`, or
  `geometric source`. That last is a genuinely new failure mode — the geometric
  source adds radial momentum without changing `E`, so it raises kinetic energy
  at fixed total energy and can drive `p < 0` in the innermost rings of a strong
  converging flow.
- **Ghost cells**: `N_GHOST = 2` per side per direction, as the MUSCL stencil
  needs, materialised per sweep exactly as `Euler1d` does.

### The 1-D Riemann solver is reused, not reimplemented

A 2-D sweep carries a transverse momentum component that `Conserved` does not
have. It does **not** need a new Riemann solver. In HLLC the transverse velocity
is constant across the acoustic waves within each star state, so the transverse
flux is

```text
F_{ρv} = F_ρ · v_K,     K = L if S* ≥ 0 else R
```

and the supersonic branches (`S_L ≥ 0`, `S_R ≤ 0`) satisfy the same identity
because they return the physical flux of one side. So a sweep packs
`(ρ, ρu_∥, E)` into the existing `Conserved`, calls the existing `hllc_flux`,
and recovers the transverse flux with one multiplication by the upwind
transverse velocity.

**Zero duplicated Riemann code** — which is the point. The alternative,
generalising `Conserved` over a component count, would touch every line of the
flux, reconstruction and update path, i.e. exactly the code M6c's gates measure,
for no capability this identity does not already give free.

## Beam ↔ plasma coupling

```text
per hydro step dt:
  1. per ring j: march I_j(x) through the current α_pl(r_j, x) — Beer–Lambert
  2. deposit q(r, x) = (I_k − I_{k+1})/Δx, discretely conservative per ring
  3. Strang: deposit(dt/2) → R(dt/2) → X(dt) → R(dt/2) → deposit(dt/2)
  4. recompute α_pl(r, x) from the new state
```

What does **not** change from M6c, and is reused rather than reimplemented:
`Absorption` (including `GreyThreshold`, and M6c's reasoning for why the
verification gates run the grey closure), `IonizationCeiling`,
`raizer_lsd_velocity`, and the discretely conservative deposition
`q_k = (I_k − I_{k+1})/Δx` that lets the budget close to round-off instead of to
a quadrature tolerance.

What is new:

- **Per-ring marches.** `I_j(0) = S·b(r_j)` for a beam profile `b`. Because each
  ring's march is independently conservative, the `r`-weighted budget still
  closes to round-off.
- **`R_b` sits on a cell face**, so a top-hat has no partial cell and `δ` cannot
  pick up an edge-quantisation artefact.
- **Relief comes from the beam being finite, not the domain.** With `R_dom` well
  outside `R_b` the lateral rarefaction is entirely interior, so
  `boundaries_undisturbed` generalises verbatim (both end planes plus the
  outermost ring) and there is no boundary-flux accounting at all.
- **An escape accumulator** for the long demonstration runs where relief *does*
  reach `r_max`: `escaped_energy()` sums `∮(E+p)u_r dA` per step, so
  `deposited = ΔE + escaped`. The escaped fraction is itself the physically
  meaningful number — it is literally the energy relief carries away — and it
  is what makes G12's second leg possible.
- **The seed gains a radius.** That is a second free parameter beside its
  pressure, so M6c's seed-independence discipline (G3c) is extended to it.
  Otherwise the headline number sits on an unexamined knob.

### Inherited limitation: the beam does not bend

The pencils are straight. A real beam crossing a radially structured plasma
refracts, and near the front it would be defocused by the density depression.
This is stated, not modelled (decision 5), and it is the second reason — after
the missing dataset — that M6d does not claim to close G7.

## Gates

Numbering continues from M6c's G1–G8. Following M6a's and M6c's discipline, each
gate is labelled by what it actually establishes. **Verification** = "the code
solves the equations I wrote down". **Validation** = "the equations describe the
world". **Pinned** = "a known departure, asserted green so its size cannot
drift".

M6d contains **no validation gate**, and that is a deliberate consequence of
decision 6 rather than an oversight.

### G9 — the planar limit reproduces `Euler1d` bit for bit (verification)

`euler2d_planar_limit_reproduces_euler1d_bit_for_bit`

Sod data uniform in `r`, `Geometry::Planar`, marched alongside an `Euler1d` of
the same `n_x`, `dx` and CFL. Assert `==` on every component of every cell at
every step.

Two non-vacuity legs, both required: perturbing one ring by 1e-12 must make the
comparison **fail** (the test can see a difference), and an axisymmetric run with
a radial gradient must diverge measurably (the geometric source is switched on).

This is also the **standing guard** on decision 2: it keeps proving the reused
HLLC has not drifted long after the one-time byte comparison has scrolled away.
Cost: milliseconds.

### G10 — Sedov–Taylor point blast (verification)

`sedov_blast_matches_the_self_similar_solution`

**Which Sedov, and why.** In `(r, x)` axisymmetric coordinates a *point* deposit
gives the **spherical** blast (`ν = 3`, `R ∝ (E/ρ₀)^{1/5} t^{2/5}`); a *line*
deposit uniform in `x` gives the cylindrical one (`ν = 2`) and degenerates to a
purely radial problem that exercises one sweep. **The point blast is gated**: it
is the only problem here that drives both sweeps, the geometric source and the
axis simultaneously.

Three legs:

- **Exponent (parameter-free, and the tightest).** Fit `log R` against `log t`
  over the self-similar window with `validate::loglog_slope_xy` and gate the
  slope inside ±0.01 of 2/5. No `ξ₀` enters, so there is no constant to get
  wrong. This is G4's idea applied to a geometry gate.
- **Level.** `R(t) = ξ₀·(E t²/ρ₀)^{1/5}` against the published `ξ₀` for
  `γ = 1.4`, at a few percent — a finite-size initial deposit is only
  asymptotically self-similar and a coarse mesh under-resolves the shock.
- **Jump.** The immediate post-shock density ratio against strong-shock
  Rankine–Hugoniot, `ρ₂/ρ₁ = (γ+1)/(γ−1) = 6`: a closed form with no constant
  in it at all.

The reference is a full parametric Sedov solution in `src/validate.rs`, with its
own unit tests (published `ξ₀`, the exponent identity, the jump, and the energy
integral returning `E`). CI runs a coarse grid; the L1-under-refinement ladder is
measured once and recorded in the gate's doc comment, per the out-of-band
practice G4's `γ` table already uses.

### G11 — 2nd order on smooth axisymmetric flow (verification)

`euler2d_is_second_order_on_smooth_axisymmetric_flow`

Self-convergence against a fine reference, refining `(Δr, Δx, Δt)` together at
fixed CFL — the structure of M6c's `coupled_pressure_profile` /
`restrict_profile` / `coupled_orders`, in two dimensions.

Made affordable by choosing a problem that is **r-structured and x-uniform**
(a smooth radial pressure blob, `n_x` small). It still exercises the r-sweep,
the axis and the geometric source — the only new terms — while the 8× reference
costs 8×, not 64×. The x-sweep's order is already G2, on unchanged code.

**Non-vacuity is mandatory**, mirroring G2b: a contrast run applying the
geometric source Godunov-style outside the sweep must read ≈1. Without it, a
measurement that cannot tell 1st from 2nd order passes silently.

### G12 — conservation in the `r`-weighted measure (verification)

`euler2d_conserves_mass_and_energy_in_the_r_weighted_measure`

Closed box, no source: `∫ρ r dr dx` and `∫E r dr dx` constant to ~1e-13. With
the laser on and the boundaries undisturbed: `ΔE = deposited` to ~1e-10 — G5's
2-D twin.

Non-vacuity, two legs. Axial momentum must **also** be conserved, and radial
momentum must **not** be — a version that conserved it would be wrong, and this
is the gate a naive `−G/r` source fails. Second: with the outer wall far out,
`escaped ≈ 0` and the budget closes; with it close in, the budget closes **only**
when the escape term is included. That is what proves the accounting rather than
assuming it.

### G13 — the axis is not a wall (verification)

`the_axis_boundary_does_not_heat_or_starve_the_on_axis_cells`

- **Leg (i)**, an in-module unit test mirroring `euler1d`'s
  `uniform_flow_is_a_fixed_point`: a uniform state with uniform axial flow is a
  **bit-exact** fixed point of the *axisymmetric* operator. Well-balancedness in
  one assertion.
- **Leg (ii)**, on the G10 run: on-axis entropy `p/ρ^γ` deviates from the
  neighbouring rings by less than a pinned bound, and that bound **falls** under
  radial refinement.

**Non-vacuity:** an even-parity `u_r` ghost fill must break leg (ii) loudly.
Without that contrast the gate proves nothing.

### A cylindrical Sod is deliberately not gated

Recorded so a reader does not wonder why it is missing. A radial Riemann problem
is self-similar in `r/t` but has **no closed form**, so it could only be a
self-convergence check — which G11 already provides, more cheaply and with a
non-vacuity contrast. Adding it would grow the suite without adding an anchor.

### G14 — the wide-beam limit reproduces the 1-D column (verification)

`lsd2d_with_a_full_width_beam_reproduces_the_one_dimensional_column`

A beam wider than the domain, radially uniform: the axisymmetric column must
reproduce `LsdColumn`'s front speed to round-off.

**This must land before G15.** It is G15's non-vacuity partner: it proves that
any deficit G15 measures is relief, and not an artefact of the geometric source,
the ring binning, or the front tracker.

### G15 — radial relief lowers the front speed by a pinned amount (**pinned**)

`radial_relief_lowers_the_lsd_front_speed_by_a_pinned_amount`

**The milestone's headline.** M6c's G3 configuration exactly — same `S`, `ρ₀`,
`γ`, `α`, seed multiple and settle discipline — except the beam has a finite
radius. Measure the on-axis front speed and report `δ = 1 − D_2D/D_1D`.

Three legs:

1. **Sign and monotonicity**, as predictions rather than fits: `D_2D < D_1D`
   always, and `δ` decreasing in `R_b`.
2. **The pinned number**: `δ` at a named `(R_b, α_pl)`, with a band. This is the
   ledger row.
3. **The failure radius**: below some `R_b` the wave does not sustain. Its
   existence and its value are pinned.

Before anything is pinned, three insensitivities must be demonstrated, because
each of them is a way for this number to be measuring something else:

- **grid** — `δ` converged under refinement in both directions;
- **seed** — `δ` insensitive to seed pressure *and* seed radius (G3c extended);
- **threshold** — `δ` insensitive to the `GreyThreshold` ignition multiple. In
  2-D the beam edge can cool below the threshold and the plasma edge retreat, so
  the wave can self-narrow for reasons belonging to the threshold rather than to
  relief. This is G4's amendment-3 problem in a new geometry. **If `δ` is
  sensitive to it, the number is not measuring relief and must not be pinned.**

Status **pinned**: not `verified`, because no closed form predicts it; not
`validated`, because nothing measured is being compared to.

### G16 — the one-third scaling survives radial relief (verification)

`the_one_third_scaling_survives_radial_relief`

M6c's G4 argues that relief can only enter as a *coefficient*, and that no
coefficient can produce a 1/3 exponent. In M6c that was an argument. Here it is
testable: sweep `S` over a decade at fixed `R_b` and check the exponent is still
≈1/3 while the level has moved by `δ`.

Two to four runs, and it upgrades the strongest claim the project owns from an
argument to a measurement.

### G7 — absolute velocity vs measurement: still UNGATED, for a smaller reason

M6d does not close G7 and does not claim to. What it changes is the
justification. After this milestone, "the geometry removed the effect" is no
longer available: relief is modelled, and its size is pinned. G7 remains ungated
because **there is no anchored measured dataset** — [M6C_SPEC.md](M6C_SPEC.md)
open question 1, inherited here unchanged.

Whatever `δ` turns out to be gets written into M6c's G7 section and into
`MODELS.md`. In particular, **if `δ` accounts for only a fraction of the ~2× gap
to measurement, that is the result**, and the remaining candidates — radiation
losses, incomplete absorption, the production EOS, and the un-bent beam — are
named there rather than left implicit.

## Failure modes (new codepaths)

| Failure | Detection | Response | Silent? |
|---|---|---|---|
| Axis ghost parity wrong → on-axis wall heating | G13 leg (ii) + its even-parity contrast | gate fails | no |
| Geometric source written as `p/r` → not well-balanced | G13 leg (i) (bit-exact fixed point) | gate fails | no |
| Geometric source applied outside the sweep | G11 reads ≈1 against the contrast | gate fails | no |
| Geometric source drives `p < 0` in the inner rings | positivity guard, stage-named | **bails**, names `(j, i)`, `(r, x)`, step, stage | no (loud) |
| Beam radially unresolved → `δ` is a mesh artefact | `check_regime` cell-count check | bails | no |
| Domain too narrow → the wall does the relief | `check_regime` + `boundaries_undisturbed` | bails | no |
| Escape flux unaccounted → budget silently open | G12 second leg | gate fails | no |
| Sweep order slipped (R and X swapped) | G9 bit-identity | gate fails | no |
| Curved front leaves the domain on the axis | `check_regime` front-in-domain check | bails | no |
| CFL taken from one direction only | G9 (planar) stays green but G10/G11 destabilise | loud crash or gate failure | no |
| M6c regression during the refactor | byte comparison of `lsd` artifacts + full suite | caught at step 1, before any 2-D code exists | no |

## NOT in scope (M6d)

- **Beam refraction and diffraction in the plasma.** Decision 5. The pencils are
  straight; a two-way beam↔plasma loop is a later milestone with its own gate,
  as [M6C_SPEC.md](M6C_SPEC.md) open question 3 already says.
- **3-D hydro.** Axisymmetry assumes no azimuthal structure. A tilted or
  astigmatic beam, or an azimuthal instability of the front, is outside this.
- **Swirl** (`u_θ`). Zero by assumption.
- **The production table EOS in the 2-D hydro.** Verification mode (constant
  `γ`) only, so every M6d gate measures the solver rather than the table's
  interpolation. M6c's G6 already gates the table itself.
- **Non-LTE, two-temperature plasma, finite-rate ionization** — inherited from
  M6c, unchanged.
- **Radiation transport** — inherited from M6c, unchanged, and still one of the
  named reasons G7 is expected high.
- **The M6a ignition stage in the 2-D case.** `lsd2d` seeds directly. The `lsd`
  case already owns the ignition-is-not-sustaining story, and seeding keeps
  `lsd2d` from re-inheriting M6a's explicitly ungated absolute threshold.
- **Python bindings.** Every case since bindings v2 has one; `lsd2d` will not,
  until a bindings v3 change adds it deliberately rather than as a side effect.
- **The measured LSD dataset for G7** — see § Open questions.

## Open questions

1. **Which measured LSD dataset anchors G7.** Inherited verbatim from
   [M6C_SPEC.md](M6C_SPEC.md) open question 1, and **still open**. It needs the
   treatment `tests/data/tt2012_*.csv` got: a named paper, the figure number
   pinned in the provenance header at digitization time, and the setup quoted so
   the solver's inputs are fixed by the source rather than chosen. M6d makes
   this debt *more* worth paying, not less — with relief modelled and pinned,
   the comparison would finally be against a model that contains the effect the
   comparison is about.
2. **Does relief explain the experimental gap?** Answered by this milestone,
   either way, and the negative answer is as publishable as the positive one.
3. **Full Sedov profile, or trajectory plus jump only?** Provisionally the full
   parametric profile, because it gives an L1-under-refinement leg in the G1
   style and is the repo's only multidimensional reference. Revisit if the
   algebra proves disproportionate.
4. **Is the `1/R` diameter-effect law an anchor or a description?** A
   description, for now, recorded in a doc comment. Promoting it to
   `validate.rs` requires the coefficient to come from outside this solver —
   see § "The circularity, stated up front".
5. **Top-hat or super-Gaussian beam?** *Resolved as spec'd.* Top-hat for the
   gates, because `R_b` is then unambiguous and the diameter effect reads
   cleanly; `--beam-order` offers a super-Gaussian for the demonstration run,
   where the profile is a picture rather than a measurement.
6. **Where is the failure radius?** New, and open. The eight-cells-across-`R_b`
   guard puts the smallest affordable beam at `R_b·α` = 1.6, where δ = 0.305 and
   the wave is healthy. Reaching the failure radius needs `Δr` decoupled from
   `Δx`, which costs CFL. Worth doing: it is the sharpest qualitative prediction
   the diameter effect makes.
7. **What sets the transverse instability's wavelength, and does it change the
   mean front speed?** New, and open. M6d establishes the instability exists and
   separates it from relief; it does not characterise it. Cell size, growth
   rate, and whether the saturated state runs slower than the smooth one are all
   unmeasured, and the last of those would bear directly on G7.

## Results (filled in as the milestone landed)

Recorded here so the spec is not left describing an intention. Every number
below is in a gate's doc comment with the run that produced it.

- **G9** bit-identical; **G10** exponent 0.38628 vs 2/5, `ξ₀` derived at 1.03278
  against the published ≈1.033; **G11** 1.861/1.964 vs a 1.030/1.155 contrast;
  **G12** <1e-13; **G13** 2.99e-7 → 3.12e-8 vs 3.02e-6 broken; **G14** 3.1e-13
  in the smooth window; **G15** δ = 0.230 at `R_b·α` = 3.2, 0.305 at 1.6;
  **G16** `S^0.34666`.
- **Two implementation defects were found by these gates and are recorded with
  their measured before-numbers**: ghost cells borrowing a neighbour's face
  areas leaked 3.3e-4 of the mass (a reflective wall is only exact when the
  ghost's *metric* mirrors too), and the Hancock predictor built its geometric
  source from face rather than cell pressures, which is well-balanced and costs
  an order (0.86/1.12 against planar's 1.71/1.89).
- **The unlooked-for result: the front is transversely unstable.** Not
  anticipated by this spec. See § G14 and `docs/MODELS.md` § M6d.

### Amendments this document owes the reader

1. **G15's reference changed from the 1-D column to the wide-beam 2-D run.**
   The original text compares `D_2D` against `D_1D`. That is wrong now that the
   instability is known: it is present at *every* beam radius including
   infinite, so a 1-D reference reports it as relief. Both runs in G15 carry it,
   and it cancels.
2. **G15 pins a band, not a value.** The spec asked for "the pinned deficit `δ`
   at a named `(R_b, α_pl)`, with a band". The band is ±13 %, and its width was
   measured — grid +6 %, seed −7 %, ignition threshold ±8 % — rather than
   chosen. A tighter pin would assert precision three knobs say is not there.
3. **No failure radius is gated.** Predicted, and not reached: the eight-cells-
   across-`R_b` guard means the smallest affordable beam still carries a healthy
   wave. Open item, below.
4. **`boundaries_undisturbed` was split.** A radially uniform seed disturbs the
   rim at `t = 0` by construction, and the seed's own blast always leaves
   downstream, so a single flag cried wolf on every run. The laser-side plane is
   the validity condition; the rim and the downstream plane are information. The
   question the rim flag was standing in for — is the wall doing the relief? —
   is answered directly instead: δ = 21.1/21.3/21.3 % at 3/5/8 beam radii.

## Implementation order

0. This document, and the README `M6d.0` row. **Record the baseline `lsd` CLI
   artifact hashes before touching any code.**
1. `src/euler1d.rs` — visibility-only refactor, and nothing else. Verified by
   the full suite, the M4 blooming gate, and the byte comparison.
2. `src/euler2d.rs` — state, geometry, boundaries, area weights, sweeps, the
   Strang sandwich, CFL, guards. Gates **G9**, **G13(i)**. Measure the real CI
   cost here, before any gate grid is chosen.
3. `src/validate.rs` — the Sedov reference and its own unit tests.
4. Gates **G10**, **G11**, **G12**.
5. `src/lsd2d.rs` — the coupled axisymmetric column. Gate **G14**.
6. The measurement: **G15**, **G16**.
7. The `lsd2d` CLI case, `scripts/render_lsd2d.py`, and the documentation pass —
   `MODELS.md` rows and census, this document's amendments, M6c's G7 and
   NOT-in-scope amendments, and the README.

## References

- L. I. Sedov, *Similarity and Dimensional Methods in Mechanics*, Academic Press
  (1959) — the self-similar blast solution.
- L. D. Landau and E. M. Lifshitz, *Fluid Mechanics*, 2nd ed., §106 — the Sedov
  solution and the published `ξ₀`.
- J. R. Kamm and F. X. Timmes, LA-UR-07-2849 (2007) — the standard write-up of
  the Sedov solution as a verification reference.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd
  ed., Springer (2009) — HLLC, MUSCL-Hancock, and dimensional splitting.
- G. Strang, SIAM J. Numer. Anal. **5**, 506 (1968) — the operator splitting.
- H. Eyring et al., Chem. Rev. **45**, 69 (1949); W. W. Wood and J. G. Kirkwood,
  J. Chem. Phys. **22**, 1920 (1954) — the diameter effect and the
  curvature–velocity relation, cited as context for `δ` and **not** as an anchor.
- Yu. P. Raizer, *Laser-Induced Discharge Phenomena*, Consultants Bureau (1977)
  — LSD wave theory and the velocity closed form.
- [M6C_SPEC.md](M6C_SPEC.md) — the 1-D milestone this one extends, and the
  source of the G7 debt inherited here.
