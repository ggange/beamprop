# M6a.2 pre-spec — turbulence-degraded ignition statistics

Written **before** any M6a.2 code, per the project's pre-spec discipline (cf.
[M4_SPEC.md](M4_SPEC.md), [M6A_SPEC.md](M6A_SPEC.md), [M6C_SPEC.md](M6C_SPEC.md)):
pin the coupling, the estimator, and — most importantly — *which* checks
validate physics versus which are solver verification, and which quantity in
this rung is not gateable at all. If any of this proves wrong during
implementation, amend this document first, then the code.

Scope: the statistical rung. M6a answers "does a given peak intensity ignite
air?" at one point. M6c answers "what does the resulting plasma do?" This rung
answers the question between them: **a real beam does not arrive at its focus
undamaged**, so given turbulence of a stated strength, *how often* does the
spark light, and *where*?

Conventions follow [MODELS.md](MODELS.md): SI units, `f64`. Symbols: aperture
diameter `D` (m), Fried parameter `r₀` (m), focal length `f` (m), aperture field
`U(x, y)`, residual phase variance `σ_φ²` (rad²), Strehl ratio `S`
(dimensionless), ignition probability `P_ig`.

## The units mismatch, and why there is no focal grid

The obvious construction — propagate to the focal plane, read the peak, feed it
to the kernel — does not work, and the reason is arithmetic rather than physics.
Turbulence is resolved on a centimetre grid over a kilometre path; the focal
spot that ignites a spark is micrometres across. One grid cannot carry both:
resolving `λf/D` while spanning `D` needs `N ≳ 10⁴` per side, and the run
becomes a focal-plane sampling exercise rather than a turbulence one.

**It is also unnecessary.** In the Fraunhofer regime the focal field is the
Fourier transform of the aperture field, so the *on-axis* focal amplitude is its
DC component — a plain integral over the aperture:

```text
U_focus(0) = (1/(λf)) · ∫∫ U(x, y) dx dy
```

The peak focal intensity therefore follows from the aperture field alone, with
no focal grid anywhere:

```text
I_focus = |∫ U dA|² / (λf)²
```

and the degradation the ignition test actually cares about is that quantity
relative to the same beam through vacuum:

```text
𝒮 ≡ I_focus / I_focus,vac = |∫ U dA|² / |∫ U_vac dA|²
```

This is the estimator M6a.2 is built on. Two properties worth pinning now:

- **`𝒮` folds in scintillation, not just wavefront error.** `U` carries
  amplitude as well as phase, so amplitude scintillation on the aperture reduces
  the coherent sum too. Reporting it as "the Strehl ratio" without qualification
  would be wrong; the code and the docs call it the **focal-intensity ratio**,
  and the phase-only Strehl `S = |∫U dA|²/(∫|U| dA)²` is reported beside it as a
  separate diagnostic so the two contributions are distinguishable.
- **It is exact, not an approximation**, given Fraunhofer and a thin lens. There
  is no small-aberration assumption in it. The Maréchal form `S ≈ exp(−σ_φ²)` is
  a *weak-aberration limit* of it, which is why that appears below as a limit
  gate rather than as the definition.

## What this rung can and cannot claim

Stated up front, in the D5 spirit that M6a learned and M6c applied.

**Not gateable: the position of the `P_ig` curve.** Whether a given `Cn²`
ignites the air depends on M6a's absolute breakdown threshold, which is M6a's
explicitly ungated quantity (4.8–7.0× above the measured Thiyagarajan & Thompson
curve, inside the 3–10× inter-lab scatter). Every `P_ig(Cn²)` this rung produces
inherits that offset. The curve's location on the `Cn²` axis is therefore **a
statement about the model, not about the world**, and must be labelled so
wherever it is plotted or quoted.

**Gateable, and genuinely independent of M6a:** everything upstream of the
kernel call. The aperture phase statistics, the focal-intensity estimator, and
the ensemble machinery are all checkable against closed forms that know nothing
about breakdown. That is where the gates go, and it is a larger fraction of this
rung than it first appears — the kernel evaluation is one boolean at the end of
a long chain, and the chain is what M6a.2 adds.

## Model

Per realization `i`, with all randomness derived from `i` (the M3 contract):

```text
1. build a TurbulentPath for (Cn², L0, z, screens) at stream i
2. propagate a collimated beam to the aperture plane           [M1/M3]
3. mask to a circular aperture of diameter D
4. 𝒮_i = |∫ U dA|² / |∫ U_vac dA|²                              [above]
5. I_peak,i = 𝒮_i · I_focus,vac                                 [IntensityScale, T4]
6. ignited_i = AirBreakdown::breaks_down(I_peak,i, …)          [M6a]
7. wander_i = f · (intensity-weighted mean phase tilt over the aperture)
```

Ensemble reductions: `P_ig = mean(ignited)`, and the wander distribution's RMS
radius. `seeded_ensemble` supplies the parallelism; the reduction runs in
realization order so it is thread-count independent (M3's existing contract).

`I_focus,vac` is fixed once from the vacuum run at the same beam power through
`IntensityScale`, which is the T4 helper's third consumer.

## Gates

Labelled by what they establish, per the house convention. Verification = "the
code computes what I wrote down". Validation = "what I wrote down describes the
world".

### N1 — Noll tip/tilt-removed residual variance (PHYSICS)

Over a circular aperture, Kolmogorov phase with piston **and** tip/tilt removed
has residual variance

```text
σ_φ² = 0.134 · (D/r₀)^(5/3)      (Noll 1976, Δ₃)
```

The coefficient is parameter-free — it comes out of Kolmogorov statistics and
the Zernike basis, with nothing to tune. **Measured on this repo's screens:
0.1352, i.e. +0.9 %**, and constant to four decimals across `D/r₀ = 0.5 → 4`.

This is a genuinely independent projection of the same statistics M3 already
gates through the structure function: M3 checks `D_φ(r)` in the plane, N1 checks
an integral over a circular pupil in the Zernike basis. Passing one does not
imply the other.

**Amended during implementation: the band is ±12 %, not ±5 %, and the sweep is
gone.** The 0.1352 above was measured at 24 screens on one seed set. Repeating
it across three independent seed sets at 64 and 128 screens gives 0.129–0.143
(−4 % to +7 %), and the spread does *not* shrink with screen count, because it
is dominated by how much low-order power a given ensemble happened to draw
rather than by counting error. The gate therefore quotes 0.1407 at the pinned
seed and bands it at ±12 %. A tighter band would be gating which realizations
were drawn; ±12 % still catches a pupil integral or a normalisation wrong by
tens of percent or by a factor, which is what it is for.

### N2 — the piston-removed variance approaches Noll as `L₀/D → ∞` (PHYSICS)

```text
σ_φ² = 1.0299 · (D/r₀)^(5/3)     (Noll 1976, Δ₁)
```

Noll assumes pure Kolmogorov, i.e. an infinite outer scale; the screens are von
Kármán with a finite `L₀`. Piston-removed variance is dominated by the largest
scales, so it is **strongly `L₀`-dependent**, and measured it is:

| `L₀/D` | piston coeff | tip/tilt coeff |
|---|---|---|
| 10 | 0.3405 | 0.1286 |
| 40 | 0.5354 | 0.1341 |
| 200 | 0.8328 | 0.1351 |
| 2 000 | 0.9913 | 0.1352 |
| 20 000 | 1.0128 | 0.1352 |

The gate is the **convergence**, not a single number: the coefficient must climb
monotonically toward 1.0299, start well below it at `L₀/D = 10`, and reach ≥ 0.9
by `L₀/D = 2000`. The same table is what licenses N1 gating a level where this
gates only a trend — tip/tilt removal strips exactly the large-scale terms that
carry the `L₀` sensitivity.

**A trend gate is the stronger choice here, not a concession.** The absolute
piston-removed coefficient swings 1.02–1.23 between seed sets, far worse than
tip/tilt's, because the largest scales carry the fewest independent samples per
screen. That noise is *common-mode* across an `L₀` sweep run on the same seeds,
so it cancels in the trend while dominating any single level. Gating the level
here would be gating the draw.

### N3 — the `(D/r₀)^(5/3)` exponent — SPECIFIED, THEN WITHDRAWN

**Not implemented, and the reason is a trap worth recording.**

Sweeping `r₀` at fixed screens is a **tautology**. `phase_psd` takes `r₀` only
through the multiplicative `0.4896·r₀^(−5/3)`, so for identical random draws the
screen scales as `r₀^(−5/6)` and the variance as `r₀^(−5/3)` *exactly, by
construction*. Measured that way the exponent came back **1.66667 for both
modes** — five decimals of agreement that establish nothing about Kolmogorov
statistics, only that the generator multiplies correctly. This is the M6a "D5"
trap in a new costume: a gate that passes for a reason unrelated to its claim,
and it passed for two runs before the perfection of the number gave it away.

Sweeping the **aperture** is a real geometric change and does test the spatial
statistics, but it is Monte-Carlo limited. Over four seed sets the fitted
exponent deviates from 5/3 by up to 0.09 at 24 screens, 0.05 at 96, and 0.007 at
256 (tip/tilt-removed; piston-removed is still at 0.03 at 256). A band worth
gating costs ~1000 screen generations and states the same content as N1 less
directly — a coefficient constant across apertures *is* a 5/3 exponent. N1 is
the better-conditioned form of the claim, so N1 is what ships.

### S1 — the estimator reduces to Maréchal in the weak limit (verification)

As `σ_φ² → 0`, `S → exp(−σ_φ²)`. Checked on synthetic phase (not turbulence) so
it isolates the estimator, and with the residual gated against the closed form
as the aberration shrinks. Establishes that step 4 is the integral it claims.

### S2 — estimator sanity (verification)

`𝒮 = 1` exactly for a flat wavefront; `0 ≤ 𝒮 ≤ 1`; a pure tilt leaves `𝒮`
unchanged but moves the wander (tilt steers the spot, it does not blur it) —
that last one is the sharpest unit test of the two-quantity split.

### E1 — ensemble convergence (verification)

`P_ig` moves by less than **0.02** when the realization count doubles, at a
`Cn²` chosen near `P_ig ≈ 0.5` where the estimator's variance is worst.

### E2 — thread-count reproducibility (verification)

Bitwise identical `P_ig` and wander across rayon pool sizes, extending M3's
existing `monte_carlo_reproducible_across_thread_counts` contract to this
driver.

### U1 — the `P_ig` level: DOCUMENTED, UNGATED

Per "What this rung can and cannot claim". Recorded in `MODELS.md` when M6a.2
lands, next to M6a's own level limitation, and stated on any figure.

## Failure modes (new codepaths)

| Failure | Detection | Response | Silent? |
|---|---|---|---|
| Aperture larger than the grid | `D` vs grid extent at construction | bail with both numbers | no |
| Aperture under-resolved (`D/dx` small) | samples-across-`D` check | bail with the `dx` needed | no |
| Beam not contained (guard band) | existing `guard_absorbed` check | bail — the M1 precedent | no |
| `L₀` too small for the Noll gates | `L₀/D` asserted in the gate itself | gate fails, not the run | no |
| Every realization ignites / none does | `P_ig` at 0 or 1 exactly | clean report; the sweep says so | no |
| Zero power in the aperture | `∫\|U\|dA` non-positive | bail rather than divide | no |

## NOT in scope (M6a.2)

- **Any focal-plane field.** The estimator is an aperture integral; there is no
  focal grid, and adding one is the units mismatch this spec exists to avoid.
- **Adaptive-optics correction.** N1's tip/tilt removal is a *diagnostic
  projection* for the gate, not a correction applied to the beam.
- **Time evolution / frozen-flow.** Each realization is an independent frozen
  atmosphere; no temporal spectrum, no servo lag.
- **Non-Kolmogorov spectral indices.** Von Kármán only, as M3.
- **Feeding ignition into M6c.** This rung reports *whether and where*; the
  coupled wave is M6c's, and it is deliberately driven from a seeded spark so
  its gates carry nothing from here.

## Open questions

1. **Whether `P_ig` deserves a shape gate at all.** The level is ungated (U1),
   but the curve's *width* in `log Cn²` may be parameter-free in the D5 sense —
   set by the log-normal statistics of `𝒮` rather than by the threshold. If it
   is, that is a second physics gate and it should be found before this rung is
   called closed. Resolve during implementation with a measurement, not an
   argument.
2. **Whether the wander RMS has a usable closed form here.** The angle-of-arrival
   variance for a circular aperture is standard, but the conversion to a focal
   displacement depends on the tilt convention (Zernike vs gradient). Either pin
   it against the literature coefficient or gate the exponent only, and say
   which.

## References

- R. J. Noll, *Zernike polynomials and atmospheric turbulence*, J. Opt. Soc. Am.
  **66**, 207 (1976) — the residual-variance coefficients N1 and N2 gate.
- V. N. Mahajan, *Strehl ratio for primary aberrations*, J. Opt. Soc. Am. **73**,
  860 (1983) — the Maréchal limit S1 checks.
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed. (2005) — the focal
  field as the aperture's Fourier transform, § 5.2.
- L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through Random Media*,
  2nd ed., SPIE Press (2005) — angle of arrival, beam wander.
- `docs/M6A_SPEC.md` — the ignition kernel and the ungated level this rung
  inherits.
