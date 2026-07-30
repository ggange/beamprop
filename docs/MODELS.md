# Physical models and references

Every physical model in `beamprop`, with its governing equation, where it is
implemented, the validation gate that pins it, and the literature it comes
from. This file is the citation record for the solver: if a formula is in the
code, it is in this table.

Conventions: `λ` vacuum wavelength (m), `k = 2π/λ` (rad/m), `z` propagation
distance (m), `κ` transverse spatial frequency (rad/m), intensity `I = |u|²`.

`D`- and `T`-numbered tags (`D5`, `T4`, …) label the project's design decisions
and implementation tasks. They are **shorthand, not citations**: wherever one
carries weight the substance is stated in full alongside it, so nothing here
depends on looking a number up.

## M1 — Diffraction

### Scalar paraxial propagation, split-step spectral method

The field `u(x, y, z)` obeys the paraxial Helmholtz equation; each slab `dz`
is advanced by the symmetric (Strang) splitting

```text
u(z + dz) = D(dz/2) · M(dz) · D(dz/2) · u(z)
```

with `D` the free-space (vacuum) spectral propagator and `M` the medium
operator applied at the slab centre — second-order accurate in `dz`
(verified: observed order ≈ 2).

- `D` uses the **angular-spectrum transfer function**
  `H(κ) = exp(−i·κ²·dz/(2k))` when the grid resolves it, switching to the
  **Fresnel impulse-response** form for long throws
  (criterion `z_c = N·dx²/λ`).
- Implemented in `src/propagate.rs`; gates in `tests/validation.rs`
  (Gaussian width/divergence < 1 %, power conservation ~1e-14, second-order
  convergence, long-throw Fresnel path).

References:
- J. A. Fleck, J. R. Morris, M. D. Feit, *Time-dependent propagation of high
  energy laser beams through the atmosphere*, Appl. Phys. **10**, 129–160
  (1976) — the original split-step beam-propagation method.
- G. Strang, *On the construction and comparison of difference schemes*,
  SIAM J. Numer. Anal. **5**, 506–517 (1968) — symmetric operator splitting.
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Co.
  (2005) — angular spectrum and Fresnel propagators.
- J. D. Schmidt, *Numerical Simulation of Optical Wave Propagation with
  Examples in MATLAB*, SPIE Press (2010) — sampling criteria, TF vs IR
  propagator selection.

### Gaussian beam evolution (validation reference)

```text
w(z) = w0·√(1 + (z/zR)²),   zR = π·w0²/λ,   θ = λ/(π·w0)
```

Implemented in `src/validate.rs` (`GaussianBeam`).

Reference: A. E. Siegman, *Lasers*, University Science Books (1986), ch. 17.

## M2 — Attenuation

### Beer–Lambert extinction

Power extinction coefficient `α` (1/m) applied as amplitude decay inside the
medium operator: `u ← u·exp(−α·dz/2)`, giving transmission
`T(z) = exp(−α·z)`. Supports transversely varying `α(x, y)` per slab.

Implemented in `src/medium.rs` (`Medium::extinction`, `UniformExtinction`) and
`src/propagate.rs`; gates: uniform extinction matches `exp(−α·z)` to ~1e-13,
transverse absorber removes exactly the predicted power, `α = 0` bit-identical
to vacuum.

Reference: standard radiative transfer (Bouguer–Lambert–Beer); see e.g.
E. J. McCartney, *Optics of the Atmosphere*, Wiley (1976).

### Kruse visibility model (aerosol extinction)

```text
α = (3.912 / V) · (λ / 550 nm)^(−q)
q = 1.6 (V > 50 km),  1.3 (6–50 km),  0.585·V_km^(1/3) (V ≤ 6 km)
```

with `V` the meteorological visibility (Koschmieder 2 % contrast). Implemented
in `src/medium.rs` (`kruse_extinction`).

References:
- P. W. Kruse, L. D. McGlauchlin, R. B. McQuistan, *Elements of Infrared
  Technology*, Wiley (1962).
- I. I. Kim, B. McArthur, E. Korevaar, *Comparison of laser beam propagation
  at 785 nm and 1550 nm in fog and haze*, Proc. SPIE **4214**, 26–37 (2001) —
  the q-exponent branches.

## M3 — Turbulence

### Von Kármán / Kolmogorov phase spectrum

Refractive-index fluctuations with structure constant `Cn²` (m^(−2/3))
integrated over a slab give a phase screen with power spectral density

```text
Φ_φ(κ) = 0.4896 · r0^(−5/3) · (κ² + κ0²)^(−11/6),   κ0 = 2π/L0
```

(`L0` outer scale; the pure Kolmogorov `κ^(−11/3)` limit for `κ ≫ κ0`), with
the plane-wave Fried parameter of the slab

```text
r0 = (0.423 · k² · Cn² · dz)^(−3/5)
```

Implemented in `src/turbulence.rs`, `src/validate.rs` (`fried_r0`).

References:
- A. N. Kolmogorov, Dokl. Akad. Nauk SSSR **30**, 301 (1941) — the −11/3
  inertial-range spectrum.
- D. L. Fried, *Optical resolution through a randomly inhomogeneous medium
  for very long and very short exposures*, J. Opt. Soc. Am. **56**, 1372
  (1966) — `r0`.
- L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through Random
  Media*, 2nd ed., SPIE Press (2005) — von Kármán form, coefficient values.

### FFT phase-screen synthesis with subharmonic compensation

Screens are drawn as `φ = N²·Re(IFFT(a))` with complex-Gaussian mode
amplitudes `a(κ) = (g₁ + i·g₂)·√Φ_φ(κ)·Δκ`, plus 6 levels of Lane-style
subharmonics (3×3 modes at spacing `Δκ/3^p`) to restore the large-scale power
the FFT grid cannot represent. Subharmonic modes use a cell-averaged PSD
(5×5 midpoint rule); the FFT modes use the point value — see the quadrature
note in `src/turbulence.rs::cell_mean_psd`.

Gate: Kolmogorov structure function `D_φ(r) = 6.88·(r/r0)^(5/3)` reproduced
to < 10 % over a decade of separations; screen variance vs the von Kármán
total `σ² ≈ 0.0863·(L0/r0)^(5/3)` within 15 %.

References:
- B. L. McGlamery, *Computer simulation studies of compensation of turbulence
  degraded images*, Proc. SPIE **74**, 225–233 (1976) — FFT screen method.
- R. G. Lane, A. Glindemann, J. C. Dainty, *Simulation of a Kolmogorov phase
  screen*, Waves in Random Media **2**, 209–224 (1992) — subharmonic
  compensation.

### Weak-fluctuation propagation statistics (validation references)

```text
Rytov variance (plane wave):  σ_R² = 1.23·Cn²·k^(7/6)·z^(11/6)
Scintillation index:          σ_I² = ⟨I²⟩/⟨I⟩² − 1 ≈ σ_R²   (σ_R² ≲ 0.3)
Long-exposure beam radius:    W_LT = W(z)·√(1 + 1.33·σ_R²·Λ^(5/6)),
                              Λ = 2z/(k·W(z)²)
```

Implemented in `src/validate.rs`; gates: long-exposure spread 0.5 % off
theory, scintillation index 1.6 % off Rytov.

Reference: L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through
Random Media*, 2nd ed., SPIE Press (2005), chs. 6–8.

### Monte-Carlo reproducibility

Realization `i` draws from `ChaCha12Rng::seed_from_u64(master).set_stream(i)`
with fixed draw order, so ensembles are bitwise reproducible across thread
counts (gated).

Reference: D. J. Bernstein, *ChaCha, a variant of Salsa20* (2008); rand /
rand_chacha crates.

## M4 — Thermal blooming

### Steady-state convection-dominated heating (isobaric)

A high-power beam deposits `α_abs·I` (W/m³) into the air; with crosswind `v`
along `+x` and Péclet number `Pe = ρ·c_p·v·w/κ_t ≫ 1` (asserted > 100 at
construction) the CW steady state is the upwind line integral

```text
ΔT(x, y, z) = (α_abs / (ρ·c_p·v)) · ∫_{−∞}^{x} I(x', y, z) dx'
δn(x, y, z) = −(n₀ − 1)/T₀ · ΔT     (isobaric: Δρ/ρ₀ = −ΔT/T₀, Gladstone–Dale)
```

evaluated slab-locally as a trapezoid cumulative sum. Implemented in
`src/blooming.rs` (`ThermalBlooming`) through the field-aware
`Medium::index_response` path: the propagator hands over the intensity after
the leading half-step of diffraction (the slab-centre field), and the medium
applies the half-slab Beer–Lambert factor `e^(−α_abs·dz/2)` so the heating is
a midpoint rule in absorbed power — without that factor the coupling
measurably degrades to 1st order. Ambient `ρ, c_p, κ_t, n₀−1` come from the
frozen air table (`src/airprops.rs`, embedded `data/air_properties.npy`,
bilinear in `(T, p)`, refractivity rescaled to the run wavelength by Ciddor
dispersion). The absorbed power also leaves the beam (`extinction = α_abs`).

The absolute intensity the heating needs comes from
[`IntensityScale`](../src/field.rs) — `I_phys = (P_beam/P_field)·|u|²`, pinned
from the **launch** field, since propagation conserves `P_field` and any
extinction is already carried by `|u|²`. Blooming computed this inline until
M6c's T4 extraction moved it to `src/field.rs` for the LSD driver to share;
the move is gated as an exact no-op (G0a below).

Gates (`tests/blooming.rs`):
- **B1** closed-form crosswind phase (erf profile) reproduced to 0.39 % max
  over all points with `I > 10⁻⁶·I₀`;
- coupling order by self-convergence: observed slope **2.000 / 2.000**;
- **B2** weak-blooming limit vs an analytic first-order (no back-reaction)
  screen reference, transmission-normalized: 0.008 % agreement at `N_φ = 0.1`,
  back-reaction residual scaling ratio 3.65 (theory: 4, quadratic);
- upwind bend sign, strong-blooming (`N_φ = 20`) boundedness with a closed
  power budget, and the qualitative Smith/Gebhardt signatures (upwind peak
  shift, downwind crescent, peak-irradiance rollover);
- **B3 quantitative** — the coupled solver reproduces Smith's (1977)
  whole-beam steady-state peak-irradiance curve `I_REL(N)` along the `F₀ = 5`
  branch to **7.2 % max** over `N ∈ [0.5, 1.8]` (rollover minimum at `N ≈ 1`
  matched to 0.7 %), against the WebPlotDigitizer trace in
  `tests/data/smith1977_F5.csv`. The largest deviation is at the high-N end,
  where the wave solver shows a mild diffractive recovery (`I_REL` rising
  0.757 → 0.807) that Smith's flat `F₀ = 5` curve does not — it behaves like a
  marginally higher effective Fresnel number there; still inside the ±15 %
  gate.

Two distortion numbers appear. The **phase** number (spec convention, reported
in run notes) measures peak on-axis blooming phase in radians:

```text
N_φ = √(2/π) · k·(n₀−1)·α_abs·P·L / (T₀·ρ·c_p·v·w)
```

Smith's **geometrical-optics** number `N_c` (the x-axis of his curves) carries
no wavenumber — it is a ray-bending measure — with `a = w/√2` the 1/e amplitude
radius and an absorption path factor that → 1 as `α·z → 0`:

```text
N_c = (n₀−1)/T₀ · I₀·α·z² / (n₀·ρ·c_p·v·a) · [2/(αz) − 2/(αz)²·(1 − e^(−αz))]
```

Smith plots against an effective `N = N_c·{path factor with diffraction Q, Ω}`;
for the B3 sub-Rayleigh geometry (`F₀ = 5` ⟹ `z_R = 5z`, `Q ≤ 1.02`; `αz =
0.05`) that factor is within a few percent of unity, so `N ≈ N_c` to ≲4 %.

Both implemented in `src/validate.rs` (`BloomingCase`: closed-form ΔT/phase
references, `distortion_number`, `smith_distortion_number`, A&S 7.1.26 `erf`).

References:
- D. C. Smith, *High-power laser propagation: Thermal blooming*, Proc. IEEE
  **65**, 1679–1714 (1977) — steady-state crosswind theory, the `N_c`
  distortion number and the whole-beam `I_REL(N)` curves (B3, F₀ = 5 branch
  digitized in `tests/data/smith1977_F5.csv`).
- F. G. Gebhardt, *High power laser propagation*, Appl. Opt. **15**,
  1479–1493 (1976) — scaling laws, distortion-number phenomenology.
- R. M. Manning, NASA/TM—2012-217634 (2012) — forced-convection heat budget
  and the closed-form erf thin-screen phase behind B1.
- P. E. Ciddor, Appl. Opt. **35**, 1566–1573 (1996) — air refractivity and
  dispersion.
- E. W. Lemmon et al., J. Phys. Chem. Ref. Data **33**, 111 (2004) — NIST air
  data gating the property table's `κ_t` (see `docs/M4_SPEC.md`).

## M6a — Optical breakdown threshold (0-D kernel)

### Electron-avalanche rate balance

At a point in dry air the electron density obeys the avalanche balance

```text
dn_e/dt = (ν_i(I, p) − ν_att(p) − ν_diff(p))·n_e + S_mpi(I, p)
```

with cascade ionization driven by the **net** power — inverse-bremsstrahlung
heating minus the inelastic excitation losses the electron pays climbing to
`U_i`:

```text
ν_i(I, p)     = max(0, heating − L)/U_i
heating(I, p) = (e²·I)/(m_e·c·ε₀) · ν_m/(ν_m² + ω²),   ν_m = K_m·p,   ω = 2πc/λ
L(p)          = δ_eff·ν_m·⟨ε⟩ ≡ L′·p
```

Both terms scale `∝ p`, so their difference gives `I_thr(p) = L′/h + …/p` — a
constant high-pressure **plateau** on top of the `1/p` avalanche term. Without
`L` the model's flattest reachable slope is exactly `p^-1`; the plateau is what
lets it approach the measured trend at all. `L′ = δ_eff·K_m·⟨ε⟩` is one lumped
constant taken from the centre of its literature range and never tuned.

The loss terms are attachment from measured rate coefficients (dissociative
`k₂·n_O₂ ∝ p` plus three-body `k₃·n_O₂·n ∝ p²`; the two-body channel leads at
1 atm, `5.4×10⁷` against `1.4×10⁷ s⁻¹`, with three-body overtaking it only above
`n = k₂/k₃ = 10²⁶ m⁻³` ≈ 4 atm) and free-electron diffusion,
`ν_diff = D_e(p)/Λ²` with `D_e ∝ 1/p`. Attachment is negligible against
diffusion throughout the gate window — `6.7×10⁷` vs `3.3×10⁹ s⁻¹` at 1 atm.

`Λ` is the diffusion length of T&T's **divergence-limited** focus, from their
Eq. 5: `(1/Λ)² = (π/l₀)² + (2.405/r₀)²` with `r₀ = f·α/2 = 20 µm` and
`l₀ = 0.414·(α/d)·f² = 66 µm`, giving **`Λ = 7.74 µm`** — matching the 8 µm the
paper states. **Pinned from geometry, never fit.** (Two earlier guesses were
wrong in opposite directions: a *sphere*, `Λ = r₀/π = 6.37 µm`, overstated
`ν_diff` by 1.48×; a diffraction-limited *filament* put the depth of focus at
2.4 mm instead of 66 µm.) Because the focus is set by the beam's 1 mrad
divergence rather than by diffraction, `Λ` and the focal volume are
wavelength-independent — which is what makes the wavelength gate below a clean
one-variable test.

Finally a swappable multiphoton seed `S_mpi = σ_K·I^K·N` (off by default; the
seed is one electron in the focal volume). Breakdown at `n_e ≥ n_bd = 10²³ m⁻³`.

The per-slice ODE is advanced by its exact solution. With `σ_K = 0` the slice
is Bernoulli — growth is **logistic**, since ionization depletes the neutrals it
feeds on — and is evaluated as `n_e' = n_e/(e^{−βdt} + b·n_e·(1 − e^{−βdt})/β)`
with `β = ν_i − ν_loss`, `b = ν_i/N`; that form underflows harmlessly instead of
overflowing to `NaN` far above threshold, and `n_e` saturates at full ionization
rather than running away. Threshold intensity is found by log-bisection; a
pressure sweep gives `I_thr(p)`.

Implemented in `src/breakdown0d.rs`.

Gate (internal, model self-consistency): solving the avalanche criterion gives

```text
I_thr(p) = L′/h + U_i·(ν_att + ν_diff + G)/(h·p),   G = ln(n_bd/n_seed)/τ
```

Attachment is negligible here, so the exponent runs between two **exact**
limits — plateau-dominated `n → 0` and diffusion-dominated `n → 2` — with the
growth-limited `p^-1` in between. Fitted over the pinned range **300–2000
Torr** (8 log-spaced points, 6 ns FWHM) the model gives **n = 0.095**, and the
gate asserts `n ∈ (0, 1)`: the plateau must be doing work, since without `L`
the model is stuck at `n ≥ 1`. Sweeping the literature ranges of `L′`
(δ_eff ≈ 0.01–0.05) gives an envelope `n ∈ [0.023, 0.231]`, separately pinned so
it cannot drift. The level at 760 Torr is **1.18×10¹² W/cm²**, and the
`FixedMeanEnergy` variant gives `n = 0.468` (level 4.58×10¹¹) as the other end
of the bracket. Absolute threshold level is **not** gated (3–10× inter-lab
scatter). Integrator sub-gates are unit tests of the exact per-slice solver,
not physics validation.

External gates (Thiyagarajan & Thompson 2012, digitized into
`tests/data/tt2012_*.csv`; the paper's setup — 1064 nm, 6 ns FWHM, 20 µm radius
focus, 10–2000 Torr — is exactly what the kernel assumes, so nothing is fitted):

- **`K_m` collision frequency — passes.** `E_eff/E_B = ν_m/√(ν_m²+ω²)`, so the
  paper's two curves measure `ν_m` independently of this crate. Implied
  `K_m = 4.21×10⁷` vs the kernel's `3.90×10⁷ s⁻¹Pa⁻¹`; ratio flat at
  1.05 ± 0.01 over 46–1858 Torr. Non-circular anchor.
- **`E_eff(p)` slope — passes.** Predicted `p^+0.642`, measured `p^+0.695`;
  the positive sign confirms the `ν_m ≪ ω` branch.
- **`I_thr(p)` slope vs the MEASURED curve — RED, and now known to be the
  wrong target.** Measured `p^-0.329`; kernel `p^-0.095`. The measured curve
  is not cascade-only (88 % cascade / 12 % MPI at 760 Torr per the paper), so
  no cascade-only kernel can match it. This gate was previously green at `p^-0.356` (an "8 % match"),
  but that rested on an integration artifact: the seed decayed by `e^-60`
  before the pulse arrived, so an arbitrary integration bound supplied most of
  the threshold requirement and, being pressure-dependent, manufactured slope.
  Corrected route: `p^-1.74` → `p^-0.468` (inelastic loss) → `p^-0.095`
  (`⟨ε⟩` eliminated). What survives is a **bracket** — the two cascade limits
  give 0.095 and 0.468 and straddle the measurement — gated by
  `the_two_cascade_models_bracket_the_measurement`. Read narrowly: the two
  variants differ only in where the cascade cuts off (`⟨ε⟩` = 3 eV vs
  `U_i` = 12.06 eV), so the bracket is a *one-parameter sensitivity*, not two
  independent limits, and not a bound — `⟨ε⟩` ≈ 5 eV gives `n` = 0.346 on its
  own, and `⟨ε⟩ = U_i` gives 0.192, inside the interval.
- **"Too flat" is window-specific — the real error is CURVATURE.** The
  `n = 0.095` above is fitted over 300–2000 Torr. Chylek's 532 nm curve extends
  the comparison two decades lower and shows the kernel is not a power law at
  all: local exponents **1.951** (10–100 Torr), **1.047** (100–300), **0.170**
  (300–786), against a measurement that holds **0.41–0.47** throughout on 1.5–6 %
  scatter. The kernel is 4.6× too steep at the bottom, 2.8× too flat at the top,
  and crosses the data near 250 Torr — an 11.5× swing where the measurement
  varies by 1.13×. Same behaviour at 1064 nm, so this is shape, not level or
  wavelength. Gated as
  `chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not`. Any statement
  that the kernel is simply "too flat" holds only above ~250 Torr.
- **Cascade theory (T&T Eq. 4) — PASSES, and is the apples-to-apples
  reference.** `I_B(CC) = 1.44×10⁶(p_atm² + 2.2×10⁵λ_µm⁻²)` W/cm², implemented
  as `validate::tt2012_cascade_threshold`. Flat at 1064 nm (`n = −0.00002`)
  because `λ⁻²` dominates `p²` by 10⁵ — so the kernel's flatness *agrees* with
  cascade theory. Level: `SelfConsistentClimb` 4.1–5.1× high, `FixedMeanEnergy`
  1.3–3.2×.
- **Wavelength scaling vs Eq. 4 — PASSES, and is the strongest shape check in
  M6a.** Both terms of `I_thr` carry `1/h ∝ ω²`, so the kernel predicts
  `I_thr ∝ λ⁻²`, the same exponent as Eq. 4's dominant term. Over
  **0.53–10.6 µm** (a 20× span, geometry frozen — legitimate, since the focus is
  divergence- not diffraction-limited) both give **−2.000**, with the ratio
  between them constant to `2×10⁻⁵`; the residual is the `(ν_m/ω)²` correction
  at 10.6 µm. Level offsets are flat at 1.64× (`FixedMeanEnergy`) and 4.22×
  (`SelfConsistentClimb`). Sharper still: the plateau `L′/h` and Eq. 4's `λ⁻²`
  coefficient are the *same physical quantity* — `ω²` times the inelastic energy
  loss per collision — and agree to **1.01×** at the literature centre
  (`δ_eff` = 0.02, `⟨ε⟩` = 3 eV). This is a shape agreement on an axis where
  nothing is tunable: `δ_eff·⟨ε⟩` sets the level and cannot produce a `λ`
  exponent. It does *not* independently discover the scaling (the `λ⁻²` is
  analytic in the `ν_m ≪ ω` limit) — it establishes that the two theories share
  it exactly, and fails loudly if that limit is left. Gated by
  `tt2012_wavelength_scaling_matches_cascade_theory`. **Not a pin:**
  `δ_eff·⟨ε⟩` stays asserted from its literature range; re-pinning it *from*
  Eq. 4 would make the level assertions here and in
  `tt2012_cascade_theory_reference` circular, and both would have to be retired
  in the same change, leaving only the exponent.
- **Level vs the measured curve — bounded, drifting, ungated.** The model sits
  4.84× above the data at 380 Torr and 6.97× at 1896 Torr, inside the ungated
  3–10× inter-lab scatter. The 1.48× drift is the residual slope error in
  absolute clothing; an earlier "flat within 1.16×" claim was withdrawn with the
  artifact that produced it. Converting `E_B` to intensity uses
  `I = ε₀cE_rms²`, since the `E_eff` ratio establishes `E_B` is an RMS
  amplitude.
- **MPI calibrated to the paper's own estimate — implemented, left OFF.**
  Anchoring a rate to their `I_B(MPI) = 4.42×10⁹ W/cm²` collapses the threshold
  to 5.5×10⁹, 37× below their own measurement, and contradicts the paper's own
  88 %-cascade accounting. Their number is an order-of-magnitude significance
  indicator (Nelson's flux-density criterion, whose constant the paper never
  states), not a rate anchor. A real `σ_K` from multiphoton cross-section data
  is the open item.

**D5 debt — DISCHARGED 2026-07-30, in the negative.** With the slope gate red,
the plan's fallback clause required an anchor independent of the kernel's own
coefficients. Eq. 4 supplied that for the **exponent** (`λ⁻²` is untouched by
any coefficient choice) but not for the **level**: it is the same paper, and its
`λ⁻²` coefficient implies the same `δ_eff·⟨ε⟩ = 0.060 eV` the kernel already
assumes, so the 1.01× agreement is intra-lineage consistency, not corroboration.

The second dataset the clause asked for is now in the suite — **Chylek et al.
1990, clean air at 532 nm** (`tests/data/chylek1990_air_threshold_vs_pressure.csv`,
digitized programmatically by `scripts/digitize_chylek1990.py`). It is the anchor
D5 specified: different group, different apparatus, and a different wavelength —
exactly half T&T's 1064 nm, at a nearly identical pulse length (6.5 vs 6 ± 1 ns)
and focal radius (16.5 vs 20 µm). That last point is what makes it usable: the
paper's own Sec. II names pulse duration and focal spot as the reason literature
values of `α` contradict one another, and here they are matched, so the 532/1064
comparison measures the *wavelength* scaling rather than two different benches.

It does not corroborate the model — it falsifies the `λ⁻²` prediction against
measurement:

```text
cascade / kernel:  I_th(532)/I_th(1064) = 3.99   (= λ⁻²; shorter λ costs more)
measured:          I_th(532)/I_th(1064) ≈ 0.80   (532 nm breaks down EASIER)
```

Wrong by ~5×, and wrong in **sign**. At 532 nm the multiphoton order falls from
`K = ⌈12.06/1.166⌉ = 11` photons to `⌈12.06/2.33⌉ = 6`, so the MPI channel the
kernel leaves OFF is enormously stronger exactly where the measurement drops —
the same missing channel the pressure-slope gates indict, seen on a second axis.
Gated as `chylek1990_tt2012_wavelength_ratio_falsifies_cascade_lambda_squared`.

**Consequence for how M6a is described.** `λ⁻²` remains a correct statement about
the kernel's internal structure and its gate stays — it fails loudly if the IB
Lorentzian limit is ever left. It may **not** be called external agreement with
measurement, and it is no longer M6a's headline. M6a's honest status: verified
against cascade theory, falsified against air on both the pressure axis and the
wavelength axis. Does not block M6c (gated separately on Chapman–Jouguet
velocity). See `docs/M6A_SPEC.md` § Fallback.

Open question M6a hands forward: the measured `n` = 0.329 is unreachable by any
cascade-only model, since accepted cascade theory is flat at this wavelength.
Closing it means the MPI contribution the paper itself invokes (12 % at
760 Torr, dominant below 100 Torr), not a flatter cascade — and separately, a
distribution-resolved cascade rate, since the default variant's near-flatness
comes from putting every electron on the mean trajectory (at threshold it runs
within 0.8 % of the `ε_∞ = U_i` pole at 2000 Torr, where that idealization is
least defensible).

**The obvious MPI candidate has been tried and it fails — 2026-07-30.** Keldysh
photoionization is implemented (`breakdown0d::keldysh_rate`) and **verified**
against both closed-form limits of its own exponent: the multiphoton branch
recovers the photon order `U_i/ħω` to better than 0.2 % (10.35 at 1064 nm, 5.17
at 532 nm) and the tunnelling branch reproduces the static-field exponent
`4√(2m)U_i^{3/2}/(3ħeE)` to 1 part in 10⁶. There is nothing tunable in that
exponent, which is what makes it a legitimate test rather than a fit.

It does not close the wavelength gap:

| prefactor × ω | `I_th(532)/I_th(1064)` |
|---|---|
| 0 (cascade only) | 3.99 |
| **1 (order unity)** | **3.87** |
| 10³ | 1.84 |
| 10⁶ | 0.48 |
| **measured** | **0.80** |

At an order-unity prefactor MPI closes 3 % of the gap. Reaching the measurement
needs `~10⁵·ω`, i.e. an ionization rate faster than the optical frequency, which
is not a rate. Gated as `keldysh_mpi_does_not_close_the_wavelength_gap`.

Two by-products worth keeping:

- **The seed density is unphysical, and it is a latent defect.** `n_e0 = 1/V_focal
  = 1.2×10¹³ m⁻³` is ~10⁴ above the cosmic-ray background (10⁹–10¹⁰ m⁻³), which
  puts ~10⁻⁴ electrons in an `8.3×10⁻¹⁴ m³` focus — the focus essentially never
  holds one. Seeding *should* therefore be MPI's job, importing the photon-order
  asymmetry (at prefactor 1, MPI makes 295 electrons per pulse in the focus at
  532 nm and 2×10⁻⁸ at 1064 nm — ten orders of magnitude). Removing the seed
  changes the ratio only 3.99 → 3.85, because the model's threshold is already
  5.7–28× too high and MPI is copious there at both wavelengths. So the defect is
  real but masked by the level error. Exposed as
  `AirBreakdown::with_seed_density`.
- **An earlier claim of mine, withdrawn.** The wavelength ratio is *not*
  prefactor-insensitive. The `x^(1/K)` suppression argument only holds once MPI
  dominates at both wavelengths; the ratio then scales as `x^(−0.097)`, so three
  decades of prefactor still move it 1.9×, and across the transition it runs 3.99
  → 0.48. Any future prefactor claim has to be justified to ~2 orders, not waved
  through.

The open item is therefore narrower than "add MPI": either a PPT-corrected rate
for molecular O₂ (Coulomb corrections can lift the prefactor by orders of
magnitude — checkable against published `σ_K`), or a systematic in the two-paper
comparison. It is *not* a missing channel at these intensities.

Chylek's 532 nm data sharpens the remaining question into a quantitative target
rather than a direction. Any candidate MPI channel now has to do three things at
once:
lift the 532 nm threshold's *ratio* to 1064 nm from 3.99 down to ≈0.80, flatten
the low-pressure branch from 1.951 to ≈0.43, and steepen the high-pressure
branch from 0.170 to ≈0.47 — with `K = 6` photons at 532 nm against `K = 11` at
1064 nm supplying most of the wavelength leverage for free. The three
`chylek1990_*` gates pin all three numbers, so a channel that fixes one while
breaking another cannot land quietly.

Full model and constants: `docs/M6A_SPEC.md`.

References:
- Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991) — cascade ionization
  and the inverse-bremsstrahlung heating rate.
- N. Kroll, K. M. Watson, *Theoretical study of ionization of air by intense
  laser pulses*, Phys. Rev. A **5**, 1883 (1972).
- C. G. Morgan, *Laser-induced breakdown of gases*, Rep. Prog. Phys. **38**,
  621 (1975) — regime map and threshold scaling.
- A. Thiyagarajan, J. B. Thompson, *Optical breakdown threshold investigation
  of 1064 nm laser induced air plasmas*, J. Appl. Phys. **111**, 073302 (2012)
  — the external anchor: `E_B` and `E_eff` curves digitized into
  `tests/data/tt2012_*.csv`, focal geometry from their Eq. 5, and the cascade
  closed form from their Eq. 4.
- P. Chylek, M. A. Jarzembski, V. Srivastava, R. G. Pinnick, *Pressure dependence
  of the laser-induced breakdown thresholds of gases and droplets*, Appl. Opt.
  **29**, 2303 (1990) — the independent D5 anchor: the clean-air threshold at
  532 nm (their Fig. 3, `α = 0.45 ± 0.01`), digitized into
  `tests/data/chylek1990_air_threshold_vs_pressure.csv` by
  `scripts/digitize_chylek1990.py`. Second group, second apparatus, second
  wavelength, matched pulse and focus. Their **Fig. 2** additionally gives clean
  He, Ar and Xe thresholds on the same bench, hand-traced into
  `tests/data/chylek1990_{he,ar,xe}_threshold_vs_pressure.csv` and cross-checked
  against an independent programmatic trace to 0.3 % on Ar and Xe. Those three
  gases span `U_i` = 24.59 / 15.76 / 12.13 eV, i.e. multiphoton order
  **K = 11 / 7 / 6 at a single wavelength and a single apparatus** — the one
  dataset here that separates photon order from `λ` — and having no attachment
  channel they isolate cascade + diffusion + MPI. No physics gate consumes them
  yet: the kernel is `AirBreakdown`, so a noble gas needs a general-gas model
  first. Carried with an integrity gate
  (`chylek1990_fig2_digitization_reproduces_the_published_slopes`) meanwhile.

## M6a.2 — Aperture optics and pupil phase statistics

### On-axis focal intensity from the pupil field

In the Fraunhofer regime the focal field is the Fourier transform of the
aperture field, so the on-axis focal amplitude is its DC component:

```text
U_focus(0) = (1/(λf))·∫∫ U(x, y) dA
I_focus    = |∫ U dA|² / (λf)²
```

Site: `src/aperture.rs` (`Aperture`). **Exact** given Fraunhofer and a thin
lens — not a small-aberration approximation; the Maréchal form `S ≈ exp(−σ_φ²)`
is a weak-aberration limit of it, and is gated as a limit rather than used as
the definition.

This is what lets M6a.2 exist at all. Turbulence is resolved on a centimetre
grid over a kilometre path while the focal spot that ignites a spark is
micrometres across; resolving `λf/D` while spanning `D` needs `N ≳ 10⁴` per
side. **There is no focal grid anywhere** — every quantity is a pupil integral
on the grid the propagator already produced.

Two degradations are reported and deliberately never conflated:

- `focal_intensity_ratio` — against the same beam through vacuum. Total
  degradation, wavefront **and** amplitude scintillation, because the pupil
  field carries both. This is the quantity that feeds an ignition test.
- `phase_only_strehl` = `|∫U dA|²/(∫|U| dA)²` — normalised against the beam's
  own amplitude, so scintillation divides out and the wavefront contribution is
  isolated. Diagnostic only. Calling the first one "the Strehl ratio" would be
  wrong, which is why both exist.

Gates (`src/aperture.rs` unit tests): a flat wavefront gives exactly `S = 1` and
nothing exceeds it (the triangle inequality on the coherent sum); a pure tilt
steers the spot without dimming it — `S` unchanged, wander `= f·θ` — which is
the sharpest statement of why the two quantities differ; an amplitude-only
perturbation leaves `phase_only_strehl` at 1 while costing focal intensity; and
`S → exp(−σ_φ²)` as the aberration shrinks, with the residual gated to fall.

Reference: J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts &
Co. (2005), § 5.2. V. N. Mahajan, J. Opt. Soc. Am. **73**, 860 (1983) — the
Maréchal limit.

### Residual pupil phase variance (Noll coefficients)

Kolmogorov phase over a circular pupil, with the low-order Zernike terms
projected out:

```text
piston removed        σ_φ² = 1.0299·(D/r₀)^(5/3)     (Noll 1976, Δ₁)
piston + tilts removed σ_φ² = 0.134 ·(D/r₀)^(5/3)     (Noll 1976, Δ₃)
```

Site: `src/aperture.rs` (`Aperture::residual_phase_variance`, `TiltRemoval`).
Both coefficients are parameter-free. Taken on **phase screens, not propagated
fields**: `arg(u)` is only recoverable modulo 2π and wraps many times at these
`D/r₀`, so a variance from a propagated field would be measuring the wrapping.

These are an independent projection of the statistics M3 already gates through
the structure function — a pupil integral in the Zernike basis versus `D_φ(r)`
in the plane. Passing one does not imply the other.

- **N1** (`noll_tip_tilt_removed_variance_matches_the_closed_form`) — measured
  **0.1407** at the pinned seed, +5.0 % on Noll, banded at ±12 %. The band is
  set by the ensemble spread, not the central value: across three seed sets at
  64 and 128 screens the coefficient runs 0.129–0.143, and that spread does
  **not** shrink with screen count because it is dominated by how much low-order
  power an ensemble happened to draw. A tighter band would gate the draw.
- **N2** (`noll_piston_removed_variance_converges_to_kolmogorov`) — Noll assumes
  an infinite outer scale; the screens are von Kármán. Piston-removed variance
  is dominated by the largest scales, so it is strongly `L₀`-dependent:
  `L₀/D` = 10 → 0.345, 40 → 0.541, 200 → 0.842, 2000 → 1.001, against Noll's
  1.0299. Gated as the convergence. Runs 32 screens, not N1's 128: the trend is
  identical at either count because the noise is common-mode across the sweep,
  so the extra screens buy nothing and cost 4× the runtime. A trend gate is the *stronger* choice here — the
  absolute coefficient swings 1.02–1.23 between seed sets, and that noise is
  common-mode across an `L₀` sweep on the same seeds, so it cancels in the trend
  while dominating any level.

**A `(D/r₀)^(5/3)` exponent gate was specified and withdrawn**, and the reason is
recorded because it is the M6a "D5" trap in new costume. Sweeping `r₀` at fixed
screens is a *tautology*: `phase_psd` takes `r₀` only through the multiplicative
`0.4896·r₀^(−5/3)`, so identical draws scale the screen as `r₀^(−5/6)` and the
variance as `r₀^(−5/3)` by construction. Measured that way the exponent came back
**1.66667 for both modes** — five decimals that establish nothing but correct
multiplication. Sweeping the *aperture* is a real geometric change but is
Monte-Carlo limited (deviation from 5/3 up to 0.09 at 24 screens, 0.05 at 96,
0.007 at 256), and a coefficient constant across apertures *is* a 5/3 exponent,
so N1 is the better-conditioned form of the same claim.

Reference: R. J. Noll, *Zernike polynomials and atmospheric turbulence*,
J. Opt. Soc. Am. **66**, 207 (1976).

### Turbulence-degraded ignition statistics (the driver)

`cases::run_ignition`. Per realization: propagate a launch beam through a
`TurbulentPath`, take the pupil integral at the receiver, turn it into W/m²
through `IntensityScale` (the T4 helper's third consumer), and hand that one
number to `AirBreakdown`. Reductions over the ensemble give the ignition
probability, the focal-intensity ratio distribution, and the focal-spot wander.
`seeded_ensemble` supplies the parallelism; realizations derive all randomness
from their index and come back in index order, so every reduction is bitwise
thread-count independent (**E2**).

**The position of `P_ig` on the `Cn²` axis is not a claim about the world.** It
carries `AirBreakdown`'s absolute threshold, which is M6a's explicitly ungated
quantity, and must be labelled so wherever it is plotted. Everything upstream of
that one boolean is independent of it and is gated.

- **E1** (`ignition_ensemble_converges`) — the spec asked for `P_ig` within
  ±0.02 on a realization doubling; that is **not achievable** and the gate says
  so. `P_ig` is a Bernoulli mean whose binomial standard error is 0.030 at
  n = 256 and 0.022 at n = 512, so ±0.02 at any affordable `n` would gate the
  draw. Gated instead: the continuous reductions converge (`wander_rms` under
  5 % on doubling, measured 1.15 → 1.11 ×10⁻⁴ m over n = 32 → 512) and `P_ig`
  moves within two binomial standard errors (measured 1.8σ, 0.0σ, 0.6σ, 0.7σ),
  plus a non-vacuity check that `P_ig` is not saturated.
- **W1** (`wander_follows_the_square_root_of_cn2`) — **PHYSICS.** RMS wander
  `∝ Cn²^(1/2)`; measured **0.4953 / 0.4977 / 0.4987** over two decades and
  three seeds, gated at ±0.02. Parameter-free: path length, aperture, outer
  scale and beam all enter as coefficients, and none can produce a 1/2.
- **W2 — retired, not gated.** An aperture-dependence gate was landed and then
  withdrawn as seed-dependent. The observation stands: the textbook
  `σ_α² ∝ D^(−1/3)` implies `D^(−1/6)` = −0.167 for the RMS and the default
  geometry does not show it (−0.003), because the tilt estimator is
  intensity-weighted and a 5 cm beam in a 15–40 cm pupil is weighted by its own
  footprint while the closed form assumes *uniform* illumination. The
  *measurement* cannot carry a gate: it fits a slope across four nested
  apertures on shared screens, and across three seeds the overfilled exponent
  runs −0.183/−0.249/+0.004 at 16 realizations and −0.102/−0.318/−0.143 at 32,
  a spread that does not shrink with ensemble size. The gate passed only at the
  seed it pinned. Documented observation, not a validated claim.

The `ignition` CLI case (`cases::run_ignition_sweep`, rendered by
`scripts/render_ignition.py`) sweeps `Cn²` and reports the ignition probability
with its **binomial** error bars, the focal-intensity distribution behind it,
and the wander law. The figure carries the shape/position caveat inside the
panel rather than in a caption.

It also reports the transition width — the span from `P_ig` = 0.9 to 0.1,
measured **1.42 decades**, and roughly invariant (1.49 at 2× drive, 1.39 at
0.5×, 1.44 at a 1.33× larger pupil) while a 4× drive change slides the curve
0.6 decades sideways. **Deliberately not gated**: no closed form for the width
has been derived, so gating it would check the measured number against itself —
the same trap that retired the `(D/r₀)^(5/3)` exponent gate above.

Reference: L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press (2005) — angle of arrival and beam wander.

## M6c — 1-D gas dynamics (the LSD substrate)

### Compressible Euler equations, HLLC + MUSCL-Hancock

```text
∂U/∂t + ∂F(U)/∂x = Ṡ

U = (ρ, ρu, E)ᵀ,   F = (ρu, ρu² + p, (E + p)u)ᵀ,   E = p/(γ−1) + ½ρu²
```

Site: `src/euler1d.rs`. Ideal gas at constant `γ` (the verification EOS of the
two `docs/M6C_SPEC.md` pins; the plasma-range table EOS attaches later without
touching the flux routines). HLLC approximate Riemann solver with
Einfeldt/Davis wave speeds, MUSCL-Hancock reconstruction under a minmod
limiter, CFL ≤ 0.8 recomputed per step from the current wave speeds, and a
positivity guard that **bails with the cell and step** rather than clamping.

**No laser physics is in this module**, deliberately (spec gate decision 4).
`Ṡ` is exposed only as `step_with_source`, the seam the LSD driver attaches to;
deposition, ignition, and the plasma column live one layer up. That is what
keeps the two gates below independent of the model that will use them.

Gate **G0** guards the T4 extraction that this milestone required
(`tests/blooming.rs`): **G0a** reproduces the pre-extraction `δn` arithmetic
from scratch and demands bit-for-bit equality with `ThermalBlooming`'s output —
a tolerance would not see a reassociation — and **G0b** pins the size and hash
of `data/air_properties.npy`, since M4's numbers are calibrated to that table
and M6c's plasma properties belong in a separate file (D8). G0a deliberately
avoids FFTs so it stays deterministic on every platform the M5 wheels target;
a whole-run field fingerprint would look stronger and be flakier, as FFT
results are not bit-portable across libraries.

The G1/G2 gates are **verification** — "the code solves the equations written
down" — not validation. Nothing here is yet a claim about the world; the physics
gate for M6c is the parameter-free `D ∝ S^(1/3)`, `ρ₀^(−1/3)` scaling (G4),
which arrives with the deposition layer.

- **G1 — Sod shock tube vs the exact Riemann solution**
  (`sod_shock_tube_matches_exact_riemann_solution`). Exact solution from the
  Newton-iterated star-state solver in `src/validate.rs` (`RiemannProblem`,
  which shares only the plain `(ρ, u, p)` struct with the solver under test).
  Measured L1(ρ) = 6.55e-3 at n = 100 falling to 6.55e-4 at n = 1600, observed
  rate 0.79–0.88. First order is the ceiling on a solution containing a shock
  and a contact; ~0.8 is the textbook minmod value.
- **G2 — observed order on smooth flow**
  (`euler_muscl_hancock_is_second_order_on_smooth_flow`). Isentropic advection
  of `ρ = 1 + 0.2·sin(2πx)` at uniform `u`, `p` over one period. L1(ρ) falls
  7.50e-4 → 3.78e-6 over n = 128 → 2048, observed order rising monotonically
  1.86 → 1.94. It approaches 2 **from below** because minmod clips the two
  smooth extrema, degrading the scheme to 1st order in a shrinking region — so
  the gate is on the finest pair (> 1.85) plus the monotone climb, not on
  hitting 2 exactly.

### Equilibrium plasma-range air properties (frozen table)

Equilibrium air from 200 K to 30,000 K and 10⁴–10⁸ Pa, tabulated offline by
`scripts/make_plasma_table.py` from Mutation++ (`air_11` mixture, RRHO thermo
database, equilibrium state model) into `data/plasma_properties.npy`, and read
through `src/plasmaprops.rs`. Shape `(4, 597, 33)`, property axis
`[ln ρ, e, γ_eff, ln n_e]`, on a grid uniform in `T` and uniform in `log₁₀ p`,
bilinearly interpolated. The pressure ceiling is set by the CJ state behind an
LSD front (~1.5×10⁷ Pa) and the temperature ceiling by its post-front
temperature. No runtime FFI, no LGPL in the build or the M5 wheels — the
`airprops.rs` discipline. `data/air_properties.npy` (M4) is a separate file and
is untouched (G0b).

`ρ` and `n_e` are stored as **logarithms**: both cross dozens of decades
through the ionization onset, where interpolating raw values fails badly.
`e` changes sign near 800 K so no log is available, and neither it nor `γ_eff`
needs one.

Two quantities are deliberately absent. **`n_i` is not stored** — every ion in
the mixture is singly charged, so quasi-neutrality makes it identical to `n_e`
(verified to 6.2×10⁻¹¹ wherever `n_e` matters); it is returned as `n_e`.
**`Z̄` is not stored and is not a prediction**: it is identically 1 and cannot
be otherwise, because the RRHO database ships no doubly ionized N or O (He⁺⁺ is
the only `++` species in it).

**Limitation — no second ionization.** Real equilibrium air begins to doubly
ionize above ~20,000–25,000 K; `air_11` structurally cannot. Toward the top of
the range the table therefore understates `n_e`, and so understates the
inverse-bremsstrahlung absorption `α_IB ∝ n_e·n_i·Z̄²` the LSD front runs on.
The range is still built to 30,000 K because the CJ state needs it, but above
`SECOND_IONIZATION_K` = 20,000 K the table is a singly-ionized approximation,
flagged by `PlasmaTable::is_singly_ionized_approximation`.

Per D8, Mutation++ is **trusted for the physics** and the gate is on the
**tabulation**:
- **G6** (`plasma_table_matches_direct_mutationpp_off_grid`) interpolates the
  frozen table to 99 points deliberately **off** its grid — 75 of them in the
  6,000–18,000 K ionization onset, where bilinear interpolation is worst — and
  compares against direct Mutation++ evaluations frozen into
  `tests/data/plasma_reference_samples.csv` at generation time (the
  `tt2012_*.csv` precedent, since the solver cannot call Mutation++). Measured
  max relative error: **ρ 4.10e-4, e 8.61e-4, γ_eff 1.68e-4, n_e 1.48e-3**.
- The generator runs its own harsher cell-midpoint sweep over 17,683 points
  before it will write anything (ρ 4.53e-4, e 1.05e-3, γ_eff 2.40e-4,
  n_e 3.61e-3), plus a quasi-neutrality gate and a cold-limit check that
  `γ_eff(300 K, 1 atm)` = 1.39883 — the one point whose answer is known without
  Mutation++.
- Neither gate makes an accuracy claim below `NE_ACCURACY_FLOOR` = 10¹⁵ m⁻³.
  There `ln n_e` is nearly linear in `1/T` rather than `T`, so the uniform-`T`
  grid interpolates it poorly — but the values are ~10³ m⁻³ against ~10²³ in an
  LSD plasma, and gating that error would be theatre.

### Laser-supported detonation (the coupled wave)

The beam ignites a plasma and the resulting absorption wave runs **back up the
beam** toward the laser as a detonation. `src/lsd.rs`, driven by
`LsdColumn::advance`.

The column's `x` axis is the beam axis: the laser sits beyond `x_min`, so the
beam travels in `+x` and the front travels in `−x`. Everything upstream of the
front is cold transparent air, which is why the front sees the full incident
intensity `S`. Beam attenuation is Beer–Lambert in the direction of travel,
`dI/dx = −α_pl·I`, and the source term in the energy equation is the absorbed
power density:

```text
∂U/∂t + ∂F(U)/∂x = (0, 0, q)ᵀ,     I_{k+1} = I_k·exp(−α_k·dx),
q_k = (I_k − I_{k+1})/dx
```

The deposition is discretised **conservatively** — each cell takes exactly what
it removes from the beam, so `Σ q·dx ≡ I_in − I_out` with no quadrature error.
Hydro and source are coupled by **Strang splitting**
(`source(dt/2) → hydro(dt) → recompute α, I → source(dt/2)`), matching the
propagator's own splitting discipline and for the same reason. The step is sized
from the *post-deposition* wave speeds: the leading half-step raises `p` and so
`c` before the flux update sees it, and sizing from the pre-deposition state
overshoots the CFL bound.

Two absorption closures:
- `Absorption::GreyThreshold` — verification. A fixed `α` wherever the specific
  internal energy exceeds a threshold, zero below. Nothing that can drift
  between refinements. Keying on internal energy rather than temperature keeps
  it well defined under the constant-`γ` EOS, which has no gas constant.
- `Absorption::InverseBremsstrahlung` — production. Thermal free-free
  absorption, `α_IB = C·Z̄² n_e n_i T^(−1/2) ν^(−3) (1 − e^(−hν/kT))·ḡ`, with
  `n_e` from the plasma table above (`n_i = n_e`, `Z̄ ≡ 1`, both structural).
  `C = 3.7×10⁻²` in SI, converted from the CGS `3.7×10⁸` of Rybicki & Lightman
  Eq. 5.18b. The Gaunt factor `ḡ` is a dimensionless multiplier of order unity,
  set to 1: a proper Gaunt table is out of scope, and it need not be in scope,
  because `ḡ` enters as a *coefficient* and no coefficient can shift the `1/3`
  exponent the physics gate measures. One honest inconsistency: the hydro
  carries a constant-`γ` ideal gas while the ionization comes from the
  equilibrium table, so `T` is the table's inversion of the hydro's `(ρ, p)`
  rather than a self-consistent EOS.

Per D7 the plasma couples to the beam through **absorption only**. `PlasmaColumn`
is the read-only `Medium` the propagator sees: `extinction(z_slab)` from the
hydro state, and `δn ≡ 0` — no Drude index, which is what keeps M6c clear of the
near-critical failure a Drude plasma column would hit in a paraxial envelope.

In the M4 Péclet spirit, `check_regime` refuses rather than mis-models: the
absorption length must be resolved by ≥ 4 cells and be under a quarter of the
domain (thicker is the LSC regime, out of scope), ≥ 90 % of the beam must reach
the front, and the front must be a strong detonation (`p₁ ≥ 10·p₀`).

"Reaching the front" is measured at the leading edge of the **strongly
absorbing** region (first cell with `α ≥ ½·α_max`), not at the pressure
half-maximum that `front_position` reports. The two genuinely differ — in a
detonation the reaction zone leads the pressure peak — and using the pressure
front here would score normal front structure as upstream extinction. Under the
grey model this check is ~1 by construction, since cold gas is exactly
transparent; it earns its keep under inverse bremsstrahlung, where a long
weakly ionized precursor really can eat the beam before it arrives.

**`SECOND_IONIZATION_K` is enforced, not just documented.** The CJ state behind
a strong LSD front sits at 1.5–2.5×10⁴ K and so legitimately crosses the table's
singly-ionized ceiling. `IonizationCeiling::Refuse` (the default) bails naming
the temperature; `::Flag` proceeds and records it on the run
(`used_singly_ionized_approximation`). Extending the table is not currently
possible from shipped data — no Mutation++ thermodynamic database contains
doubly ionized N or O — so the bias is converted into an explicit boundary
instead of being carried silently.

Gates (`tests/validation.rs`):
- **G3** (`lsd_front_speed_matches_the_raizer_closed_form`) — **VERIFICATION,
  not validation.** At `S = 10¹¹ W/m²`, `ρ₀ = 1.225 kg/m³`, `γ = 1.4`, and an
  absorption length `1/α = 50 µm`, the measured front speed is **5402 m/s**
  against Raizer's `D = [2(γ²−1)S/ρ₀]^(1/3)` = **5392 m/s**, `+0.19 %`;
  gated below 1 %. Refining `dx` from 10 µm to 5 µm moves it by 5×10⁻⁵, so the
  answer is grid-converged and the residual is physical, not numerical. Both
  boundaries are asserted undisturbed alongside `check_regime`: once the wave
  runs off the laser-side end, `front_position` degrades to the first cell
  centre and would report a plausible speed for a front that no longer exists.
  G3b carries the same assertion.

  The label matters. Raizer's expression is **not** an independent check on this
  model — it is the Chapman–Jouguet construction the deposition model is built
  from, with the chemical heat release replaced by `q = S/(ρ₀D)`. Reproducing it
  establishes that HLLC, the Strang-split source, and the EOS together solve a
  nontrivial self-similar problem correctly. It establishes nothing about
  whether that problem describes the world. `raizer_lsd_velocity` therefore
  lives in `src/lsd.rs` beside the model and *not* in `src/validate.rs` among
  the independent reference solutions — filing it there would quietly assert
  otherwise, which is precisely the M6a "D5" trap.
- **G3b** (`lsd_front_speed_converges_as_the_absorption_layer_thins`) — the
  refinement that actually bites. Over `1/α = 400 → 200 → 100 → 50 µm` at
  `dx = 10 µm` the residual runs **−8.26 % → −2.66 % → −0.43 % → +0.19 %**,
  monotone, each halving taking at least 2.3× off the error.

  The residual is a **relaxation transient**, not a permanent thick-layer
  deficit: held at `1/α = 400 µm` and given longer to settle (0.15 → 0.30 → 0.50
  of the domain) it runs `−8.3 % → −3.7 % → −1.5 %` and does not plateau. A
  thicker deposition zone relaxes onto the self-sustaining speed more slowly, so
  at a fixed settle it sits further from it; given long enough they all reach the
  same CJ speed. That is the textbook result — a CJ velocity depends on total
  heat release, not reaction-zone length — and it is worth stating because an
  earlier version of this entry claimed a steady-state deficit instead, which
  contradicted the theory the gate checks against.
- **G3c** (`lsd_front_speed_is_seed_independent`) — the answer must not depend
  on how the wave was lit. A seeded detonation starts overdriven and relaxes
  onto the CJ speed slowly, and G3's 1 % tolerance is the same order as that
  transient, so the seed is a free parameter sitting directly under the headline
  number unless it is checked. Between a 1× and a 2× CJ-pressure seed the
  results differ by `5.3e-3` at a 1.0 µs settle, `2.7e-3` at 1.4 µs and
  `1.1e-3` at the 1.8 µs used, both converging on `≈ +0.2 %`. Gated below
  `2e-3`, with both boundaries asserted undisturbed.
- **G2b** (`lsd_source_coupling_is_second_order`) — the hydro↔source coupling is
  2nd order, the M6c counterpart of M1's `split_step_is_second_order` and M4's
  `coupling_is_second_order`. Refining `dx` and `dt` **together at fixed CFL**
  (the limit the claim is stated in; MUSCL-Hancock degenerates to forward Euler
  in time if `dt→0` at fixed `dx`), the Strang cadence gives observed order
  **1.99 / 2.03 / 1.99**. The gate also runs a deliberate 1st-order contrast —
  folding the source into the update gives **0.88 / 1.02 / 1.07** — so it
  demonstrably resolves the difference rather than passing vacuously.

  This is why `Euler1d::step_with_source` carries a warning: used on its own it
  *is* that 1st-order contrast. Nothing about its output looks wrong; it simply
  converges half as fast. The driver's four-call Strang sandwich is the
  supported path.
- **G5** (`lsd_energy_budget_closes`) — absorbed laser energy versus the
  domain's energy gain. Measured relative residual **2.1×10⁻¹⁶ to 4.6×10⁻¹⁵**,
  five orders inside the 10⁻¹⁰ the spec asks. Exact rather than approximate for
  two reasons: the conservative deposition above, and a boundary flux that is
  *verified* zero rather than estimated — with transmissive ends and undisturbed
  ambient gas at both, `(E + p)u` vanishes identically, and
  `boundaries_undisturbed` asserts that premise rather than assuming it.

- **G4** (`lsd_velocity_follows_the_parameter_free_one_third_scaling`) — **THE
  PHYSICS GATE.** `D ∝ S^(1/3)` over 1.52 decades of absorbed intensity and
  `D ∝ ρ₀^(−1/3)` over 1.50 decades of ambient density, exponents gated inside
  `±0.01`. Measured at `γ = 1.4`: **+0.33190** and **−0.33020**.

  Everything above this line is verification — it establishes that the code
  solves the equations it was given, and G3 in particular is checked against a
  closed form the model is *derived from*. This gate is different in kind. Every
  quantity uncertain about the *level* of `D` — `γ_eff`, the absorbed fraction,
  radial relief, radiation losses, the Gaunt factor — enters as a coefficient,
  and no coefficient can produce a `1/3` exponent. The exponent is what the
  model predicts independently of the coefficient soup, and it is what measured
  LSD velocities are reported to follow.

  The EOS-independence leg is done by moving `γ`, since the table EOS is not
  wired into the hydro. `2(γ²−1)` runs 0.88 → 3.56 from `γ = 1.2` to `5/3`,
  shifting the level of `D` by 1.59×, while the fitted exponents move by 0.001
  and 0.002:

  | γ | 2(γ²−1) | D at S = 10¹¹ | S exponent | ρ₀ exponent |
  |---|---|---|---|---|
  | 1.2 | 0.88 | 4169 m/s | +0.33127 | −0.32895 |
  | 1.3 | 1.38 | 4842 m/s | +0.33176 | −0.32977 |
  | 1.4 | 1.92 | 5400 m/s | +0.33190 | −0.33020 |
  | 5/3 | 3.56 | 6632 m/s | +0.33213 | −0.33096 |

  This is **not** a demonstration that a real equilibrium EOS leaves the exponent
  alone — a `γ_eff` varying with local state is not a different constant `γ` —
  and the gate does not claim it is.

  The density sweep holds ambient **temperature** fixed (`p₀ ∝ ρ₀`), not
  pressure: at fixed `p₀` a decade of `ρ₀` moves the ambient internal energy by
  a decade, and at the thin end the undisturbed gas would cross the ignition
  threshold and the whole column would absorb. The threshold itself is 5×
  ambient `e₀`, not G3's fixed 2 MJ/kg — at the sweep corner (`γ = 1.2`,
  `ρ₀ = 12.25`) the post-shock state is only 11× ambient, so a 10× threshold
  there starts *controlling* the front rather than enabling it and drives the
  fitted exponent to −0.459. That the exponents agree to 1e-3 between 3× and 5×
  thresholds is the evidence the threshold is out of the loop.
- **`lsd_velocity_level_tracks_the_eos_coefficient`** — the counterpart to G4,
  pinning what the *level* is worth. Moving `γ` 1.4 → 1.2 must scale `D` by
  `(0.88/1.92)^(1/3) = 0.772`; measured **0.7722**. The solver tracks the
  coefficient exactly where the coefficient is knowable, which is the sharpest
  statement of why agreement on the level would not be evidence about the
  physics.

- **G8** (`plasma_column_absorbs_as_beer_lambert`) — the D7 coupling itself,
  end to end. A real beam marched through a `PlasmaColumn` built from G3's
  settled hydro state, against `exp(−τ)`, `τ = Σ α_k·dx`. The M2 twin
  (`beer_lambert_matches_closed_form`) does this for a constant absorber; this
  is its M6c counterpart with the absorber coming from gas dynamics. Added at
  step 6: until then D7's claim was carried by `PlasmaColumn`'s unit tests,
  which exercise its `Medium` methods in isolation, and no field had ever been
  marched through one. `δn ≡ 0` is asserted at every slab rather than assumed —
  a Drude index appearing there is the near-critical failure D7 avoids.

  Measured at `τ = 339`: **1.7e-13 at 500 slabs, 8.4e-14 at 100**, across 500
  successive amplitude multiplications against a single exponential. Two slab
  resolutions because `PlasmaColumn::from_column_resampled` — mean `α` over each
  bin, so `α_slab·dz = Σ α_k·dx` exactly — is what makes marching a 2500-cell
  hydro state through an FFT propagator affordable.

Not yet gated: **G7**, absolute velocity against measurement, which is expected
to land high and is documented-but-ungated because a planar 1-D solver has no
radial relief. See `docs/M6C_SPEC.md`.

### The `lsd` demonstration run (CLI case)

`beamprop lsd` (`src/cases.rs::run_lsd`, written by `src/main.rs`, rendered by
`scripts/render_lsd.py`) is the case that puts M6a and M6c in the same run: a
spark is lit at M6a's breakdown threshold and the absorption wave it launches is
tracked back up the beam. The igniting pulse's peak intensity comes from its
power and focal radius through `IntensityScale` — the T4 extraction's second
consumer, and the reason it was extracted.

**Its headline is a result, not a demonstration.** The case takes a short
*igniting* pulse and a separate long *sustaining* drive, and the two models
together say the second could never have produced the first:

- M6a's threshold in air at 1 atm saturates at **≈1.14×10¹⁶ W/m² and does not
  fall with pulse length** (6 ns → 1.18×10¹⁶, 1 ms → 1.14×10¹⁶). It is an
  intensity floor, not a fluence one: below it the inelastic losses paid
  climbing to the ionization potential exceed the inverse-bremsstrahlung
  heating, the net cascade rate is negative, and no exposure time rescues it.
  Widening the focus moves it by 4 % over a 500× range of spot radius.
- The sustaining LSD drive is ~10¹¹ W/m² — five orders of magnitude below.

So the detonation must be *initiated* by something far brighter than what
*sustains* it, which is the known experimental situation: LSD waves in clean air
are started on a target, on an aerosol, or by a separate spike. Pinned by
`the_sustaining_drive_is_far_below_the_breakdown_threshold`, so a future change
to either model that closes the gap fails rather than quietly invalidating the
write-up.

**What each half is worth.** *When and where* the spark lights inherits M6a's
explicitly ungated absolute level (4.8–7.0× above the measured T&T curve). The
front speed does not: it depends on the absorbed intensity at the front and on
`ρ₀`, not on where the spark was lit — which is why G3/G3b/G3c and the G4
physics gate all use seeded ignition and never touch `AirBreakdown`. The gap
above is likewise untouched by that uncertainty: 10⁵ against ~7×. Default run:
`D` = 5401 m/s against Raizer's 5391 (+0.19 %), energy budget closing to 1.3e-16,
final column optical depth 374.

**Why the grey closure drives it.** `GreyThreshold` is what G3–G5 gate and it
introduces nothing that can drift. The run *evaluates* the production
inverse-bremsstrahlung closure at its own measured post-front state rather than
asserting a reason for not using it, and the answer is informative: `α ≈ 6.8 1/m`
at 1064 nm, making the whole 2.5 cm column **0.17 optical depths** — nearly
transparent to the beam driving it, with no front, and `check_regime` would
correctly refuse it as volumetric — against `α ≈ 1.1×10³ 1/m` at 10.6 µm, an
absorption length of 0.92 mm that is 92 cells on the demo grid and 3.7 % of the
domain. Free-free absorption falls steeply toward short wavelengths, so this is
the model reproducing **why LSD experiments are done with CO₂ lasers**. What
blocks running that closure coupled is cost, and specifically the table
inversion: `PlasmaTable::temperature` bisects ~45 times per cell per deposition
call, three deposition calls per step. A faster inversion, not a finer grid, and
a separate change with its own gate.

References:
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer (2009) — HLLC (§10.4–10.6), MUSCL-Hancock (§14.4), and the
  exact Riemann solver and Test 1 tabulation (§4.3, §6.4) the gates use.
- Yu. P. Raizer, *Laser-Induced Discharge Phenomena*, Consultants Bureau (1977)
  — LSD wave theory, the detonation analogy, and the velocity closed form.
- G. B. Rybicki, A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley
  (1979), Eq. 5.18b — the thermal free-free absorption coefficient.
- G. Strang, SIAM J. Numer. Anal. **5**, 506 (1968) — the operator splitting.
- J. B. Scoggins et al., *Mutation++: multicomponent thermodynamic and
  transport properties for ionized gases*, SoftwareX **12**, 100575 (2020) —
  the equilibrium thermochemistry behind the plasma table.
- B. Einfeldt, *On Godunov-type methods for gas dynamics*, SIAM J. Numer. Anal.
  **25**, 294 (1988) — the wave-speed estimates.
- G. A. Sod, *A survey of several finite difference methods…*, J. Comput. Phys.
  **27**, 1 (1978) — the shock-tube problem.

Full model, the remaining gates (G3–G7), and which of them are verification
versus validation: `docs/M6C_SPEC.md`.

## Rendering (not physics)

The solver writes data only (`.npy` arrays + `_meta.json`/`_notes.md`
sidecars; collection helpers in `src/viz.rs`). All images come from
`scripts/render.py` (matplotlib): the perceptually-uniform **magma** colormap
applied to `t = (I/I_max)^γ` with `γ = 0.5` to lift the dim wings; `I_max` is
the global peak (across all frames of a GIF), so brightness differences
between frames are physical. Colorbars are labeled in `I/I_max`, axes in
metres.

Reference: S. van der Walt, N. Smith, *matplotlib colormaps* (magma),
<https://bids.github.io/colormap/>.
