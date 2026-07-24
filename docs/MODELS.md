# Physical models and references

Every physical model in `beamprop`, with its governing equation, where it is
implemented, the validation gate that pins it, and the literature it comes
from. This file is the citation record for the solver: if a formula is in the
code, it is in this table.

Conventions: `λ` vacuum wavelength (m), `k = 2π/λ` (rad/m), `z` propagation
distance (m), `κ` transverse spatial frequency (rad/m), intensity `I = |u|²`.

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

attachment from measured rate coefficients (dissociative `k₂·n_O₂ ∝ p` plus
three-body `k₃·n_O₂·n ∝ p²`, the latter dominant at atmospheric density, and
both negligible here — `6.7×10⁷` vs `4.9×10⁹ s⁻¹` for diffusion at 1 atm),
free-electron diffusion loss
`ν_diff = D_e(p)/Λ²` with `D_e ∝ 1/p` over the geometry-pinned diffusion length
`Λ` (T&T 20 µm focus as a sphere, `Λ = r/π` — **pinned, never fit**), and a
swappable multiphoton seed `S_mpi = σ_K·I^K·N` (off by default; the seed is one
electron in the focal volume). Breakdown at `n_e ≥ n_bd = 10²³ m⁻³`.

The linear-per-slice ODE is advanced by its exact exponential solution
`n_e' = n_e·e^{βdt} + S·expm1(βdt)/β` (`β = ν_i − ν_loss`). Threshold intensity
is found by log-bisection; a pressure sweep gives `I_thr(p)`.

Implemented in `src/breakdown0d.rs`.

Gate (internal, model self-consistency): solving the avalanche criterion gives

```text
I_thr(p) = L′/h + U_i·(ν_att + ν_diff + G)/(h·p),   G = ln(n_bd/n_seed)/τ
```

Attachment is negligible here, so the exponent runs between two **exact**
limits — plateau-dominated `n → 0` and diffusion-dominated `n → 2` — with the
growth-limited `p^-1` in between. Fitted over the pinned range **300–2000
Torr** (8 log-spaced points, 6 ns FWHM) the model gives **n = 0.356**, and the
gate asserts `n ∈ (0, 1)`: the plateau must be doing work, since without `L`
the model is stuck at `n ≥ 1`. Sweeping the literature ranges of `L′`
(δ_eff ≈ 0.01–0.05) gives an envelope `n ∈ [0.154, 0.583]`, separately pinned so
it cannot drift. The level at 760 Torr is **1.32×10¹² W/cm²**, and the
`FixedMeanEnergy` variant gives `n = 0.800` as the other end of the bracket.
Absolute threshold level is **not** gated (3–10× inter-lab
scatter). Integrator sub-gates are unit tests of the exact-exponential solver,
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
- **`I_thr(p)` slope — PASSES to 8%.** Measured `p^-0.329`, kernel `p^-0.356`.
  The route: `p^-1.74` originally (where `n = 1` was a hard floor and the
  measurement was unreachable at any parameter value), `p^-0.80` once inelastic
  losses were subtracted from the heating, `p^-0.356` once `⟨ε⟩` was eliminated
  by solving the climb ODE exactly. `δ_eff = 0.02` predates the self-consistent
  model, so this is a prediction, not a fit. Gated as containment in the
  `δ_eff` literature envelope plus a factor-1.5 band on the central value. This first looked
  like a 2× attachment problem; implementing attachment from measured rate
  coefficients showed real attachment is ~150× *smaller* than the
  order-of-magnitude constant it replaced, moving the model from `p^-0.72` to
  `p^-1.74` — the earlier near-agreement was an artifact of a wrong constant.
  Since `ν_i ∝ I·p`, a threshold set by a fixed growth requirement gives
  `I_thr ∝ p^-1` exactly when losses vanish — `n = 1` is the model's **flat
  floor**, and diffusion only steepens it, so the measured 0.33 lies below what
  the model can produce. Only a loss growing faster than `p` flattens past it;
  three-body attachment qualifies but at its measured coefficient that regime
  starts near 10⁴ Torr, so reaching 0.33 in-window would need `k₃` ≈ 100× the
  measured value. The defensible route is the cascade coefficient itself
  falling with pressure (`A ∝ p^-0.67`), as an effective `U_i` growing with
  collision frequency would produce. That is new physics and is M6a's open
  question.
  On level, the model now sits uniformly above the data by a bounded amount —
  model/measured runs 4.69× at 380 Torr to 2.41× at 1896 Torr, inside the
  ungated inter-lab scatter (before the inelastic term it *crossed* the data,
  3.10× → 0.33×). Converting `E_B` to intensity uses `I = ε₀cE_rms²`, since the
  `E_eff` ratio establishes `E_B` is an RMS amplitude.

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
  — the threshold-vs-pressure anchor (external gate, pending digitization).

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
