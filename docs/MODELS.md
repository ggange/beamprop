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
