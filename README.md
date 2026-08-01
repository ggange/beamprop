# beamprop

An open, validation-first solver for **laser beam propagation through the atmosphere**, written in Rust.

![A Gaussian beam through Kolmogorov turbulence: side view of one realization, speckled receiver-plane intensity, and the smooth 48-realization long-exposure mean](docs/turbulence.png)

*One Monte-Carlo realization of a 1 µm beam over 1 km of turbulence (`beamprop turbulence`, rendered by `scripts/render.py`): the instantaneous beam wanders and breaks into speckle; averaging 48 realizations recovers the smooth long-exposure profile that theory predicts to within 0.5%.*

Several effects stack when a laser crosses air, and `beamprop` aims to model each one rigorously and reproducibly:

- **Diffraction** — split-step wave-optics propagation.
- **Attenuation** — molecular and aerosol extinction (Beer–Lambert).
- **Turbulence** — Kolmogorov/von Kármán phase screens: beam wander, spreading, scintillation.
- **Thermal blooming** — the beam heats the air, the refractive index changes, wind and slew clear it, and the beam self-distorts. A coupled radiative-transport ↔ thermal-fluid problem.
- **Optical breakdown** *(in progress)* — above a threshold irradiance the air ionizes: an electron avalanche, then a plasma that absorbs the beam that made it. A standalone 0-D rate kernel (M6a) supplies the ignition threshold.
- **Laser-supported detonation** — once lit, that plasma does not sit still: the absorption wave runs *back up the beam* toward the laser as a detonation, at kilometres per second, closing the channel behind it. A 1-D Euler solver with laser deposition (M6c), coupled to the propagator as a pure absorber.
- **Radial relief** — a real beam has a finite diameter, so the shocked gas escapes sideways and the detonation runs slower than the planar theory says. An axisymmetric `(r, x)` Euler solver (M6d) measures the cost: about a quarter of the front speed. It also shows the front going **transversely unstable**, growing the cellular structure real detonations have.

## Scope

This repository is **pure propagation physics** — how a beam evolves through air. It deliberately contains no application-specific modeling of any kind, and none is planned here. The physics has broad civilian use: free-space optical communications, lidar, adaptive optics and astronomy, laser machining, and atmospheric science.

Every physical effect is anchored to a closed-form solution or a published benchmark **before** the next effect is added. The validation suite is the project's reason to be trusted.

## Status

Early, built one validated milestone at a time.

The table below is a summary; the **[claims ledger](docs/MODELS.md#claims-ledger)**
is the precise version. It states, claim by claim, what is *verified* against a
closed form, *validated* against an external measurement, *pinned* as a known
disagreement asserted green so it cannot drift, and simply *ungated*. The suite's
243 passing tests are not 243 validations — ten claims in this solver are checked
against external measured data, and the *level* of the breakdown-threshold curve
is not one of them (its high-pressure slope now is).

| Milestone | Content | State |
|-----------|---------|-------|
| M0 | Crate skeleton, `Field`/`Grid`, `.npy`+PNG output, CI | **done** |
| M1 | Symmetric split-step propagator through a `Medium` trait, validated: Gaussian evolution & divergence <1%, power conservation ~1e-14, boundary wraparound, 2nd-order convergence, long-throw Fresnel path | **done** |
| M2 | Beer–Lambert attenuation via the `Medium` trait, Kruse visibility model, validated: uniform extinction matches `exp(−α·z)` to ~1e-13, transverse absorber removes exactly the predicted power, `α = 0` bit-identical to vacuum | **done** |
| M3 | Von Kármán phase screens (FFT + subharmonics) + reproducible Monte-Carlo, validated: Kolmogorov structure function <10% over a decade of lags, long-exposure spread 0.5% off Andrews–Phillips, scintillation index 1.6% off Rytov, bitwise thread-count reproducibility | **done** |
| M3.5 | M4 pre-spec gate ([spec](docs/M4_SPEC.md)): fluid model, coupling scheme, stability bounds and the two anchor benchmarks (closed-form erf blooming phase, Gebhardt/Smith trend curve) pinned before code; air properties tabulated, no FFI | **done** |
| M4 | Coupled thermal blooming (steady-state isobaric, convection-dominated) through a field-aware `Medium`, frozen air-property table, validated: closed-form erf blooming phase 0.39% max, coupling 2nd-order by self-convergence (slope 2.000), weak-blooming first-order limit 0.008% with quadratic back-reaction residual (ratio 3.65 vs 4), stable at N_φ = 20 with closed power budget, upwind bend + crescent + irradiance-rollover signatures, and the Smith-1977 whole-beam I_REL(N) curve reproduced to 7.2% over N ∈ [0.5, 1.8] (F₀ = 5) | **done** |
| M5 | Python bindings (PyO3, abi3) + CI wheels ([spec](docs/M5_SPEC.md)): `import beamprop` exposes the core classes and a `run_*` helper for **every** CLI case. Gated: Python results bit-identical to the CLI, closed-form Gaussian width <1% (≈2e-11 observed), seed-exact Monte-Carlo determinism, solver validity errors surfaced as `ValueError`; wheels built and gated on linux/macOS/windows | **done** |
| M6a | 0-D optical-breakdown threshold kernel ([spec](docs/M6A_SPEC.md)): electron-avalanche balance (inverse-bremsstrahlung heating − inelastic loss − attachment − diffusion) with a distribution-resolved cascade closure, PPT photoionization, a produced (not assumed) seed, free-molecular escape, an exact per-slice logistic integrator and the `breakdown` CLI case. **Validated:** the PPT rate in absolute magnitude against a measured O₂ cross-section, landing 1.99× high (Sci. Rep. **8**, 2874); collision frequency 1.05× of literature and flat over 46–1858 Torr; the threshold slope against Thiyagarajan & Thompson 2012 (Fig. 4, digitized) — the measured 0.329 sits inside the model's `δ_eff` envelope [0.174, 0.382] (centre 0.264); Chylek's 10–100 Torr branch at 0.501 vs 0.428 measured. **Pinned (known disagreements, asserted green):** the wavelength ratio at 2.854 vs 0.80 measured, and a mid-pressure shape defect — 70–350 Torr runs 2.0–2.3× too steep, unreachable by any constant the model carries, and M6a's sharpest open question. **Ungated:** the absolute level, which sits inside the 3–10× inter-lab scatter | **done** |
| M6c.0 | M6c pre-spec gate ([spec](docs/M6C_SPEC.md)): 1-D Euler + laser deposition for the laser-supported-detonation wave — laser-agnostic HLLC/MUSCL-Hancock core, Strang-split source, plasma column coupled to the propagator as absorption-only, offline plasma-property table. Gates pinned before code: Sod and observed-order as solver verification; Raizer's LSD velocity closed form labelled **verification, not validation** (it is the Chapman–Jouguet construction the model is built from); the parameter-free `D ∝ S^(1/3)`, `ρ₀^(−1/3)` scaling as the physics gate; energy-budget closure; and absolute velocity vs measurement documented but **ungated** | **done** |
| M6c | Laser-supported detonation wave ([spec](docs/M6C_SPEC.md)): 1-D Euler core (HLLC + MUSCL-Hancock, minmod, per-step CFL, a positivity guard that bails rather than clamps), frozen equilibrium plasma table to 30,000 K, the coupled `LsdColumn` driver, and the `lsd` CLI case. **Physics gate (G4):** `D ∝ S^(+0.33190)` over 1.52 decades and `ρ₀^(−0.33020)` over 1.50 decades, gated inside ±0.01 of ±1/3 — with the level moving 59% under `γ` while the exponents move by 0.001. **Verification:** Sod vs the exact Riemann solution, L1(ρ) 6.55e-3 → 6.55e-4 over n = 100 → 1600 (G1); observed order 1.86 → 1.94 on smooth flow (G2); coupled hydro↔source order ≈1.99 against a deliberate 1st-order contrast (G2b); Raizer's velocity 5402 vs 5392 m/s, +0.19% (G3 — **verification, not validation**, and the spec says why); energy budget closed to 2.1e-16 (G5); plasma table vs direct Mutation++, worst 1.48e-3 (G6); a beam marched through the column matching `exp(−τ)` to 1.7e-13 at τ = 339 (G8). The demonstration run's headline is that M6a's threshold is an *intensity floor*, so the sustaining drive sits 10⁵ below the intensity that could light the wave — the model reproducing why an LSD wave needs a separate initiating spark. G7 (absolute velocity vs measurement) stays **ungated** — since M6d, for one reason only: no measured dataset has been anchored | **done** |
| M6a.2 | Turbulence-degraded ignition statistics ([spec](docs/M6A2_SPEC.md)): pupil optics (`src/aperture.rs`), the Monte-Carlo driver and the `ignition` CLI case. **No focal grid** — in the Fraunhofer regime peak focal intensity is a pupil integral, and one grid cannot carry both centimetre samples over a kilometre and a micrometre-wide focal spot. **Physics gates:** pupil residual phase variance against Noll (1976) — tip/tilt-removed 0.1407 vs 0.134, banded at ±12% by the measured ensemble spread (N1), and piston-removed converging to 1.0299 as `L₀/D → ∞` (N2); RMS focal-spot wander `∝ Cn²^(+0.495–0.499)` against the theoretical 1/2 (W1). **Verification:** the estimator reduces to Maréchal in the weak limit, a pure tilt steers without dimming, ensemble convergence and bitwise thread-count reproducibility (E1, E2). **Ungated by design:** where the ignition curve sits on the `Cn²` axis rides M6a's absolute threshold, so the shape is the result and the position is not — the figure says so in-panel. Three gates were specified and then **withdrawn** — a `(D/r₀)^(5/3)` exponent as a tautology, an aperture-dependence gate as seed-dependent, and a width gate for having no independent anchor; the spec records each | **done** |
| M6d.0 | M6d pre-spec gate ([spec](docs/M6D_SPEC.md)): axisymmetric `(r, x)` Euler, so a **finite-diameter** beam can relieve laterally — the one effect M6c's planar geometry removed by assumption, and the entire justification its G7 gave for being ungated. Pinned before code: an area-weighted finite-volume discretisation on annular cells, which puts zero area on the axis interface and cancels the geometric source against the pressure flux bit-exactly for a radially uniform state; Strang with the radial sweep outside, so the planar limit is bit-for-bit `Euler1d`; the 1-D HLLC reused unmodified; the axis as odd-parity ghosts. Gates pinned before code: the planar limit, **Sedov–Taylor** (the repo's first multidimensional anchor), 2nd order on smooth axisymmetric flow, conservation in the `r`-weighted measure, the wide-beam limit, and — the headline — the **relief deficit `δ = 1 − D_2D/D_1D` measured and pinned**, required to be insensitive to grid, seed and ignition threshold before it is pinned at all. **No validation gate, on purpose** | **spec'd** |
| M6d | Axisymmetric gas dynamics and radial relief ([spec](docs/M6D_SPEC.md)): `src/euler2d.rs`, `src/lsd2d.rs` (the coupled column with a finite-diameter beam) and the `lsd2d` CLI case. **Pinned physics:** radial relief costs **δ = 0.230** of the front speed at `R_b·α` = 3.2 and 0.305 at 1.6, monotone in beam radius, banded at ±13 % — a width *measured* from the grid (+6 %), seed (−7 %) and ignition-threshold (±8 %) sensitivities rather than chosen, and shown not to be a boundary effect. M6c's G4 argued relief can only enter as a coefficient; M6d measures it — the exponent survives at `S^0.34666` while the level moves 23 % (G16). **Verification:** the planar limit reproduces `Euler1d` **bit for bit** (G9); Sedov–Taylor exponent 0.38628 vs the exact 2/5, with level and peak compression gated as trends under refinement (G10); 2nd order on smooth axisymmetric flow, 1.861/1.964 against a split-source contrast at 1.030/1.155 (G11); conservation in the `r dr dx` measure to <1e-13 with an escape-flux leg (G12); the axis does not heat, 2.99e-7 → 3.12e-8 under refinement against 3.02e-6 for an even-parity contrast (G13); the wide-beam limit reproduces the 1-D column to 3.1e-13 (G14). **The unlooked-for result:** the modelled front is **transversely unstable**, growing cellular structure out of round-off and saturating at \|u_r\| ≈ 200–400 m/s in planar and axisymmetric geometry alike. A 1-D solver structurally cannot show it, and it had to be separated from relief before the relief number meant anything. G7 stays **ungated**, now for one reason only — the missing measured dataset | **done** |

## Build & run

```sh
cargo build --release
cargo test

# write a Gaussian field's intensity to out/beam.npy
cargo run --release -- gaussian --n 512 --dx 1e-3 --w0 5e-2 --out beam

# propagate a beam over 2 Rayleigh ranges (side-view map, snapshots, final field)
cargo run --release -- propagate --w0 1e-2 --steps 400 --frames 4 --out beam

# same, through a 5 km-visibility haze (Kruse aerosol extinction at the beam wavelength)
cargo run --release -- propagate --w0 1e-2 --z 200 --visibility 5000 --out hazy

# Monte-Carlo turbulence: receiver-plane + side-view frame stacks and the long-exposure mean
cargo run --release -- turbulence --n 256 --dx 2e-3 --w0 1e-2 --z 1000 --cn2 1.5e-14 --out turb

# thermal blooming: a 20 kW beam heating the air, bending into a 2 m/s crosswind
cargo run --release -- blooming --w0 5e-2 --power 2e4 --wind 2 --alpha-abs 1e-4 --z 500 --out bloom

# 0-D optical breakdown (M6a): threshold vs pressure + the electron avalanche
cargo run --release -- breakdown --out breakdown

# laser-supported detonation (M6c): a spark at the M6a threshold, then the
# absorption wave running back up the beam toward the laser
cargo run --release -- lsd --out lsd

# turbulence-degraded ignition (M6a.2): how often a focused beam still lights
# the air across a Cn^2 sweep, and where the spark lands
cargo run --release -- ignition --out ignition

# axisymmetric LSD (M6d): the same detonation driven by a finite-diameter beam,
# so the shocked gas can escape sideways and the front slows down
cargo run --release -- lsd2d --out lsd2d

# render the images: GIFs/PNGs with physical axes and a labeled colorbar (matplotlib)
python3 scripts/render.py out/turb
python3 scripts/render_breakdown.py out/breakdown   # breakdown runs
python3 scripts/render_lsd.py out/lsd               # LSD runs
python3 scripts/render_ignition.py out/ignition      # ignition sweeps
python3 scripts/render_lsd2d.py out/lsd2d           # axisymmetric LSD runs

# remove generated results (images, .npy and sidecars in the output directory)
cargo run --release -- clean
```

The solver writes **data** (`.npy` arrays plus `_meta.json`/`_notes.md` sidecars); all images come from `python3 scripts/render.py <basename>` (or `scripts/render_breakdown.py` for `breakdown` runs, which are 0-D rate physics rather than fields). Generated files land in `out/` by default (`--out-dir` overrides). Each `propagate`/`turbulence` run's `<out>_notes.md` describes the test case: parameters, derived physical quantities (Rayleigh range, Fried parameter, Rytov variance, …), what each file contains with its physical axes, and how the images are normalized. `cargo run --release -- --help` (or `--help` on any subcommand) lists all options.

## Python bindings

The same solver is importable from Python (PyO3/maturin, `beamprop-py/`):

```sh
pip install maturin
maturin develop --release -m beamprop-py/Cargo.toml   # or: pip install a CI wheel
```

```python
import beamprop as bp

# high-level: one call per field-propagation case, numpy arrays + diagnostics back
r = bp.run_blooming(w0=5e-2, power=2e4, wind=2.0, alpha_abs=1e-4, z=500.0)
print(r["n_phi"], r["centroid_x"], r["final"].shape)

# or compose the pieces: propagate a field step by step through a medium
g = bp.Grid(512, 1e-3)
f = bp.Field.gaussian(g, 1.0e-6, 1e-2)
m = bp.Medium.turbulence(g, 1.0e-6, cn2=1.5e-14, l0=1e3, z=1000.0, screens=10, seed=1)
bp.Propagator(g, 1.0e-6).propagate(f, m, dz=1000.0 / 10, steps=10)
speckle = f.intensity          # numpy, ready to plot
```

The bindings return **data only** (rendering stays in `scripts/render.py`), and their gate suite requires Python results to be bit-identical to the CLI (`beamprop-py/tests/`, run in CI and against every built wheel). They cover the three field-propagation cases; the 0-D `breakdown` case is Rust/CLI-only for now.

Every physical model in the solver — equation, implementation site, validation gate, and literature reference — is catalogued in [docs/MODELS.md](docs/MODELS.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or <https://www.apache.org/licenses/LICENSE-2.0>). Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.
