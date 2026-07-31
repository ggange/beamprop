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

## Scope

This repository is **pure propagation physics** — how a beam evolves through air. It deliberately contains no application-specific modeling of any kind, and none is planned here. The physics has broad civilian use: free-space optical communications, lidar, adaptive optics and astronomy, laser machining, and atmospheric science.

Every physical effect is anchored to a closed-form solution or a published benchmark **before** the next effect is added. The validation suite is the project's reason to be trusted.

## Status

Early, built one validated milestone at a time.

The table below is a summary; the **[claims ledger](docs/MODELS.md#claims-ledger)**
is the precise version. It states, claim by claim, what is *verified* against a
closed form, *validated* against an external measurement, *pinned* as a known
disagreement asserted green so it cannot drift, and simply *ungated*. The suite's
213 passing tests are not 213 validations — ten claims in this solver are checked
against external measured data, and the *level* of the breakdown-threshold curve
is not one of them (its high-pressure slope now is).

| Milestone | Content | State |
|-----------|---------|-------|
| M0 | Crate skeleton, `Field`/`Grid`, `.npy`+PNG output, CI | **done** |
| M1 | Symmetric split-step propagator through a `Medium` trait, validated: Gaussian evolution & divergence <1%, power conservation ~1e-14, boundary wraparound, 2nd-order convergence, long-throw Fresnel path | **done** |
| M2 | Beer–Lambert attenuation via the `Medium` trait, Kruse visibility model, validated: uniform extinction matches `exp(−α·z)` to ~1e-13, transverse absorber removes exactly the predicted power, `α = 0` bit-identical to vacuum | **done** |
| M3 | Von Kármán phase screens (FFT + subharmonics) + reproducible Monte-Carlo, validated: Kolmogorov structure function <10% over a decade of lags, long-exposure spread 0.5% off Andrews–Phillips, scintillation index 1.6% off Rytov, bitwise thread-count reproducibility | **done** |
| M3.5 | M4 pre-spec gate ([docs/M4_SPEC.md](docs/M4_SPEC.md)): fluid model (steady-state isobaric, convection-dominated), slab-local predictor–corrector coupling with a 2nd-order gate, stability/resolution bounds, closed-form anchor benchmark (erf blooming phase) + Gebhardt/Smith trend curve, air-property tabulation pinned (no FFI) | **done** |
| M4 | Coupled thermal blooming (steady-state isobaric, convection-dominated) through a field-aware `Medium`, frozen air-property table, validated: closed-form erf blooming phase 0.39% max, coupling 2nd-order by self-convergence (slope 2.000), weak-blooming first-order limit 0.008% with quadratic back-reaction residual (ratio 3.65 vs 4), stable at N_φ = 20 with closed power budget, upwind bend + crescent + irradiance-rollover signatures, and the Smith-1977 whole-beam I_REL(N) curve reproduced to 7.2% over N ∈ [0.5, 1.8] (F₀ = 5) | **done** |
| M5 | Python bindings (PyO3, abi3) + CI wheels ([docs/M5_SPEC.md](docs/M5_SPEC.md)): `import beamprop` exposes the core classes and `run_*` helpers — since bindings v2 that is **every** CLI case, the three propagation ones plus `run_breakdown`, `run_lsd` and `run_ignition` — validated: CLI compute loops extracted to shared pure runners with bit-identical `.npy` outputs, Python results bit-identical to the CLI for all three cases, closed-form Gaussian width <1% (≈2e-11 observed), seed-exact Monte-Carlo determinism, solver validity errors as `ValueError` (including M6's refuse-don't-mis-model guards, and the LSD below-threshold *clean report* arriving as data rather than an exception); wheels built+gated on linux/macOS/windows in CI | **done** |
| M6a | 0-D optical-breakdown threshold kernel ([docs/M6A_SPEC.md](docs/M6A_SPEC.md)): electron-avalanche balance (inverse-bremsstrahlung heating − inelastic loss − attachment − diffusion), exact per-slice logistic integrator, log-bisection threshold, pressure sweep, `breakdown` CLI case. Validated against Thiyagarajan & Thompson 2012 (Fig. 4, digitized): collision frequency 1.05× of literature and flat over 46–1858 Torr, `E_eff(p)` slope `p^+0.695` vs predicted `p^+0.642`, wavelength scaling `λ^-2.000` over a 20× span matching the paper's cascade closed form (Eq. 4) with the plateau coefficient agreeing to 1.01×. Absolute level sits inside the ungated 3–10× inter-lab scatter. **`D_e` gated (2026-07-30):** kinetic theory ties it to the already-validated `K_m`, so `D_e,ref` = 0.2 m²/s *is* the statement `ε` = 6.740 eV; sweeping the whole band that admits (`ε` = 2 eV → `U_i`, a 6.0× range) moves the fitted slope only 0.053 → 0.155 (mean-trajectory closure) against a measured 0.329, so `D_e` **cannot** explain the slope gap — and it exposes a pinned 2.25× inconsistency with the `⟨ε⟩` = 3 eV the loss term assumes. **General-gas kernel (2026-07-30):** gas constants split into a `Gas` value (bit-identical output), unlocking Chylek's He/Ar/Xe curves. The cascade's plateau floor `δ_eff·U_i·m_e c ε₀ ω²/e²` contains *no* transport constant — `ν_m` cancels exactly — so for a monatomic gas, where `δ = 2m_e/M` is the atomic mass, it is a **parameter-free** prediction: the He > Ar > Xe ordering is right, the spacing is not (He/Ar 15.6 predicted vs ≈2.5 measured), and the headroom above the floor runs 1.85× / 7.8× / 13.2×, mass-ordered — though that floor is a hard bound only for the mean-trajectory closure, which the next entry replaces. **Distribution-resolved cascade (2026-07-30):** the mean-trajectory closure ionizes only above `ε_∞ = U_i`, a hard bifurcation the model sits on top of (`ε_∞/U_i` = 1.032 at 760 Torr). Replacing the trajectory with the Ornstein–Uhlenbeck process the photon shot noise `D_ε = ½·P_heat·ħω` implies — **no new constant** — and solving first passage to `U_i` by Siegert's formula **fixes the high-pressure branch**: at the untouched literature centre the T&T slope goes 0.0951 → 0.2793 (measured 0.329) and Chylek's 0.1717 → 0.4665 (measured 0.468), moving the measurement from *outside* the model's one-free-constant envelope to *inside* it. It does **not** fix the low-pressure branch (1.952 → 1.954 vs 0.428, so that failure is diffusion, not the cascade closure) and does **not** close the wavelength gap (4.00 → 3.39 vs ≈0.80 — the right sign at last, 15 % of a 5× gap). **Promoted to default** once those numbers were measured, which **retired M6a's red gate**: `tt2012_threshold_slope_matches_measurement` had been `#[ignore]`d and failing since 2026-07-25, and now passes with no tolerance moved and no constant touched — the measured 0.329 sits inside the `δ_eff` envelope `[0.183, 0.407]` where the old closure's `[0.023, 0.231]` excluded it. The suite has no ignored gates left. Promotion also refined an M6c claim: M6a's threshold is an *asymptotic* intensity floor rather than a flat one — 8.510e15 at 6 ns converging to 6.797e15 by ~10 µs, a bounded 1.25× fall, not the fluence criterion that would break M6c's two-stage argument. **Free-molecular escape (2026-07-30):** the low-pressure branch turned out to be a *validity* failure, not a value one. `ν = D_e/Λ²` is a continuum random-walk result, and the Knudsen number `Kn = λ_mfp/ℓ` runs 0.013 at 760 Torr to **0.96 at 10 Torr** — the kernel was applying a continuum formula in the collisionless regime, across the whole window where it was worst. Escape time is the diffusive time *plus* the ballistic transit time, `ν_esc = 1/(Λ²/D_e + ℓ/v̄)`, with **no new constant** (`v̄ = √(3·D_e,ref·p_ref·K_m)` = 6.740 eV, the same energy `D_e` implies) and `ℓ = 4V/S` the Cauchy mean chord of the pinned focus — which is **4.0× `Λ`**, since `Λ` is a diffusion eigenvalue and not a distance. That takes the low-pressure slope **1.954 → 1.293** against a measured 0.428: about half the error, and the gate asserts the remaining 2.6× as loudly as the improvement. Both source papers say MPI dominates below 100 Torr, which is what a cascade-plus-loss model cannot reach.  **PPT for molecular O₂ (2026-07-31):** the multiphoton question closes, and not the way it was framed. Keldysh's soft prefactor was replaced by PPT's *derived* one — fully determined once `Z_eff` is given, and `Z_eff` = 0.53 for O₂ is published (Talebpour 1999), so nothing is fitted. The rate is then **validated in absolute magnitude against a measured cross-section** rather than against threshold data: `σ₈ = (3.3 ± 0.3)×10⁻¹³⁰ W⁻⁸m¹⁶s⁻¹` for O₂ at 800 nm, obtained by counting the electrons directly with Rayleigh microwave scattering (Sci. Rep. **8**, 2874 (2018)) — and it lands at **1.99×**, high, the direction that paper reports for purely theoretical predictions. `K` = 8 there sits between the kernel's `K` = 11 and 6, so it is interpolation. Two consequences. **The prefactor escape hatch is shut**: `n* = Z_eff/κ` = 0.563 makes the Coulomb exponent `2n* − 3/2` *negative*, so the correction that lifts an atomic rate by orders of magnitude is order-unity for a molecule — the λ ratio moves only 3.349 → 2.947 against a measured 0.80, 16 % of the gap, and that is now a statement about a validated rate rather than about a free multiplier. **And the two anchor experiments turn out not to be in the same regime**: evaluated at each paper's own measured threshold, with no model threshold in the calculation at all, multiphoton ionization supplies 3.15 electrons per pulse at Chylek's 532 nm point — the measured threshold *is* the seeding threshold, to 17 % — against 5.4×10⁻⁹ at T&T's 1064 nm point, which is 5.7× short. So the residual wavelength discrepancy is part cascade closure and part a comparison between two different mechanisms. A gate written to assert PPT's ponderomotively shifted order also found, by failing, that the above-threshold sum returns the **integer** photon order (10.998 at 1064 nm where `ν` = 10.34) where the bare Keldysh exponential gives a fractional one. The independent-anchor debts this leaves open are recorded in [docs/MODELS.md](docs/MODELS.md) | **done** |
| M6c.0 | M6c pre-spec gate ([docs/M6C_SPEC.md](docs/M6C_SPEC.md)): 1-D Euler + laser deposition for the laser-supported-detonation wave — laser-agnostic HLLC/MUSCL-Hancock core, Strang-split source, plasma column coupled to the propagator as absorption-only (read-only `Medium`, no Drude index), offline plasma-property table. Gates pinned before code: Sod + observed-order (solver verification), Raizer's LSD velocity closed form (**verification, not validation** — it is the Chapman–Jouguet construction the model is built from), the parameter-free `D ∝ S^(1/3)`, `ρ₀^(−1/3)` scaling as the physics gate, energy-budget closure, table consistency, and absolute velocity vs measurement documented but ungated (a planar solver has no radial relief, so the known experimental gap is a prediction of the omissions) | **done** |
| M6c | Laser-supported detonation wave ([docs/M6C_SPEC.md](docs/M6C_SPEC.md)): laser-agnostic 1-D Euler core (HLLC + MUSCL-Hancock, minmod, per-step CFL, positivity guard that bails rather than clamps), `IntensityScale` extracted from M4 (T4), frozen equilibrium plasma table to 30,000 K, the coupled `LsdColumn` driver (Beer–Lambert attenuation, discretely conservative deposition, Strang-split source), and the `lsd` CLI case + `scripts/render_lsd.py`. **Verification:** Sod vs the exact Riemann solution, L1(ρ) 6.55e-3 → 6.55e-4 over n = 100 → 1600 at rate 0.79–0.88 (G1); observed order 1.86 → 1.94 on smooth flow (G2); coupled hydro↔source order 1.99/2.03/1.99 against a deliberate 1st-order contrast at 0.88/1.02/1.07 (G2b); Raizer's LSD velocity 5402 vs 5392 m/s, +0.19% (G3 — **verification, not validation**: it is the Chapman–Jouguet construction the model is built from); residual −8.26% → +0.19% as the absorption layer halves (G3b); seed-independent to 1.1e-3 (G3c); energy budget closed to 2.1e-16 (G5); plasma table vs direct Mutation++ off-grid, worst 1.48e-3 in `n_e` (G6); a real beam marched through the plasma column matching `exp(−τ)` to 1.7e-13 at τ = 339, with `δn ≡ 0` asserted at every slab (G8 — D7's absorption-only coupling, gated end to end). **Physics gate (G4):** `D ∝ S^(+0.33190)` over 1.52 decades and `ρ₀^(−0.33020)` over 1.50 decades, gated inside ±0.01 of ±1/3, with the level shown to move 59% under `γ` while the exponents move by 0.001. The demonstration run ignites at the M6a threshold and tracks the wave (`D` = 5401 vs 5391 m/s); its headline is that M6a's threshold is an *intensity floor* — it does not fall with pulse length — so the sustaining drive sits 10⁵ below the intensity that could light the wave, which is the model reproducing why an LSD wave needs a separate initiating spark. G7 (absolute velocity vs measurement) is documented and **ungated on purpose**: a planar solver has no radial relief | **done** |
| M6a.2 | Turbulence-degraded ignition statistics ([docs/M6A2_SPEC.md](docs/M6A2_SPEC.md)): pupil optics (`src/aperture.rs`) + the Monte-Carlo driver + the `ignition` CLI case and `scripts/render_ignition.py`. **No focal grid** — in the Fraunhofer regime the on-axis focal amplitude is the DC component of the pupil field's Fourier transform, so peak focal intensity is a pupil integral; turbulence needs centimetre samples over a kilometre while the focal spot is micrometres across, and one grid cannot carry both. **Physics gates:** pupil residual phase variance against Noll (1976) — tip/tilt-removed **0.1407 vs 0.134**, banded at ±12% by the measured ensemble spread (N1), and piston-removed converging to 1.0299 as `L₀/D → ∞`, 0.34 → 0.99 over `L₀/D` = 10 → 2000 (N2); RMS focal-spot wander `∝ Cn²^(+0.4953/0.4977/0.4987)` against the theoretical 1/2 (W1); an aperture-dependence gate (W2) was landed and then **retired as seed-dependent** — the observation (the pupil only matters once it truncates the beam) is documented, but the fitted exponent swings −0.10 to −0.32 across seeds at any affordable ensemble size, so it was measuring the draw. **Verification:** the estimator reduces to Maréchal in the weak limit, a pure tilt steers without dimming, an amplitude-only perturbation is not counted as wavefront error, tilt survives phase wrapping where a plane fit does not; ensemble convergence and bitwise thread-count reproducibility (E1, E2). **Ungated by design:** where the ignition curve sits on the `Cn²` axis rides M6a's absolute threshold, so the shape is the result and the position is not — the figure says so in-panel. A `(D/r₀)^(5/3)` exponent gate was specified, implemented, passed at 1.66667, and **withdrawn as a tautology** (the generator scales the screen linearly in `r₀`); so was a width gate, for having no independent anchor | **done** |

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

# render the images: GIFs/PNGs with physical axes and a labeled colorbar (matplotlib)
python3 scripts/render.py out/turb
python3 scripts/render_breakdown.py out/breakdown   # breakdown runs
python3 scripts/render_lsd.py out/lsd               # LSD runs
python3 scripts/render_ignition.py out/ignition      # ignition sweeps

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
