# beamprop

An open, validation-first solver for **laser beam propagation through the atmosphere**, written in Rust.

Four effects stack when a laser crosses air, and `beamprop` aims to model each one rigorously and reproducibly:

- **Diffraction** — split-step wave-optics propagation.
- **Attenuation** — molecular and aerosol extinction (Beer–Lambert).
- **Turbulence** — Kolmogorov/von Kármán phase screens: beam wander, spreading, scintillation.
- **Thermal blooming** — the beam heats the air, the refractive index changes, wind and slew clear it, and the beam self-distorts. A coupled radiative-transport ↔ thermal-fluid problem.

## Scope

This repository is **pure propagation physics**. It contains no application-specific content — no weapon, comms, or lidar modeling. Those are downstream consumers of the delivered-field output and live in separate projects. What is here has obvious civilian homes: free-space optical communications, lidar, adaptive optics and astronomy, laser machining, and atmospheric science.

Every physical effect is anchored to a closed-form solution or a published benchmark **before** the next effect is added. The validation suite is the project's reason to be trusted.

## Status

Early, built one validated milestone at a time.

| Milestone | Content | State |
|-----------|---------|-------|
| M0 | Crate skeleton, `Field`/`Grid`, `.npy`+PNG output, CI | **done** |
| M1 | Symmetric split-step propagator through a `Medium` trait, validated: Gaussian evolution & divergence <1%, power conservation ~1e-14, boundary wraparound, 2nd-order convergence, long-throw Fresnel path | **done** |
| M2 | Beer–Lambert attenuation via the `Medium` trait, Kruse visibility model, validated: uniform extinction matches `exp(−α·z)` to ~1e-13, transverse absorber removes exactly the predicted power, `α = 0` bit-identical to vacuum | **done** |
| M3 | Von Kármán phase screens (FFT + subharmonics) + reproducible Monte-Carlo, validated: Kolmogorov structure function <10% over a decade of lags, long-exposure spread 0.5% off Andrews–Phillips, scintillation index 1.6% off Rytov, bitwise thread-count reproducibility | **done** |
| M4 | Coupled thermal blooming | planned |
| M5 | Python bindings (PyO3) + wheels | planned |

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

# render the images: GIFs/PNGs with physical axes and a labeled colorbar (matplotlib)
python3 scripts/render.py out/turb

# remove generated results (images, .npy and sidecars in the output directory)
cargo run --release -- clean
```

The solver writes **data** (`.npy` arrays plus `_meta.json`/`_notes.md` sidecars); all images come from `python3 scripts/render.py <basename>`. Generated files land in `out/` by default (`--out-dir` overrides). Each `propagate`/`turbulence` run's `<out>_notes.md` describes the test case: parameters, derived physical quantities (Rayleigh range, Fried parameter, Rytov variance, …), what each file contains with its physical axes, and how the images are normalized. `beamprop --help` lists all options; further analysis happens in Python/NumPy against the `.npy` output until the PyO3 bindings arrive at M5.

Every physical model in the solver — equation, implementation site, validation gate, and literature reference — is catalogued in [docs/MODELS.md](docs/MODELS.md).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or <https://www.apache.org/licenses/LICENSE-2.0>). Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.
