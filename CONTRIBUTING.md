# Contributing to beamprop

Thanks for your interest. `beamprop` is a **validation-first** solver: the
validation suite is the reason the project can be trusted, and it comes before
features. Please read the rules below before opening a pull request.

## Core principles

- **No new physics without a gate.** Every physical effect is anchored to a
  closed-form solution or a published benchmark, with a passing validation
  test, *before* the next effect is built on top of it. A change that adds or
  alters physics must add or update the corresponding gate and cite its
  reference.
- **Pure propagation physics only.** This repository models how a beam evolves
  through air (diffraction, attenuation, turbulence, thermal blooming). It
  contains no application-specific modeling, and none is planned here.
- **SI units and `f64` throughout.** No mixed unit systems; keep quantities in
  SI at every interface.
- **Data and images are separate.** The Rust solver writes *data* — `.npy`
  arrays plus `_meta.json` / `_notes.md` sidecars. All images come from
  `python3 scripts/render.py <basename>`. Do not add plotting to the Rust side.

## Before you push

CI (`.github/workflows/ci.yml`) builds with `RUSTFLAGS="-D warnings"`, so any
compiler or Clippy warning fails the build. Run all four gates locally, in this
order, and make sure they pass:

```sh
cargo fmt --all -- --check          # format check — a SEPARATE gate; easy to miss
cargo clippy --all-targets --all-features
cargo build --all-targets
cargo test
```

`cargo fmt --check` is the one most easily forgotten: code can compile, lint,
and test clean while still failing the format gate. Run it every time.

## Documentation to keep in sync

When you change or add a model, update the docs in the same PR:

- **`docs/MODELS.md`** — the catalog of every physical model: its equation,
  implementation site, validation gate (with the measured numbers), and
  literature reference.
- **`docs/M*_SPEC.md`** — per-milestone pre-spec gates and design records. New
  milestone work is spec'd here before it is implemented.

## Pull requests

- Keep PRs scoped to one milestone or one self-contained change.
- Include the validation numbers your gate produces in the PR description.
- Make sure the four gates above pass before requesting review.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License, Version 2.0, consistent with the rest of the project.
