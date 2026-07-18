# M5 — Python bindings (PyO3) + wheels: implementation record

M5 adds no physics, so unlike M4 it needed no pre-spec gate. This document
records the decisions taken, the gate definitions, and the measured results —
the same closeout discipline as every other milestone.

## Decisions (recorded)

1. **API surface: core classes + run helpers.** The Python module `beamprop`
   exposes thin wrappers over the Rust types (`Grid`, `Field`, `Medium`,
   `Propagator`, `GaussianBeam`, `fried_r0`, `rytov_variance`,
   `kruse_extinction`) *and* high-level `run_propagate` / `run_turbulence` /
   `run_blooming` helpers mirroring the CLI cases, returning dicts of numpy
   arrays + derived diagnostics. Composability and notebook ergonomics in one
   surface.
2. **One source of truth.** The CLI compute loops were extracted to pure
   functions in `src/cases.rs` (no I/O); both `src/main.rs` and the bindings
   call them. Refactor gate: the CLI `.npy` outputs are **bit-identical**
   before/after (9/9 sha256 across the three cases).
3. **Layout: cargo workspace.** Root crate untouched at `.`; `beamprop-py/` is
   a `cdylib` member (pyo3 0.29 + rust-numpy 0.29, `abi3-py310`, maturin with
   `module-name = "beamprop"`). The Rust crate name stays `beamprop`; the
   Python package installs as `beamprop`.
4. **`Medium` across the FFI: one wrapper class.** Python's `Medium` holds a
   `Box<dyn Medium + Send + Sync>` built by static constructors (`vacuum`,
   `constant_delta_n`, `uniform_extinction`, `turbulence`,
   `thermal_blooming`), collapsing the trait to a single pyclass instead of
   N parallel ones.
5. **Errors, not panics.** Solver `anyhow::Result` errors surface as Python
   `ValueError` with the message intact (Péclet guard, ΔT ceiling, guard-band
   containment). Arguments that Rust constructors `assert!` on (grid parity,
   positive waist/wavelength) are pre-validated on the Python side. Residual
   panics from deep inside the ensemble runner would arrive as pyo3
   `PanicException` rather than aborting — none observed in the gates.
6. **Wheels: CI artifacts, no PyPI.** `wheels.yml` (PyO3/maturin-action)
   builds abi3 wheels for linux x86_64, macOS universal2, and windows x64,
   smoke-tests each wheel against the full pytest gate suite, uploads
   artifacts, and attaches wheels to GitHub releases on `v*` tags. PyPI
   publishing is deliberately deferred until the project wants a public
   release.

## Gates and measured results

No new physics ⇒ the M5 gates are **parity and reproducibility** (pytest,
`beamprop-py/tests/`, 17 tests; also run in CI on every push and against every
built wheel):

| Gate | Definition | Measured |
|---|---|---|
| CLI parity | `run_*()` arrays vs the CLI's `.npy` files, same params/seed, `np.array_equal` (no tolerance) | **bit-identical**, all three cases (propagate / turbulence / blooming) |
| Closed form | vacuum Gaussian width at 2 z_R vs `GaussianBeam.width_at` | < 1% gate; ~2e-11 relative observed |
| Power conservation | vacuum propagation, relative power drift | < 1e-12 |
| Beer–Lambert | `run_propagate` transmission vs `e^(−αz)` | < 1e-10 relative |
| Determinism | same seed ⇒ identical arrays; different seed ⇒ different; `realization` selects ensemble member | exact (bitwise) |
| Round-trip | `Field.u` numpy get/set: intensity/power consistent, shape mismatch raises | exact |
| Error mapping | Péclet guard, ΔT ceiling, bad grid/beam → `ValueError`, message preserved | pass |
| Callback | `on_step(i, field)` sees every step; Python exceptions propagate out of `propagate` | pass |

## CI

- `ci.yml` gained a `python` job: maturin build → install wheel → pytest gates
  (ubuntu, Python 3.12). The Rust job is unchanged and now covers the
  workspace (fmt, clippy `-D warnings`, build, test).
- `wheels.yml`: 3-platform wheel matrix + per-wheel gate run, artifacts,
  release assets on tags; `workflow_dispatch` for manual runs.

## Non-goals (v1 of the bindings)

- No PyPI release (see decision 6).
- No `Propagator.step`-level or custom-`Medium`-from-Python API: Python-defined
  media would put a Python callback inside the hot loop; revisit if a real use
  case appears.
- Rendering stays in `scripts/render.py`; the bindings return data only,
  matching the solver's data/image split.
