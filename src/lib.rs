//! `beamprop` — an open solver for laser beam propagation through the atmosphere.
//!
//! Scope is **pure propagation physics**: diffraction, molecular/aerosol
//! attenuation, atmospheric turbulence, and thermal blooming. Applications
//! (weapon lethality, comms link budgets, lidar returns) are downstream
//! consumers built in separate crates — they are deliberately not here.
//!
//! Units are **SI throughout** unless a field's documentation says otherwise.
//!
//! ## Build ladder
//! This crate grows one validated milestone at a time:
//! - **M0:** the [`Field`](field::Field) container on a [`Grid`](grid::Grid),
//!   plus its `.npy` output path — the data interface; images are rendered
//!   from it in Python (`scripts/render.py`).
//! - **M1:** the symmetric split-step
//!   [`Propagator`](propagate::Propagator) advancing a field through any
//!   [`Medium`](medium::Medium), validated against the analytic
//!   [`GaussianBeam`](validate::GaussianBeam) plus energy-conservation,
//!   boundary-wraparound, and order-of-accuracy tests.
//! - **M2:** Beer–Lambert attenuation through the same `Medium` trait.
//! - **M3:** von Kármán
//!   [turbulence phase screens](turbulence::ScreenGenerator) stacked into a
//!   [`TurbulentPath`](turbulence::TurbulentPath), with reproducible
//!   [Monte-Carlo ensembles](montecarlo::seeded_ensemble), validated against
//!   the Kolmogorov structure function, long-exposure spread, and
//!   weak-turbulence scintillation.
//! - **M4:** coupled thermal blooming. **M5:** PyO3 bindings.
//! - **M6a:** the 0-D optical-breakdown threshold kernel
//!   ([`breakdown0d`]) — the electron-avalanche balance that decides *whether*
//!   the air ionizes, validated against Thiyagarajan & Thompson (2012).
//! - **M6c (this milestone):** the laser-supported detonation wave — what the
//!   plasma does once lit. A laser-agnostic 1-D Euler core ([`euler1d`]), a
//!   frozen equilibrium plasma-property table ([`plasmaprops`]), and the
//!   coupled driver ([`lsd`]) that deposits the beam into the gas and exposes
//!   the resulting column to the propagator as a pure absorber. See
//!   `docs/M6C_SPEC.md` for which of its gates are verification and which is
//!   the physics gate — the distinction is load-bearing here.

pub mod airprops;
pub mod aperture;
pub mod blooming;
pub mod breakdown0d;
pub mod cases;
pub mod euler1d;
pub mod field;
pub mod grid;
pub mod lsd;
pub mod medium;
pub mod montecarlo;
pub mod plasmaprops;
pub mod propagate;
pub mod turbulence;
pub mod validate;
pub mod viz;
