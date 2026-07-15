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
//!   plus its `.npy`/PNG output path — the inspection interface through M2.
//! - **M1 (this milestone):** the symmetric split-step
//!   [`Propagator`](propagate::Propagator) advancing a field through any
//!   [`Medium`](medium::Medium), validated against the analytic
//!   [`GaussianBeam`](validate::GaussianBeam) plus energy-conservation,
//!   boundary-wraparound, and order-of-accuracy tests.
//! - **M2:** Beer–Lambert attenuation. **M3:** turbulence phase screens.
//! - **M4:** coupled thermal blooming. **M5:** PyO3 bindings.

pub mod field;
pub mod grid;
pub mod medium;
pub mod propagate;
pub mod validate;
pub mod viz;
