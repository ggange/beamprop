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
//! - **M0 (this milestone):** the [`Field`](field::Field) container on a
//!   [`Grid`](grid::Grid), plus its `.npy`/PNG output path — the interface used
//!   for inspection through M2. No propagator yet.
//! - **M1:** the symmetric split-step propagator taking a `Medium` trait.
//! - **M2:** Beer–Lambert attenuation. **M3:** turbulence phase screens.
//! - **M4:** coupled thermal blooming. **M5:** PyO3 bindings.

pub mod field;
pub mod grid;
