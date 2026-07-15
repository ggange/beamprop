//! The medium interface: what the beam propagates *through*.
//!
//! The propagator is deliberately ignorant of what produces the refractive
//! index field. A `Medium` supplies the index perturbation `δn(x, y)` for each
//! z-slab; a turbulence phase screen (M3) and a computed volumetric thermal
//! blooming field (M4) implement the same trait. This interface is fixed at M1
//! so the later physics needs no propagator rewrite.

use ndarray::Array2;

/// A propagation medium, sampled as one `δn(x, y)` field per z-slab.
///
/// `δn` is the dimensionless refractive-index perturbation about the
/// background (`n = n₀ + δn`); the propagator turns it into the phase
/// `k·δn·dz` accumulated across the slab.
pub trait Medium {
    /// Refractive-index perturbation `δn(x, y)` for slab `z_slab`,
    /// shape `[n, n]` matching the propagation grid.
    fn index_perturbation(&self, z_slab: usize) -> Array2<f64>;
}

/// Vacuum (or unperturbed air): `δn = 0` everywhere.
#[derive(Debug, Clone, Copy)]
pub struct Vacuum {
    n: usize,
}

impl Vacuum {
    /// Vacuum sampled on an `n × n` grid.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Medium for Vacuum {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::zeros((self.n, self.n))
    }
}

/// A uniform index offset, constant across the grid and along z.
///
/// Physically this only adds a global phase, which makes it a useful
/// validation medium: intensity must be identical to vacuum propagation.
#[derive(Debug, Clone, Copy)]
pub struct ConstantDeltaN {
    n: usize,
    delta_n: f64,
}

impl ConstantDeltaN {
    /// Uniform perturbation `delta_n` on an `n × n` grid.
    pub fn new(n: usize, delta_n: f64) -> Self {
        Self { n, delta_n }
    }
}

impl Medium for ConstantDeltaN {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::from_elem((self.n, self.n), self.delta_n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuum_is_zero() {
        let m = Vacuum::new(16);
        let dn = m.index_perturbation(3);
        assert_eq!(dn.dim(), (16, 16));
        assert!(dn.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_is_constant_across_slabs() {
        let m = ConstantDeltaN::new(8, 1e-6);
        assert_eq!(m.index_perturbation(0), m.index_perturbation(99));
        assert!(m.index_perturbation(0).iter().all(|&v| v == 1e-6));
    }
}
