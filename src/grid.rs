//! Transverse computational grid geometry (SI units).

/// A square `n × n` transverse grid with uniform spacing `dx` (metres).
///
/// Coordinates are centred on zero, so sample `n / 2` sits at the origin. This
/// is the sampling that the M1 split-step propagator and its FFTs will assume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// Number of samples per side.
    pub n: usize,
    /// Sample spacing in metres.
    pub dx: f64,
}

impl Grid {
    /// Create a grid of `n` samples per side spaced `dx` metres apart.
    ///
    /// # Panics
    /// Panics if `n == 0` or `dx` is not a positive, finite number.
    pub fn new(n: usize, dx: f64) -> Self {
        assert!(n > 0, "grid size must be positive");
        assert!(
            dx > 0.0 && dx.is_finite(),
            "grid spacing must be positive and finite"
        );
        Self { n, dx }
    }

    /// Physical side length of the grid in metres (`n · dx`).
    pub fn extent(&self) -> f64 {
        self.n as f64 * self.dx
    }

    /// Coordinate of sample `i` along one axis, centred on zero (metres).
    pub fn coord(&self, i: usize) -> f64 {
        (i as f64 - self.n as f64 / 2.0) * self.dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_is_n_times_dx() {
        let g = Grid::new(512, 1e-3);
        assert_eq!(g.extent(), 512.0 * 1e-3);
    }

    #[test]
    fn origin_is_centred() {
        let g = Grid::new(8, 0.25);
        assert_eq!(g.coord(4), 0.0);
        assert_eq!(g.coord(3), -0.25);
    }

    #[test]
    #[should_panic]
    fn rejects_zero_size() {
        Grid::new(0, 1.0);
    }

    #[test]
    #[should_panic]
    fn rejects_nonpositive_spacing() {
        Grid::new(8, 0.0);
    }
}
