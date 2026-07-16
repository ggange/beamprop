//! Visualization data collection.
//!
//! No image encoding lives in Rust: the solver writes `.npy` arrays (plus
//! `_meta.json`/`_notes.md` sidecars) and `scripts/render.py` produces the
//! figures — physical axes, titles, labeled colorbar — with matplotlib.
//! This module only accumulates the arrays worth rendering.

use ndarray::Array2;

use crate::field::Field;

/// Accumulates the beam's central intensity slice `I(x)` at each z-step into
/// an `x`–`z` map: the classic side-view of a propagating beam.
pub struct XzSliceMap {
    rows: Vec<Vec<f64>>,
}

impl XzSliceMap {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Record the central row of the field's intensity (a fixed-y slice).
    pub fn record(&mut self, field: &Field) {
        let inten = field.intensity();
        let mid = field.grid.n / 2;
        self.rows.push(inten.row(mid).to_vec());
    }

    /// Number of recorded slices.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Assemble into an array with x down the rows and z across the columns.
    pub fn to_array(&self) -> Array2<f64> {
        let nz = self.rows.len();
        let nx = self.rows.first().map_or(0, Vec::len);
        Array2::from_shape_fn((nx, nz), |(ix, iz)| self.rows[iz][ix])
    }
}

impl Default for XzSliceMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn xz_map_shape_is_x_by_z() {
        let grid = Grid::new(16, 1e-3);
        let field = Field::gaussian(grid, 1e-6, 5e-3);
        let mut xz = XzSliceMap::new();
        assert!(xz.is_empty());
        xz.record(&field);
        xz.record(&field);
        assert_eq!(xz.len(), 2);
        assert_eq!(xz.to_array().dim(), (16, 2));
    }
}
