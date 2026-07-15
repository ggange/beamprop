//! Colormapped intensity rendering (the M1 visual deliverable, T6).
//!
//! Self-contained: a small perceptual colormap interpolated from anchor
//! points, no plotting dependency. Publication figures come later from Python
//! against the `.npy` output; this is the "look at the beam" path.

use std::path::Path;

use anyhow::Result;
use ndarray::Array2;

use crate::field::Field;

/// Magma-like anchor colors, position in [0, 1] → (r, g, b).
const ANCHORS: [(f64, [f64; 3]); 6] = [
    (0.00, [0.0, 0.0, 4.0]),
    (0.20, [45.0, 17.0, 96.0]),
    (0.40, [114.0, 31.0, 129.0]),
    (0.60, [183.0, 55.0, 121.0]),
    (0.80, [245.0, 125.0, 21.0]),
    (1.00, [252.0, 253.0, 191.0]),
];

/// Map a value in [0, 1] to an RGB pixel via the anchor gradient.
pub fn colormap(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut lo = ANCHORS[0];
    let mut hi = ANCHORS[ANCHORS.len() - 1];
    for pair in ANCHORS.windows(2) {
        if t >= pair[0].0 && t <= pair[1].0 {
            lo = pair[0];
            hi = pair[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(f64::EPSILON);
    let f = (t - lo.0) / span;
    let mut rgb = [0u8; 3];
    for (i, px) in rgb.iter_mut().enumerate() {
        *px = (lo.1[i] + f * (hi.1[i] - lo.1[i]))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    rgb
}

/// Render a non-negative array to a colormapped PNG.
///
/// Values are normalised to `max` (global peak) and gamma-compressed with
/// `t = (v/max)^gamma`; `gamma < 1` lifts the dim wings of a beam into
/// visibility. `gamma = 0.5` is a good default for intensity.
pub fn save_colormapped_png(data: &Array2<f64>, gamma: f64, path: impl AsRef<Path>) -> Result<()> {
    let max = data.iter().copied().fold(0.0_f64, f64::max);
    let (ny, nx) = data.dim();
    let mut img = image::RgbImage::new(nx as u32, ny as u32);
    for ((iy, ix), &v) in data.indexed_iter() {
        let t = if max > 0.0 {
            (v / max).powf(gamma)
        } else {
            0.0
        };
        img.put_pixel(ix as u32, iy as u32, image::Rgb(colormap(t)));
    }
    img.save(path)?;
    Ok(())
}

/// Render a field's intensity to a colormapped PNG (`gamma = 0.5`).
pub fn save_intensity_render(field: &Field, path: impl AsRef<Path>) -> Result<()> {
    save_colormapped_png(&field.intensity(), 0.5, path)
}

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

    /// Write the x–z map as a colormapped PNG.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        save_colormapped_png(&self.to_array(), 0.5, path)
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

    #[test]
    fn colormap_endpoints() {
        assert_eq!(colormap(0.0), [0, 0, 4]);
        assert_eq!(colormap(1.0), [252, 253, 191]);
    }

    #[test]
    fn colormap_is_monotone_in_brightness() {
        let lum =
            |rgb: [u8; 3]| 0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64;
        let mut prev = -1.0;
        for i in 0..=20 {
            let l = lum(colormap(i as f64 / 20.0));
            assert!(l >= prev, "brightness must not decrease");
            prev = l;
        }
    }
}
