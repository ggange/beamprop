//! Pure case runners: the compute loops behind the CLI subcommands, with no
//! I/O, shared by `src/main.rs` and the M5 Python bindings.
//!
//! Each runner takes a plain parameter struct, runs the propagation, and
//! returns the result arrays plus the derived diagnostics the caller needs to
//! report. File writing, notes/metadata formatting, and printing stay with the
//! caller (the CLI writes `.npy` + sidecars; the bindings hand numpy arrays to
//! Python). The M5 refactor gate requires the CLI outputs to be bit-identical
//! to the pre-refactor implementation, so the operation order here mirrors the
//! original `main.rs` loops exactly.

use anyhow::Result;
use ndarray::{Array2, Array3, s};

use crate::airprops::AirTable;
use crate::blooming::ThermalBlooming;
use crate::field::Field;
use crate::grid::Grid;
use crate::medium::{UniformExtinction, kruse_extinction};
use crate::montecarlo::seeded_ensemble;
use crate::propagate::{Propagator, beam_width, centroid};
use crate::turbulence::TurbulentPath;
use crate::validate::{BloomingCase, GaussianBeam};
use crate::viz::XzSliceMap;

/// Stack transverse maps into a `[frame, y, x]` array.
fn stack(maps: &[Array2<f64>]) -> Array3<f64> {
    let (ny, nx) = maps[0].dim();
    let mut s = Array3::<f64>::zeros((maps.len(), ny, nx));
    for (i, m) in maps.iter().enumerate() {
        s.index_axis_mut(ndarray::Axis(0), i).assign(m);
    }
    s
}

/// Crop a side-view map to the middle half in x (the first axis).
fn crop_middle_x(full: &Array2<f64>) -> Array2<f64> {
    let nx = full.dim().0;
    full.slice(s![nx / 4..3 * nx / 4, ..]).to_owned()
}

/// Parameters for [`run_propagate`] (the M1/M2 `propagate` case).
#[derive(Debug, Clone)]
pub struct PropagateParams {
    pub n: usize,
    pub dx: f64,
    pub wavelength: f64,
    pub w0: f64,
    /// Total distance (m); default 2 Rayleigh ranges when `None`.
    pub z: Option<f64>,
    pub steps: usize,
    /// Number of transverse snapshots recorded along the path.
    pub frames: usize,
    /// Uniform power extinction (1/m); takes precedence over `visibility`.
    pub alpha: Option<f64>,
    /// Meteorological visibility (m) → Kruse extinction at the wavelength.
    pub visibility: Option<f64>,
}

/// Results of the `propagate` case.
pub struct PropagateRun {
    pub grid: Grid,
    pub z_total: f64,
    pub dz: f64,
    /// The resolved extinction coefficient actually applied (1/m).
    pub alpha: f64,
    /// Side view I(x, 0, z), cropped to the middle half in x.
    pub xz: Array2<f64>,
    /// Transverse snapshots `[frame, y, x]` (launch plane included).
    pub snapshots: Array3<f64>,
    /// z position of each snapshot (m).
    pub snapshot_z: Vec<f64>,
    /// Receiver-plane intensity.
    pub final_intensity: Array2<f64>,
    /// Receiver 1/e² intensity half-width along x (m).
    pub width_x: f64,
    /// Final power / initial power.
    pub transmission: f64,
    /// Guard-band absorbed power as a fraction of initial power.
    pub guard_frac: f64,
}

/// Propagate a Gaussian beam through vacuum or uniform Beer–Lambert
/// extinction (the `propagate` CLI case), returning data + diagnostics.
pub fn run_propagate(p: &PropagateParams) -> Result<PropagateRun> {
    let grid = Grid::new(p.n, p.dx);
    let analytic = GaussianBeam {
        w0: p.w0,
        wavelength: p.wavelength,
    };
    let z_total = p.z.unwrap_or(2.0 * analytic.rayleigh_range());
    let dz = z_total / p.steps as f64;

    let alpha = match (p.alpha, p.visibility) {
        (Some(a), _) => a,
        (None, Some(v)) => kruse_extinction(p.wavelength, v),
        (None, None) => 0.0,
    };
    let medium = UniformExtinction::new(grid.n, alpha);

    let mut field = Field::gaussian(grid, p.wavelength, p.w0);
    let p0 = field.power();
    let mut prop = Propagator::new(grid, p.wavelength)?;

    let frame_every = (p.steps / p.frames.max(1)).max(1);
    let mut xz = XzSliceMap::new();
    xz.record(&field);
    let mut snapshots = vec![field.intensity()];
    let mut snapshot_z = vec![0.0_f64];

    prop.propagate(&mut field, &medium, dz, 0, p.steps, |i, f| {
        xz.record(f);
        let step = i + 1;
        if step % frame_every == 0 || step == p.steps {
            snapshots.push(f.intensity());
            snapshot_z.push(step as f64 * dz);
        }
    })?;

    Ok(PropagateRun {
        grid,
        z_total,
        dz,
        alpha,
        xz: crop_middle_x(&xz.to_array()),
        snapshots: stack(&snapshots),
        snapshot_z,
        final_intensity: field.intensity(),
        width_x: beam_width(&field).0,
        transmission: field.power() / p0,
        guard_frac: prop.guard_absorbed() / p0,
    })
}

/// Parameters for [`run_turbulence`] (the M3 Monte-Carlo case).
#[derive(Debug, Clone)]
pub struct TurbulenceParams {
    pub n: usize,
    pub dx: f64,
    pub wavelength: f64,
    pub w0: f64,
    pub z: f64,
    /// Number of phase screens (= split-step slabs).
    pub screens: usize,
    /// Refractive-index structure constant Cn² (m^(-2/3)).
    pub cn2: f64,
    /// Outer scale L0 (m).
    pub l0: f64,
    pub realizations: usize,
    /// Master seed for the reproducible screen ensemble.
    pub seed: u64,
}

/// Results of the `turbulence` case.
pub struct TurbulenceRun {
    pub grid: Grid,
    /// Diffraction-only substeps inserted between screens (side-view smoothness).
    pub substeps: usize,
    /// Receiver-plane intensity per realization `[realization, y, x]`.
    pub frames: Array3<f64>,
    /// Side views per realization, middle half in x `[realization, x, z]`.
    pub xz_frames: Array3<f64>,
    /// Ensemble-mean receiver intensity (the long-exposure image).
    pub longexp: Array2<f64>,
    /// Ensemble-mean guard-band absorbed power fraction.
    pub guard_frac_mean: f64,
}

/// Propagate a Gaussian beam through a reproducible Monte-Carlo ensemble of
/// von Kármán turbulence (the `turbulence` CLI case).
pub fn run_turbulence(p: &TurbulenceParams) -> Result<TurbulenceRun> {
    let grid = Grid::new(p.n, p.dx);

    // Diffraction-only substeps between screens give the side view a smooth
    // z-axis (~240 columns) without changing the screen physics.
    let substeps = (240 / p.screens).max(1);
    // Each realization is fallible (an under-resolved or uncontained beam is
    // rejected by the propagator); return a Result per member and surface the
    // first failure rather than panicking inside the parallel closure.
    let results = seeded_ensemble(p.realizations, |i| -> Result<_> {
        let path = TurbulentPath::new(grid, p.wavelength, p.cn2, p.l0, p.z, p.screens, p.seed, i)
            .with_substeps(substeps);
        let mut field = Field::gaussian(grid, p.wavelength, p.w0);
        let p0 = field.power();
        let mut prop = Propagator::new(grid, p.wavelength)?;
        let mut xz = XzSliceMap::new();
        xz.record(&field);
        prop.propagate(&mut field, &path, path.dz(), 0, path.n_slabs(), |_, f| {
            xz.record(f);
        })?;
        Ok((field.intensity(), xz.to_array(), prop.guard_absorbed() / p0))
    });
    let mut frames = Vec::with_capacity(p.realizations);
    let mut xz_maps = Vec::with_capacity(p.realizations);
    let mut guard_sum = 0.0;
    for result in results {
        let (frame, xz_map, guard_frac) = result?;
        frames.push(frame);
        xz_maps.push(xz_map);
        guard_sum += guard_frac;
    }
    let guard_frac_mean = guard_sum / p.realizations as f64;

    // Side view (x-z plane, beam travelling left to right), cropped to the
    // middle half in x.
    let xz_frames: Vec<_> = xz_maps.iter().map(crop_middle_x).collect();

    // Long-exposure (ensemble-mean) receiver intensity.
    let mut mean = Array2::<f64>::zeros((grid.n, grid.n));
    for f in &frames {
        mean += f;
    }
    mean /= p.realizations as f64;

    Ok(TurbulenceRun {
        grid,
        substeps,
        frames: stack(&frames),
        xz_frames: stack(&xz_frames),
        longexp: mean,
        guard_frac_mean,
    })
}

/// Parameters for [`run_blooming`] (the M4 thermal-blooming case).
#[derive(Debug, Clone)]
pub struct BloomingParams {
    pub n: usize,
    pub dx: f64,
    pub wavelength: f64,
    pub w0: f64,
    /// Total beam power (W).
    pub power: f64,
    /// Crosswind speed (m/s, along +x).
    pub wind: f64,
    /// Absorbed-power coefficient (1/m).
    pub alpha_abs: f64,
    /// Ambient temperature (K).
    pub t0: f64,
    /// Ambient pressure (Pa).
    pub p0: f64,
    pub z: f64,
    pub steps: usize,
    /// Number of transverse snapshots recorded along the path.
    pub frames: usize,
}

/// Results of the `blooming` case.
pub struct BloomingRun {
    pub grid: Grid,
    pub dz: f64,
    /// Phase distortion number N_φ (spec convention, docs/M4_SPEC.md).
    pub n_phi: f64,
    /// Péclet number (convection-dominated model needs ≫ 100).
    pub peclet: f64,
    /// Saturated downwind ΔT of the launch beam (K, closed form).
    pub delta_t_sat: f64,
    /// Side view I(x, 0, z), cropped to the middle half in x.
    pub xz: Array2<f64>,
    /// Transverse snapshots `[frame, y, x]` (launch plane included).
    pub snapshots: Array3<f64>,
    /// z position of each snapshot (m).
    pub snapshot_z: Vec<f64>,
    /// Receiver-plane intensity: the bloomed profile.
    pub final_intensity: Array2<f64>,
    /// Receiver centroid x (m); negative = bent upwind.
    pub centroid_x: f64,
    /// Final power / initial power.
    pub transmission: f64,
    /// Guard-band absorbed power as a fraction of initial power.
    pub guard_frac: f64,
}

/// Propagate a high-power Gaussian beam through steady-state thermal blooming
/// (the `blooming` CLI case).
pub fn run_blooming(p: &BloomingParams) -> Result<BloomingRun> {
    let grid = Grid::new(p.n, p.dx);
    let air = AirTable::load()?.at(p.t0, p.p0, p.wavelength)?;
    let case = BloomingCase {
        alpha_abs: p.alpha_abs,
        power: p.power,
        w: p.w0,
        wind: p.wind,
        rho: air.rho,
        cp: air.cp,
        n_minus_1: air.n_minus_1,
        t0: p.t0,
        wavelength: p.wavelength,
    };
    let n_phi = case.distortion_number(p.z);
    let peclet = air.rho * air.cp * p.wind * p.w0 / air.kappa_t;
    // Saturated downwind temperature rise of the launch beam (closed form).
    let delta_t_sat = case.delta_t_ref(1e3 * p.w0, 0.0);

    let mut field = Field::gaussian(grid, p.wavelength, p.w0);
    let p_init = field.power();
    let medium = ThermalBlooming::new(grid, air, p.alpha_abs, p.wind, p.power, p_init, p.w0, p.t0)?;
    let mut prop = Propagator::new(grid, p.wavelength)?;

    let dz = p.z / p.steps as f64;
    let frame_every = (p.steps / p.frames.max(1)).max(1);
    let mut xz = XzSliceMap::new();
    xz.record(&field);
    let mut snapshots = vec![field.intensity()];
    let mut snapshot_z = vec![0.0_f64];
    prop.propagate(&mut field, &medium, dz, 0, p.steps, |i, f| {
        xz.record(f);
        let step = i + 1;
        if step % frame_every == 0 || step == p.steps {
            snapshots.push(f.intensity());
            snapshot_z.push(step as f64 * dz);
        }
    })?;

    Ok(BloomingRun {
        grid,
        dz,
        n_phi,
        peclet,
        delta_t_sat,
        xz: crop_middle_x(&xz.to_array()),
        snapshots: stack(&snapshots),
        snapshot_z,
        final_intensity: field.intensity(),
        centroid_x: centroid(&field).0,
        transmission: field.power() / p_init,
        guard_frac: prop.guard_absorbed() / p_init,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagate_run_shapes_and_transmission() {
        let r = run_propagate(&PropagateParams {
            n: 256,
            dx: 1e-3,
            wavelength: 1e-6,
            w0: 5e-3,
            z: Some(50.0),
            steps: 10,
            frames: 2,
            alpha: Some(1e-3),
            visibility: None,
        })
        .unwrap();
        assert_eq!(r.final_intensity.dim(), (256, 256));
        assert_eq!(r.xz.dim().0, 128); // middle half in x
        assert_eq!(r.snapshots.dim().0, r.snapshot_z.len());
        // Beer–Lambert transmission to ~machine precision (M2 gate re-check).
        let t_ref = (-1e-3 * 50.0_f64).exp();
        assert!((r.transmission - t_ref).abs() / t_ref < 1e-10);
    }

    #[test]
    fn turbulence_run_is_seed_deterministic() {
        let p = TurbulenceParams {
            n: 256,
            dx: 2e-3,
            wavelength: 1e-6,
            w0: 1e-2,
            z: 500.0,
            screens: 3,
            cn2: 1e-14,
            l0: 1e3,
            realizations: 2,
            seed: 42,
        };
        let a = run_turbulence(&p).unwrap();
        let b = run_turbulence(&p).unwrap();
        assert_eq!(a.frames, b.frames);
        assert_eq!(a.longexp, b.longexp);
        let c = run_turbulence(&TurbulenceParams { seed: 43, ..p }).unwrap();
        assert_ne!(a.frames, c.frames);
    }

    #[test]
    fn blooming_run_bends_upwind() {
        let r = run_blooming(&BloomingParams {
            n: 256,
            dx: 1e-3,
            wavelength: 1e-6,
            w0: 2e-2,
            power: 2e4,
            wind: 2.0,
            alpha_abs: 1e-4,
            t0: 288.15,
            p0: 101_325.0,
            z: 200.0,
            steps: 40,
            frames: 2,
        })
        .unwrap();
        // The beam bends into the wind (−x) and loses the absorbed power.
        assert!(r.centroid_x < 0.0);
        assert!(r.transmission < 1.0 && r.transmission > 0.9);
        assert!(r.n_phi > 0.0 && r.peclet > 100.0);
    }
}
