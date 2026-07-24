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

use anyhow::{Context, Result};
use ndarray::{Array2, Array3, s};

use crate::airprops::AirTable;
use crate::blooming::ThermalBlooming;
use crate::breakdown0d::AirBreakdown;
use crate::field::Field;
use crate::grid::Grid;
use crate::medium::{UniformExtinction, kruse_extinction};
use crate::montecarlo::seeded_ensemble;
use crate::propagate::{Propagator, beam_width, centroid};
use crate::turbulence::TurbulentPath;
use crate::validate::{BloomingCase, GaussianBeam, loglog_slope};
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

/// Parameters of the `breakdown` case (M6a, 0-D — no grid, no propagator).
pub struct BreakdownParams {
    /// Vacuum wavelength (m).
    pub wavelength: f64,
    /// Pulse FWHM (s).
    pub fwhm: f64,
    /// Lowest pressure of the sweep (Torr).
    pub p_min_torr: f64,
    /// Highest pressure of the sweep (Torr).
    pub p_max_torr: f64,
    /// Number of log-spaced sweep pressures (= animation frames).
    pub points: usize,
    /// Time slices per pulse for the rate integration.
    pub steps: usize,
    /// Drive intensity for the n_e(t) traces, as a multiple of the threshold
    /// at `p_max_torr`. A single fixed intensity across all pressures is what
    /// makes the animation physical: the same pulse ignites the gas at high
    /// pressure and fizzles at low.
    pub drive: f64,
}

/// Results of the `breakdown` case.
pub struct BreakdownRun {
    /// Sweep pressures (Torr).
    pub pressure_torr: Vec<f64>,
    /// Threshold peak intensity at the literature-central `δ_eff` (W/cm²).
    pub threshold: Vec<f64>,
    /// Threshold at `δ_eff = 0.05` — the flat-slope edge of the literature
    /// envelope (W/cm²).
    pub threshold_lo_slope: Vec<f64>,
    /// Threshold at `δ_eff = 0.01` — the steep-slope edge (W/cm²).
    pub threshold_hi_slope: Vec<f64>,
    /// Fitted log-log slope `n` of the central curve (`I_thr ∝ p^-n`).
    pub slope: f64,
    /// Envelope of `n` over the `δ_eff` literature range.
    pub slope_envelope: (f64, f64),
    /// The fixed drive intensity used for the traces (W/cm²).
    pub drive_intensity: f64,
    /// Electron-density traces `[pressure, time]` (m⁻³) at `drive_intensity`.
    pub ne_traces: Array2<f64>,
    /// Trace time axis (s), relative to the pulse peak.
    pub trace_time: Vec<f64>,
    /// Breakdown criterion density (m⁻³) — the line the traces must cross.
    pub n_bd: f64,
    /// Seed density (m⁻³): one electron in the focal volume, where every
    /// trace starts.
    pub n_seed: f64,
}

/// Sweep the 0-D optical-breakdown threshold against pressure and record the
/// electron-density avalanche at a fixed drive intensity (the `breakdown` CLI
/// case). Pure rate physics: no field, no grid, no propagator.
pub fn run_breakdown(p: &BreakdownParams) -> Result<BreakdownRun> {
    use beamprop_torr::TORR;
    if !(p.p_min_torr > 0.0 && p.p_max_torr > p.p_min_torr) {
        anyhow::bail!(
            "need 0 < p_min < p_max, got {} .. {} Torr",
            p.p_min_torr,
            p.p_max_torr
        );
    }
    if p.points < 2 {
        anyhow::bail!("need at least 2 sweep points, got {}", p.points);
    }
    let model = if (p.wavelength - 1064e-9).abs() < 1e-12 {
        AirBreakdown::air_1064nm()
    } else {
        // Same focal geometry as the pinned T&T case, retuned to λ.
        let r_focus = 20e-6;
        AirBreakdown::new(
            p.wavelength,
            12.06,
            r_focus / std::f64::consts::PI,
            288.0,
            4.0 / 3.0 * std::f64::consts::PI * r_focus.powi(3),
        )?
    };

    let curve = |m: &AirBreakdown| -> Result<Vec<(f64, f64)>> {
        (0..p.points)
            .map(|i| {
                let frac = i as f64 / (p.points - 1) as f64;
                let torr = p.p_min_torr * (p.p_max_torr / p.p_min_torr).powf(frac);
                let it = m.threshold_intensity(p.fwhm, torr * TORR, p.steps)?;
                Ok((torr, it))
            })
            .collect()
    };

    let central = curve(&model)?;
    // Envelope edges: δ_eff = 0.05 flattens the slope, 0.01 steepens it.
    let lo_slope = curve(&model.with_inelastic_loss(0.05, 3.0))?;
    let hi_slope = curve(&model.with_inelastic_loss(0.01, 3.0))?;

    let slope = -loglog_slope(&central).context("fitting the threshold slope")?;
    let env = (
        -loglog_slope(&lo_slope).context("fitting the flat envelope edge")?,
        -loglog_slope(&hi_slope).context("fitting the steep envelope edge")?,
    );

    // One fixed intensity for every trace, referenced to the highest-pressure
    // threshold so the low-pressure frames genuinely fail to ignite.
    let drive_intensity = p.drive * central.last().expect("non-empty sweep").1;

    let n_time = p.steps;
    let dt = 4.0 * p.fwhm / n_time as f64;
    let c = 4.0 * std::f64::consts::LN_2 / (p.fwhm * p.fwhm);
    let trace_time: Vec<f64> = (0..n_time)
        .map(|s| -2.0 * p.fwhm + (s as f64 + 0.5) * dt)
        .collect();

    let mut ne_traces = Array2::<f64>::zeros((p.points, n_time));
    for (row, &(torr, _)) in central.iter().enumerate() {
        let pressure = torr * TORR;
        let mut n_e = model.seed_density();
        for (col, &t) in trace_time.iter().enumerate() {
            let intensity = drive_intensity * (-c * t * t).exp();
            n_e = model.advance(n_e, intensity, pressure, dt);
            // Traces are plotted on a log axis; keep them finite for the
            // renderer even when the avalanche saturates the exponent.
            ne_traces[[row, col]] = n_e.min(1e40);
        }
    }

    Ok(BreakdownRun {
        pressure_torr: central.iter().map(|c| c.0).collect(),
        threshold: central.iter().map(|c| c.1 / 1e4).collect(),
        threshold_lo_slope: lo_slope.iter().map(|c| c.1 / 1e4).collect(),
        threshold_hi_slope: hi_slope.iter().map(|c| c.1 / 1e4).collect(),
        slope,
        slope_envelope: env,
        drive_intensity: drive_intensity / 1e4,
        ne_traces,
        trace_time,
        n_bd: model.criterion_density(),
        n_seed: model.seed_density(),
    })
}

/// Pa per Torr — the pressure unit the breakdown literature and the T&T data
/// are quoted in, kept local to the one case that needs it.
mod beamprop_torr {
    pub const TORR: f64 = 133.322_368_4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breakdown_run_shapes_and_physics() {
        let r = run_breakdown(&BreakdownParams {
            wavelength: 1064e-9,
            fwhm: 6e-9,
            p_min_torr: 300.0,
            p_max_torr: 2000.0,
            points: 6,
            steps: 200,
            drive: 1.5,
        })
        .unwrap();
        assert_eq!(r.pressure_torr.len(), 6);
        assert_eq!(r.ne_traces.shape(), &[6, 200]);
        assert_eq!(r.trace_time.len(), 200);
        // Threshold falls with pressure, and the envelope straddles the centre.
        assert!(r.threshold[0] > *r.threshold.last().unwrap());
        assert!(r.slope > 0.0);
        assert!(r.slope_envelope.0 < r.slope && r.slope < r.slope_envelope.1);
        // Traces start from the seed and stay finite despite the avalanche.
        assert!(r.ne_traces.iter().all(|v| v.is_finite() && *v > 0.0));
        // The drive is above threshold at p_max and below it at p_min, which is
        // what makes the animation show ignition switching on with pressure.
        assert!(r.drive_intensity > *r.threshold.last().unwrap());
        assert!(r.drive_intensity < r.threshold[0]);
    }

    #[test]
    fn breakdown_rejects_bad_pressure_range() {
        let bad = BreakdownParams {
            wavelength: 1064e-9,
            fwhm: 6e-9,
            p_min_torr: 2000.0,
            p_max_torr: 300.0,
            points: 6,
            steps: 100,
            drive: 1.5,
        };
        assert!(run_breakdown(&bad).is_err());
    }

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
