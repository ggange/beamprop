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
use crate::aperture::Aperture;
use crate::blooming::ThermalBlooming;
use crate::breakdown0d::AirBreakdown;
use crate::euler1d::{IdealGas, Primitive};
use crate::field::{Field, IntensityScale};
use crate::grid::Grid;
use crate::lsd::{Absorption, IonizationCeiling, LsdColumn, SeededIgnition, raizer_lsd_velocity};
use crate::medium::{UniformExtinction, kruse_extinction};
use crate::montecarlo::seeded_ensemble;
use crate::plasmaprops::PlasmaTable;
use crate::propagate::{Propagator, beam_width, centroid};
use crate::turbulence::TurbulentPath;
use crate::validate::{BloomingCase, GaussianBeam, loglog_slope, tt2012_cascade_threshold};
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
    /// Neutral density at each sweep pressure (m⁻³) — the ceiling each trace
    /// saturates against, since it is full ionization.
    pub neutral_density: Vec<f64>,
    /// T&T Eq. 4 cascade-theory threshold at each sweep pressure (W/cm²) —
    /// the apples-to-apples reference for a cascade-only kernel.
    pub cascade_theory: Vec<f64>,
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
            n_e = model
                .advance(n_e, intensity, pressure, dt)
                .max(model.seed_density());
            // Bounded by construction: the logistic term caps n_e at full
            // ionization, so no plotting clamp is needed (there used to be one
            // at 1e40, which was the visible ceiling in the first release).
            ne_traces[[row, col]] = n_e;
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
        neutral_density: central
            .iter()
            .map(|c| model.neutral_density(c.0 * TORR))
            .collect(),
        cascade_theory: central
            .iter()
            .map(|c| tt2012_cascade_threshold(c.0 * TORR, p.wavelength) / 1e4)
            .collect(),
    })
}

/// Pa per Torr — the pressure unit the breakdown literature and the T&T data
/// are quoted in, kept local to the one case that needs it.
mod beamprop_torr {
    pub const TORR: f64 = 133.322_368_4;
}

/// Specific gas constant for dry air, `R_u/M` (J/(kg·K)) — the one place the
/// `lsd` case needs to turn an ambient `(T, p)` into the `ρ₀` the closed form
/// is written in.
const R_AIR: f64 = 287.052_874;

/// The CO₂ wavelength (m). Not a parameter: it is the wavelength LSD
/// experiments are done at, and the `lsd` case quotes the inverse-bremsstrahlung
/// closure there as a fixed reference point against whatever `--wavelength` the
/// run itself used.
const CO2_WAVELENGTH: f64 = 10.6e-6;

/// Where the focus (and so the spark) sits along the column, as a fraction of
/// its length. The front runs back toward the laser from here, so this is also
/// how much room it has: three quarters of the domain.
const LSD_FOCUS_FRACTION: f64 = 0.75;

/// Seed pressure as a multiple of the Chapman–Jouguet pressure.
///
/// A free parameter, and deliberately so: M6a's kernel says *whether* the gas
/// breaks down, not how much energy the spark carries, so nothing in the model
/// fixes this. It does not need fixing — `lsd_front_speed_is_seed_independent`
/// (G3c) gates the front speed as insensitive to it at the 1e-3 level, which is
/// exactly the statement that the number below cannot influence the result the
/// run reports. Mildly overdriven, so the wave relaxes *down* onto its
/// self-sustaining speed.
const LSD_SEED_MULTIPLE: f64 = 2.0;

/// Parameters of the `lsd` case (M6c step 6 — the demonstration run).
///
/// The two intensities are deliberately separate, and the reason is the
/// headline result of this case — see [`run_lsd`].
#[derive(Debug, Clone)]
pub struct LsdParams {
    /// Vacuum wavelength (m).
    pub wavelength: f64,
    /// Igniting-pulse FWHM (s) for the M6a test.
    pub fwhm: f64,
    /// Peak power of the **igniting** pulse (W) — the short spark, not the
    /// sustaining beam.
    pub ignite_power: f64,
    /// Focal spot `1/e²` intensity radius of the igniting pulse (m).
    pub w_focus: f64,
    /// **Sustaining** drive intensity into the column (W/m²) — the long-pulse
    /// beam the detonation runs on.
    pub drive: f64,
    /// Ambient pressure (Pa).
    pub p0: f64,
    /// Ambient temperature (K).
    pub t0: f64,
    /// Column length (m).
    pub length: f64,
    /// Hydro cells across the column.
    pub cells: usize,
    /// Grey-plasma absorption coefficient (1/m).
    pub alpha: f64,
    /// Fraction of the column the front is asked to cross; sets the run
    /// duration from the expected speed rather than from a chosen time.
    pub cross_fraction: f64,
    /// Number of recorded profile snapshots.
    pub frames: usize,
    /// Time slices per pulse for the M6a rate integration.
    pub ignition_steps: usize,
}

/// Results of the `lsd` case.
pub struct LsdRun {
    /// Peak on-axis intensity of the igniting pulse (W/m²), via
    /// [`IntensityScale`](crate::field::IntensityScale).
    pub i_peak: f64,
    /// M6a breakdown threshold at the ambient pressure (W/m²).
    pub i_threshold: f64,
    /// Sustaining drive intensity the column runs on (W/m²).
    pub drive: f64,
    /// Whether the igniting pulse lit the gas at all.
    pub ignited: bool,
    /// Ambient density derived from `(T₀, p₀)` (kg/m³).
    pub rho_0: f64,
    /// Raizer's closed-form LSD velocity for this drive (m/s).
    pub d_raizer: f64,
    /// Front speed fitted over the settled half of the trajectory (m/s).
    pub d_measured: f64,
    /// Cell axis (m).
    pub x: Vec<f64>,
    /// Snapshot times (s).
    pub frame_time: Vec<f64>,
    /// Profiles `[frame, quantity, cell]`, quantities ordered
    /// `[p (Pa), ρ (kg/m³), u (m/s), α (1/m), I (W/m²)]`.
    pub profiles: Array3<f64>,
    /// Front position at each frame (m); `NaN` before a front exists.
    pub front_x: Vec<f64>,
    /// Column optical depth at each frame (dimensionless).
    pub optical_depth: Vec<f64>,
    /// `log₁₀` of the transmitted fraction at each frame. Logged because an
    /// established LSD plasma reaches hundreds of optical depths, where the
    /// fraction itself is numerically indistinguishable from zero.
    pub log10_transmission: Vec<f64>,
    /// Absorbed laser energy (J/m²) and the relative closure of the budget.
    pub deposited_energy: f64,
    pub energy_residual: f64,
    /// Both ends still ambient, so the run is uncontaminated by the boundary.
    pub boundaries_undisturbed: bool,
    /// What the inverse-bremsstrahlung closure gives at this run's own measured
    /// post-front state, at the run's wavelength: `α` (1/m) and the column's
    /// optical depth under it. Reported, not used — see [`run_lsd`].
    pub ib_alpha: f64,
    pub ib_optical_depth: f64,
    /// The same two at 10.6 µm, the CO₂ wavelength LSD experiments are actually
    /// done at. The comparison is the point, not either number alone.
    pub ib_alpha_co2: f64,
    pub ib_optical_depth_co2: f64,
}

/// Ignite a spark at the M6a threshold and run the laser-supported detonation
/// wave the sustaining beam then drives back up itself (the `lsd` CLI case,
/// M6c step 6).
///
/// # Why there are two intensities, and why that is the result
///
/// The case takes a short **igniting** pulse and a separate long **sustaining**
/// drive, which looks like an extra knob and is really this run's headline
/// finding. Putting M6a's kernel and M6c's wave in the same file forces the
/// question "does the beam that drives the detonation also light it?", and the
/// answer the two models give together is **no, by five orders of magnitude**:
///
/// - M6a's breakdown threshold in air at 1 atm **saturates at ≈1.14×10¹⁶ W/m²
///   and does not fall with pulse length** — 6 ns and 1 ms give 1.18×10¹⁶ and
///   1.14×10¹⁶. It is an intensity floor, not a fluence one: below it the
///   inelastic losses paid climbing to the ionization potential exceed the
///   inverse-bremsstrahlung heating, the net cascade rate is negative, and no
///   exposure time rescues it. Widening the focus does not help either — over a
///   500× range of spot radius the threshold moves by 4 %, because diffusion
///   loss is not what sets it at this pressure.
/// - The sustaining drive an LSD wave runs on is ~10¹¹ W/m² (10⁷ W/cm², the
///   spec's representative value).
///
/// So the beam that sustains the detonation cannot have started it. That is not
/// a defect in either model — it is the known experimental situation, where LSD
/// waves in clean air are initiated on a target, on an aerosol, or by a separate
/// high-intensity spike, and are then *sustained* far below breakdown by the
/// plasma that already exists. M6a's ungated absolute level (4.8–7.0× above the
/// measured Thiyagarajan & Thompson curve) does not touch the conclusion: the
/// gap is 10⁵ and the uncertainty is ~7×.
///
/// # What each half is worth
///
/// The **ignition** half is M6a's. The igniting pulse's peak intensity comes
/// from its power and focal radius through
/// [`IntensityScale`](crate::field::IntensityScale) — the T4 extraction's second
/// consumer, and the reason it was extracted. Whether and when the gas lights
/// therefore inherits M6a's explicitly ungated level, and the run reports the
/// threshold and the margin so that is visible rather than buried.
///
/// The **propagation** half is M6c's, and does not inherit it: the front speed
/// depends on the absorbed intensity at the front and on `ρ₀`, not on where or
/// when the spark was lit. G3/G3b/G3c and the G4 physics gate all run seeded
/// ignition for exactly this reason, so the velocity this case reports is
/// backed by gates that never touch [`AirBreakdown`].
///
/// # Why the grey closure, when the production one exists
///
/// [`Absorption::GreyThreshold`] drives the run, because it is what G3–G5 gate
/// and it introduces nothing that can drift. [`Absorption::InverseBremsstrahlung`]
/// is implemented and unit-tested but not in the loop, and rather than assert a
/// reason the run *evaluates* it at its own measured post-front state and
/// reports the answer — which turns out to be more interesting than the
/// deferral:
///
/// At 1064 nm the closure gives `α ≈ 6.8 1/m`, so the whole 2.5 cm column is
/// **0.17 optical depths** — the plasma is very nearly transparent to the beam
/// driving it, there would be no front, and `check_regime` would correctly
/// refuse the run as volumetric (the LSC regime, out of scope). At 10.6 µm the
/// same closure at the same gas state gives `α ≈ 1.1×10³ 1/m`: an absorption
/// length of 0.92 mm, which is 92 cells on this grid and 3.7 % of the domain —
/// comfortably inside `check_regime` on both counts, and an optical depth of 27.
///
/// Free-free absorption falls steeply toward short wavelengths, so **this is
/// the model reproducing why LSD experiments are done with CO₂ lasers**, not a
/// limitation of the implementation. What actually blocks running the closure
/// coupled is cost, and it is a specific cost: `PlasmaTable::temperature`
/// bisects ~45 times per cell per deposition call, and the driver calls
/// deposition three times per step. That is a faster table inversion, not a
/// finer grid — a separate change with its own gate.
///
/// If the beam does not break the gas down, this returns a run with
/// `ignited: false` and empty trajectories — a clean report, not a hang and not
/// an error, per the spec's failure-mode table.
pub fn run_lsd(p: &LsdParams) -> Result<LsdRun> {
    if !(p.length > 0.0 && p.cells >= 8) {
        anyhow::bail!(
            "lsd: need a positive length and at least 8 cells, got {} m / {}",
            p.length,
            p.cells
        );
    }
    if p.frames < 2 {
        anyhow::bail!("lsd: need at least 2 frames, got {}", p.frames);
    }
    if !(p.cross_fraction > 0.0 && p.cross_fraction < LSD_FOCUS_FRACTION) {
        anyhow::bail!(
            "lsd: --cross must be in (0, {LSD_FOCUS_FRACTION}) so the front stays \
             inside the domain, got {}",
            p.cross_fraction
        );
    }
    if !(p.drive > 0.0 && p.drive.is_finite()) {
        anyhow::bail!("lsd: drive intensity must be positive, got {}", p.drive);
    }
    if !(p.t0 > 0.0 && p.p0 > 0.0) {
        anyhow::bail!(
            "lsd: need positive ambient T and p, got {} K / {} Pa",
            p.t0,
            p.p0
        );
    }

    // Peak intensity at the focus, through the T4 scale — the same path the
    // blooming case pins its absolute intensity with.
    let grid = Grid::new(128, p.w_focus / 16.0);
    let focal_field = Field::gaussian(grid, p.wavelength, p.w_focus);
    let scale = IntensityScale::from_beam_power(p.ignite_power, focal_field.power())?;
    let i_peak = focal_field.peak_physical_intensity(scale);

    // Ignition: M6a's kernel, in T&T's focal geometry at this wavelength.
    let igniter = AirBreakdown::dry_air_tt2012_focus(p.wavelength)?;
    let i_threshold = igniter.threshold_intensity(p.fwhm, p.p0, p.ignition_steps)?;
    let ignited = igniter.breaks_down(i_peak, p.fwhm, p.p0, p.ignition_steps);

    let rho_0 = p.p0 / (R_AIR * p.t0);
    let gas = IdealGas::AIR;
    // The wave runs on the SUSTAINING drive, not on the igniting spike.
    let d_raizer = raizer_lsd_velocity(&gas, p.drive, rho_0);

    if !ignited {
        return Ok(LsdRun {
            i_peak,
            i_threshold,
            drive: p.drive,
            ignited: false,
            rho_0,
            d_raizer,
            d_measured: f64::NAN,
            x: Vec::new(),
            frame_time: Vec::new(),
            profiles: Array3::zeros((0, 5, 0)),
            front_x: Vec::new(),
            optical_depth: Vec::new(),
            log10_transmission: Vec::new(),
            deposited_energy: 0.0,
            energy_residual: f64::NAN,
            boundaries_undisturbed: true,
            ib_alpha: f64::NAN,
            ib_optical_depth: f64::NAN,
            ib_alpha_co2: f64::NAN,
            ib_optical_depth_co2: f64::NAN,
        });
    }

    let ambient = Primitive {
        rho: rho_0,
        u: 0.0,
        p: p.p0,
    };
    // The threshold is a multiple of ambient internal energy, as G4 established:
    // far enough above ambient that undisturbed air is transparent, far enough
    // below the shocked state that it enables the front rather than controls it.
    let e_ignite = 5.0 * gas.specific_internal_energy(rho_0, p.p0);
    let mut column = LsdColumn::seeded(
        gas,
        p.cells,
        p.length,
        ambient,
        SeededIgnition {
            centre: LSD_FOCUS_FRACTION * p.length,
            width: 6e-4,
            pressure: LSD_SEED_MULTIPLE * rho_0 * d_raizer * d_raizer / (gas.gamma + 1.0),
        },
        Absorption::GreyThreshold {
            alpha: p.alpha,
            e_ignite,
        },
        p.drive,
    )?;

    let dx = p.length / p.cells as f64;
    let x: Vec<f64> = (0..p.cells).map(|i| (i as f64 + 0.5) * dx).collect();
    let t_end = p.cross_fraction * p.length / d_raizer;

    let mut profiles = Array3::<f64>::zeros((p.frames, 5, p.cells));
    let mut frame_time = Vec::with_capacity(p.frames);
    let mut front_x = Vec::with_capacity(p.frames);
    let mut optical_depth = Vec::with_capacity(p.frames);
    let mut log10_transmission = Vec::with_capacity(p.frames);

    for frame in 0..p.frames {
        let t = frame as f64 * t_end / (p.frames - 1) as f64;
        column.advance_to(t)?;
        if frame == 1 {
            // Refuse rather than mis-model, in the M4 Péclet spirit: once the
            // wave exists, the run must be in the LSD regime the model is
            // written for. Checked at the first frame that has a front rather
            // than at t = 0, where there is only a seed and nothing to judge.
            column
                .check_regime()
                .context("the lsd run is outside the LSD regime")?;
        }

        let w = column.hydro().primitives();
        let alpha = column.alpha_profile()?;
        let intensity = column.intensity_profile()?;
        for (i, c) in w.iter().enumerate() {
            profiles[[frame, 0, i]] = c.p;
            profiles[[frame, 1, i]] = c.rho;
            profiles[[frame, 2, i]] = c.u;
            profiles[[frame, 3, i]] = alpha[i];
            profiles[[frame, 4, i]] = intensity[i];
        }
        let tau = column.optical_depth()?;
        frame_time.push(column.hydro().time());
        front_x.push(column.front_position().unwrap_or(f64::NAN));
        optical_depth.push(tau);
        // exp(−τ) underflows past τ ≈ 745; the log is the honest carrier.
        log10_transmission.push(-tau / std::f64::consts::LN_10);
    }

    // Fit the speed over the settled half of the trajectory — the wave starts
    // overdriven and relaxes onto its self-sustaining speed (G3c).
    let half = p.frames / 2;
    let settled: Vec<(f64, f64)> = frame_time[half..]
        .iter()
        .zip(&front_x[half..])
        .filter(|(_, x)| x.is_finite())
        .map(|(&t, &x)| (t, x))
        .collect();
    let d_measured = if settled.len() >= 2 {
        // Least-squares slope of x(t), negated: D is positive toward the laser.
        let n = settled.len() as f64;
        let mt = settled.iter().map(|s| s.0).sum::<f64>() / n;
        let mx = settled.iter().map(|s| s.1).sum::<f64>() / n;
        let cov: f64 = settled.iter().map(|s| (s.0 - mt) * (s.1 - mx)).sum();
        let var: f64 = settled.iter().map(|s| (s.0 - mt).powi(2)).sum();
        -cov / var
    } else {
        f64::NAN
    };

    // What the production closure says about this run, measured at the run's own
    // peak post-front state rather than assumed. `Flag` rather than `Refuse`:
    // the CJ state behind a strong front legitimately crosses the table's
    // singly-ionized ceiling, which is the case the flag exists for, and this is
    // a diagnostic rather than a quantity anything downstream depends on.
    let peak = column
        .hydro()
        .primitives()
        .into_iter()
        .fold(ambient, |best, c| if c.p > best.p { c } else { best });
    let ib_at = |wavelength: f64| -> f64 {
        PlasmaTable::load()
            .ok()
            .and_then(|table| {
                Absorption::InverseBremsstrahlung {
                    wavelength,
                    gaunt: 1.0,
                    table,
                    ceiling: IonizationCeiling::Flag,
                }
                .coefficient(&gas, peak)
                .ok()
            })
            .unwrap_or(f64::NAN)
    };
    let ib_alpha = ib_at(p.wavelength);
    let ib_alpha_co2 = ib_at(CO2_WAVELENGTH);

    Ok(LsdRun {
        i_peak,
        i_threshold,
        drive: p.drive,
        ignited: true,
        rho_0,
        d_raizer,
        d_measured,
        x,
        frame_time,
        profiles,
        front_x,
        optical_depth,
        log10_transmission,
        deposited_energy: column.deposited_energy(),
        energy_residual: column.energy_residual(),
        boundaries_undisturbed: column.boundaries_undisturbed(),
        ib_alpha,
        ib_optical_depth: ib_alpha * p.length,
        ib_alpha_co2,
        ib_optical_depth_co2: ib_alpha_co2 * p.length,
    })
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
            drive: 1.08,
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
        // The drive must straddle the sweep — above threshold at p_max, below
        // it at p_min — or the animation shows nothing switching on. The
        // corrected model makes this a narrow target: I_thr spans only ~1.17x
        // across 300-2000 Torr, so `drive` has to sit inside that.
        let (lo, hi) = (*r.threshold.last().unwrap(), r.threshold[0]);
        assert!(
            r.drive_intensity > lo && r.drive_intensity < hi,
            "drive {:.4e} does not straddle the sweep [{lo:.4e}, {hi:.4e}]",
            r.drive_intensity
        );
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

#[cfg(test)]
mod lsd_tests {
    use super::*;

    /// The pinned demonstration configuration, so the tests and the CLI
    /// defaults cannot drift apart silently.
    fn demo() -> LsdParams {
        LsdParams {
            wavelength: 1064e-9,
            fwhm: 6e-9,
            ignite_power: 1.5e7,
            w_focus: 20e-6,
            drive: 1e11,
            p0: 101_325.0,
            t0: 288.0,
            length: 2.5e-2,
            cells: 2500,
            alpha: 2e4,
            cross_fraction: 0.5,
            frames: 24,
            ignition_steps: 400,
        }
    }

    #[test]
    fn lsd_demo_ignites_and_reproduces_the_closed_form() {
        let r = run_lsd(&demo()).unwrap();
        assert!(r.ignited, "the demo pulse must light the gas");
        assert!(
            r.i_peak > r.i_threshold,
            "ignited without clearing the threshold: {:.3e} vs {:.3e}",
            r.i_peak,
            r.i_threshold
        );
        // The velocity is the case's headline number and it is the one G3
        // gates; if the demo geometry ever drifts out of that agreement, the
        // run is no longer demonstrating what it claims.
        let err = r.d_measured / r.d_raizer - 1.0;
        assert!(
            err.abs() < 0.01,
            "D = {:.1} m/s vs Raizer {:.1} ({:+.3} %)",
            r.d_measured,
            r.d_raizer,
            100.0 * err
        );
        assert!(
            r.boundaries_undisturbed,
            "the wave reached a boundary, so the run is contaminated"
        );
        assert!(
            r.energy_residual < 1e-10,
            "energy budget off by {:.3e}",
            r.energy_residual
        );
        // Shapes the render script indexes.
        assert_eq!(r.profiles.dim(), (24, 5, 2500));
        assert_eq!(r.x.len(), 2500);
        assert_eq!(r.frame_time.len(), 24);
        assert_eq!(r.front_x.len(), 24);
        // The front runs toward the laser, monotonically, once it exists.
        let track: Vec<f64> = r
            .front_x
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(track.len() > 20, "the front vanished mid-run");
        for w in track.windows(2) {
            assert!(w[1] < w[0], "the front reversed: {:?}", track);
        }
        // The column becomes a shutter, not a partial filter.
        assert!(
            *r.log10_transmission.last().unwrap() < -100.0,
            "final log10 transmission {:.1}",
            r.log10_transmission.last().unwrap()
        );
    }

    /// The case's headline finding, pinned: the beam that *sustains* the
    /// detonation is orders of magnitude below the one that could *light* it.
    /// If a future change to either model closes that gap, this fails and the
    /// write-up gets revisited rather than quietly going stale.
    #[test]
    fn the_sustaining_drive_is_far_below_the_breakdown_threshold() {
        let r = run_lsd(&demo()).unwrap();
        let ratio = r.drive / r.i_threshold;
        assert!(
            ratio < 1e-4,
            "the sustaining drive is only {ratio:.2e} of the breakdown threshold; \
             the two-stage argument in run_lsd's docs no longer holds"
        );
        // And it is an intensity floor, not a fluence one: a pulse a hundred
        // thousand times longer does not lower it appreciably.
        let igniter = AirBreakdown::dry_air_tt2012_focus(1064e-9).unwrap();
        let short = igniter.threshold_intensity(6e-9, 101_325.0, 400).unwrap();
        let long = igniter.threshold_intensity(1e-3, 101_325.0, 400).unwrap();
        assert!(
            long > 0.9 * short,
            "the threshold fell from {short:.3e} to {long:.3e} over 6 ns → 1 ms; \
             it is not the intensity floor run_lsd's docs describe"
        );
    }

    /// A beam below threshold must produce a clean report, not a hang and not
    /// an error — the spec's failure-mode table.
    #[test]
    fn a_beam_that_cannot_ignite_reports_cleanly() {
        let r = run_lsd(&LsdParams {
            ignite_power: 1.0,
            ..demo()
        })
        .unwrap();
        assert!(!r.ignited);
        assert!(r.i_peak < r.i_threshold);
        assert!(r.profiles.is_empty() && r.front_x.is_empty());
        assert!(r.d_measured.is_nan());
    }

    #[test]
    fn degenerate_lsd_parameters_are_refused() {
        for bad in [
            LsdParams { cells: 4, ..demo() },
            LsdParams {
                frames: 1,
                ..demo()
            },
            LsdParams {
                length: -1.0,
                ..demo()
            },
            LsdParams {
                drive: 0.0,
                ..demo()
            },
            LsdParams { t0: 0.0, ..demo() },
            // Asking the front to cross more of the column than it has room
            // for would run it off the boundary and quietly report nonsense.
            LsdParams {
                cross_fraction: 0.9,
                ..demo()
            },
        ] {
            assert!(run_lsd(&bad).is_err(), "accepted {bad:?}");
        }
    }
}

/// Parameters of the `ignition` case (M6a.2 — turbulence-degraded ignition
/// statistics).
#[derive(Debug, Clone)]
pub struct IgnitionParams {
    pub n: usize,
    pub dx: f64,
    pub wavelength: f64,
    /// Launch `1/e²` waist (m).
    pub w0: f64,
    /// Path length (m).
    pub z: f64,
    /// Phase screens along the path.
    pub screens: usize,
    /// Refractive-index structure constant (m^(−2/3)).
    pub cn2: f64,
    /// Turbulence outer scale (m).
    pub l0: f64,
    /// Receiver aperture diameter (m).
    pub aperture: f64,
    /// Focal length of the focusing optic (m) — sets the wander scale only.
    pub focal_length: f64,
    /// Total beam power (W).
    pub power: f64,
    /// Igniting-pulse FWHM (s) for the M6a test.
    pub fwhm: f64,
    /// Ambient pressure (Pa).
    pub p0: f64,
    /// Time slices per pulse for the M6a rate integration.
    pub ignition_steps: usize,
    pub realizations: usize,
    /// Master seed for the reproducible ensemble.
    pub seed: u64,
}

/// Results of the `ignition` case.
pub struct IgnitionRun {
    /// Fraction of realizations that ignited.
    pub p_ignite: f64,
    /// Peak focal intensity with no turbulence (W/m²).
    pub i_focus_vacuum: f64,
    /// M6a breakdown threshold at the ambient pressure (W/m²).
    pub i_threshold: f64,
    /// Per realization: focal intensity relative to vacuum.
    pub focal_ratio: Vec<f64>,
    /// Per realization: the phase-only Strehl, as the diagnostic that separates
    /// wavefront error from scintillation.
    pub strehl: Vec<f64>,
    /// Per realization: focal-spot displacement (m).
    pub wander: Vec<(f64, f64)>,
    /// RMS radial wander (m).
    pub wander_rms: f64,
    /// Per realization: did it light.
    pub ignited: Vec<bool>,
    /// Mean guard-band absorbed fraction — non-negligible means the grid, not
    /// the turbulence, shaped the answer.
    pub guard_frac_mean: f64,
}

/// Run an ensemble of turbulence realizations and report how often the focal
/// spot still ignites the air, and where it lands (the M6a.2 driver).
///
/// # What is and is not a claim about the world
///
/// **The position of `p_ignite` on the `cn2` axis is not.** Whether a given
/// realization lights depends on [`AirBreakdown`]'s absolute threshold, which is
/// M6a's explicitly ungated quantity (4.8–7.0× above the measured Thiyagarajan
/// & Thompson curve, inside the 3–10× inter-lab scatter). Every ignition
/// probability here carries that offset, and it must be labelled so wherever it
/// is plotted (`docs/M6A2_SPEC.md` § "What this rung can and cannot claim").
///
/// Everything upstream of that one boolean *is* independent of M6a and is
/// gated: the pupil phase statistics against Noll, the focal-intensity
/// estimator against its closed forms, and this driver's own convergence and
/// thread-count reproducibility.
///
/// # No focal grid
///
/// The peak focal intensity comes from the pupil integral
/// ([`Aperture`](crate::aperture::Aperture)), not from a focal-plane field —
/// turbulence needs centimetre samples over a kilometre and the focal spot is
/// micrometres across, so one grid cannot carry both.
pub fn run_ignition(p: &IgnitionParams) -> Result<IgnitionRun> {
    if p.realizations == 0 {
        anyhow::bail!("ignition: need at least one realization");
    }
    let grid = Grid::new(p.n, p.dx);
    let pupil = Aperture::new(grid, p.aperture)?;

    // Physical intensity scale, pinned once from the launch field (T4).
    let launch = Field::gaussian(grid, p.wavelength, p.w0);
    let p_field = launch.power();
    let scale = IntensityScale::from_beam_power(p.power, p_field)?;
    // |∫U dA|² / (λf)² is an intensity in the field's own units; the T4 scale
    // takes it to W/m².
    let lf2 = (p.wavelength * p.focal_length).powi(2);
    let to_physical = |focal_power: f64| scale.to_physical(focal_power / lf2);

    // Vacuum reference: the same launch, the same path, no turbulence.
    let i_focus_vacuum = {
        let mut field = Field::gaussian(grid, p.wavelength, p.w0);
        let mut prop = Propagator::new(grid, p.wavelength)?;
        let medium = crate::medium::Vacuum::new(grid.n);
        prop.propagate(
            &mut field,
            &medium,
            p.z / p.screens as f64,
            0,
            p.screens,
            |_, _| {},
        )?;
        to_physical(pupil.focal_power(&field))
    };
    if !(i_focus_vacuum > 0.0 && i_focus_vacuum.is_finite()) {
        anyhow::bail!("ignition: vacuum reference has no focal intensity ({i_focus_vacuum})");
    }

    let igniter = AirBreakdown::dry_air_tt2012_focus(p.wavelength)?;
    let i_threshold = igniter.threshold_intensity(p.fwhm, p.p0, p.ignition_steps)?;

    // Each realization derives all randomness from its index (the M3 contract),
    // and results come back in index order, so every reduction below is
    // thread-count independent.
    let results = seeded_ensemble(p.realizations, |i| -> Result<_> {
        let path = TurbulentPath::new(grid, p.wavelength, p.cn2, p.l0, p.z, p.screens, p.seed, i);
        let mut field = Field::gaussian(grid, p.wavelength, p.w0);
        let p_in = field.power();
        let mut prop = Propagator::new(grid, p.wavelength)?;
        prop.propagate(&mut field, &path, path.dz(), 0, path.n_slabs(), |_, _| {})?;

        let i_focus = to_physical(pupil.focal_power(&field));
        let ratio = i_focus / i_focus_vacuum;
        let strehl = pupil.phase_only_strehl(&field)?;
        let wander = pupil.focal_wander_of_field(&field, p.wavelength, p.focal_length)?;
        let ignited = igniter.breaks_down(i_focus, p.fwhm, p.p0, p.ignition_steps);
        Ok((ratio, strehl, wander, ignited, prop.guard_absorbed() / p_in))
    });

    let mut focal_ratio = Vec::with_capacity(p.realizations);
    let mut strehl = Vec::with_capacity(p.realizations);
    let mut wander = Vec::with_capacity(p.realizations);
    let mut ignited = Vec::with_capacity(p.realizations);
    let mut guard = 0.0;
    for r in results {
        let (a, b, c, d, g) = r?;
        focal_ratio.push(a);
        strehl.push(b);
        wander.push(c);
        ignited.push(d);
        guard += g;
    }
    let n = p.realizations as f64;
    let wander_rms = (wander.iter().map(|(x, y)| x * x + y * y).sum::<f64>() / n).sqrt();

    Ok(IgnitionRun {
        p_ignite: ignited.iter().filter(|&&b| b).count() as f64 / n,
        i_focus_vacuum,
        i_threshold,
        focal_ratio,
        strehl,
        wander,
        wander_rms,
        ignited,
        guard_frac_mean: guard / n,
    })
}

/// Parameters of the `ignition` CLI case: an [`IgnitionParams`] geometry swept
/// over turbulence strength.
#[derive(Debug, Clone)]
pub struct IgnitionSweepParams {
    /// Geometry and ensemble size. Its `cn2` is ignored — the sweep sets it.
    pub base: IgnitionParams,
    /// Lowest and highest `Cn²` of the sweep (m^(−2/3)).
    pub cn2_min: f64,
    pub cn2_max: f64,
    /// Number of log-spaced sweep points.
    pub points: usize,
}

/// Results of the `ignition` CLI case.
pub struct IgnitionSweepRun {
    pub cn2: Vec<f64>,
    /// Fried parameter at each point (m).
    pub r0: Vec<f64>,
    /// `D/r₀` at each point — the parameter the pupil statistics are written in.
    pub d_over_r0: Vec<f64>,
    pub p_ignite: Vec<f64>,
    /// Binomial standard error of each `p_ignite`. Reported because it is the
    /// honest error bar on a Bernoulli mean and the figure must carry it.
    pub p_ignite_se: Vec<f64>,
    pub mean_ratio: Vec<f64>,
    pub median_ratio: Vec<f64>,
    pub wander_rms: Vec<f64>,
    /// Per-point, per-realization focal-intensity ratio `[point, realization]`.
    pub focal_ratio: Array2<f64>,
    /// Per-point, per-realization phase-only Strehl `[point, realization]`.
    pub strehl: Array2<f64>,
    pub i_focus_vacuum: f64,
    pub i_threshold: f64,
    /// Width of the ignition transition, in decades of `Cn²`, measured as the
    /// span between `p_ignite` = 0.9 and 0.1 by linear interpolation in
    /// `log₁₀ Cn²`. `NaN` if the sweep does not bracket both.
    pub transition_decades: f64,
}

/// Sweep turbulence strength and report how the ignition probability, the focal
/// intensity and the spot wander respond (the `ignition` CLI case, M6a.2).
///
/// # What the figure may and may not claim
///
/// The **shape** of the response is this rung's contribution: how fast the
/// ignition window closes as turbulence strengthens, how the focal-intensity
/// distribution broadens, and the `Cn²^(1/2)` wander law (gated, W1).
///
/// The **position** of the curve on the `Cn²` axis is not a claim about the
/// world. It is set by where `AirBreakdown`'s absolute threshold falls, and
/// that threshold is M6a's explicitly ungated quantity — 4.8–7.0× above the
/// measured Thiyagarajan & Thompson curve, inside the 3–10× inter-lab scatter.
/// Shifting the threshold slides the whole curve sideways without changing its
/// shape. Any plot of this must say so.
pub fn run_ignition_sweep(p: &IgnitionSweepParams) -> Result<IgnitionSweepRun> {
    if p.points < 2 {
        anyhow::bail!("ignition sweep: need at least 2 points, got {}", p.points);
    }
    if !(p.cn2_min > 0.0 && p.cn2_max > p.cn2_min) {
        anyhow::bail!(
            "ignition sweep: need 0 < cn2_min < cn2_max, got {} .. {}",
            p.cn2_min,
            p.cn2_max
        );
    }
    let n_real = p.base.realizations;
    let mut out = IgnitionSweepRun {
        cn2: Vec::with_capacity(p.points),
        r0: Vec::with_capacity(p.points),
        d_over_r0: Vec::with_capacity(p.points),
        p_ignite: Vec::with_capacity(p.points),
        p_ignite_se: Vec::with_capacity(p.points),
        mean_ratio: Vec::with_capacity(p.points),
        median_ratio: Vec::with_capacity(p.points),
        wander_rms: Vec::with_capacity(p.points),
        focal_ratio: Array2::zeros((p.points, n_real)),
        strehl: Array2::zeros((p.points, n_real)),
        i_focus_vacuum: 0.0,
        i_threshold: 0.0,
        transition_decades: f64::NAN,
    };

    for i in 0..p.points {
        let frac = i as f64 / (p.points - 1) as f64;
        let cn2 = p.cn2_min * (p.cn2_max / p.cn2_min).powf(frac);
        let run = run_ignition(&IgnitionParams {
            cn2,
            ..p.base.clone()
        })?;
        let r0 = crate::validate::fried_r0(cn2, p.base.wavelength, p.base.z);
        let n = n_real as f64;
        let mut sorted = run.focal_ratio.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite focal ratios"));

        out.cn2.push(cn2);
        out.r0.push(r0);
        out.d_over_r0.push(p.base.aperture / r0);
        out.p_ignite.push(run.p_ignite);
        out.p_ignite_se
            .push((run.p_ignite * (1.0 - run.p_ignite) / n).sqrt());
        out.mean_ratio.push(run.focal_ratio.iter().sum::<f64>() / n);
        out.median_ratio.push(sorted[sorted.len() / 2]);
        out.wander_rms.push(run.wander_rms);
        for (j, (&r, &s)) in run.focal_ratio.iter().zip(&run.strehl).enumerate() {
            out.focal_ratio[[i, j]] = r;
            out.strehl[[i, j]] = s;
        }
        out.i_focus_vacuum = run.i_focus_vacuum;
        out.i_threshold = run.i_threshold;
    }

    out.transition_decades = transition_width_decades(&out.cn2, &out.p_ignite);
    Ok(out)
}

/// Span in decades of `cn2` between `p_ignite` = 0.9 and 0.1, by linear
/// interpolation in `log₁₀ cn2`. `NaN` when the sweep does not bracket both.
fn transition_width_decades(cn2: &[f64], p: &[f64]) -> f64 {
    let cross = |level: f64| -> Option<f64> {
        // p_ignite falls with cn2, so find the first descending crossing.
        for w in p.windows(2).enumerate() {
            let (i, pair) = w;
            if pair[0] >= level && pair[1] < level {
                let t = (pair[0] - level) / (pair[0] - pair[1]);
                return Some(cn2[i].log10() + t * (cn2[i + 1].log10() - cn2[i].log10()));
            }
        }
        None
    };
    match (cross(0.9), cross(0.1)) {
        (Some(hi), Some(lo)) => lo - hi,
        _ => f64::NAN,
    }
}
