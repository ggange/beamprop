//! `beamprop` command-line interface.
//!
//! Build a field, propagate it, dump `.npy` arrays (+ `_meta.json`/`_notes.md`
//! sidecars); images are rendered from those by `scripts/render.py`. The
//! compute loops live in `beamprop::cases`, shared with the M5 Python
//! bindings — this binary is the file-writing shell around them.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use beamprop::cases::{
    BloomingParams, BreakdownParams, LsdParams, PropagateParams, TurbulenceParams, run_blooming,
    run_breakdown, run_lsd, run_propagate, run_turbulence,
};
use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::validate::GaussianBeam;

#[derive(Parser, Debug)]
#[command(name = "beamprop", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Grid and beam parameters shared by all subcommands.
#[derive(clap::Args, Debug)]
struct BeamArgs {
    /// Grid samples per side.
    #[arg(long, default_value_t = 512)]
    n: usize,
    /// Grid spacing in metres.
    #[arg(long, default_value_t = 1e-3)]
    dx: f64,
    /// Vacuum wavelength in metres.
    #[arg(long, default_value_t = 1.0e-6)]
    wavelength: f64,
    /// Gaussian 1/e² waist radius w0 in metres. Default is the Smith 1977
    /// F₀=5 waist w0 = √(2·F₀·z/k) at z=500 m, λ=1 µm — the blooming default
    /// run then reproduces the M4 B3 validation geometry end-to-end.
    #[arg(long, default_value_t = 2.8209e-2)]
    w0: f64,
    /// Output basename (within --out-dir).
    #[arg(long, default_value = "beam")]
    out: String,
    /// Directory for all generated files; created if missing.
    #[arg(long, default_value = "out")]
    out_dir: PathBuf,
}

impl BeamArgs {
    /// Ensure the output directory exists and resolve `<out-dir>/<name>`.
    fn out_path(&self, name: &str) -> Result<PathBuf> {
        fs::create_dir_all(&self.out_dir)
            .with_context(|| format!("creating output directory {}", self.out_dir.display()))?;
        Ok(self.out_dir.join(name))
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Write a Gaussian field's intensity to <out>.npy (M0).
    Gaussian {
        #[command(flatten)]
        beam: BeamArgs,
    },
    /// Propagate a Gaussian beam (M1/M2) and write the data: side-view map
    /// <out>_xz.npy, snapshot stack <out>_frames.npy, final <out>_final.npy,
    /// plus _meta.json/_notes.md sidecars (render images with
    /// `python3 scripts/render.py`). Lossless unless --alpha or --visibility
    /// sets a Beer-Lambert extinction.
    Propagate {
        #[command(flatten)]
        beam: BeamArgs,
        /// Total propagation distance in metres (default: 2 Rayleigh ranges).
        #[arg(long)]
        z: Option<f64>,
        /// Number of split-step slabs.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Number of transverse snapshots to record along the path.
        #[arg(long, default_value_t = 5)]
        frames: usize,
        /// Uniform power extinction coefficient in 1/m (Beer-Lambert).
        #[arg(long, conflicts_with = "visibility")]
        alpha: Option<f64>,
        /// Meteorological visibility in metres; sets alpha via the Kruse
        /// model at the beam wavelength.
        #[arg(long)]
        visibility: Option<f64>,
    },
    /// Propagate a Gaussian beam through Kolmogorov/von Karman turbulence
    /// (M3) and write the Monte-Carlo data: receiver-plane and side-view
    /// frame stacks <out>_frames.npy / <out>_xz_frames.npy, long-exposure
    /// mean <out>_longexp.npy, _meta.json/_notes.md sidecars, and a
    /// comparison against weak-turbulence theory (render GIFs/PNGs with
    /// `python3 scripts/render.py`).
    Turbulence {
        #[command(flatten)]
        beam: BeamArgs,
        /// Total propagation distance in metres.
        #[arg(long, default_value_t = 1000.0)]
        z: f64,
        /// Number of phase screens (= split-step slabs).
        #[arg(long, default_value_t = 10)]
        screens: usize,
        /// Refractive-index structure constant Cn^2 in m^(-2/3).
        #[arg(long, default_value_t = 1.5e-14)]
        cn2: f64,
        /// Turbulence outer scale L0 in metres.
        #[arg(long, default_value_t = 1e3)]
        l0: f64,
        /// Number of Monte-Carlo realizations (= GIF frames).
        #[arg(long, default_value_t = 48)]
        realizations: usize,
        /// Master seed for the reproducible screen ensemble.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Propagate a high-power Gaussian beam through steady-state thermal
    /// blooming (M4): the beam heats the air, the heated air defocuses and
    /// bends the beam into the wind. Writes the same data family as
    /// `propagate` (side view, snapshots, final field, sidecars; render with
    /// `python3 scripts/render.py`).
    Blooming {
        #[command(flatten)]
        beam: BeamArgs,
        /// Total beam power in watts. Default places the default geometry at
        /// Smith number N_c ≈ 1 (N_φ ≈ 8.9) — the blooming rollover, inside the
        /// M4 B3 validated range N_c ∈ [0.5, 1.8].
        #[arg(long, default_value_t = 2600.0)]
        power: f64,
        /// Crosswind speed in m/s (blows along +x).
        #[arg(long, default_value_t = 2.0)]
        wind: f64,
        /// Absorbed-power coefficient in 1/m (heats the air and depletes the
        /// beam; scattering is not blooming-active and belongs to M2). Default
        /// matches the α used by every M4 validation gate and the power sweep
        /// (the Smith 1977 F₀=5 regime; see tests/blooming.rs, scripts/sweep_blooming.py).
        #[arg(long, default_value_t = 1e-4)]
        alpha_abs: f64,
        /// Ambient temperature in K.
        #[arg(long, default_value_t = 288.15)]
        t0: f64,
        /// Ambient pressure in Pa.
        #[arg(long, default_value_t = 101_325.0)]
        p0: f64,
        /// Total propagation distance in metres.
        #[arg(long, default_value_t = 500.0)]
        z: f64,
        /// Number of split-step slabs.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Number of transverse snapshots to record along the path.
        #[arg(long, default_value_t = 5)]
        frames: usize,
    },
    /// Sweep the 0-D optical-breakdown threshold against pressure (M6a) and
    /// record the electron avalanche at a fixed drive intensity. Writes
    /// <out>_threshold.csv (central curve + the δ_eff literature envelope),
    /// <out>_ne_traces.npy (the animation frames), and _meta.json/_notes.md.
    /// Render with `python3 scripts/render_breakdown.py`. Pure rate physics:
    /// no grid, no propagator — none of the beam arguments apply.
    Breakdown {
        /// Vacuum wavelength in metres (1064 nm is the validated case).
        #[arg(long, default_value_t = 1064e-9)]
        wavelength: f64,
        /// Pulse FWHM in seconds (T&T: 6 ns).
        #[arg(long, default_value_t = 6e-9)]
        fwhm: f64,
        /// Lowest sweep pressure in Torr.
        #[arg(long, default_value_t = 300.0)]
        p_min: f64,
        /// Highest sweep pressure in Torr. The default 300–2000 Torr window is
        /// the one the external T&T slope gate is validated over.
        #[arg(long, default_value_t = 2000.0)]
        p_max: f64,
        /// Number of log-spaced sweep pressures (= animation frames).
        #[arg(long, default_value_t = 48)]
        points: usize,
        /// Time slices per pulse for the rate integration.
        #[arg(long, default_value_t = 400)]
        steps: usize,
        /// Drive intensity for the traces, as a multiple of the threshold at
        /// --p-max. Must straddle the sweep to be interesting, and the usable
        /// range is narrow: the corrected model's threshold spans only ~1.17x
        /// across 300-2000 Torr, so anything above that ignites every frame.
        #[arg(long, default_value_t = 1.08)]
        drive: f64,
        /// Output basename (within --out-dir).
        #[arg(long, default_value = "breakdown")]
        out: String,
        /// Directory for all generated files; created if missing.
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
    },
    /// Ignite a spark at the M6a breakdown threshold and run the
    /// laser-supported detonation wave the sustaining beam drives back up
    /// itself (M6c). Writes <out>_profiles.npy (p, rho, u, alpha, I per frame),
    /// <out>_trajectory.csv (the front track and the column's optical depth),
    /// and _meta.json/_notes.md. Render with `python3 scripts/render_lsd.py`.
    /// 1-D gas dynamics: the beam arguments do not apply.
    Lsd {
        /// Vacuum wavelength in metres.
        #[arg(long, default_value_t = 1064e-9)]
        wavelength: f64,
        /// Igniting-pulse FWHM in seconds.
        #[arg(long, default_value_t = 6e-9)]
        fwhm: f64,
        /// Peak power of the igniting pulse in watts. The default 15 MW in a
        /// 6 ns pulse (~90 mJ, an ordinary Q-switched Nd:YAG) clears the M6a
        /// threshold by 2.0x at the default focus.
        #[arg(long, default_value_t = 1.5e7)]
        ignite_power: f64,
        /// Igniting focal spot 1/e^2 intensity radius in metres. The default
        /// 20 um is T&T's focal geometry, which is what the M6a kernel this
        /// case triggers from is built on.
        #[arg(long, default_value_t = 20e-6)]
        w_focus: f64,
        /// Sustaining drive intensity in W/m^2 — the long-pulse beam the
        /// detonation runs on, NOT the igniting spike. The default 1e11
        /// (1e7 W/cm^2) is the spec's representative LSD drive, and sits five
        /// orders of magnitude below the breakdown threshold; that gap is the
        /// point of the case, not an oversight.
        #[arg(long, default_value_t = 1e11)]
        drive: f64,
        /// Ambient pressure in pascals.
        #[arg(long, default_value_t = 101_325.0)]
        p0: f64,
        /// Ambient temperature in kelvin.
        #[arg(long, default_value_t = 288.0)]
        t0: f64,
        /// Column length in metres.
        #[arg(long, default_value_t = 2.5e-2)]
        length: f64,
        /// Hydro cells across the column. The default 2500 puts the 50 um
        /// absorption length across 5 cells, which is what check_regime needs.
        #[arg(long, default_value_t = 2500)]
        cells: usize,
        /// Grey-plasma absorption coefficient in 1/m.
        #[arg(long, default_value_t = 2e4)]
        alpha: f64,
        /// Fraction of the column the front is asked to cross. Sets the run
        /// duration from the expected speed; must stay under 0.75 so the wave
        /// never reaches the boundary.
        #[arg(long, default_value_t = 0.5)]
        cross: f64,
        /// Number of recorded profile snapshots (= animation frames).
        #[arg(long, default_value_t = 48)]
        frames: usize,
        /// Time slices per pulse for the M6a ignition test.
        #[arg(long, default_value_t = 400)]
        ignition_steps: usize,
        /// Output basename (within --out-dir).
        #[arg(long, default_value = "lsd")]
        out: String,
        /// Directory for all generated files; created if missing.
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
    },
    /// Remove generated results (.npy, .png, .gif files and *_notes.md /
    /// *_meta.json sidecars) from the output directory. Only those are
    /// touched; other files and the directory itself are left alone.
    Clean {
        /// Directory to clean.
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Gaussian { beam } => gaussian(&beam),
        Cmd::Propagate {
            beam,
            z,
            steps,
            frames,
            alpha,
            visibility,
        } => propagate(&beam, z, steps, frames, alpha, visibility),
        Cmd::Turbulence {
            beam,
            z,
            screens,
            cn2,
            l0,
            realizations,
            seed,
        } => turbulence(&beam, z, screens, cn2, l0, realizations, seed),
        Cmd::Blooming {
            beam,
            power,
            wind,
            alpha_abs,
            t0,
            p0,
            z,
            steps,
            frames,
        } => blooming(&beam, power, wind, alpha_abs, t0, p0, z, steps, frames),
        Cmd::Breakdown {
            wavelength,
            fwhm,
            p_min,
            p_max,
            points,
            steps,
            drive,
            out,
            out_dir,
        } => breakdown(
            wavelength, fwhm, p_min, p_max, points, steps, drive, &out, &out_dir,
        ),
        Cmd::Lsd {
            wavelength,
            fwhm,
            ignite_power,
            w_focus,
            drive,
            p0,
            t0,
            length,
            cells,
            alpha,
            cross,
            frames,
            ignition_steps,
            out,
            out_dir,
        } => lsd(
            &LsdParams {
                wavelength,
                fwhm,
                ignite_power,
                w_focus,
                drive,
                p0,
                t0,
                length,
                cells,
                alpha,
                cross_fraction: cross,
                frames,
                ignition_steps,
            },
            &out,
            &out_dir,
        ),
        Cmd::Clean { out_dir } => clean(&out_dir),
    }
}

fn gaussian(args: &BeamArgs) -> Result<()> {
    let grid = Grid::new(args.n, args.dx);
    let field = Field::gaussian(grid, args.wavelength, args.w0);
    let npy = args.out_path(&format!("{}.npy", args.out))?;
    field.save_intensity_npy(&npy)?;
    println!(
        "wrote {}  (n={}, dx={} m, lambda={} m, w0={} m, power={:.6})",
        npy.display(),
        grid.n,
        grid.dx,
        args.wavelength,
        args.w0,
        field.power()
    );
    Ok(())
}

fn turbulence(
    args: &BeamArgs,
    z: f64,
    screens: usize,
    cn2: f64,
    l0: f64,
    realizations: usize,
    seed: u64,
) -> Result<()> {
    use beamprop::validate::{fried_r0, rytov_variance};

    let beam = GaussianBeam {
        w0: args.w0,
        wavelength: args.wavelength,
    };
    println!(
        "turbulent path: z = {z} m, {screens} screens, Cn2 = {cn2:.2e} m^-2/3, \
         r0 = {:.3} m, Rytov sigma_R^2 = {:.3}",
        fried_r0(cn2, args.wavelength, z),
        rytov_variance(cn2, args.wavelength, z)
    );

    let run = run_turbulence(&TurbulenceParams {
        n: args.n,
        dx: args.dx,
        wavelength: args.wavelength,
        w0: args.w0,
        z,
        screens,
        cn2,
        l0,
        realizations,
        seed,
    })?;
    let grid = run.grid;
    let substeps = run.substeps;
    let guard_frac_mean = run.guard_frac_mean;

    // Frame stacks + run metadata: the inputs to scripts/render.py, which
    // does all image rendering.
    let frames_npy = args.out_path(&format!("{}_frames.npy", args.out))?;
    ndarray_npy::write_npy(&frames_npy, &run.frames)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", frames_npy.display()))?;
    let xz_npy = args.out_path(&format!("{}_xz_frames.npy", args.out))?;
    ndarray_npy::write_npy(&xz_npy, &run.xz_frames)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", xz_npy.display()))?;
    let meta = format!(
        "{{\n  \"case\": \"turbulence\",\n  \"n\": {n},\n  \"dx\": {dx},\n  \
         \"wavelength\": {wl},\n  \"w0\": {w0},\n  \"z\": {z},\n  \
         \"screens\": {screens},\n  \"substeps\": {substeps},\n  \"cn2\": {cn2},\n  \
         \"l0\": {l0},\n  \"realizations\": {realizations},\n  \"seed\": {seed},\n  \
         \"xz_x_min\": {x_min},\n  \"xz_x_max\": {x_max}\n}}\n",
        n = grid.n,
        dx = args.dx,
        wl = args.wavelength,
        w0 = args.w0,
        x_min = -grid.extent() / 4.0,
        x_max = grid.extent() / 4.0,
    );
    let meta_path = args.out_path(&format!("{}_meta.json", args.out))?;
    fs::write(&meta_path, meta).with_context(|| format!("writing {}", meta_path.display()))?;

    // Long-exposure (ensemble-mean) receiver intensity.
    let npy = args.out_path(&format!("{}_longexp.npy", args.out))?;
    ndarray_npy::write_npy(&npy, &run.longexp)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", npy.display()))?;

    println!(
        "  long-exposure width (theory): {:.2} mm vs vacuum {:.2} mm",
        beam.long_exposure_width(z, cn2) * 1e3,
        beam.width_at(z) * 1e3
    );
    println!("  guard-band absorption (ensemble mean): {guard_frac_mean:.2e} of initial power");

    let r0 = fried_r0(cn2, args.wavelength, z);
    let notes = format!(
        "# Test case: Gaussian beam through Kolmogorov/von Kármán turbulence\n\
         \n\
         Collimated Gaussian beam, Monte-Carlo ensemble of {realizations} independent\n\
         atmospheres (von Kármán phase screens, FFT synthesis + 6 subharmonic levels),\n\
         symmetric split-step propagation. Models and references: docs/MODELS.md in the\n\
         beamprop repository.\n\
         \n\
         ## Parameters\n\
         \n\
         | Quantity | Value |\n\
         |---|---|\n\
         | Grid | {n} x {n}, dx = {dx:.3e} m (extent {extent:.3} m) |\n\
         | Wavelength λ | {wavelength:.3e} m |\n\
         | Waist w0 (1/e² intensity radius) | {w0:.3e} m |\n\
         | Rayleigh range zR | {zr:.1} m |\n\
         | Path length z | {z} m |\n\
         | Phase screens | {screens} (spacing {dz_screen:.1} m, {substeps} diffraction substeps each) |\n\
         | Cn² | {cn2:.3e} m^(-2/3) |\n\
         | Outer scale L0 | {l0:.3e} m |\n\
         | Realizations / master seed | {realizations} / {seed} |\n\
         \n\
         ## Derived quantities\n\
         \n\
         | Quantity | Value |\n\
         |---|---|\n\
         | Fried parameter r0 (full path) | {r0:.4} m ({r0_dx:.1} grid samples) |\n\
         | Rytov variance σ_R² (plane wave) | {rytov:.3} (weak fluctuation if ≲ 0.3) |\n\
         | Long-exposure radius W_LT, theory | {wlt:.2} mm (vacuum w(z) = {wvac:.2} mm) |\n\
         | Guard-band absorbed power (ensemble mean) | {guard_frac_mean:.2e} of initial (grid-edge artifact unless ≈ 0) |\n\
         \n\
         ## Files\n\
         \n\
         - `{out}_frames.npy` — receiver-plane intensity |u(x,y,z={z})|², shape\n\
           {realizations} × {n} × {n} (float64). Each slice is one **independent\n\
           atmospheric realization**, not a time sequence; it spans the full grid,\n\
           {extent:.3} m per side.\n\
         - `{out}_xz_frames.npy` — side views: central slice I(x, 0, z) per realization.\n\
           Second axis is x over the middle half of the grid ({half_extent:.3} m), third\n\
           axis is z from 0 to {z} m.\n\
         - `{out}_longexp.npy` — ensemble-mean receiver intensity (the long-exposure\n\
           image), float64, same extent as the receiver-plane frames.\n\
         - `{out}_meta.json` — the run parameters.\n\
         \n\
         ## Rendering\n\
         \n\
         The solver writes data only. `python3 scripts/render.py <out-dir>/{out}`\n\
         renders `{out}_turb.gif` and `{out}_xz.gif` (one frame per realization,\n\
         one global normalization so brightness differences between frames are\n\
         physical) and `{out}_longexp.png`, with physical axes and a colorbar in\n\
         I/I_max (magma colormap on (I/I_max)^0.5 to lift the dim wings). Intensity\n\
         is in units of the initial on-axis peak; no absolute radiometric\n\
         calibration.\n",
        n = grid.n,
        dx = args.dx,
        extent = grid.extent(),
        half_extent = grid.extent() / 2.0,
        wavelength = args.wavelength,
        w0 = args.w0,
        zr = beam.rayleigh_range(),
        dz_screen = z / screens as f64,
        r0_dx = r0 / args.dx,
        rytov = rytov_variance(cn2, args.wavelength, z),
        wlt = beam.long_exposure_width(z, cn2) * 1e3,
        wvac = beam.width_at(z) * 1e3,
        out = args.out,
    );
    let notes_path = args.out_path(&format!("{}_notes.md", args.out))?;
    fs::write(&notes_path, notes).with_context(|| format!("writing {}", notes_path.display()))?;

    println!(
        "  wrote {}, {} ({realizations} realizations), {}, {} and {}",
        frames_npy.display(),
        xz_npy.display(),
        npy.display(),
        meta_path.display(),
        notes_path.display()
    );
    println!(
        "  render images: python3 scripts/render.py {}/{}",
        args.out_dir.display(),
        args.out
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn blooming(
    args: &BeamArgs,
    power: f64,
    wind: f64,
    alpha_abs: f64,
    t0: f64,
    p0: f64,
    z_total: f64,
    steps: usize,
    frames: usize,
) -> Result<()> {
    let analytic = GaussianBeam {
        w0: args.w0,
        wavelength: args.wavelength,
    };
    let run = run_blooming(&BloomingParams {
        n: args.n,
        dx: args.dx,
        wavelength: args.wavelength,
        w0: args.w0,
        power,
        wind,
        alpha_abs,
        t0,
        p0,
        z: z_total,
        steps,
        frames,
    })?;
    let grid = run.grid;
    let (n_phi, peclet, delta_t_max, dz) = (run.n_phi, run.peclet, run.delta_t_sat, run.dz);
    println!(
        "thermal blooming: P = {power} W, v = {wind} m/s, alpha_abs = {alpha_abs:.2e} 1/m, \
         N_phi = {n_phi:.2}, Pe = {peclet:.0}, saturated dT = {delta_t_max:.3} K"
    );

    let xz_npy = args.out_path(&format!("{}_xz.npy", args.out))?;
    ndarray_npy::write_npy(&xz_npy, &run.xz)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", xz_npy.display()))?;

    let frames_npy = args.out_path(&format!("{}_frames.npy", args.out))?;
    ndarray_npy::write_npy(&frames_npy, &run.snapshots)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", frames_npy.display()))?;

    let final_npy = args.out_path(&format!("{}_final.npy", args.out))?;
    ndarray_npy::write_npy(&final_npy, &run.final_intensity)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", final_npy.display()))?;

    let frames_z = run
        .snapshot_z
        .iter()
        .map(|z| format!("{z}"))
        .collect::<Vec<_>>()
        .join(", ");
    let meta = format!(
        "{{\n  \"case\": \"blooming\",\n  \"n\": {n},\n  \"dx\": {dx},\n  \
         \"wavelength\": {wl},\n  \"w0\": {w0},\n  \"z\": {z_total},\n  \
         \"steps\": {steps},\n  \"power\": {power},\n  \"wind\": {wind},\n  \
         \"alpha_abs\": {alpha_abs},\n  \"t0\": {t0},\n  \"p0\": {p0},\n  \
         \"n_phi\": {n_phi},\n  \"frames_z\": [{frames_z}],\n  \
         \"xz_x_min\": {x_min},\n  \"xz_x_max\": {x_max}\n}}\n",
        n = grid.n,
        dx = args.dx,
        wl = args.wavelength,
        w0 = args.w0,
        x_min = -grid.extent() / 4.0,
        x_max = grid.extent() / 4.0,
    );
    let meta_path = args.out_path(&format!("{}_meta.json", args.out))?;
    fs::write(&meta_path, meta).with_context(|| format!("writing {}", meta_path.display()))?;

    let transmission = run.transmission;
    let t_ref = (-alpha_abs * z_total).exp();
    let guard_frac = run.guard_frac;
    let cx = run.centroid_x;
    println!(
        "  receiver: centroid x = {:.2} mm (negative = upwind), transmission {transmission:.4} \
         (Beer-Lambert {t_ref:.4}), guard-band absorption {guard_frac:.2e}",
        cx * 1e3
    );

    let notes = format!(
        "# Test case: Gaussian beam through steady-state thermal blooming\n\
         \n\
         Collimated Gaussian beam heating the air it crosses: convection-dominated\n\
         steady state (wind along +x), isobaric density/index response, coupled to the\n\
         beam through the field-aware medium interface. Models, gates, and references:\n\
         docs/MODELS.md and docs/M4_SPEC.md in the beamprop repository.\n\
         \n\
         ## Parameters\n\
         \n\
         | Quantity | Value |\n\
         |---|---|\n\
         | Grid | {n} x {n}, dx = {dx:.3e} m (extent {extent:.3} m) |\n\
         | Wavelength λ | {wavelength:.3e} m |\n\
         | Waist w0 (1/e² intensity radius) | {w0:.3e} m |\n\
         | Rayleigh range zR | {zr:.1} m |\n\
         | Path length z | {z_total:.1} m ({steps} slabs, dz = {dz:.2} m) |\n\
         | Beam power P | {power:.3e} W |\n\
         | Crosswind v (+x) | {wind} m/s |\n\
         | Absorption α_abs | {alpha_abs:.3e} 1/m |\n\
         | Ambient T₀, p₀ | {t0} K, {p0} Pa |\n\
         \n\
         ## Derived quantities\n\
         \n\
         | Quantity | Value |\n\
         |---|---|\n\
         | Distortion number N_φ | {n_phi:.2} (spec convention, docs/M4_SPEC.md) |\n\
         | Péclet number | {peclet:.0} (convection-dominated model needs ≫ 100) |\n\
         | Saturated downwind ΔT (launch beam) | {delta_t_max:.3} K |\n\
         | Receiver centroid x | {cx_mm:.2} mm (negative = bent upwind) |\n\
         | Transmission | {transmission:.4} (Beer–Lambert {t_ref:.4}) |\n\
         | Guard-band absorbed power | {guard_frac:.2e} of initial (grid-edge artifact unless ≈ 0) |\n\
         \n\
         ## Files\n\
         \n\
         - `{out}_xz.npy` — side view: central slice I(x, 0, z), float64. First axis is\n\
           x over the middle half of the grid ({half_extent:.3} m), second axis is z\n\
           from 0 to {z_total:.1} m. The beam visibly bends toward −x (upwind).\n\
         - `{out}_frames.npy` — transverse intensity snapshots along the path (z\n\
           positions in `{out}_meta.json`); each spans the full grid, {extent:.3} m per\n\
           side.\n\
         - `{out}_final.npy` — receiver-plane intensity at z = {z_total:.1} m: the\n\
           bloomed profile (upwind-shifted peak, downwind crescent).\n\
         - `{out}_meta.json` — the run parameters.\n\
         \n\
         ## Rendering\n\
         \n\
         The solver writes data only. `python3 scripts/render.py <out-dir>/{out}`\n\
         renders the side view `{out}_xz.png` (note: axes not to equal scale), the\n\
         snapshot animation `{out}_prop.gif`, and `{out}_final.png`, with physical\n\
         axes and a colorbar in I/I_max (magma colormap on (I/I_max)^0.5 to lift the\n\
         dim wings). Intensity is in units of the initial on-axis peak; no absolute\n\
         radiometric calibration.\n",
        n = grid.n,
        dx = args.dx,
        extent = grid.extent(),
        half_extent = grid.extent() / 2.0,
        wavelength = args.wavelength,
        w0 = args.w0,
        zr = analytic.rayleigh_range(),
        cx_mm = cx * 1e3,
        out = args.out,
    );
    let notes_path = args.out_path(&format!("{}_notes.md", args.out))?;
    fs::write(&notes_path, notes).with_context(|| format!("writing {}", notes_path.display()))?;

    println!(
        "  wrote {}, {}, {}, {} and {}",
        xz_npy.display(),
        frames_npy.display(),
        format_args!("{}/{}_final.npy", args.out_dir.display(), args.out),
        meta_path.display(),
        notes_path.display()
    );
    println!(
        "  render images: python3 scripts/render.py {}/{}",
        args.out_dir.display(),
        args.out
    );
    Ok(())
}

/// M6a: 0-D breakdown threshold sweep + avalanche traces. No grid, no
/// propagator — the compute is a pure rate integration in `beamprop::cases`.
#[allow(clippy::too_many_arguments)]
fn breakdown(
    wavelength: f64,
    fwhm: f64,
    p_min: f64,
    p_max: f64,
    points: usize,
    steps: usize,
    drive: f64,
    out: &str,
    out_dir: &Path,
) -> Result<()> {
    let run = run_breakdown(&BreakdownParams {
        wavelength,
        fwhm,
        p_min_torr: p_min,
        p_max_torr: p_max,
        points,
        steps,
        drive,
    })?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    let path = |name: &str| out_dir.join(name);

    println!(
        "breakdown sweep: {p_min:.0}–{p_max:.0} Torr, λ = {:.0} nm, {:.1} ns FWHM",
        wavelength * 1e9,
        fwhm * 1e9
    );
    println!(
        "  I_thr ∝ p^-{:.3}   (δ_eff literature envelope n ∈ [{:.3}, {:.3}])",
        run.slope, run.slope_envelope.0, run.slope_envelope.1
    );
    println!(
        "  I_thr({:.0} Torr) = {:.3e} W/cm²;  traces driven at {:.3e} W/cm²",
        run.pressure_torr[0], run.threshold[0], run.drive_intensity
    );

    // Threshold curve + envelope, as CSV: the render script overlays the
    // digitized T&T points on this.
    let mut csv = String::from(
        "# beamprop M6a 0-D optical-breakdown threshold sweep\n\
         # columns: pressure_torr, i_thr_w_per_cm2, i_thr_delta005, i_thr_delta001,\n\
         #          n_neutral_per_m3, i_cascade_theory (T&T 2012 Eq. 4)\n\
         # i_thr_delta005/001 are the edges of the delta_eff literature envelope\n\
         # (0.05 flattens the slope, 0.01 steepens it); see docs/M6A_SPEC.md.\n\
         pressure_torr,i_thr_w_per_cm2,i_thr_delta005,i_thr_delta001,n_neutral_per_m3,i_cascade_theory\n",
    );
    for i in 0..run.pressure_torr.len() {
        csv.push_str(&format!(
            "{:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            run.pressure_torr[i],
            run.threshold[i],
            run.threshold_lo_slope[i],
            run.threshold_hi_slope[i],
            run.neutral_density[i],
            run.cascade_theory[i]
        ));
    }
    let csv_path = path(&format!("{out}_threshold.csv"));
    fs::write(&csv_path, csv).with_context(|| format!("writing {}", csv_path.display()))?;

    // Avalanche traces [pressure, time] — the animation frames.
    let npy_path = path(&format!("{out}_ne_traces.npy"));
    ndarray_npy::write_npy(&npy_path, &run.ne_traces)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", npy_path.display()))?;

    let meta = format!(
        "{{\n  \"case\": \"breakdown\",\n  \"wavelength\": {wavelength},\n  \
         \"fwhm\": {fwhm},\n  \"p_min_torr\": {p_min},\n  \"p_max_torr\": {p_max},\n  \
         \"points\": {points},\n  \"steps\": {steps},\n  \"drive\": {drive},\n  \
         \"drive_intensity_w_per_cm2\": {di},\n  \"n_bd\": {nbd},\n  \"n_seed\": {nseed},\n  \
         \"slope\": {slope},\n  \"slope_envelope\": [{env_lo}, {env_hi}],\n  \
         \"t_min\": {t_min},\n  \"t_max\": {t_max}\n}}\n",
        di = run.drive_intensity,
        nbd = run.n_bd,
        nseed = run.n_seed,
        slope = run.slope,
        env_lo = run.slope_envelope.0,
        env_hi = run.slope_envelope.1,
        t_min = run.trace_time[0],
        t_max = run.trace_time[run.trace_time.len() - 1],
    );
    let meta_path = path(&format!("{out}_meta.json"));
    fs::write(&meta_path, meta).with_context(|| format!("writing {}", meta_path.display()))?;

    let notes = format!(
        "# Test case: 0-D optical-breakdown threshold in dry air (M6a)\n\
         \n\
         A single point in the gas sees a {fwhm_ns:.1} ns Gaussian pulse at\n\
         {lambda_nm:.0} nm. Its electron density avalanches by inverse-bremsstrahlung\n\
         cascade ionization — driven by the **net** power, heating minus the\n\
         inelastic excitation losses paid climbing to the ionization potential —\n\
         and is drained by attachment and diffusion. Breakdown is declared when\n\
         n_e reaches {nbd:.1e} m^-3 within the pulse.\n\
         \n\
         There is no beam and no propagator here: this is a pure rate balance at\n\
         one point, the kernel any ignition test calls per realization.\n\
         \n\
         ## What the sweep shows\n\
         \n\
         Threshold peak intensity against pressure over {p_min:.0}–{p_max:.0} Torr:\n\
         I_thr ∝ p^-{slope:.3}, against Thiyagarajan & Thompson's measured p^-0.329.\n\
         The shaded band in the render is the model's own uncertainty envelope from\n\
         the δ_eff literature range (n ∈ [{env_lo:.3}, {env_hi:.3}]) — **containment of\n\
         the measurement in that band is the validated claim**, not the\n\
         central-curve agreement, which leans on ungated order-of-magnitude\n\
         constants (see the sensitivity audit in docs/M6A_SPEC.md).\n\
         \n\
         The absolute level sits ~7x above T&T, as a *flat* offset — the shape is\n\
         reproduced, the normalization is not, and it is deliberately not gated\n\
         (published thresholds scatter 3–10x across labs).\n\
         \n\
         ## What the animation shows\n\
         \n\
         Every frame drives the *same* peak intensity, {di:.3e} W/cm^2, at a\n\
         different pressure. The avalanche is a knife edge: below threshold the\n\
         seed electron never multiplies, above it n_e climbs many orders of\n\
         magnitude within the pulse and saturates at full ionization. Raising\n\
         the pressure alone flips the gas from transparent to broken down --\n\
         though note the corrected model's threshold spans only ~1.17x over\n\
         300-2000 Torr, so the switch happens over a narrow pressure range.\n\
         \n\
         ## Files\n\
         \n\
         - `{out}_threshold.csv` — pressure (Torr), threshold (W/cm^2), and the\n\
           two envelope edges.\n\
         - `{out}_ne_traces.npy` — n_e(t) as `[pressure, time]`, {points} x {steps},\n\
           in m^-3, bounded above by the local full-ionization density.\n\
         - `{out}_meta.json` — the run parameters, fitted slope, and time axis.\n\
         \n\
         Measured reference data: `tests/data/tt2012_*.csv`.\n",
        fwhm_ns = fwhm * 1e9,
        lambda_nm = wavelength * 1e9,
        nbd = run.n_bd,
        slope = run.slope,
        env_lo = run.slope_envelope.0,
        env_hi = run.slope_envelope.1,
        di = run.drive_intensity,
    );
    let notes_path = path(&format!("{out}_notes.md"));
    fs::write(&notes_path, notes).with_context(|| format!("writing {}", notes_path.display()))?;

    println!(
        "  wrote {}, {} ({points} x {steps}), {} and {}",
        csv_path.display(),
        npy_path.display(),
        meta_path.display(),
        notes_path.display()
    );
    Ok(())
}

/// M6c: ignite at the M6a threshold, then run the laser-supported detonation
/// wave the sustaining beam drives back up itself. 1-D gas dynamics; the
/// compute is in `beamprop::cases`.
fn lsd(p: &LsdParams, out: &str, out_dir: &Path) -> Result<()> {
    let run = run_lsd(p)?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    let path = |name: &str| out_dir.join(name);

    println!(
        "lsd: {:.0} nm, igniting {:.2e} W into a {:.0} µm focus → I_peak = {:.3e} W/m²",
        p.wavelength * 1e9,
        p.ignite_power,
        p.w_focus * 1e6,
        run.i_peak
    );
    println!(
        "  M6a threshold {:.3e} W/m² → {} ({:.2}× {})",
        run.i_threshold,
        if run.ignited {
            "IGNITED"
        } else {
            "no breakdown"
        },
        run.i_peak / run.i_threshold,
        if run.ignited { "over" } else { "under" }
    );

    if !run.ignited {
        // The spec's failure-mode table: a clean report, not a hang and not an
        // error. Nothing to write — there is no wave.
        println!(
            "  the igniting pulse is below threshold, so no wave was launched. \
             Raise --ignite-power above {:.3e} W (or tighten --w-focus).",
            p.ignite_power * run.i_threshold / run.i_peak
        );
        return Ok(());
    }

    println!(
        "  sustaining drive {:.3e} W/m² = {:.2e}× the threshold — the beam that \
         runs the wave could never have lit it",
        run.drive,
        run.drive / run.i_threshold
    );
    println!(
        "  D = {:.1} m/s vs Raizer {:.1} m/s ({:+.3} %); ρ₀ = {:.4} kg/m³",
        run.d_measured,
        run.d_raizer,
        100.0 * (run.d_measured / run.d_raizer - 1.0),
        run.rho_0
    );
    println!(
        "  absorbed {:.4e} J/m² (budget closes to {:.2e}); final τ = {:.1}, \
         log₁₀ T = {:.1}",
        run.deposited_energy,
        run.energy_residual,
        run.optical_depth[run.optical_depth.len() - 1],
        run.log10_transmission[run.log10_transmission.len() - 1]
    );
    println!(
        "  inverse-bremsstrahlung at the post-front state: α = {:.3e} 1/m at \
         {:.2} µm (τ_column = {:.3}), {:.3e} 1/m at 10.6 µm (τ_column = {:.1})",
        run.ib_alpha,
        p.wavelength * 1e6,
        run.ib_optical_depth,
        run.ib_alpha_co2,
        run.ib_optical_depth_co2
    );
    if !run.boundaries_undisturbed {
        println!("  WARNING: the wave reached a domain boundary; results are contaminated.");
    }

    // Profiles [frame, quantity, cell] — the animation frames.
    let npy_path = path(&format!("{out}_profiles.npy"));
    ndarray_npy::write_npy(&npy_path, &run.profiles)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", npy_path.display()))?;

    let mut csv = String::from(
        "# beamprop M6c laser-supported detonation: front track and column opacity\n\
         # columns: time_s, front_x_m, optical_depth, log10_transmission\n\
         # front_x is the laser-side half-maximum of the pressure rise; it\n\
         # DECREASES because the wave runs back up the beam toward the laser.\n\
         # log10_transmission is logged because an established LSD column\n\
         # reaches hundreds of optical depths, where exp(-tau) underflows.\n\
         time_s,front_x_m,optical_depth,log10_transmission\n",
    );
    for i in 0..run.frame_time.len() {
        csv.push_str(&format!(
            "{:.9e},{:.9e},{:.6e},{:.6e}\n",
            run.frame_time[i], run.front_x[i], run.optical_depth[i], run.log10_transmission[i]
        ));
    }
    let csv_path = path(&format!("{out}_trajectory.csv"));
    fs::write(&csv_path, csv).with_context(|| format!("writing {}", csv_path.display()))?;

    let meta = format!(
        "{{\n  \"case\": \"lsd\",\n  \"wavelength\": {wl},\n  \"fwhm\": {fwhm},\n  \
         \"ignite_power\": {ip},\n  \"w_focus\": {wf},\n  \"drive\": {drive},\n  \
         \"p0\": {p0},\n  \"t0\": {t0},\n  \"rho_0\": {rho},\n  \"length\": {len},\n  \
         \"cells\": {cells},\n  \"alpha\": {alpha},\n  \"frames\": {frames},\n  \
         \"i_peak\": {ipk},\n  \"i_threshold\": {ithr},\n  \
         \"d_measured\": {dm},\n  \"d_raizer\": {dr},\n  \
         \"deposited_energy\": {dep},\n  \"energy_residual\": {res},\n  \
         \"ib_alpha\": {iba},\n  \"ib_optical_depth\": {ibt},\n  \
         \"ib_alpha_co2\": {ibac},\n  \"ib_optical_depth_co2\": {ibtc},\n  \
         \"x_min\": {xmin},\n  \"x_max\": {xmax},\n  \
         \"quantities\": [\"p_Pa\", \"rho_kg_m3\", \"u_m_s\", \"alpha_1_m\", \"I_W_m2\"]\n}}\n",
        wl = p.wavelength,
        fwhm = p.fwhm,
        ip = p.ignite_power,
        wf = p.w_focus,
        drive = p.drive,
        p0 = p.p0,
        t0 = p.t0,
        rho = run.rho_0,
        len = p.length,
        cells = p.cells,
        alpha = p.alpha,
        frames = p.frames,
        ipk = run.i_peak,
        ithr = run.i_threshold,
        dm = run.d_measured,
        dr = run.d_raizer,
        dep = run.deposited_energy,
        res = run.energy_residual,
        iba = run.ib_alpha,
        ibt = run.ib_optical_depth,
        ibac = run.ib_alpha_co2,
        ibtc = run.ib_optical_depth_co2,
        xmin = run.x[0],
        xmax = run.x[run.x.len() - 1],
    );
    let meta_path = path(&format!("{out}_meta.json"));
    fs::write(&meta_path, meta).with_context(|| format!("writing {}", meta_path.display()))?;

    let notes = format!(
        "# Test case: laser-supported detonation wave (M6c)\n\
         \n\
         A spark is lit at the M6a breakdown threshold, and the absorption wave\n\
         it launches runs **back up the beam toward the laser** as a detonation.\n\
         The x axis is the beam axis: the laser sits beyond x = 0, the beam\n\
         travels in +x, and the front travels in -x. Everything upstream of the\n\
         front is cold transparent air, which is why the front sees the full\n\
         drive intensity.\n\
         \n\
         ## The headline: the beam that sustains the wave could not have lit it\n\
         \n\
         Igniting pulse: {ipk:.3e} W/m^2 peak ({ratio_ig:.2}x the M6a threshold\n\
         of {ithr:.3e} W/m^2). Sustaining drive: {drive:.3e} W/m^2, which is\n\
         {ratio_dr:.2e} of that threshold -- five orders of magnitude below.\n\
         \n\
         That gap is a result, not a modelling convenience. M6a's threshold is an\n\
         intensity floor rather than a fluence one: below it the inelastic losses\n\
         paid climbing to the ionization potential exceed the inverse-\n\
         bremsstrahlung heating, so the net cascade rate is negative and no\n\
         exposure time rescues it (6 ns and 1 ms give thresholds within 4%).\n\
         Widening the focus does not help either. So the detonation has to be\n\
         *initiated* by something far brighter than what *sustains* it -- which\n\
         is the known experimental situation, where LSD waves in clean air are\n\
         started on a target, on an aerosol, or by a separate spike.\n\
         \n\
         M6a's absolute level is explicitly ungated (4.8-7.0x above the measured\n\
         Thiyagarajan & Thompson curve, inside the 3-10x inter-lab scatter), so\n\
         *when and where* the spark lights carries that uncertainty. The gap\n\
         above does not: it is 10^5 against a ~7x uncertainty.\n\
         \n\
         ## What the velocity is worth\n\
         \n\
         D = {dm:.1} m/s against Raizer's closed form {dr:.1} m/s ({derr:+.3}%).\n\
         This half of the run does **not** inherit M6a's uncertainty: the front\n\
         speed depends on the absorbed intensity at the front and on rho_0, not\n\
         on where or when the spark was lit -- which is why the M6c gates\n\
         (G3/G3b/G3c and the G4 physics gate) all use seeded ignition and never\n\
         touch AirBreakdown.\n\
         \n\
         Note also what the agreement above is and is not. Raizer's expression is\n\
         the Chapman-Jouguet construction this model is *built from*, so\n\
         reproducing it is **solver verification**, not validation. The gate that\n\
         speaks about the world is G4, the parameter-free D ~ S^(1/3),\n\
         rho_0^(-1/3) scaling. See docs/MODELS.md.\n\
         \n\
         ## The plasma as a shutter\n\
         \n\
         The column ends the run at tau = {tau:.1}, i.e. log10 of the transmitted\n\
         fraction is {logt:.1}. An established LSD plasma is not a partial\n\
         filter -- it closes the channel completely. That is the D7 coupling in\n\
         one number: the propagator sees this column as pure Beer-Lambert\n\
         absorption and nothing else (no Drude index), gated by\n\
         `plasma_column_absorbs_as_beer_lambert`.\n\
         \n\
         ## Why the grey absorber, and what the real one says\n\
         \n\
         The run uses the grey verification closure (fixed alpha = {alpha:.1e} 1/m\n\
         above an internal-energy threshold), which is what G3-G5 gate. The\n\
         production inverse-bremsstrahlung closure is implemented and unit-tested,\n\
         and evaluating it at this run's own post-front state is informative:\n\
         \n\
         - at {lam_um:.2} um:  alpha = {iba:.3e} 1/m, so the whole column is only\n\
           {ibt:.3} optical deep -- the plasma is nearly transparent to the beam\n\
           driving it, and check_regime would refuse the run as volumetric.\n\
         - at 10.6 um:  alpha = {ibac:.3e} 1/m, an absorption length of {ibl_mm:.2} mm\n\
           and a column {ibtc:.1} optical deep -- a real, well-resolved front.\n\
         \n\
         Free-free absorption falls steeply toward short wavelengths, so this is\n\
         the model reproducing why LSD experiments are done with CO2 lasers. What\n\
         blocks running that closure coupled is cost, not physics: the table\n\
         inversion bisects ~45 times per cell per deposition call. That is a\n\
         separate change with its own gate.\n\
         \n\
         ## Files\n\
         \n\
         - `{out}_profiles.npy` -- [frame, quantity, cell], {frames} x 5 x {cells},\n\
           quantities ordered p (Pa), rho (kg/m^3), u (m/s), alpha (1/m), I (W/m^2).\n\
         - `{out}_trajectory.csv` -- time, front position, optical depth,\n\
           log10 transmission.\n\
         - `{out}_meta.json` -- run parameters and the derived numbers above.\n\
         \n\
         Energy budget closes to {res:.2e} relative (G5). Render with\n\
         `python3 scripts/render_lsd.py {out_dir}/{out}`.\n",
        ipk = run.i_peak,
        ithr = run.i_threshold,
        ratio_ig = run.i_peak / run.i_threshold,
        drive = run.drive,
        ratio_dr = run.drive / run.i_threshold,
        dm = run.d_measured,
        dr = run.d_raizer,
        derr = 100.0 * (run.d_measured / run.d_raizer - 1.0),
        tau = run.optical_depth[run.optical_depth.len() - 1],
        logt = run.log10_transmission[run.log10_transmission.len() - 1],
        alpha = p.alpha,
        lam_um = p.wavelength * 1e6,
        iba = run.ib_alpha,
        ibt = run.ib_optical_depth,
        ibac = run.ib_alpha_co2,
        ibl_mm = 1e3 / run.ib_alpha_co2,
        ibtc = run.ib_optical_depth_co2,
        frames = p.frames,
        cells = p.cells,
        res = run.energy_residual,
        out_dir = out_dir.display(),
    );
    let notes_path = path(&format!("{out}_notes.md"));
    fs::write(&notes_path, notes).with_context(|| format!("writing {}", notes_path.display()))?;

    println!(
        "  wrote {} ({} x 5 x {}), {}, {} and {}",
        npy_path.display(),
        p.frames,
        p.cells,
        csv_path.display(),
        meta_path.display(),
        notes_path.display()
    );
    Ok(())
}

/// Delete `.npy`/`.png`/`.gif` files and `*_notes.md`/`*_meta.json` sidecars
/// directly inside `dir` (non-recursive).
fn clean(dir: &Path) -> Result<()> {
    if !dir.exists() {
        println!("nothing to clean: {} does not exist", dir.display());
        return Ok(());
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_result = path.is_file()
            && (path
                .extension()
                .is_some_and(|e| e == "npy" || e == "png" || e == "gif")
                || path.file_name().is_some_and(|f| {
                    let f = f.to_string_lossy();
                    f.ends_with("_notes.md") || f.ends_with("_meta.json")
                }));
        if is_result {
            fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            removed += 1;
        }
    }
    println!("removed {removed} result file(s) from {}", dir.display());
    Ok(())
}

fn propagate(
    args: &BeamArgs,
    z: Option<f64>,
    steps: usize,
    frames: usize,
    alpha: Option<f64>,
    visibility: Option<f64>,
) -> Result<()> {
    let run = run_propagate(&PropagateParams {
        n: args.n,
        dx: args.dx,
        wavelength: args.wavelength,
        w0: args.w0,
        z,
        steps,
        frames,
        alpha,
        visibility,
    })?;
    let grid = run.grid;
    let analytic = GaussianBeam {
        w0: args.w0,
        wavelength: args.wavelength,
    };
    let (z_total, dz, alpha) = (run.z_total, run.dz, run.alpha);
    if let Some(v) = visibility {
        println!("visibility {v} m -> alpha = {alpha:.4e} 1/m (Kruse)");
    }

    let xz_npy = args.out_path(&format!("{}_xz.npy", args.out))?;
    ndarray_npy::write_npy(&xz_npy, &run.xz)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", xz_npy.display()))?;

    let frames_npy = args.out_path(&format!("{}_frames.npy", args.out))?;
    ndarray_npy::write_npy(&frames_npy, &run.snapshots)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", frames_npy.display()))?;

    let final_npy = args.out_path(&format!("{}_final.npy", args.out))?;
    ndarray_npy::write_npy(&final_npy, &run.final_intensity)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", final_npy.display()))?;

    let frames_z = run
        .snapshot_z
        .iter()
        .map(|z| format!("{z}"))
        .collect::<Vec<_>>()
        .join(", ");
    let meta = format!(
        "{{\n  \"case\": \"propagate\",\n  \"n\": {n},\n  \"dx\": {dx},\n  \
         \"wavelength\": {wl},\n  \"w0\": {w0},\n  \"z\": {z_total},\n  \
         \"steps\": {steps},\n  \"alpha\": {alpha},\n  \
         \"frames_z\": [{frames_z}],\n  \
         \"xz_x_min\": {x_min},\n  \"xz_x_max\": {x_max}\n}}\n",
        n = grid.n,
        dx = args.dx,
        wl = args.wavelength,
        w0 = args.w0,
        x_min = -grid.extent() / 4.0,
        x_max = grid.extent() / 4.0,
    );
    let meta_path = args.out_path(&format!("{}_meta.json", args.out))?;
    fs::write(&meta_path, meta).with_context(|| format!("writing {}", meta_path.display()))?;

    let wx = run.width_x;
    let w_ref = analytic.width_at(z_total);
    println!(
        "propagated z = {z_total:.1} m in {steps} steps (dz = {dz:.2} m, zR = {:.1} m)",
        analytic.rayleigh_range()
    );
    println!(
        "  width: {:.4} mm numeric vs {:.4} mm analytic ({:+.3}%)",
        wx * 1e3,
        w_ref * 1e3,
        100.0 * (wx - w_ref) / w_ref
    );
    if alpha > 0.0 {
        let t_num = run.transmission;
        let t_ref = (-alpha * z_total).exp();
        println!(
            "  transmission: {t_num:.6e} numeric vs {t_ref:.6e} Beer-Lambert ({:+.2e} rel)",
            (t_num - t_ref) / t_ref
        );
    } else {
        let drift = (run.transmission - 1.0).abs();
        println!("  power drift: {drift:.2e}");
    }
    let guard_frac = run.guard_frac;
    println!("  guard-band absorption: {guard_frac:.2e} of initial power");
    let extinction_row = match (alpha > 0.0, visibility) {
        (true, Some(v)) => {
            format!("| Extinction α (Kruse, visibility {v} m) | {alpha:.4e} 1/m |\n")
        }
        (true, None) => format!("| Extinction α | {alpha:.4e} 1/m |\n"),
        (false, _) => "| Extinction | none (vacuum) |\n".to_string(),
    };
    let notes = format!(
        "# Test case: Gaussian beam free-space / Beer–Lambert propagation\n\
         \n\
         Collimated Gaussian beam, symmetric split-step propagation, optionally through\n\
         a uniform Beer–Lambert extinction (Kruse aerosol model when set via visibility).\n\
         Models and references: docs/MODELS.md in the beamprop repository.\n\
         \n\
         ## Parameters\n\
         \n\
         | Quantity | Value |\n\
         |---|---|\n\
         | Grid | {n} x {n}, dx = {dx:.3e} m (extent {extent:.3} m) |\n\
         | Wavelength λ | {wavelength:.3e} m |\n\
         | Waist w0 (1/e² intensity radius) | {w0:.3e} m |\n\
         | Rayleigh range zR | {zr:.1} m |\n\
         | Path length z | {z_total:.1} m ({steps} slabs, dz = {dz:.2} m) |\n\
         {extinction_row}\
         | Guard-band absorbed power | {guard_frac:.2e} of initial (grid-edge artifact unless ≈ 0) |\n\
         \n\
         ## Files\n\
         \n\
         - `{out}_xz.npy` — side view: central slice I(x, 0, z), float64. First axis is\n\
           x over the middle half of the grid ({half_extent:.3} m), second axis is z\n\
           from 0 to {z_total:.1} m (the full path length).\n\
         - `{out}_frames.npy` — transverse intensity snapshots along the path (z\n\
           positions in `{out}_meta.json`); each spans the full grid, {extent:.3} m per\n\
           side.\n\
         - `{out}_final.npy` — receiver-plane intensity at z = {z_total:.1} m, float64.\n\
         - `{out}_meta.json` — the run parameters.\n\
         \n\
         ## Rendering\n\
         \n\
         The solver writes data only. `python3 scripts/render.py <out-dir>/{out}`\n\
         renders the side view `{out}_xz.png` (note: axes not to equal scale), the\n\
         snapshot animation `{out}_prop.gif`, and `{out}_final.png`, with physical\n\
         axes and a colorbar in I/I_max (magma colormap on (I/I_max)^0.5 to lift the\n\
         dim wings). Intensity is in units of the initial on-axis peak; no absolute\n\
         radiometric calibration.\n",
        n = grid.n,
        dx = args.dx,
        extent = grid.extent(),
        half_extent = grid.extent() / 2.0,
        wavelength = args.wavelength,
        w0 = args.w0,
        zr = analytic.rayleigh_range(),
        out = args.out,
    );
    let notes_path = args.out_path(&format!("{}_notes.md", args.out))?;
    fs::write(&notes_path, notes).with_context(|| format!("writing {}", notes_path.display()))?;

    println!(
        "  wrote {}, {}, {}, {} and {}",
        xz_npy.display(),
        frames_npy.display(),
        format_args!("{}/{}_final.npy", args.out_dir.display(), args.out),
        meta_path.display(),
        notes_path.display()
    );
    println!(
        "  render images: python3 scripts/render.py {}/{}",
        args.out_dir.display(),
        args.out
    );
    Ok(())
}
