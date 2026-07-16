//! `beamprop` command-line interface.
//!
//! The inspection interface until the M5 PyO3 bindings arrive: build a field,
//! propagate it, dump `.npy` arrays (+ `_meta.json`/`_notes.md` sidecars).
//! Images are rendered from those by `scripts/render.py`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::{UniformExtinction, kruse_extinction};
use beamprop::propagate::{Propagator, beam_width};
use beamprop::validate::GaussianBeam;
use beamprop::viz::XzSliceMap;

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
    /// Gaussian 1/e² waist radius w0 in metres.
    #[arg(long, default_value_t = 1.0e-2)]
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
    use beamprop::montecarlo::seeded_ensemble;
    use beamprop::turbulence::TurbulentPath;
    use beamprop::validate::{fried_r0, rytov_variance};

    let grid = Grid::new(args.n, args.dx);
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

    // Diffraction-only substeps between screens give the side view a smooth
    // z-axis (~240 columns) without changing the screen physics.
    let substeps = (240 / screens).max(1);
    let results = seeded_ensemble(realizations, |i| {
        let path = TurbulentPath::new(grid, args.wavelength, cn2, l0, z, screens, seed, i)
            .with_substeps(substeps);
        let mut field = Field::gaussian(grid, args.wavelength, args.w0);
        let mut prop = Propagator::new(grid, args.wavelength).expect("valid propagator");
        let mut xz = XzSliceMap::new();
        xz.record(&field);
        prop.propagate(&mut field, &path, path.dz(), 0, path.n_slabs(), |_, f| {
            xz.record(f);
        })
        .expect("propagation");
        (field.intensity(), xz.to_array())
    });
    let (frames, xz_maps): (Vec<_>, Vec<_>) = results.into_iter().unzip();

    // Side view (x-z plane, beam travelling left to right), cropped to the
    // middle half in x.
    let xz_frames: Vec<_> = xz_maps
        .iter()
        .map(|m| {
            let nx = m.dim().0;
            m.slice(ndarray::s![nx / 4..3 * nx / 4, ..]).to_owned()
        })
        .collect();

    // Frame stacks + run metadata: the inputs to scripts/render.py, which
    // does all image rendering.
    let stack = |maps: &[ndarray::Array2<f64>]| {
        let (ny, nx) = maps[0].dim();
        let mut s = ndarray::Array3::<f64>::zeros((maps.len(), ny, nx));
        for (i, m) in maps.iter().enumerate() {
            s.index_axis_mut(ndarray::Axis(0), i).assign(m);
        }
        s
    };
    let frames_npy = args.out_path(&format!("{}_frames.npy", args.out))?;
    ndarray_npy::write_npy(&frames_npy, &stack(&frames))
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", frames_npy.display()))?;
    let xz_npy = args.out_path(&format!("{}_xz_frames.npy", args.out))?;
    ndarray_npy::write_npy(&xz_npy, &stack(&xz_frames))
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
    let mut mean = ndarray::Array2::<f64>::zeros((grid.n, grid.n));
    for f in &frames {
        mean += f;
    }
    mean /= realizations as f64;
    let npy = args.out_path(&format!("{}_longexp.npy", args.out))?;
    ndarray_npy::write_npy(&npy, &mean)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", npy.display()))?;

    println!(
        "  long-exposure width (theory): {:.2} mm vs vacuum {:.2} mm",
        beam.long_exposure_width(z, cn2) * 1e3,
        beam.width_at(z) * 1e3
    );

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
    let grid = Grid::new(args.n, args.dx);
    let analytic = GaussianBeam {
        w0: args.w0,
        wavelength: args.wavelength,
    };
    let z_total = z.unwrap_or(2.0 * analytic.rayleigh_range());
    let dz = z_total / steps as f64;

    let alpha = match (alpha, visibility) {
        (Some(a), _) => a,
        (None, Some(v)) => {
            let a = kruse_extinction(args.wavelength, v);
            println!("visibility {v} m -> alpha = {a:.4e} 1/m (Kruse)");
            a
        }
        (None, None) => 0.0,
    };
    let medium = UniformExtinction::new(grid.n, alpha);

    let mut field = Field::gaussian(grid, args.wavelength, args.w0);
    let p0 = field.power();
    let mut prop = Propagator::new(grid, args.wavelength)?;

    let frame_every = (steps / frames.max(1)).max(1);
    let mut xz = XzSliceMap::new();
    xz.record(&field);
    let mut snapshots = vec![field.intensity()];
    let mut snapshot_z = vec![0.0_f64];

    prop.propagate(&mut field, &medium, dz, 0, steps, |i, f| {
        xz.record(f);
        let step = i + 1;
        if step % frame_every == 0 || step == steps {
            snapshots.push(f.intensity());
            snapshot_z.push(step as f64 * dz);
        }
    })?;

    // Side view: central x-slice vs z, cropped to the middle half in x.
    let full = xz.to_array();
    let nx = full.dim().0;
    let cropped = full.slice(ndarray::s![nx / 4..3 * nx / 4, ..]).to_owned();
    let xz_npy = args.out_path(&format!("{}_xz.npy", args.out))?;
    ndarray_npy::write_npy(&xz_npy, &cropped)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", xz_npy.display()))?;

    let mut snap_stack = ndarray::Array3::<f64>::zeros((snapshots.len(), grid.n, grid.n));
    for (i, s) in snapshots.iter().enumerate() {
        snap_stack.index_axis_mut(ndarray::Axis(0), i).assign(s);
    }
    let frames_npy = args.out_path(&format!("{}_frames.npy", args.out))?;
    ndarray_npy::write_npy(&frames_npy, &snap_stack)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", frames_npy.display()))?;

    field.save_intensity_npy(args.out_path(&format!("{}_final.npy", args.out))?)?;

    let frames_z = snapshot_z
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

    let (wx, _) = beam_width(&field);
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
        let t_num = field.power() / p0;
        let t_ref = (-alpha * z_total).exp();
        println!(
            "  transmission: {t_num:.6e} numeric vs {t_ref:.6e} Beer-Lambert ({:+.2e} rel)",
            (t_num - t_ref) / t_ref
        );
    } else {
        let drift = (field.power() - p0).abs() / p0;
        println!("  power drift: {drift:.2e}");
    }
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
