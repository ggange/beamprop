//! `beamprop` command-line interface.
//!
//! The inspection interface until the M5 PyO3 bindings arrive: build a field,
//! propagate it, dump `.npy` arrays and PNG renders.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::{UniformExtinction, kruse_extinction};
use beamprop::propagate::{Propagator, beam_width};
use beamprop::validate::GaussianBeam;
use beamprop::viz::{XzSliceMap, save_intensity_render};

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
    /// Write a Gaussian field's intensity to <out>.npy and <out>.png (M0).
    Gaussian {
        #[command(flatten)]
        beam: BeamArgs,
    },
    /// Propagate a Gaussian beam and render the result (M1/M2): side-view
    /// <out>_xz.png, snapshot frames, final <out>_final.{npy,png}. Lossless
    /// unless --alpha or --visibility sets a Beer-Lambert extinction.
    Propagate {
        #[command(flatten)]
        beam: BeamArgs,
        /// Total propagation distance in metres (default: 2 Rayleigh ranges).
        #[arg(long)]
        z: Option<f64>,
        /// Number of split-step slabs.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Number of snapshot frames to render along the path.
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
    /// (M3): Monte-Carlo realizations rendered into animated GIFs — receiver
    /// plane <out>_turb.gif and side view <out>_xz.gif — plus the
    /// long-exposure mean <out>_longexp.{npy,png} and a comparison against
    /// weak-turbulence theory.
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
    /// Remove generated results (.npy, .png, .gif files and *_notes.md
    /// sidecars) from the output directory. Only those are touched; other
    /// files and the directory itself are left alone.
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
    let png = args.out_path(&format!("{}.png", args.out))?;
    field.save_intensity_npy(&npy)?;
    field.save_intensity_png(&png)?;
    println!(
        "wrote {} and {}  (n={}, dx={} m, lambda={} m, w0={} m, power={:.6})",
        npy.display(),
        png.display(),
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

    let gif = args.out_path(&format!("{}_turb.gif", args.out))?;
    beamprop::viz::save_colormapped_gif(&frames, 0.5, 80, &gif)?;

    // Side view (x-z plane, beam travelling left to right), cropped to the
    // middle half in x like the `propagate` render.
    let xz_frames: Vec<_> = xz_maps
        .iter()
        .map(|m| {
            let nx = m.dim().0;
            m.slice(ndarray::s![nx / 4..3 * nx / 4, ..]).to_owned()
        })
        .collect();
    let xz_gif = args.out_path(&format!("{}_xz.gif", args.out))?;
    beamprop::viz::save_colormapped_gif(&xz_frames, 0.5, 80, &xz_gif)?;

    // Long-exposure (ensemble-mean) receiver intensity.
    let mut mean = ndarray::Array2::<f64>::zeros((grid.n, grid.n));
    for f in &frames {
        mean += f;
    }
    mean /= realizations as f64;
    let npy = args.out_path(&format!("{}_longexp.npy", args.out))?;
    let png = args.out_path(&format!("{}_longexp.png", args.out))?;
    ndarray_npy::write_npy(&npy, &mean)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", npy.display()))?;
    beamprop::viz::save_colormapped_png(&mean, 0.5, &png)?;

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
         - `{out}_turb.gif` — receiver-plane intensity |u(x,y,z={z})|². Each frame is one\n\
           **independent atmospheric realization**, not a time sequence. Frame spans the\n\
           full grid, {extent:.3} m per side.\n\
         - `{out}_xz.gif` — side view: central slice I(x, 0, z), beam travelling left\n\
           (z = 0) to right (z = {z} m), vertical axis the middle half of the grid\n\
           ({half_extent:.3} m). One frame per realization.\n\
         - `{out}_longexp.npy` / `.png` — ensemble-mean receiver intensity (the\n\
           long-exposure image), float64, same extent as `{out}_turb.gif`.\n\
         \n\
         ## Rendering\n\
         \n\
         Magma-like colormap (black → purple → orange → light yellow) on\n\
         t = (I/I_max)^0.5; I_max is the global peak across all frames of a GIF, so\n\
         brightness differences between frames are physical. Intensity is in units of\n\
         the initial on-axis peak; no absolute radiometric calibration.\n",
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
        "  wrote {} and {} ({realizations} frames), {}, {} and {}",
        gif.display(),
        xz_gif.display(),
        npy.display(),
        png.display(),
        notes_path.display()
    );
    Ok(())
}

/// Delete `.npy`/`.png`/`.gif` files and `*_notes.md` sidecars directly
/// inside `dir` (non-recursive).
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
                || path
                    .file_name()
                    .is_some_and(|f| f.to_string_lossy().ends_with("_notes.md")));
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
    save_intensity_render(
        &field,
        args.out_path(&format!("{}_frame_000.png", args.out))?,
    )?;

    prop.propagate(&mut field, &medium, dz, 0, steps, |i, f| {
        xz.record(f);
        let step = i + 1;
        if (step % frame_every == 0 || step == steps)
            && let Ok(path) = args.out_path(&format!("{}_frame_{:03}.png", args.out, step))
        {
            let _ = save_intensity_render(f, path);
        }
    })?;

    // Side view: central x-slice vs z, cropped to the middle half in x.
    let full = xz.to_array();
    let nx = full.dim().0;
    let cropped = full.slice(ndarray::s![nx / 4..3 * nx / 4, ..]).to_owned();
    let xz_png = args.out_path(&format!("{}_xz.png", args.out))?;
    beamprop::viz::save_colormapped_png(&cropped, 0.5, &xz_png)?;

    field.save_intensity_npy(args.out_path(&format!("{}_final.npy", args.out))?)?;
    save_intensity_render(&field, args.out_path(&format!("{}_final.png", args.out))?)?;

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
         - `{out}_xz.png` — side view: central slice I(x, 0, z), beam travelling left\n\
           (z = 0) to right (z = {z_total:.1} m), vertical axis the middle half of the\n\
           grid ({half_extent:.3} m).\n\
         - `{out}_frame_XXX.png` — transverse intensity snapshots at slab XXX; each\n\
           spans the full grid, {extent:.3} m per side.\n\
         - `{out}_final.npy` / `.png` — receiver-plane intensity at z = {z_total:.1} m,\n\
           float64.\n\
         \n\
         ## Rendering\n\
         \n\
         Magma-like colormap (black → purple → orange → light yellow) on\n\
         t = (I/I_max)^0.5, normalized per image. Intensity is in units of the initial\n\
         on-axis peak; no absolute radiometric calibration.\n",
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
        "  wrote {}, frames, {} and {} to {}/",
        xz_png.display(),
        format_args!("{}_final.npy/png", args.out),
        notes_path.display(),
        args.out_dir.display()
    );
    Ok(())
}
