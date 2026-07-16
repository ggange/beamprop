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
    /// Remove generated results (.npy and .png files) from the output
    /// directory. Only those extensions are touched; other files and the
    /// directory itself are left alone.
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

/// Delete `.npy`/`.png` files directly inside `dir` (non-recursive).
fn clean(dir: &Path) -> Result<()> {
    if !dir.exists() {
        println!("nothing to clean: {} does not exist", dir.display());
        return Ok(());
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_result =
            path.is_file() && path.extension().is_some_and(|e| e == "npy" || e == "png");
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
    println!(
        "  wrote {}, frames, and {} to {}/",
        xz_png.display(),
        format_args!("{}_final.npy/png", args.out),
        args.out_dir.display()
    );
    Ok(())
}
