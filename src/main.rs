//! `beamprop` command-line interface.
//!
//! The inspection interface until the M5 PyO3 bindings arrive: build a field,
//! propagate it, dump `.npy` arrays and PNG renders.

use anyhow::Result;
use clap::{Parser, Subcommand};

use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::Vacuum;
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
    /// Output basename.
    #[arg(long, default_value = "beam")]
    out: String,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Write a Gaussian field's intensity to <out>.npy and <out>.png (M0).
    Gaussian {
        #[command(flatten)]
        beam: BeamArgs,
    },
    /// Propagate a Gaussian beam through vacuum and render the result (M1):
    /// side-view <out>_xz.png, snapshot frames, final <out>_final.{npy,png}.
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
        } => propagate(&beam, z, steps, frames),
    }
}

fn gaussian(args: &BeamArgs) -> Result<()> {
    let grid = Grid::new(args.n, args.dx);
    let field = Field::gaussian(grid, args.wavelength, args.w0);
    let npy = format!("{}.npy", args.out);
    let png = format!("{}.png", args.out);
    field.save_intensity_npy(&npy)?;
    field.save_intensity_png(&png)?;
    println!(
        "wrote {npy} and {png}  (n={}, dx={} m, lambda={} m, w0={} m, power={:.6})",
        grid.n,
        grid.dx,
        args.wavelength,
        args.w0,
        field.power()
    );
    Ok(())
}

fn propagate(args: &BeamArgs, z: Option<f64>, steps: usize, frames: usize) -> Result<()> {
    let grid = Grid::new(args.n, args.dx);
    let analytic = GaussianBeam {
        w0: args.w0,
        wavelength: args.wavelength,
    };
    let z_total = z.unwrap_or(2.0 * analytic.rayleigh_range());
    let dz = z_total / steps as f64;

    let mut field = Field::gaussian(grid, args.wavelength, args.w0);
    let p0 = field.power();
    let mut prop = Propagator::new(grid, args.wavelength)?;
    let vacuum = Vacuum::new(grid.n);

    let frame_every = (steps / frames.max(1)).max(1);
    let mut xz = XzSliceMap::new();
    xz.record(&field);
    save_intensity_render(&field, format!("{}_frame_000.png", args.out))?;

    prop.propagate(&mut field, &vacuum, dz, 0, steps, |i, f| {
        xz.record(f);
        let step = i + 1;
        if step % frame_every == 0 || step == steps {
            let _ = save_intensity_render(f, format!("{}_frame_{:03}.png", args.out, step));
        }
    })?;

    // Side view: central x-slice vs z, cropped to the middle half in x.
    let full = xz.to_array();
    let nx = full.dim().0;
    let cropped = full.slice(ndarray::s![nx / 4..3 * nx / 4, ..]).to_owned();
    let xz_png = format!("{}_xz.png", args.out);
    beamprop::viz::save_colormapped_png(&cropped, 0.5, &xz_png)?;

    field.save_intensity_npy(format!("{}_final.npy", args.out))?;
    save_intensity_render(&field, format!("{}_final.png", args.out))?;

    let (wx, _) = beam_width(&field);
    let w_ref = analytic.width_at(z_total);
    let drift = (field.power() - p0).abs() / p0;
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
    println!("  power drift: {drift:.2e}");
    println!("  wrote {xz_png}, frames, and {}_final.npy/png", args.out);
    Ok(())
}
