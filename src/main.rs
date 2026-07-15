//! `beamprop` command-line interface.
//!
//! M0 scope: build a sampled field and dump its intensity to `.npy` and PNG.
//! This is the inspection interface until the M5 PyO3 bindings arrive; the
//! propagation subcommands land as the M1+ physics is written.

use anyhow::Result;
use clap::Parser;

use beamprop::{field::Field, grid::Grid};

/// Dump a sampled optical field to `.npy` and PNG (M0 I/O path).
#[derive(Parser, Debug)]
#[command(name = "beamprop", version, about)]
struct Cli {
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
    #[arg(long, default_value_t = 5.0e-2)]
    w0: f64,
    /// Output basename; writes <out>.npy and <out>.png.
    #[arg(long, default_value = "beam")]
    out: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let grid = Grid::new(cli.n, cli.dx);
    let field = Field::gaussian(grid, cli.wavelength, cli.w0);

    let npy = format!("{}.npy", cli.out);
    let png = format!("{}.png", cli.out);
    field.save_intensity_npy(&npy)?;
    field.save_intensity_png(&png)?;

    println!(
        "wrote {npy} and {png}  (n={}, dx={} m, lambda={} m, w0={} m, power={:.6})",
        grid.n,
        grid.dx,
        cli.wavelength,
        cli.w0,
        field.power()
    );
    Ok(())
}
