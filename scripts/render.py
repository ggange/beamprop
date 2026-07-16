#!/usr/bin/env python3
"""Render beamprop results: the solver writes .npy data, this makes the images.

Reads the .npy arrays and the _meta.json sidecar a `beamprop propagate` or
`beamprop turbulence` run writes, and renders matplotlib figures with
physical axes and a labeled colorbar.

Usage:
    python3 scripts/render.py out/demo        # basename = <out-dir>/<out>
    python3 scripts/render.py out/demo --fps 12

Requires: numpy, matplotlib (pip install numpy matplotlib).

Outputs (turbulence): <base>_turb.gif, <base>_xz.gif, <base>_longexp.png
Outputs (propagate):  <base>_xz.png, <base>_prop.gif, <base>_final.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

GAMMA = 0.5  # display gamma: images show (I/I_max)^GAMMA
CBAR_LABEL = "$I\\,/\\,I_{\\mathrm{max}}$"


def add_colorbar(fig, ax, im):
    """Colorbar with ticks at physical relative intensities."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks_i = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    cbar.set_ticks(ticks_i**GAMMA)
    cbar.set_ticklabels([f"{t:g}" for t in ticks_i])
    cbar.set_label(CBAR_LABEL)
    return cbar


def make_axes(extent, xlabel, ylabel, aspect):
    fig, ax = plt.subplots(figsize=(8, 5) if aspect == "auto" else (7, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_still(data, extent, xlabel, ylabel, title, path, aspect="equal"):
    """Render a single 2D intensity map, normalized to its own peak."""
    fig, ax = make_axes(extent, xlabel, ylabel, aspect)
    im = ax.imshow(
        (data / data.max()) ** GAMMA,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect=aspect,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11)
    add_colorbar(fig, ax, im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def save_animation(stack, extent, xlabel, ylabel, frame_title, path, fps, aspect="equal"):
    """Encode a (frames, ny, nx) stack as a GIF with one global normalization.

    `frame_title(i)` supplies the per-frame title.
    """
    norm = (stack / stack.max()) ** GAMMA
    fig, ax = make_axes(extent, xlabel, ylabel, aspect)
    im = ax.imshow(
        norm[0],
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect=aspect,
        interpolation="nearest",
    )
    add_colorbar(fig, ax, im)

    def update(i):
        im.set_data(norm[i])
        ax.set_title(frame_title(i), fontsize=11)
        return (im,)

    update(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(norm), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path}")


def render_turbulence(base, meta, fps):
    half = meta["n"] * meta["dx"] / 2.0
    params = (
        f"$C_n^2$ = {meta['cn2']:.2g} m$^{{-2/3}}$, z = {meta['z']:g} m, "
        f"$\\lambda$ = {meta['wavelength']:.2g} m, $w_0$ = {meta['w0']:.2g} m"
    )
    n_real = meta["realizations"]

    frames = np.load(base.parent / f"{base.name}_frames.npy")
    save_animation(
        frames,
        extent=(-half, half, -half, half),
        xlabel="x [m]",
        ylabel="y [m]",
        frame_title=lambda i: f"Receiver plane — realization {i + 1}/{n_real}\n{params}",
        path=base.parent / f"{base.name}_turb.gif",
        fps=fps,
    )

    xz = np.load(base.parent / f"{base.name}_xz_frames.npy")
    save_animation(
        xz,
        extent=(0.0, meta["z"], meta["xz_x_min"], meta["xz_x_max"]),
        xlabel="z [m]",
        ylabel="x [m]",
        frame_title=lambda i: f"Side view — realization {i + 1}/{n_real}\n{params}",
        path=base.parent / f"{base.name}_xz.gif",
        fps=fps,
        aspect="auto",
    )

    longexp = np.load(base.parent / f"{base.name}_longexp.npy")
    save_still(
        longexp,
        extent=(-half, half, -half, half),
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"Long-exposure mean ({n_real} realizations)\n{params}",
        path=base.parent / f"{base.name}_longexp.png",
    )


def render_propagate(base, meta, fps):
    half = meta["n"] * meta["dx"] / 2.0
    alpha = meta["alpha"]
    extinction = f", $\\alpha$ = {alpha:.2g} m$^{{-1}}$" if alpha > 0.0 else ""
    params = (
        f"z = {meta['z']:g} m, $\\lambda$ = {meta['wavelength']:.2g} m, "
        f"$w_0$ = {meta['w0']:.2g} m{extinction}"
    )

    xz = np.load(base.parent / f"{base.name}_xz.npy")
    save_still(
        xz,
        extent=(0.0, meta["z"], meta["xz_x_min"], meta["xz_x_max"]),
        xlabel="z [m]",
        ylabel="x [m]",
        title=f"Side view (central slice)\n{params}",
        path=base.parent / f"{base.name}_xz.png",
        aspect="auto",
    )

    frames = np.load(base.parent / f"{base.name}_frames.npy")
    frames_z = meta["frames_z"]
    save_animation(
        frames,
        extent=(-half, half, -half, half),
        xlabel="x [m]",
        ylabel="y [m]",
        frame_title=lambda i: f"Transverse plane at z = {frames_z[i]:.1f} m\n{params}",
        path=base.parent / f"{base.name}_prop.gif",
        fps=fps,
    )

    final = np.load(base.parent / f"{base.name}_final.npy")
    save_still(
        final,
        extent=(-half, half, -half, half),
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"Receiver plane\n{params}",
        path=base.parent / f"{base.name}_final.png",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="run basename, e.g. out/demo")
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    args = ap.parse_args()
    base = Path(args.base)

    meta = json.loads((base.parent / f"{base.name}_meta.json").read_text())
    case = meta.get("case")
    if case == "turbulence":
        render_turbulence(base, meta, args.fps)
    elif case == "propagate":
        render_propagate(base, meta, args.fps)
    else:
        raise SystemExit(f"unknown case {case!r} in {base}_meta.json")


if __name__ == "__main__":
    main()
