#!/usr/bin/env python3
"""Publication-quality figures for beamprop turbulence runs.

Reads the .npy frame stacks and _meta.json a `beamprop turbulence` run
writes, and renders matplotlib GIFs/PNGs with physical axes and a labeled
colorbar — the counterpart to the solver's self-contained quick-look renders.

Usage:
    python scripts/render.py out/demo          # basename = <out-dir>/<out>
    python scripts/render.py out/demo --fps 12

Requires: numpy, matplotlib (pip install numpy matplotlib).
Outputs: <base>_turb_mpl.gif, <base>_xz_mpl.gif, <base>_longexp_mpl.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

GAMMA = 0.5  # display gamma, matches the Rust renders (t = (I/I_max)^0.5)


def animate(stack, extent, xlabel, ylabel, title, subtitle, cbar_label, path, fps, aspect):
    """Encode a (frames, ny, nx) stack as a GIF with one global normalization."""
    vmax = stack.max()
    norm = (stack / vmax) ** GAMMA

    fig, ax = plt.subplots(figsize=(8, 8 * 0.62 if aspect == "auto" else 8))
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # Colorbar labeled in physical relative intensity, not display units.
    ticks_i = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    cbar.set_ticks(ticks_i**GAMMA)
    cbar.set_ticklabels([f"{t:g}" for t in ticks_i])
    cbar.set_label(cbar_label)

    def update(i):
        im.set_data(norm[i])
        ax.set_title(
            f"{title} — realization {i + 1}/{len(norm)}\n{subtitle}", fontsize=11
        )
        return (im,)

    update(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(norm), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="run basename, e.g. out/demo")
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    args = ap.parse_args()
    base = Path(args.base)

    meta = json.loads((base.parent / f"{base.name}_meta.json").read_text())
    half = meta["n"] * meta["dx"] / 2.0
    subtitle = (
        f"$C_n^2$ = {meta['cn2']:.2g} m$^{{-2/3}}$, z = {meta['z']:g} m, "
        f"$\\lambda$ = {meta['wavelength']:.2g} m, $w_0$ = {meta['w0']:.2g} m"
    )
    cbar_label = "$I / I_{max}$ (display: $(I/I_{max})^{1/2}$)"

    frames = np.load(base.parent / f"{base.name}_frames.npy")
    animate(
        frames,
        extent=(-half, half, -half, half),
        xlabel="x [m]",
        ylabel="y [m]",
        title="Receiver plane",
        subtitle=subtitle,
        cbar_label=cbar_label,
        path=base.parent / f"{base.name}_turb_mpl.gif",
        fps=args.fps,
        aspect="equal",
    )

    xz = np.load(base.parent / f"{base.name}_xz_frames.npy")
    animate(
        xz,
        extent=(0.0, meta["z"], meta["xz_x_min"], meta["xz_x_max"]),
        xlabel="z [m]",
        ylabel="x [m]",
        title="Side view (central slice)",
        subtitle=subtitle,
        cbar_label=cbar_label,
        path=base.parent / f"{base.name}_xz_mpl.gif",
        fps=args.fps,
        aspect="auto",
    )

    longexp = np.load(base.parent / f"{base.name}_longexp.npy")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        (longexp / longexp.max()) ** GAMMA,
        origin="lower",
        extent=(-half, half, -half, half),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Long-exposure mean ({meta['realizations']} realizations)\n{subtitle}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks_i = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    cbar.set_ticks(ticks_i**GAMMA)
    cbar.set_ticklabels([f"{t:g}" for t in ticks_i])
    cbar.set_label(cbar_label)
    fig.tight_layout()
    png = base.parent / f"{base.name}_longexp_mpl.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
