#!/usr/bin/env python3
"""Render M6d axisymmetric laser-supported-detonation results: the solver writes
data, this makes the images.

Reads the NPY/CSV/meta a `beamprop lsd2d` run writes and renders:

  <base>_wave.gif    — the wave in (x, ±r), mirrored about the axis, one frame
                       per recorded snapshot, with the beam edge marked
  <base>_relief.png  — the front track on the axis and at the beam edge against
                       the wide-beam limit, and the front's shape
  <base>_budget.png  — deposited vs gained vs escaped energy

Usage:
    python3 scripts/render_lsd2d.py out/lsd2d
    python3 scripts/render_lsd2d.py out/lsd2d --fps 10

Requires: numpy, matplotlib (pip install numpy matplotlib).

Three things the figures are drawn to make unmissable, because they are the
three easiest to misread about this case.

1. **The front runs in −x.** The laser is off the left edge and the beam travels
   left-to-right, so the detonation moving *toward* the laser means the front
   position *decreases*. Inherited from `render_lsd.py`, and still true.

2. **The field is mirrored about r = 0 on purpose.** An axisymmetric solution
   only exists for r ≥ 0; showing ±r is the standard presentation, and it makes
   an axis artifact instantly visible as a discontinuity down the centre line.
   If that seam ever appears, gate G13 has stopped doing its job.

3. **The dashed wide-beam line is not a measurement the solver failed to
   reproduce.** It is the same solver with the beam made infinitely wide — the
   closed-form limit with the effect removed — and the gap between it and the
   solid line *is* the result. Note it is deliberately the wide-beam 2-D run and
   not the 1-D column: the modelled front is transversely unstable (gate G14),
   so a 1-D reference would show that instability as if it were relief.
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LogNorm

# Matches scripts/render.py, render_breakdown.py and render_lsd.py.
P_C = "#f1605d"      # magma mid — pressure, the wave itself
ALPHA_C = "#721f81"  # magma dark — the plasma's absorption
BEAM_C = "#feca8d"   # magma light — the beam
EDGE_C = "#3b0f70"   # magma darker — the beam edge
REF_C = "#888888"
GRID_C = "#cccccc"

# Index of each quantity along axis 1 of <base>_fields.npy. Mirrors the
# "quantities" list the CLI writes into _meta.json; asserted against it below,
# so a reordering on the Rust side fails loudly here instead of silently
# plotting the wrong field.
QUANTITIES = ["p_Pa", "rho_kg_m3", "u_x_m_s", "u_r_m_s", "alpha_1_m", "I_W_m2"]
I_P, I_RHO, I_UX, I_UR, I_ALPHA, I_I = range(6)


def style(ax):
    ax.grid(True, which="both", color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mark_laser_side(ax):
    """Annotate which way the beam travels and where the laser is."""
    ax.annotate(
        "laser",
        xy=(0.0, 1.02), xycoords="axes fraction", fontsize=9,
        color=BEAM_C, fontweight="bold", ha="left", va="bottom",
    )
    ax.annotate(
        "", xy=(0.16, 1.035), xytext=(0.055, 1.035), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=BEAM_C, lw=1.6),
    )
    ax.annotate(
        "beam", xy=(0.175, 1.02), xycoords="axes fraction", fontsize=9,
        color=BEAM_C, ha="left", va="bottom",
    )


def read_trajectory(path):
    """Read <base>_trajectory.csv into (t, x_axis, x_edge, lag)."""
    cols = [[] for _ in range(4)]
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(row[i]) for i in range(4)]
            except (ValueError, IndexError):
                continue  # header row
            for i, v in enumerate(vals):
                cols[i].append(v)
    return [np.array(c) for c in cols]


def load_fields(base, meta):
    fields = np.load(f"{base}_fields.npy")
    if fields.ndim != 4 or fields.shape[1] != len(QUANTITIES):
        raise SystemExit(
            f"expected [frame, {len(QUANTITIES)}, ring, cell], got {fields.shape}"
        )
    if meta.get("quantities") and list(meta["quantities"]) != QUANTITIES:
        raise SystemExit(
            f"field layout changed: meta says {meta['quantities']}, "
            f"this script plots {QUANTITIES}"
        )
    return fields


def mirror(field_2d):
    """Stack a (ring, cell) field into (−r … +r, cell) for display."""
    return np.vstack([field_2d[::-1, :], field_2d])


def render_wave(base, meta, fps):
    """The animation: pressure in (x, ±r), with the beam edge marked."""
    fields = load_fields(base, meta)
    n_frames, _, n_r, n_x = fields.shape
    t, x_axis, x_edge, _ = read_trajectory(f"{base}_trajectory.csv")

    x_mm = np.linspace(meta["x_min"], meta["x_max"], n_x) * 1e3
    r_mm = np.linspace(meta["r_min"], meta["r_max"], n_r) * 1e3
    extent = [x_mm[0], x_mm[-1], -r_mm[-1], r_mm[-1]]
    p0 = meta["p0"]
    rb_mm = meta["beam_radius"] * 1e3

    p_max = float(fields[:, I_P, :, :].max())
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    im = ax.imshow(
        mirror(fields[0, I_P] / p0),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max(p_max / p0, 2.0)),
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("pressure  p/p₀")

    # The beam edge: everything outside it is undriven, and the gas that crosses
    # it is the relief this case exists to measure.
    for sign in (-1.0, 1.0):
        ax.axhline(sign * rb_mm, color=BEAM_C, lw=1.2, ls="--", alpha=0.9)
    ax.annotate(
        "beam edge", xy=(x_mm[-1], rb_mm), xytext=(-6, 4),
        textcoords="offset points", ha="right", va="bottom",
        color=BEAM_C, fontsize=8,
    )
    front_line = ax.axvline(x_axis[0] * 1e3, color="#ffffff", lw=1.0, alpha=0.7)
    ax.set_xlabel("position along the beam axis  (mm)"
                  "      — the front runs toward the laser, so it moves left")
    ax.set_ylabel("radius  ±r  (mm)")
    mark_laser_side(ax)
    title = ax.set_title("")

    def draw(k):
        im.set_data(mirror(fields[k, I_P] / p0))
        front_line.set_xdata([x_axis[k] * 1e3, x_axis[k] * 1e3])
        title.set_text(
            f"t = {t[k] * 1e9:7.1f} ns    front (axis) = {x_axis[k] * 1e3:6.3f} mm"
            f"    lag at beam edge = {(x_edge[k] - x_axis[k]) * 1e6:5.1f} µm"
        )
        return im, front_line, title

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    out = f"{base}_wave.gif"
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}")


def render_relief(base, meta):
    """The result: the front track against the wide-beam limit, and its shape."""
    fields = load_fields(base, meta)
    t, x_axis, x_edge, lag = read_trajectory(f"{base}_trajectory.csv")
    n_r, n_x = fields.shape[2], fields.shape[3]
    r_mm = np.linspace(meta["r_min"], meta["r_max"], n_r) * 1e3
    rb_mm = meta["beam_radius"] * 1e3

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax_t.plot(t * 1e9, x_axis * 1e3, color=P_C, lw=2.0, label="front, on axis")
    ax_t.plot(t * 1e9, x_edge * 1e3, color=EDGE_C, lw=1.6, ls="-",
              label="front, at the beam edge")
    # The wide-beam limit, anchored at the first tracked point.
    good = np.isfinite(x_axis)
    if good.any():
        t0, x0 = t[good][0], x_axis[good][0]
        ax_t.plot(
            t * 1e9, (x0 - meta["d_wide"] * (t - t0)) * 1e3,
            color=REF_C, lw=1.4, ls="--",
            label=f"wide-beam limit, {meta['d_wide']:.0f} m/s",
        )
    ax_t.set_xlabel("time  (ns)")
    ax_t.set_ylabel("front position  (mm)")
    ax_t.set_title(
        f"radial relief costs {100 * meta['relief_deficit']:.1f} % of the front speed\n"
        f"{meta['d_measured']:.0f} m/s against the wide-beam {meta['d_wide']:.0f} m/s",
        fontsize=10,
    )
    ax_t.legend(fontsize=8, frameon=False)
    style(ax_t)

    # The front's shape at the last frame: the curvature IS the relief.
    p = fields[-1, I_P]
    p0 = meta["p0"]
    level = p0 + 0.5 * (p.max() - p0)
    x_mm = np.linspace(meta["x_min"], meta["x_max"], n_x) * 1e3
    # Only inside the beam. Outside it the gas is undriven, so the half-maximum
    # crossing there is the seed's own decaying blast rather than the
    # detonation, and plotting it would draw a second, unrelated branch.
    shape, shape_r = [], []
    for j in range(n_r):
        if r_mm[j] > rb_mm:
            break
        hit = np.nonzero(p[j] >= level)[0]
        shape.append(x_mm[hit[0]] if hit.size else np.nan)
        shape_r.append(r_mm[j])
    ax_s.plot(np.array(shape), np.array(shape_r), color=P_C, lw=2.0)
    ax_s.axhline(rb_mm, color=BEAM_C, lw=1.2, ls="--", label="beam edge")
    ax_s.set_ylim(0.0, 1.15 * rb_mm)
    ax_s.set_xlabel("front position  (mm)")
    ax_s.set_ylabel("radius  (mm)")
    ax_s.set_title(
        "the front is curved: the axis leads, the beam edge lags\n"
        "— that curvature is the relief, made visible",
        fontsize=10,
    )
    ax_s.legend(fontsize=8, frameon=False)
    style(ax_s)

    fig.tight_layout()
    out = f"{base}_relief.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def render_budget(base, meta):
    """Deposited vs escaped: where the beam's energy went."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["deposited\nby the beam", "escaped\nthrough the walls"]
    values = [meta["deposited_energy"], meta["escaped_energy"]]
    ax.bar(labels, values, color=[P_C, ALPHA_C])
    ax.set_ylabel("energy  (J/rad, in the r dr dx measure)")
    ax.set_title(
        f"energy budget closes to {meta['energy_residual']:.2e}\n"
        "— the escape term is what lets it close on a run where relief\n"
        "actually reaches the wall, not only on the runs where nothing happens",
        fontsize=9,
    )
    style(ax)
    fig.tight_layout()
    out = f"{base}_budget.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", help="output basename, e.g. out/lsd2d")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    with open(f"{args.base}_meta.json") as fh:
        meta = json.load(fh)
    case = meta.get("case")
    if case != "lsd2d":
        raise SystemExit(
            f"{args.base} is a '{case}' run; use scripts/render_lsd.py for 'lsd', "
            f"render_breakdown.py for 'breakdown', render_ignition.py for 'ignition', "
            f"or render.py for the field cases"
        )

    render_wave(args.base, meta, args.fps)
    render_relief(args.base, meta)
    render_budget(args.base, meta)


if __name__ == "__main__":
    main()
