#!/usr/bin/env python3
"""Render M6c laser-supported-detonation results: the solver writes data, this
makes the images.

Reads the NPY/CSV/meta a `beamprop lsd` run writes and renders:

  <base>_wave.gif       — the wave running back up the beam, one frame per
                          recorded snapshot: pressure, the plasma's absorption
                          coefficient, and the beam being eaten
  <base>_trajectory.png — the front track against Raizer's closed form, and the
                          column's opacity climbing

Usage:
    python3 scripts/render_lsd.py out/lsd
    python3 scripts/render_lsd.py out/lsd --fps 12

Requires: numpy, matplotlib (pip install numpy matplotlib).

Two things the figures are drawn to make unmissable, because they are the two
things easiest to misread about this case:

1. **The front runs in −x.** The laser is off the left edge and the beam travels
   left-to-right, so the detonation moving *toward* the laser means the front
   position *decreases*. Both figures mark the laser side explicitly.
2. **Agreement with Raizer is verification, not validation.** That closed form
   is the Chapman–Jouguet construction the model is built from, so the dashed
   reference line in the trajectory panel is labelled as such rather than being
   presented as a measurement the solver reproduced. The gate that speaks about
   the world is G4, the parameter-free one-third scaling. See docs/MODELS.md.
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Matches scripts/render.py and render_breakdown.py: magma family throughout.
P_C = "#f1605d"      # magma mid — pressure, the wave itself
ALPHA_C = "#721f81"  # magma dark — the plasma's absorption
BEAM_C = "#feca8d"   # magma light — the beam
REF_C = "#888888"
GRID_C = "#cccccc"

# Index of each quantity along axis 1 of <base>_profiles.npy. Mirrors the
# "quantities" list the CLI writes into _meta.json; asserted against it below,
# so a reordering on the Rust side fails loudly here instead of silently
# plotting the wrong row.
QUANTITIES = ["p_Pa", "rho_kg_m3", "u_m_s", "alpha_1_m", "I_W_m2"]
I_P, I_RHO, I_U, I_ALPHA, I_I = range(5)

# Floor for the log-scaled beam panel. The column reaches hundreds of optical
# depths, so the honest intensity underflows; clipping keeps the decay through
# the front — the part with structure in it — legible.
BEAM_FLOOR = 1e-9


def style(ax):
    ax.grid(True, which="both", color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def read_trajectory(path):
    """Read <base>_trajectory.csv into (t, x_front, tau, log10_T)."""
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


def render_wave(base, meta, fps):
    """The animation: pressure, absorption, and beam intensity per frame."""
    prof = np.load(f"{base}_profiles.npy")
    n_frames, n_q, n_cells = prof.shape
    if n_q != len(QUANTITIES):
        raise SystemExit(f"expected {len(QUANTITIES)} quantities, got {n_q}")
    if meta.get("quantities") and list(meta["quantities"]) != QUANTITIES:
        raise SystemExit(
            f"profile layout changed: meta says {meta['quantities']}, "
            f"this script plots {QUANTITIES}"
        )

    t, x_front, tau, _ = read_trajectory(f"{base}_trajectory.csv")
    x_mm = np.linspace(meta["x_min"], meta["x_max"], n_cells) * 1e3
    drive = meta["drive"]
    p0 = meta["p0"]

    fig, (ax_p, ax_b) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.16),
    )

    # --- top: the detonation itself -------------------------------------
    p_max = prof[:, I_P, :].max()
    (p_line,) = ax_p.plot([], [], color=P_C, lw=2.0, zorder=5, label="pressure")
    front_mark = ax_p.axvline(np.nan, color=P_C, lw=1.0, ls=":", alpha=0.8, zorder=4)
    ax_p.axhline(p0 / 1e5, color=REF_C, lw=0.9, ls="--", zorder=3)
    ax_p.annotate(
        f"ambient  {p0 / 1e5:.2f} bar", xy=(0.015, p0 / 1e5), xycoords=("axes fraction", "data"),
        fontsize=8, color=REF_C, ha="left", va="bottom",
    )
    ax_p.set_ylabel("pressure  (bar)")
    ax_p.set_ylim(0.0, 1.08 * p_max / 1e5)
    ax_p.set_xlim(x_mm[0], x_mm[-1])
    style(ax_p)
    mark_laser_side(ax_p)

    # Absorption coefficient on a twin axis: it is what makes the plasma a
    # plasma, and it sits exactly where the pressure front is.
    ax_a = ax_p.twinx()
    (a_line,) = ax_a.plot([], [], color=ALPHA_C, lw=1.4, alpha=0.85, zorder=4)
    ax_a.set_ylabel(r"plasma absorption $\alpha$  (1/m)", color=ALPHA_C)
    ax_a.tick_params(axis="y", colors=ALPHA_C)
    ax_a.set_ylim(0.0, 1.25 * max(prof[:, I_ALPHA, :].max(), 1e-30))
    ax_a.spines["top"].set_visible(False)

    label = ax_p.text(
        0.015, 0.95, "", transform=ax_p.transAxes, fontsize=10, va="top",
        family="monospace", zorder=6,
    )

    # --- bottom: the beam being consumed ---------------------------------
    (b_line,) = ax_b.plot([], [], color=BEAM_C, lw=2.0, zorder=5)
    ax_b.axhline(1.0, color=REF_C, lw=0.9, ls="--", zorder=2)
    ax_b.set_ylabel(r"beam intensity  $I / S$")
    ax_b.set_xlabel(
        "position along the beam axis  (mm)"
        "      — the front runs toward the laser, so it moves left"
    )
    # Log, because the decay through the front IS an exponential and a linear
    # axis renders it as a featureless cliff. On this axis Beer-Lambert is a
    # straight line whose slope is alpha, which is the point.
    ax_b.set_yscale("log")
    ax_b.set_ylim(BEAM_FLOOR, 3.0)
    style(ax_b)
    tau_text = ax_b.text(
        0.015, 0.12, "", transform=ax_b.transAxes, fontsize=10, va="top",
        family="monospace", color=ALPHA_C, zorder=6,
    )

    fig.suptitle(
        f"Laser-supported detonation: {drive:.2e} W/m$^2$ sustaining drive, "
        f"D = {meta['d_measured']:.0f} m/s",
        fontsize=11.5, y=0.98,
    )

    def update(k):
        p_line.set_data(x_mm, prof[k, I_P, :] / 1e5)
        a_line.set_data(x_mm, prof[k, I_ALPHA, :])
        b_line.set_data(x_mm, np.clip(prof[k, I_I, :] / drive, BEAM_FLOOR, None))
        front_mark.set_xdata([x_front[k] * 1e3, x_front[k] * 1e3])
        label.set_text(
            f"t = {t[k] * 1e9:7.1f} ns\nfront = {x_front[k] * 1e3:6.2f} mm"
        )
        tau_text.set_text(f"column optical depth  tau = {tau[k]:6.1f}")
        return p_line, a_line, b_line, front_mark, label, tau_text

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = f"{base}_wave.gif"
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def render_trajectory(base, meta):
    """The front track against the closed form, and the column going opaque."""
    t, x_front, tau, log_t = read_trajectory(f"{base}_trajectory.csv")
    t_ns = t * 1e9
    ok = np.isfinite(x_front)

    fig, (ax_x, ax_t) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # --- left: front position vs time ------------------------------------
    ax_x.plot(
        t_ns[ok], x_front[ok] * 1e3, color=P_C, lw=2.2, zorder=5,
        label=f"solver:  D = {meta['d_measured']:.0f} m/s",
    )
    # Raizer's slope through the first tracked point. Labelled as the
    # construction the model is built from, not as an independent measurement.
    t0, x0 = t[ok][0], x_front[ok][0]
    ax_x.plot(
        t_ns[ok], (x0 - meta["d_raizer"] * (t[ok] - t0)) * 1e3,
        color=REF_C, lw=1.6, ls="--", zorder=4,
        label=f"Raizer CJ:  D = {meta['d_raizer']:.0f} m/s",
    )
    ax_x.set_xlabel("time  (ns)")
    ax_x.set_ylabel("front position  (mm)")
    ax_x.set_title(
        "The front runs back up the beam\n"
        "(agreement here is solver VERIFICATION — see docs/MODELS.md)",
        fontsize=10,
    )
    ax_x.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax_x)
    ax_x.annotate(
        "toward the laser",
        xy=(0.03, 0.18), xytext=(0.03, 0.42), xycoords="axes fraction",
        textcoords="axes fraction", fontsize=9, color=P_C,
        arrowprops=dict(arrowstyle="->", color=P_C, lw=1.5),
    )
    err = 100.0 * (meta["d_measured"] / meta["d_raizer"] - 1.0)
    ax_x.text(
        0.03, 0.05, f"residual {err:+.2f} %", transform=ax_x.transAxes,
        fontsize=9, family="monospace", color=REF_C,
    )

    # --- right: the column closing ---------------------------------------
    ax_t.plot(t_ns, tau, color=ALPHA_C, lw=2.2, zorder=5)
    ax_t.set_xlabel("time  (ns)")
    ax_t.set_ylabel(r"column optical depth  $\tau$", color=ALPHA_C)
    ax_t.tick_params(axis="y", colors=ALPHA_C)
    ax_t.set_title(
        "The plasma is a shutter, not a filter\n"
        r"(D7: the propagator sees this as pure Beer-Lambert, $\delta n \equiv 0$)",
        fontsize=10,
    )
    style(ax_t)

    ax_l = ax_t.twinx()
    ax_l.plot(t_ns, log_t, color=BEAM_C, lw=1.6, ls="--", zorder=4)
    ax_l.set_ylabel(r"$\log_{10}$ transmitted fraction", color="#c8873a")
    ax_l.tick_params(axis="y", colors="#c8873a")
    ax_l.spines["top"].set_visible(False)
    ax_l.annotate(
        f"final: {log_t[-1]:.0f} decades",
        xy=(0.97, 0.12), xycoords="axes fraction", fontsize=9,
        color="#c8873a", ha="right",
    )

    fig.tight_layout()
    out = f"{base}_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="run basename, e.g. out/lsd")
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    args = ap.parse_args()

    base = args.base
    with open(f"{base}_meta.json") as fh:
        meta = json.load(fh)
    if meta.get("case") != "lsd":
        raise SystemExit(
            f"{base}_meta.json is not an lsd run (case={meta.get('case')!r}); "
            "use scripts/render.py for propagate/turbulence/blooming runs and "
            "scripts/render_breakdown.py for breakdown runs"
        )

    for out in (
        render_wave(base, meta, args.fps),
        render_trajectory(base, meta),
    ):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
