#!/usr/bin/env python3
"""Render M6a breakdown results: the solver writes data, this makes the images.

Reads the CSV/NPY/meta a `beamprop breakdown` run writes, plus the digitized
Thiyagarajan & Thompson curve in tests/data/, and renders:

  <base>_threshold.png  — I_thr(p), model envelope + measured points
  <base>_avalanche.gif  — the electron avalanche, one frame per pressure

Usage:
    python3 scripts/render_breakdown.py out/breakdown
    python3 scripts/render_breakdown.py out/breakdown --fps 12

Requires: numpy, matplotlib (pip install numpy matplotlib).

On the honest reading of the comparison: what M6a validates is that the
measurement lies inside the model's *envelope*, not that the central curve
agrees. The figure is drawn to make exactly that claim — the band is the
headline, the central line is thin and secondary. See docs/M6A_SPEC.md.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

EPS0 = 8.8541878128e-12
C_LIGHT = 299792458.0

# Matches scripts/render.py: magma family, physical axes, labelled everything.
MODEL_C = "#f1605d"   # magma mid — the model
BAND_C = "#721f81"    # magma dark — the envelope
DATA_C = "#feca8d"    # magma light — measurement
GRID_C = "#cccccc"


def read_csv_columns(path, n_cols):
    """Read a `#`-commented CSV with one header row into n_cols float lists."""
    cols = [[] for _ in range(n_cols)]
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(row[i]) for i in range(n_cols)]
            except (ValueError, IndexError):
                continue  # header row
            for i, v in enumerate(vals):
                cols[i].append(v)
    return [np.array(c) for c in cols]


def load_measured(repo_root):
    """Digitized T&T breakdown field -> intensity (W/cm^2).

    E_B is an RMS amplitude in 10^6 V/cm (established by the E_eff ratio; see
    the CSV provenance header), so the cycle-averaged intensity is
    I = eps0*c*E_rms^2 -- NOT the peak form, which would halve it.
    """
    path = repo_root / "tests" / "data" / "tt2012_E_B_vs_pressure.csv"
    if not path.exists():
        return None, None
    p_torr, e_mv = read_csv_columns(path, 2)
    e_rms = e_mv * 1e6 * 100.0                      # 10^6 V/cm -> V/m
    return p_torr, EPS0 * C_LIGHT * e_rms**2 / 1e4  # W/m^2 -> W/cm^2


def style(ax):
    ax.grid(True, which="both", color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def interp_loglog(x, y, x0):
    """Value of an ascending (x, y) curve at x0, interpolated in log-log."""
    return float(np.exp(np.interp(np.log(x0), np.log(x), np.log(y))))


P_REF = 760.0  # Torr — normalization anchor; every series spans 1 atm


def render_threshold(base, meta, repo_root):
    """Two panels: absolute level (left) and shape (right).

    Neither is validated, and the figure says so. The model runs ~5-7x above
    the measurement in level, and its pressure trend is too FLAT -- the
    measured points fall outside the model's own slope envelope on the right
    panel. An earlier version of this figure claimed the opposite; that claim
    rested on an integration artifact (see docs/M6A_SPEC.md).

    The right panel normalizes every series at 1 atm so only shape is compared;
    plotting a slope envelope on absolute axes would be a category error.
    """
    p, i_thr, i_d005, i_d001, _ = read_csv_columns(f"{base}_threshold.csv", 5)
    p_m, i_m = load_measured(repo_root)
    if p_m is not None:
        m = (p_m >= p.min()) & (p_m <= p.max())
        p_m, i_m = p_m[m], i_m[m]

    lo, hi = (float(v) for v in meta["slope_envelope"])
    fig, (ax_abs, ax_rel) = plt.subplots(
        1, 2, figsize=(11.6, 4.9), constrained_layout=True
    )

    def draw(ax, scale_model, scale_data):
        ax.fill_between(
            p, i_d005 / scale_model[1], i_d001 / scale_model[2], color=BAND_C,
            alpha=0.22, lw=0,
            label=f"model envelope, $\\delta_{{\\rm eff}}$ = 0.01–0.05  "
                  f"($n \\in [{lo:.2f}, {hi:.2f}]$)",
        )
        ax.plot(p, i_d005 / scale_model[1], color=BAND_C, lw=0.8, alpha=0.5)
        ax.plot(p, i_d001 / scale_model[2], color=BAND_C, lw=0.8, alpha=0.5)
        ax.plot(
            p, i_thr / scale_model[0], color=MODEL_C, lw=1.8,
            label=f"model, central constants ($n$ = {float(meta['slope']):.3f})",
        )
        if p_m is not None:
            ax.plot(
                p_m, i_m / scale_data, "o", color=DATA_C, ms=7, mec="#3b0f70",
                mew=1.0, zorder=5,
                label="Thiyagarajan & Thompson 2012 ($n$ = 0.329)",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("pressure $p$  (Torr)")
        style(ax)

    # --- left: absolute, offset and all ---
    draw(ax_abs, (1.0, 1.0, 1.0), 1.0)
    ax_abs.set_ylabel("threshold peak intensity  (W/cm$^2$)")
    ax_abs.set_title("Absolute level — not validated (ungated)", fontsize=11)
    if p_m is not None:
        ratio = interp_loglog(p, i_thr, P_REF) / interp_loglog(p_m, i_m, P_REF)
        ax_abs.annotate(
            "", xy=(P_REF, interp_loglog(p, i_thr, P_REF)),
            xytext=(P_REF, interp_loglog(p_m, i_m, P_REF)),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2),
        )
        ax_abs.text(
            P_REF * 1.08,
            np.sqrt(interp_loglog(p, i_thr, P_REF) * interp_loglog(p_m, i_m, P_REF)),
            f"{ratio:.1f}x", color="#555555", fontsize=9, va="center",
        )

    # --- right: normalized at 1 atm, i.e. shape only ---
    sm = tuple(interp_loglog(p, c, P_REF) for c in (i_thr, i_d005, i_d001))
    sd = interp_loglog(p_m, i_m, P_REF) if p_m is not None else 1.0
    draw(ax_rel, sm, sd)
    ax_rel.set_ylabel(f"threshold, normalized at {P_REF:.0f} Torr")
    ax_rel.set_title(
        "Shape — model too flat; measurement outside its envelope", fontsize=11
    )
    ax_rel.axvline(P_REF, color=GRID_C, lw=1.0, zorder=0)

    fig.suptitle(
        f"Optical breakdown threshold in dry air, "
        f"{float(meta['wavelength']) * 1e9:.0f} nm, "
        f"{float(meta['fwhm']) * 1e9:.0f} ns FWHM",
        fontsize=13,
    )
    ax_abs.legend(loc="center left", frameon=False, fontsize=7.5)
    fig.text(
        0.5, -0.02,
        "Model $n$ = %.3f vs measured 0.329: too flat, and outside the "
        "$\\delta_{\\rm eff}$ envelope — the external slope gate is RED. "
        "Level is ungated (3–10x inter-lab scatter).\n"
        "What M6a defends is a bracket: the two cascade limits give "
        "$n$ = 0.127 and 0.551, straddling the measurement. See docs/M6A_SPEC.md."
        % float(meta["slope"]),
        fontsize=8, color="#555555", va="top", ha="center",
    )

    out = f"{base}_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def render_avalanche(base, meta, fps):
    """The avalanche: one INDEPENDENT 0-D run per pressure, swept as frames.

    This is a parameter sweep, not a time-varying-pressure simulation. Each
    frame integrates the rate equation from scratch at its own fixed pressure,
    driven by the same pulse.

    The pulse envelope is drawn behind the trace because the timing is
    otherwise unreadable: t < 0 is *before* the pulse peak, and above threshold
    the avalanche completes on the leading edge (the drive crosses the ignition
    intensity ~1 ns early), so the interesting physics happens at negative t.

    Traces run to their saturation ceiling — full ionization at that frame's
    pressure — rather than being cut at n_bd. Above n_bd the model has no
    recombination, opacity or back-reaction, so that region is shaded as beyond
    its validity: the plateau is drawn, but not claimed.
    """
    ne = np.load(f"{base}_ne_traces.npy")           # [pressure, time]
    p, _, _, _, n_neutral = read_csv_columns(f"{base}_threshold.csv", 5)
    t_ns = np.linspace(meta["t_min"], meta["t_max"], ne.shape[1]) * 1e9
    # float() matters: n_bd round-trips through JSON as a bare integer, and a
    # Python big-int axis limit is not safely castable to float64.
    n_bd = float(meta["n_bd"])
    n_seed = float(meta["n_seed"])
    drive = float(meta["drive_intensity_w_per_cm2"])
    fwhm_ns = float(meta["fwhm"]) * 1e9

    floor = n_seed * 1e-2
    top = float(n_neutral.max()) * 30.0

    fig, ax = plt.subplots(figsize=(7.8, 4.9), constrained_layout=True)

    # Pulse envelope, on its own scale behind everything.
    env = ax.twinx()
    pulse = np.exp(-4.0 * np.log(2.0) * (t_ns / fwhm_ns) ** 2)
    env.fill_between(t_ns, 0, pulse, color="#fcfdbf", alpha=0.75, lw=0, zorder=0)
    env.plot(t_ns, pulse, color="#fec98d", lw=1.0, zorder=0)
    env.set_ylim(0, 3.2)
    env.set_yticks([])
    env.text(
        -1.92 * fwhm_ns, 0.06, f"laser pulse, {fwhm_ns:.0f} ns FWHM",
        color="#b8860b", fontsize=8, va="bottom",
    )

    # Beyond-model-validity band.
    ax.axhspan(n_bd, top, color=BAND_C, alpha=0.07, lw=0, zorder=1)
    ax.axhline(n_bd, color=BAND_C, lw=1.3, ls="--", zorder=3)
    ax.text(
        t_ns[-1], n_bd * 1.7,
        "breakdown criterion $n_{bd}$\nabove: beyond model validity",
        color=BAND_C, fontsize=7.5, va="bottom", ha="right", zorder=3,
    )
    ax.text(
        t_ns[-1], top / 2.5, "dash-dot: full ionization at this pressure",
        color="#3b0f70", fontsize=7.5, va="top", ha="right", zorder=6,
    )
    ax.axhline(n_seed, color="#888888", lw=1.0, ls=":", zorder=3)
    ax.text(
        t_ns[0], n_seed * 1.7, "seed: one electron in the focal volume",
        color="#888888", fontsize=7.5, va="bottom", zorder=3,
    )
    for row in ne:
        ax.plot(t_ns, np.clip(row, floor, None), color=MODEL_C, lw=0.4,
                alpha=0.09, zorder=2)

    (sat,) = ax.plot([], [], color="#3b0f70", lw=1.0, ls="-.", zorder=4)
    (line,) = ax.plot([], [], color=MODEL_C, lw=2.3, zorder=5)
    label = ax.text(
        0.015, 0.97, "", transform=ax.transAxes, fontsize=10, va="top",
        family="monospace", zorder=6,
    )
    verdict = ax.text(
        0.015, 0.88, "", transform=ax.transAxes, fontsize=10, va="top",
        fontweight="bold", zorder=6,
    )

    ax.set_yscale("log")
    ax.set_xlim(t_ns[0], t_ns[-1])
    ax.set_ylim(floor, top)
    ax.set_zorder(env.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("time relative to pulse peak  (ns)     — negative is before the peak")
    ax.set_ylabel("electron density $n_e$  (m$^{-3}$)")
    ax.set_title(
        f"Pressure sweep: {ne.shape[0]} independent runs, same pulse "
        f"({drive:.2e} W/cm$^2$)",
        fontsize=11,
    )
    style(ax)

    def update(k):
        line.set_data(t_ns, np.clip(ne[k], floor, None))
        sat.set_data([t_ns[0], t_ns[-1]], [n_neutral[k], n_neutral[k]])
        label.set_text(f"run {k + 1:2d}/{ne.shape[0]}   p = {p[k]:7.1f} Torr")
        if ne[k].max() >= n_bd:
            verdict.set_text("BREAKDOWN")
            verdict.set_color(MODEL_C)
        else:
            verdict.set_text("no breakdown")
            verdict.set_color("#888888")
        return line, sat, label, verdict

    anim = animation.FuncAnimation(fig, update, frames=ne.shape[0], blit=False)
    out = f"{base}_avalanche.gif"
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="run basename, e.g. out/breakdown")
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    args = ap.parse_args()

    base = args.base
    with open(f"{base}_meta.json") as fh:
        meta = json.load(fh)
    if meta.get("case") != "breakdown":
        raise SystemExit(
            f"{base}_meta.json is not a breakdown run (case={meta.get('case')!r}); "
            "use scripts/render.py for propagate/turbulence/blooming runs"
        )
    repo_root = Path(__file__).resolve().parent.parent

    for out in (
        render_threshold(base, meta, repo_root),
        render_avalanche(base, meta, args.fps),
    ):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
