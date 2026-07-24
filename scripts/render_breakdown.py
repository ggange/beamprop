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
    """Two panels, because the model makes two claims of very different strength.

    Left: absolute threshold. The model sits ~7x above the measurement, and
    that offset is NOT validated -- it is ungated on purpose (published
    thresholds scatter 3-10x across labs).

    Right: the same curves normalized at 1 atm, i.e. SHAPE only. This is where
    the validated claim lives: the slope envelope from delta_eff's literature
    range contains the measured trend. Plotting containment on absolute axes
    would be a category error -- the envelope spans slopes, not levels.
    """
    p, i_thr, i_d005, i_d001 = read_csv_columns(f"{base}_threshold.csv", 4)
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
    ax_abs.set_title("Absolute threshold — level is NOT validated", fontsize=11)
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
    ax_rel.set_title("Shape — this is the validated comparison", fontsize=11)
    ax_rel.axvline(P_REF, color=GRID_C, lw=1.0, zorder=0)

    fig.suptitle(
        f"Optical breakdown threshold in dry air, "
        f"{float(meta['wavelength']) * 1e9:.0f} nm, "
        f"{float(meta['fwhm']) * 1e9:.0f} ns FWHM",
        fontsize=13,
    )
    handles, labels = ax_rel.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.005, -0.055,
        "Validated: the measured pressure trend lies inside the model's "
        "slope envelope (right). Not validated: the absolute level (left), "
        "deliberately ungated — published thresholds scatter 3–10x across labs.",
        fontsize=8, color="#555555", va="top",
    )

    out = f"{base}_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def render_avalanche(base, meta, fps):
    """The avalanche, one frame per pressure, at a single fixed intensity.

    Traces are truncated once they cross n_bd. That is not cosmetic: the rate
    model is linear in n_e with no saturation -- no neutral depletion, no
    recombination, no plasma back-reaction on the beam -- so above the
    breakdown criterion it is extrapolating far outside its domain (unchecked,
    it runs to 1e40 m^-3, past solid density and 1e13x the critical density at
    1064 nm). n_bd is where the model's claim ends, so that is where the
    picture ends.
    """
    ne = np.load(f"{base}_ne_traces.npy")           # [pressure, time]
    p, _, _, _ = read_csv_columns(f"{base}_threshold.csv", 4)
    t_ns = np.linspace(meta["t_min"], meta["t_max"], ne.shape[1]) * 1e9
    # float() matters: n_bd round-trips through JSON as a bare integer, and a
    # Python big-int axis limit is not safely castable to float64.
    n_bd = float(meta["n_bd"])
    n_seed = float(meta["n_seed"])
    drive = float(meta["drive_intensity_w_per_cm2"])

    ceiling = n_bd * 10.0
    floor = n_seed * 1e-8

    def truncated(row):
        """Trace up to its first excursion past the display ceiling."""
        over = np.nonzero(row > ceiling)[0]
        stop = over[0] + 1 if over.size else row.size
        return t_ns[:stop], np.clip(row[:stop], floor, ceiling)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ax.axhline(n_bd, color=BAND_C, lw=1.3, ls="--")
    ax.text(
        t_ns[-1], n_bd * 1.6, "breakdown criterion  $n_{bd}$",
        color=BAND_C, fontsize=8.5, va="bottom", ha="right",
    )
    ax.axhline(n_seed, color="#888888", lw=1.0, ls=":")
    ax.text(
        t_ns[0], n_seed * 1.6, "seed: one electron in the focal volume",
        color="#888888", fontsize=8, va="bottom",
    )
    for row in ne:
        gt, gy = truncated(row)
        ax.plot(gt, gy, color=MODEL_C, lw=0.4, alpha=0.10)

    (line,) = ax.plot([], [], color=MODEL_C, lw=2.2)
    label = ax.text(
        0.015, 0.90, "", transform=ax.transAxes, fontsize=10, va="top",
        family="monospace",
    )
    verdict = ax.text(
        0.015, 0.81, "", transform=ax.transAxes, fontsize=10, va="top",
        fontweight="bold",
    )

    ax.set_yscale("log")
    ax.set_xlim(t_ns[0], t_ns[-1])
    ax.set_ylim(floor, ceiling * 3.0)
    ax.set_xlabel("time relative to pulse peak  (ns)")
    ax.set_ylabel("electron density $n_e$  (m$^{-3}$)")
    ax.set_title(f"Same pulse, {drive:.2e} W/cm$^2$ — only the pressure changes")
    style(ax)
    ax.text(
        0.985, 0.03,
        "traces stop at $n_{bd}$: the rate model has no saturation term\n"
        "and does not describe the plasma beyond it",
        transform=ax.transAxes, fontsize=7.5, color="#555555",
        ha="right", va="bottom",
    )

    def update(k):
        gt, gy = truncated(ne[k])
        line.set_data(gt, gy)
        label.set_text(f"p = {p[k]:7.1f} Torr")
        if ne[k].max() >= n_bd:
            verdict.set_text("BREAKDOWN")
            verdict.set_color(MODEL_C)
        else:
            verdict.set_text("no breakdown")
            verdict.set_color("#888888")
        return line, label, verdict

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
