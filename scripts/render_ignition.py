#!/usr/bin/env python3
"""Render M6a.2 ignition-statistics results: the solver writes data, this makes
the images.

Reads the CSV/NPY/meta a `beamprop ignition` run writes and renders:

  <base>_ignition.png  — three panels: the ignition probability against
                         turbulence strength, the focal-intensity distribution
                         that produces it, and the spot wander against its
                         closed form

Usage:
    python3 scripts/render_ignition.py out/ignition

Requires: numpy, matplotlib (pip install numpy matplotlib).

The figure is drawn to make one distinction unmissable, because it is the whole
honesty of this milestone: **the SHAPE of the ignition curve is a result, its
POSITION on the Cn² axis is not.** The position is set by M6a's absolute
breakdown threshold, which is explicitly ungated (4.8–7.0× above the measured
T&T curve); changing it slides the curve sideways with its shape intact. The
left panel therefore carries that caveat in the panel itself rather than in a
caption someone can crop off, and the ignition probability is drawn with the
binomial error bars it actually has.

The right panel is the opposite case: the Cn²^(1/2) wander law is gated (W1)
against a parameter-free exponent, so it is drawn against its closed form.
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

# Matches the other render scripts: magma family throughout.
IG_C = "#f1605d"     # magma mid — the ignition curve
DIST_C = "#721f81"   # magma dark — distributions
WAND_C = "#feca8d"   # magma light — wander
REF_C = "#888888"
GRID_C = "#cccccc"

# Axis 0 of <base>_realizations.npy. Mirrors "quantities" in _meta.json.
Q_RATIO, Q_STREHL = 0, 1


def style(ax):
    ax.grid(True, which="both", color=GRID_C, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def read_sweep(path):
    """Read <base>_sweep.csv into a dict of columns."""
    names = None
    cols = None
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if names is None:
                names = [c.strip() for c in row]
                cols = [[] for _ in names]
                continue
            for i, v in enumerate(row[: len(names)]):
                cols[i].append(float(v))
    if names is None:
        raise SystemExit(f"{path} has no data rows")
    return {n: np.array(c) for n, c in zip(names, cols)}


def render(base, meta):
    s = read_sweep(f"{base}_sweep.csv")
    real = np.load(f"{base}_realizations.npy")
    if real.shape[0] != 2:
        raise SystemExit(f"expected 2 quantities, got {real.shape[0]}")
    if meta.get("quantities") and list(meta["quantities"]) != [
        "focal_ratio",
        "phase_only_strehl",
    ]:
        raise SystemExit(f"realization layout changed: {meta['quantities']}")

    cn2 = s["cn2"]
    fig, (ax_p, ax_d, ax_w) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---- left: the ignition probability, with honest error bars ------------
    ax_p.errorbar(
        cn2, s["p_ignite"], yerr=s["p_ignite_se"], color=IG_C, lw=2.2,
        marker="o", ms=4, capsize=3, zorder=5,
    )
    ax_p.set_xscale("log")
    ax_p.set_xlabel(r"turbulence strength  $C_n^2$  (m$^{-2/3}$)")
    ax_p.set_ylabel("ignition probability")
    ax_p.set_ylim(-0.05, 1.08)
    ax_p.set_title("How often the spark still lights", fontsize=11, pad=30)
    style(ax_p)

    # The caveat lives in the panel, not the caption.
    ax_p.text(
        0.97, 0.97,
        "shape is the result\nposition is NOT:\nit rides M6a's ungated\nabsolute threshold",
        transform=ax_p.transAxes, fontsize=8.5, ha="right", va="top",
        color=REF_C, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=REF_C, alpha=0.85),
    )
    td = meta.get("transition_decades")
    if td and np.isfinite(td):
        ax_p.text(
            0.03, 0.06, f"0.9 → 0.1 over {td:.2f} decades",
            transform=ax_p.transAxes, fontsize=9, family="monospace", color=IG_C,
        )
    # A second x axis in D/r0, the parameter the pupil statistics live in.
    top = ax_p.twiny()
    top.set_xscale("log")
    top.set_xlim(ax_p.get_xlim())
    ticks = [c for c in cn2[:: max(1, len(cn2) // 4)]]
    top.set_xticks(ticks)
    top.set_xticklabels(
        [f"{s['d_over_r0'][list(cn2).index(t)]:.1f}" for t in ticks], fontsize=8
    )
    top.set_xlabel(r"$D/r_0$", fontsize=9)
    for side in ("right",):
        top.spines[side].set_visible(False)

    # ---- middle: the distribution behind it --------------------------------
    ratio = real[Q_RATIO]
    strehl = real[Q_STREHL]
    thr = meta["i_threshold"] / meta["i_focus_vacuum"]
    parts = ax_d.violinplot(
        [np.log10(np.clip(r, 1e-12, None)) for r in ratio],
        positions=np.log10(cn2), widths=0.18, showextrema=False, showmedians=True,
    )
    for b in parts["bodies"]:
        b.set_facecolor(DIST_C)
        b.set_alpha(0.55)
    parts["cmedians"].set_color(DIST_C)
    ax_d.axhline(
        np.log10(thr), color=IG_C, lw=1.8, ls="--", zorder=6,
        label="M6a threshold (ungated level)",
    )
    ax_d.set_xlabel(r"$\log_{10} C_n^2$")
    ax_d.set_ylabel(r"$\log_{10}$ (focal intensity / vacuum)")
    ax_d.set_title(
        "A realization ignites when it lands above the line", fontsize=11
    )
    ax_d.legend(
        frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.10)
    )
    style(ax_d)
    # Scintillation is the gap between the two: the focal-intensity ratio
    # carries amplitude as well as phase, the Strehl only phase.
    gap = float(np.median(strehl / np.clip(ratio, 1e-30, None)))
    ax_d.text(
        0.02, 0.02,
        f"median Strehl/ratio = {gap:.2f} — the degradation is "
        f"{100 * gap:.0f}% wavefront, the rest scintillation",
        transform=ax_d.transAxes, fontsize=8, ha="left", va="bottom", color=REF_C,
    )

    # ---- right: the wander law, which IS gated ------------------------------
    ax_w.loglog(cn2, s["wander_rms_m"] * 1e6, color=WAND_C, lw=2.4, marker="o",
                ms=4, zorder=5, label="solver")
    ref = s["wander_rms_m"][0] * np.sqrt(cn2 / cn2[0]) * 1e6
    ax_w.loglog(cn2, ref, color=REF_C, lw=1.5, ls="--", zorder=4,
                label=r"$\propto C_n^{2\,\,1/2}$  (gated, W1)")
    ax_w.set_xlabel(r"turbulence strength  $C_n^2$  (m$^{-2/3}$)")
    ax_w.set_ylabel(r"RMS focal-spot wander  ($\mu$m)")
    ax_w.set_title("Where the spark lands — this one is anchored", fontsize=11)
    ax_w.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax_w)

    fig.suptitle(
        f"Turbulence-degraded ignition: {meta['w0'] * 100:.0f} cm beam, "
        f"{meta['z']:.0f} m path, {meta['aperture']:.2f} m aperture, "
        f"{meta['realizations']} realizations per point",
        fontsize=11.5, y=1.06,
    )
    fig.tight_layout()
    out = f"{base}_ignition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="run basename, e.g. out/ignition")
    args = ap.parse_args()

    with open(f"{args.base}_meta.json") as fh:
        meta = json.load(fh)
    if meta.get("case") != "ignition":
        raise SystemExit(
            f"{args.base}_meta.json is not an ignition run "
            f"(case={meta.get('case')!r}); use scripts/render.py for "
            "propagate/turbulence/blooming, render_breakdown.py for breakdown, "
            "render_lsd.py for lsd"
        )
    print(f"wrote {render(args.base, meta)}")


if __name__ == "__main__":
    main()
