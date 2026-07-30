#!/usr/bin/env python3
"""Regenerate `tests/data/chylek1990_air_threshold_vs_pressure.csv` from the paper PDF.

This exists so the digitized anchor is *reproducible* rather than merely
described. The other digitized curves in `tests/data/` came out of a manual
WebPlotDigitizer session, which cannot be re-run; this one can.

    python3 scripts/digitize_chylek1990.py chylek1990.pdf > out.csv

Source: P. Chylek, M. A. Jarzembski, V. Srivastava, R. G. Pinnick, "Pressure
dependence of the laser-induced breakdown thresholds of gases and droplets",
Appl. Opt. 29, 2303-2306 (1990), Fig. 3, the clean-air series (filled diamonds).
The PDF is copyrighted and is deliberately NOT committed; supply your own copy.

Method
------
1. `pdfimages` extracts the page-3 raster (400 dpi, bilevel -- no thresholding
   choices to make).
2. Both axes are calibrated by least squares on the log-decade **minor** ticks,
   iterating the tick->value assignment to convergence so a missed tick cannot
   silently shift the fit by one minor step.
3. Filled diamonds are found by normalised cross-correlation against a 19x15
   rhombus, then refined to the dark-pixel centroid of a local window. Open
   diamonds (the water-droplet series) score low because their centre is hollow.
4. Text, the bulk-water rule and the arrow are removed by explicit masks, which
   are printed to stderr so they can be audited.

Two checks, neither of which is used to fit anything -- see `--verify`:
  * the BULK WATER rule is held out of the y calibration and must land on the
    4.7e10 W/cm^2 the paper's Sec. V states;
  * the recovered points must reproduce the alpha = 0.45 +/- 0.01 slope the
    paper's own Fig. 3 caption prints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Frame of the Fig. 3 panel in the page-3 raster, and the marker template size.
TOP, BOT, LEFT, RIGHT = 215, 1925, 1736, 2756
MARK_H, MARK_W = 19, 15

# Inked features that are not clean-air data, as (name, x0, x1, y0, y1).
MASKS = [
    ("AIR text label", 1945, 2055, 405, 470),
    ("BULK WATER text", 1750, 2110, 940, 1000),
    ("BULK WATER dashed rule", LEFT, RIGHT, 1004, 1022),
    ("S_eff arrow and label", 1900, 2010, 1030, 1400),
    ("water-droplet series", LEFT, RIGHT, 1380, 1560),
    ("WATER DROPLET IN AIR legend", 1750, 2400, 1670, 1775),
]

# The clean-air series occupies I_TH >= this; every other inked feature in the
# panel lies at least 2.7x below the lowest air marker.
AIR_FLOOR = 1.0e11


def page_raster(pdf: Path) -> np.ndarray:
    """Page 3 of `pdf` as a boolean ink mask."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["pdfimages", "-f", "3", "-l", "3", "-png", str(pdf), f"{td}/pg"],
            check=True,
        )
        pngs = sorted(Path(td).glob("pg-*.png"))
        if not pngs:
            sys.exit("pdfimages extracted no image from page 3")
        return np.array(Image.open(pngs[0]).convert("L")) < 128


def tick_centroids(band: np.ndarray, along: int, minfill: int, gap: int = 4):
    """Mass-weighted sub-pixel centroids of tick stubs in `band`."""
    prof = band.sum(axis=along)
    groups: list[list[int]] = []
    for c in np.where(prof >= minfill)[0]:
        if groups and c - groups[-1][-1] <= gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    return np.array([float(np.average(g, weights=prof[g].astype(float))) for g in groups])


def _snap(pix, lo, hi, A, B):
    cand = np.array([m * 10.0**e for e in range(lo, hi + 1) for m in range(1, 10)])
    cand = cand[(cand >= 10.0**lo * 0.99) & (cand <= 10.0**hi * 1.01)]
    return np.array(
        [cand[np.argmin(np.abs(np.log10(cand / 10 ** ((p - A) / B))))] for p in pix]
    )


def _dedupe(pix, val):
    seen, keep = set(), []
    for i, v in enumerate(val):
        if v not in seen:
            seen.add(v)
            keep.append(i)
    return pix[keep], val[keep]


def calibrate(pix, lo, hi, A0, B0, label):
    """Iterate assign-then-fit to convergence. Returns (A, B, resid_px, n)."""
    A, B = A0, B0
    for _ in range(12):
        p, v = _dedupe(pix, _snap(pix, lo, hi, A, B))
        L = np.log10(v)
        Bn, An = np.polyfit(L, p, 1)
        if abs(An - A) < 1e-9 and abs(Bn - B) < 1e-9:
            A, B = An, Bn
            break
        A, B = An, Bn
    p, v = _dedupe(pix, _snap(pix, lo, hi, A, B))
    resid = p - (A + B * np.log10(v))
    print(
        f"  {label}: n={len(p)} ticks, decade={B:+.3f}px, "
        f"resid rms {resid.std():.2f}px = {abs(resid.std() / B) * 100:.2f}% of a decade",
        file=sys.stderr,
    )
    return A, B, resid, len(p), (p, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="the Chylek 1990 PDF")
    ap.add_argument("--verify", action="store_true", help="run the two checks and exit")
    args = ap.parse_args()

    d = page_raster(args.pdf).astype(float)
    print("axis calibration (log-decade minor ticks):", file=sys.stderr)

    xt = tick_centroids(d[BOT - 14 : BOT - 2, LEFT - 6 : RIGHT + 7].astype(bool), 0, 10)
    xt += LEFT - 6
    Ax, Bx, _, nx, _ = calibrate(xt, 0, 3, float(LEFT), (RIGHT - LEFT) / 3.0, "x/Torr")

    yt = tick_centroids(d[TOP - 6 : BOT + 7, LEFT + 2 : LEFT + 14].astype(bool), 1, 10)
    yt += TOP - 6
    yv = _snap(yt, 8, 13, TOP + 13 * 342.0, -342.0)
    bulk_px = next((p for p, v in zip(yt, yv) if 4.0e10 < v < 5.6e10), None)
    keep = np.array([not (4.0e10 < v < 5.6e10) for v in yv])  # hold out the rule
    Ay, By, _, ny, _ = calibrate(
        yt[keep], 8, 13, TOP + 13 * 342.0, -342.0, "y/(W/cm^2)"
    )

    # --- marker detection -------------------------------------------------
    cy, cx = (MARK_H - 1) / 2, (MARK_W - 1) / 2
    yy, xx = np.mgrid[0:MARK_H, 0:MARK_W]
    T = (np.abs(yy - cy) / cy + np.abs(xx - cx) / cx <= 1.0).astype(float)
    Tz = T - T.mean()
    num = ndimage.correlate(d, Tz, mode="constant")
    sq = ndimage.uniform_filter(d * d, size=(MARK_H, MARK_W), mode="constant")
    mu = ndimage.uniform_filter(d, size=(MARK_H, MARK_W), mode="constant")
    var = sq * (MARK_H * MARK_W) - (MARK_H * MARK_W) * mu**2
    ncc = np.where(
        var > 1e-6, num / np.sqrt(np.maximum(var, 1e-6) * (Tz**2).sum()), 0.0
    )

    roi = np.zeros(ncc.shape, bool)
    roi[TOP + 12 : BOT - 12, LEFT + 12 : RIGHT - 12] = True
    print("masked non-data ink:", file=sys.stderr)
    for name, xa, xb, ya, yb in MASKS:
        roi[ya:yb, xa:xb] = False
        print(f"  {name:30s} x {xa}-{xb}  y {ya}-{yb}", file=sys.stderr)

    cand = (ncc >= 0.50) & roi
    peaks = (ncc == ndimage.maximum_filter(ncc, size=(11, 9))) & cand
    ys, xs = np.nonzero(peaks)
    kept: list[tuple[int, int]] = []
    for i in np.argsort(-ncc[ys, xs]):  # greedy NMS, elliptical exclusion
        y, x = int(ys[i]), int(xs[i])
        if all(
            (y - ky) ** 2 / 100.0 + (x - kx) ** 2 / 81.0 >= 1.0 for ky, kx in kept
        ):
            kept.append((y, x))

    pts = []
    for y, x in kept:
        y0, y1 = y - MARK_H // 2 - 1, y + MARK_H // 2 + 2
        x0, x1 = x - MARK_W // 2 - 1, x + MARK_W // 2 + 2
        win = d[y0:y1, x0:x1]
        if win.sum() < 40:
            continue
        gy, gx = ndimage.center_of_mass(win)
        torr = 10 ** ((x0 + gx - Ax) / Bx)
        val = 10 ** ((y0 + gy - Ay) / By)
        if val >= AIR_FLOOR:
            pts.append((torr, val))
    pts.sort()
    print(f"recovered {len(pts)} clean-air markers", file=sys.stderr)

    # --- the two held-out checks -----------------------------------------
    a = np.array(pts)
    slope, icept = np.polyfit(np.log10(a[:, 0]), np.log10(a[:, 1]), 1)
    resid = np.log10(a[:, 1]) - (slope * np.log10(a[:, 0]) + icept)
    se = resid.std(ddof=2) / (np.log10(a[:, 0]).std() * np.sqrt(len(a)))
    print("\nchecks (neither is used to fit anything):", file=sys.stderr)
    if bulk_px is not None:
        got = 10 ** ((bulk_px - Ay) / By)
        print(
            f"  held-out BULK WATER rule -> {got:.3e} W/cm^2 vs the paper's "
            f"4.7e10  ({got / 4.7e10:.3f}x)",
            file=sys.stderr,
        )
    print(
        f"  recovered slope alpha = {-slope:.4f} +/- {se:.4f} vs the paper's "
        f"printed 0.45 +/- 0.01",
        file=sys.stderr,
    )
    print(f"  scatter about that fit: {10 ** resid.std(ddof=2):.4f}x", file=sys.stderr)
    if args.verify:
        return

    print("# columns: pressure_torr, I_th_W_per_cm2")
    print("pressure_torr,I_th_W_per_cm2")
    for t, v in pts:
        print(f"{t:.4f},{v:.5e}")


if __name__ == "__main__":
    main()
