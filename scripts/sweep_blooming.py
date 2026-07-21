#!/usr/bin/env python3
"""Thermal-blooming power sweep: the delivered-irradiance rollover, animated.

Sweeps launch power through the validated Smith (1977) F0 = 5 geometry
(docs/M4_SPEC.md B3 gate: 512 x 1 mm grid, z = 500 m, 2 m/s crosswind,
alpha = 1e-4 /m) using the `beamprop` Python bindings, and renders:

  <out>/bloom_sweep.mp4        two-panel animation: receiver-plane spot
                               (absolute delivered scale) next to the
                               delivered-peak-vs-power curve tracing live
  <out>/bloom_smith_validation.png
                               solver I_REL vs Smith's digitized F0 = 5 curve

Sweep results are cached in <out>/bloom_sweep.npz so render tweaks do not
recompute the runs (delete the file to force a re-sweep).

Requires: beamprop (maturin develop), numpy, matplotlib, ffmpeg on PATH.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import beamprop as bp

# --- B3 gate geometry (tests/blooming.rs) -----------------------------------
N_GRID = 512
DX = 1e-3
WAVELENGTH = 1e-6
Z = 500.0
F0 = 5.0
STEPS = 200
WIND = 2.0
ALPHA_ABS = 1e-4
T0 = 288.15
P0 = 101_325.0

K = 2.0 * math.pi / WAVELENGTH
W0 = math.sqrt(2.0 * F0 * Z / K)  # 1/e^2 waist giving F0 = k*a^2/z, a = w0/sqrt(2)

GAMMA = 0.5  # display gamma, matching scripts/render.py house style
REPO = Path(__file__).resolve().parent.parent


def n_c_over_n_phi():
    """Ratio of Smith's geometric N_c to the spec's phase number N_phi.

    Both are linear in alpha*P/(rho*cp*v) with the same (n0-1)/T0 factor, so
    every air property cancels except n0 in N_c's denominator, and n0 = 1 to
    3e-4 relative — negligible against the 7 % validation envelope.
    """
    a = W0 / math.sqrt(2.0)
    az = ALPHA_ABS * Z
    bracket = 2.0 / az - 2.0 / (az * az) * (1.0 - math.exp(-az))
    i0_per_watt = 2.0 / (math.pi * W0 * W0)
    n_c = i0_per_watt * ALPHA_ABS * Z * Z * bracket / a
    n_phi = math.sqrt(2.0 / math.pi) * K * ALPHA_ABS * Z / W0
    return n_c / n_phi


def sweep(cache_path):
    """Run the power sweep (or load it from cache). Returns a dict of arrays."""
    if cache_path.exists():
        print(f"loading cached sweep from {cache_path}")
        return dict(np.load(cache_path))

    common = dict(
        n=N_GRID, dx=DX, wavelength=WAVELENGTH, w0=W0, wind=WIND,
        alpha_abs=ALPHA_ABS, t0=T0, p0=P0, z=Z, steps=STEPS,
    )

    # Calibrate N_c per watt from one weak probe run's N_phi (both linear in P).
    probe = bp.run_blooming(power=1e3, frames=1, **common)
    n_c_per_watt = (probe["n_phi"] / 1e3) * n_c_over_n_phi()
    print(f"w0 = {W0 * 1e2:.2f} cm, N_c per kW = {n_c_per_watt * 1e3:.3f}, "
          f"Peclet = {probe['peclet']:.0f}")

    # Bloom-free diffraction reference for I_REL (Smith's normalization
    # divides out Beer-Lambert, so the reference run carries no absorption).
    vac = bp.run_propagate(n=N_GRID, dx=DX, wavelength=WAVELENGTH, w0=W0,
                           z=Z, steps=STEPS, frames=1)
    vac_peak = vac["final"].max()

    n_targets = np.linspace(0.05, 3.0, 40)
    powers, peaks, finals, n_phis = [], [], [], []
    for n_c in n_targets:
        power = n_c / n_c_per_watt
        try:
            r = bp.run_blooming(power=power, frames=1, **common)
        except ValueError as e:
            print(f"sweep stops at N_c = {n_c:.2f}: {e}")
            break
        powers.append(power)
        peaks.append(r["final"].max())
        finals.append(r["final"])
        n_phis.append(r["n_phi"])
        print(f"N_c = {n_c:.2f}  P = {power / 1e3:6.1f} kW  "
              f"I_rel = {r['final'].max() / (vac_peak * math.exp(-ALPHA_ABS * Z)):.3f}")

    data = dict(
        n_c=n_targets[: len(powers)],
        power=np.array(powers),
        peak=np.array(peaks),
        finals=np.array(finals),
        n_phi=np.array(n_phis),
        vac_peak=np.array(vac_peak),
        vac_final=vac["final"],
    )
    np.savez_compressed(cache_path, **data)
    print(f"sweep cached to {cache_path}")
    return data


def load_smith_curve():
    path = REPO / "tests" / "data" / "smith1977_F5.csv"
    rows = [
        line.split(",")
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("N,")
    ]
    pts = np.array([[float(a), float(b)] for a, b in rows])
    return pts[:, 0], pts[:, 1]


def crop(img, half_px):
    mid = img.shape[0] // 2
    return img[mid - half_px : mid + half_px, mid - half_px : mid + half_px]


def render_animation(data, out_path, fps):
    smith_n, smith_i = load_smith_curve()
    # Stay on the validated branch: the digitized curve ends at N ~ 1.87 and
    # the B3 gate covers N in [0.5, 1.8]; beyond that the solver is untested.
    sel = data["n_c"] <= smith_n[-1]
    n_c = data["n_c"][sel]
    power_kw = data["power"][sel] / 1e3
    finals = data["finals"][sel]
    i_rel = data["peak"][sel] / (float(data["vac_peak"]) * math.exp(-ALPHA_ABS * Z))

    half_px = 96  # +-9.6 cm view around the axis (~3.4 w0)
    extent_cm = [-half_px * DX * 1e2, half_px * DX * 1e2] * 2
    # Per-watt irradiance on a fixed scale: every run launches the same
    # unit-power numerical field, so the frames are directly comparable and
    # the blooming loss shows as the spot dimming and bending upwind.
    scale = finals[0].max()

    fig, (ax_spot, ax_curve) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    fig.suptitle(
        "Thermal blooming: the atmosphere taxes your beam "
        f"(collimated, F$_0$ = {F0:g}, z = {Z:g} m, crosswind {WIND:g} m/s)",
        fontsize=13,
    )

    im = ax_spot.imshow(
        crop(finals[0] / scale, half_px) ** GAMMA,
        origin="lower", extent=extent_cm, cmap="magma", vmin=0.0, vmax=1.0,
        aspect="equal",
    )
    ax_spot.set_xlabel("x (cm)  —  wind →")
    ax_spot.set_ylabel("y (cm)")
    spot_title = ax_spot.set_title("")
    cbar = fig.colorbar(im, ax=ax_spot, fraction=0.046, pad=0.03)
    ticks_i = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    cbar.set_ticks(ticks_i**GAMMA)
    cbar.set_ticklabels([f"{t:g}" for t in ticks_i])
    cbar.set_label("irradiance per watt launched (rel.)")

    ax_curve.set_xlim(0, smith_n[-1] * 1.05)
    ax_curve.set_ylim(0.68, 1.03)
    ax_curve.set_xlabel("distortion number N (∝ power)")
    ax_curve.set_ylabel("peak irradiance vs bloom-free beam")
    ax_curve.plot(smith_n, smith_i, color="0.45", lw=2.2,
                  label="Smith 1977 theory (F$_0$ = 5)")
    (trace,) = ax_curve.plot([], [], color="tab:orange", lw=2.2,
                             label="beamprop solver")
    (dot,) = ax_curve.plot([], [], "o", color="tab:orange", ms=8)
    ax_curve.legend(loc="upper right", frameon=False)
    ax_curve.grid(alpha=0.25)
    p_label = ax_curve.text(0.03, 0.06, "", transform=ax_curve.transAxes,
                            fontsize=12)

    def update(i):
        im.set_data(crop(finals[i] / scale, half_px) ** GAMMA)
        spot_title.set_text(f"receiver plane — P = {power_kw[i]:.1f} kW")
        trace.set_data(n_c[: i + 1], i_rel[: i + 1])
        dot.set_data([n_c[i]], [i_rel[i]])
        p_label.set_text(f"N = {n_c[i]:.2f}   P = {power_kw[i]:.1f} kW")
        return im, trace, dot, spot_title, p_label

    frames = list(range(len(power_kw)))
    frames += [frames[-1]] * fps  # hold the last frame ~1 s
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"wrote {out_path}")


def render_validation(data, out_path):
    smith_n, smith_i = load_smith_curve()
    i_rel = data["peak"] / (float(data["vac_peak"]) * math.exp(-ALPHA_ABS * Z))

    in_range = (data["n_c"] >= smith_n[0]) & (data["n_c"] <= smith_n[-1])
    dev = np.abs(
        i_rel[in_range] - np.interp(data["n_c"][in_range], smith_n, smith_i)
    ) / np.interp(data["n_c"][in_range], smith_n, smith_i)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(smith_n, smith_i, color="0.25", lw=2,
            label="Smith 1977, F$_0$ = 5 (digitized)")
    ax.plot(data["n_c"][in_range], i_rel[in_range], "o", ms=6,
            color="tab:orange", label="beamprop solver")
    ax.plot(data["n_c"][~in_range], i_rel[~in_range], "o", ms=6, mfc="none",
            color="tab:orange", label="beamprop, past digitized range")
    ax.set_xlabel("distortion number N")
    ax.set_ylabel(r"$I_{\mathrm{REL}} = I_{\mathrm{bloomed}}\,/\,I_{\mathrm{no\ bloom}}$")
    ax.set_title("Peak-irradiance rollover vs Smith (1977) steady-state theory")
    lo, hi = data["n_c"][in_range][0], data["n_c"][in_range][-1]
    ax.text(0.97, 0.06,
            f"max deviation {100 * dev.max():.1f}%\n"
            f"over N ∈ [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="right", fontsize=10, color="0.35")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}  (max dev {100 * dev.max():.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(REPO / "out"))
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = sweep(out / "bloom_sweep.npz")
    render_validation(data, out / "bloom_smith_validation.png")
    render_animation(data, out / "bloom_sweep.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
