#!/usr/bin/env python3
"""Generate the frozen dry-air property table for M4 (docs/M4_SPEC.md).

Writes data/air_properties.npy  — float64, shape (4, n_T, n_p):
    [0] rho       density                    kg/m^3
    [1] cp        isobaric specific heat     J/(kg K)
    [2] kappa_t   thermal conductivity       W/(m K)
    [3] n_minus_1 refractivity at LAMBDA_REF (dimensionless)
plus data/air_properties.json — grid axes, provenance, gate results.

The committed .npy is the canonical artifact: the Rust solver and CI only
ever read it; this script exists so the table is reproducible, not so it
is regenerated routinely.

Reproducing the shipped table
-----------------------------
Requires Python 3.10 with the Mutation++ python bindings (the binding is
built per-interpreter; any Python works if mutationpp imports there) and
MPP_DATA_DIRECTORY pointing at the Mutation++ data/ directory:

    python3.10 scripts/make_air_table.py

Sources, per docs/M4_SPEC.md "Property closure":
  rho, cp    primary: Mutation++ (N2/O2/Ar equilibrium mixture);
             cross-checked against the analytic model (ideal gas +
             NASA-9 polynomials) — both must agree to < GATE_RHO_CP
             everywhere or nothing is written.
  kappa_t    single-source Sutherland-form correlation, spot-checked
             against NIST reference points (Lemmon et al. 2004) to
             < GATE_KAPPA. (Mutation++ transport targets high-T plasma
             regimes and is 4-25 % off NIST for 200-400 K air, measured
             2026-07-17 — see the M4_SPEC provenance note.)
  n_minus_1  Ciddor (1996) standard dry-air dispersion at LAMBDA_REF,
             scaled by density (Gladstone-Dale): applied to the primary
             rho, so it inherits the rho gate.

--fallback-only skips Mutation++ (and therefore the cross-check gate);
the sidecar marks the table as single-source. Not for shipping.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- grid
T_MIN, T_MAX, T_STEP = 200.0, 400.0, 2.0        # K
P_MIN, P_MAX, P_STEP = 40_000.0, 110_000.0, 2_500.0  # Pa
T_AXIS = np.arange(T_MIN, T_MAX + 0.5 * T_STEP, T_STEP)
P_AXIS = np.arange(P_MIN, P_MAX + 0.5 * P_STEP, P_STEP)

# ------------------------------------------------------- composition
# Dry air, mole fractions (CO2-free three-species approximation).
COMPOSITION = {"N2": 0.7811, "O2": 0.2096, "Ar": 0.0093}
MW = {"N2": 28.014e-3, "O2": 31.998e-3, "Ar": 39.948e-3}  # kg/mol
R_UNIV = 8.31446261815324  # J/(mol K)
MW_AIR = sum(COMPOSITION[s] * MW[s] for s in COMPOSITION)

LAMBDA_REF_UM = 1.0  # refractivity reference wavelength, micrometres

GATE_RHO_CP = 0.005  # max |mpp - analytic| / analytic for rho and cp
GATE_KAPPA = 0.02    # max deviation of kappa_t at the NIST spot points

# NIST reference points for dry air at 101.325 kPa
# (E. W. Lemmon et al., J. Phys. Chem. Ref. Data 33, 111 (2004)).
KAPPA_REF_POINTS = [(200.0, 0.0181), (300.0, 0.02635), (400.0, 0.0338)]

# NASA-9 polynomial coefficients, 200-1000 K range (McBride, Zehe,
# Gordon, NASA/TP-2002-211556): cp/R = a1/T^2 + a2/T + a3 + a4*T
# + a5*T^2 + a6*T^3 + a7*T^4.  Ar is monatomic: cp/R = 5/2.
NASA9 = {
    "N2": [2.210371497e4, -3.818461820e2, 6.082738360, -8.530914410e-3,
           1.384646189e-5, -9.625793620e-9, 2.519705809e-12],
    "O2": [-3.425563420e4, 4.847000970e2, 1.119010961, 4.293889240e-3,
           -6.836300520e-7, -2.023372700e-9, 1.039040018e-12],
}


def _cp_over_r(species: str, t: np.ndarray) -> np.ndarray:
    if species == "Ar":
        return np.full_like(t, 2.5)
    a = NASA9[species]
    return (a[0] / t**2 + a[1] / t + a[2] + a[3] * t + a[4] * t**2
            + a[5] * t**3 + a[6] * t**4)


def analytic_rho_cp(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal gas + NASA-9. t, p broadcast to the (n_T, n_p) grid."""
    rho = p * MW_AIR / (R_UNIV * t)
    cp_mole = R_UNIV * sum(x * _cp_over_r(s, t) for s, x in COMPOSITION.items())
    cp = cp_mole / MW_AIR * np.ones_like(p)
    return rho, cp


def kappa_t(t: np.ndarray) -> np.ndarray:
    """Sutherland form fit to NIST (Lemmon 2004); pressure dependence of
    kappa_t is < 0.3 % over 40-110 kPa and is neglected."""
    return 0.024113 * (t / 273.15) ** 1.5 * (273.15 + 194.4) / (t + 194.4)


def ciddor_standard_refractivity(lambda_um: float) -> float:
    """(n-1) of standard dry air (15 C, 101325 Pa, 450 ppm CO2),
    Ciddor, Appl. Opt. 35, 1566 (1996), Eq. (1)."""
    sigma2 = (1.0 / lambda_um) ** 2
    return 1e-8 * (5_792_105.0 / (238.0185 - sigma2)
                   + 167_917.0 / (57.362 - sigma2))


def mutationpp_rho_cp() -> tuple[np.ndarray, np.ndarray, dict]:
    import mutationpp as mpp

    opts = mpp.MixtureOptions()
    opts.setSpeciesDescriptor(" ".join(COMPOSITION))
    opts.setStateModel("Equil")
    mix = mpp.Mixture(opts)
    comp = ", ".join(f"{s}:{x}" for s, x in COMPOSITION.items())
    mix.addComposition(comp, True)

    rho = np.empty((T_AXIS.size, P_AXIS.size))
    cp = np.empty_like(rho)
    for i, t in enumerate(T_AXIS):
        for j, p in enumerate(P_AXIS):
            mix.equilibrate(float(t), float(p))
            rho[i, j] = mix.density()
            cp[i, j] = mix.mixtureEquilibriumCpMass()

    prov = {
        "package_version": _pip_version("mutationpp"),
        "module_path": mpp.__file__,
        "data_directory": os.environ.get("MPP_DATA_DIRECTORY"),
        "local_build_git": _git_describe(_local_mpp_repo()),
    }
    return rho, cp, prov


def _local_mpp_repo() -> str | None:
    data_dir = os.environ.get("MPP_DATA_DIRECTORY")
    return str(Path(data_dir).parent) if data_dir else None


def _git_describe(repo: str | None) -> str | None:
    if not repo or not (Path(repo) / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "-C", repo, "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _pip_version(package: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data", type=Path)
    ap.add_argument("--fallback-only", action="store_true",
                    help="skip Mutation++ and the cross-check gate "
                         "(table is marked single-source; not for shipping)")
    args = ap.parse_args()

    tt = T_AXIS[:, None]
    pp = P_AXIS[None, :]
    rho_a, cp_a = analytic_rho_cp(tt, pp)

    gates: dict = {}
    if args.fallback_only:
        rho, cp = rho_a, cp_a
        mpp_prov = None
    else:
        rho, cp, mpp_prov = mutationpp_rho_cp()
        dev_rho = float(np.abs(rho / rho_a - 1).max())
        dev_cp = float(np.abs(cp / cp_a - 1).max())
        gates["max_rel_dev_rho"] = dev_rho
        gates["max_rel_dev_cp"] = dev_cp
        print(f"cross-check gate: max|Δrho| = {dev_rho:.2e}, "
              f"max|Δcp| = {dev_cp:.2e} (limit {GATE_RHO_CP:.0e})")
        if max(dev_rho, dev_cp) >= GATE_RHO_CP:
            print("FAIL: sources disagree beyond the gate — nothing written.",
                  file=sys.stderr)
            return 1

    kap = np.broadcast_to(kappa_t(tt), rho.shape).copy()
    kap_devs = [abs(float(kappa_t(np.array(t))) - ref) / ref
                for t, ref in KAPPA_REF_POINTS]
    gates["max_rel_dev_kappa_vs_nist"] = max(kap_devs)
    print(f"kappa_t spot-check vs NIST: max dev = {max(kap_devs):.2%} "
          f"(limit {GATE_KAPPA:.0%})")
    if max(kap_devs) >= GATE_KAPPA:
        print("FAIL: kappa_t correlation off NIST references.", file=sys.stderr)
        return 1

    n1_std = ciddor_standard_refractivity(LAMBDA_REF_UM)
    rho_std = 101_325.0 * MW_AIR / (R_UNIV * 288.15)
    n_minus_1 = n1_std * rho / rho_std

    table = np.stack([rho, cp, kap, n_minus_1])
    if not np.isfinite(table).all() or (table <= 0).any():
        print("FAIL: non-finite or non-positive table entries.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "air_properties.npy"
    np.save(npy_path, table)

    sidecar = {
        "schema": "beamprop-air-table-v1",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "properties": ["rho", "cp", "kappa_t", "n_minus_1"],
        "units": ["kg/m^3", "J/(kg K)", "W/(m K)", "1"],
        "shape": list(table.shape),
        "T_axis_K": {"min": T_MIN, "max": T_MAX, "step": T_STEP},
        "p_axis_Pa": {"min": P_MIN, "max": P_MAX, "step": P_STEP},
        "interpolation": "bilinear",
        "composition_mole_fractions": COMPOSITION,
        "molar_mass_kg_per_mol": MW_AIR,
        "refractivity": {
            "lambda_ref_um": LAMBDA_REF_UM,
            "model": "Ciddor 1996 standard dry-air dispersion, "
                     "density-scaled (Gladstone-Dale)",
            "n_minus_1_standard": n1_std,
            "rho_standard_kg_m3": rho_std,
        },
        "sources": {
            "rho_cp": ("analytic fallback only (NOT for shipping)"
                       if args.fallback_only else
                       "Mutation++ N2/O2/Ar equilibrium, cross-checked vs "
                       "ideal gas + NASA-9 (McBride 2002)"),
            "kappa_t": "Sutherland-form fit, spot-checked vs Lemmon 2004",
            "n_minus_1": "Ciddor 1996 at lambda_ref, scaled by rho",
        },
        "gates": gates,
        "provenance": {
            "script": "scripts/make_air_table.py",
            "beamprop_git": _git_describe(str(Path(__file__).resolve().parents[1])),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mutationpp": mpp_prov,
        },
    }
    json_path = args.out_dir / "air_properties.json"
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"wrote {npy_path} {table.shape} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
