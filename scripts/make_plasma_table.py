#!/usr/bin/env python3
"""Generate the frozen plasma-range air property table for M6c (docs/M6C_SPEC.md, D8).

Writes data/plasma_properties.npy — float64, shape (4, n_T, n_p):
    [0] ln_rho     ln(density / 1 kg m^-3)          dimensionless
    [1] e          specific internal energy         J/kg
    [2] gamma_eff  equilibrium ratio of specific heats
    [3] ln_ne      ln(electron number density / 1 m^-3)
plus data/plasma_properties.json — axes, provenance, gate results — and
tests/data/plasma_reference_samples.csv, the off-grid direct-Mutation++ samples
the Rust G6 gate interpolates against.

The committed .npy is the canonical artifact: the Rust solver and CI only ever
read it. This script exists so the table is reproducible, not so it is
regenerated routinely. `data/air_properties.npy` (M4) is a SEPARATE file and is
never touched here — its green gate must not be perturbable by M6c (G0b).

Reproducing the shipped table
-----------------------------
Requires Python 3.10 with the Mutation++ python bindings (the binding is built
per-interpreter; note that on this machine plain `python3` hangs inside
setState and `python3.10` is the working interpreter) and MPP_DATA_DIRECTORY
pointing at the Mutation++ data/ directory:

    python3.10 scripts/make_plasma_table.py

Design decisions, and why
-------------------------
Axes are uniform in T and uniform in log10(p), so the Rust side indexes with
two divisions and interpolates bilinearly — the airprops.rs precedent. Pressure
spans 1e4-1e8 Pa because the Chapman-Jouguet state behind an LSD front sits
near 1.5e7 Pa (docs/M6C_SPEC.md), and temperature reaches 30,000 K for the same
reason.

rho and n_e are tabulated as LOGARITHMS. Both vary over dozens of decades
across the ionization onset, and linear interpolation of the raw values there
is hopeless; interpolating the log and exponentiating on read holds the error
near 1e-3 (measured below). e and gamma_eff are stored raw — e changes sign
around 800 K, so a log is not available, and both are smooth enough not to need
one.

n_i is NOT stored. With only singly charged ions in the mixture, quasi-
neutrality makes n_i identical to n_e (verified to 7.4e-11 wherever n_e is
physically relevant, gated below), so storing it would be a second copy of the
same array dressed up as data.

Z_bar is NOT stored, for a sharper reason: it is identically 1 and cannot be
anything else here. The RRHO thermodynamic database ships no doubly ionized
N or O (He++ is the only ++ species in it), so air_11 structurally cannot
represent second ionization. See the LIMITATION note below — this bounds where
the table is trustworthy, and the Rust side re-states it.

LIMITATION: second ionization is absent
---------------------------------------
Real equilibrium air begins to doubly ionize above roughly 20,000-25,000 K.
This table cannot: Z_bar == 1 by construction of the species set. So toward the
top of the T range the table understates n_e and therefore the inverse-
bremsstrahlung absorption alpha_IB (which goes as n_e * n_i * Z_bar^2). The
table is still generated to 30,000 K because the CJ state needs the range, but
above SECOND_IONIZATION_K the values are a singly-ionized approximation, not a
prediction. This is recorded in the sidecar and in src/plasmaprops.rs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- grid
T_MIN, T_MAX, T_STEP = 200.0, 30_000.0, 50.0        # K
P_MIN_LOG10, P_MAX_LOG10, P_PER_DECADE = 4.0, 8.0, 8  # Pa, log-uniform
T_AXIS = np.arange(T_MIN, T_MAX + 0.5 * T_STEP, T_STEP)
P_AXIS = np.logspace(
    P_MIN_LOG10, P_MAX_LOG10,
    int((P_MAX_LOG10 - P_MIN_LOG10) * P_PER_DECADE) + 1,
)

MIXTURE = "air_11"
THERMO_DB = "RRHO"
STATE_MODEL = "Equil"
# setState variable-set 1 is (P, T) for the equilibrium state model. Variable
# set 2 wants element mole fractions as well and hangs the equilibrium solver
# if handed only two arguments -- an easy and silent mistake.
SETSTATE_PT = 1

NA = 6.02214076e23          # Avogadro, 1/mol
QE = 1.602176565e-19        # Mutation++'s elementary charge, C

# Above this the singly-ionized species set stops being a fair description of
# equilibrium air (see LIMITATION above).
SECOND_IONIZATION_K = 20_000.0

# n_e below this (1/m^3) is physically irrelevant -- for scale, 1e15 is still
# eight orders below the ~1e23 of an LSD plasma, and the cold tail runs down to
# 1e-96. Accuracy claims are made only above it: in the tail, ln(n_e) is nearly
# linear in 1/T rather than T, so a uniform-T grid interpolates it badly, and
# gating a 100x error on 1e3 electrons per cubic metre would be theatre.
NE_FLOOR = 1e15

# Gate limits, set from the measured values (printed at generation).
GATE_INTERP_RHO = 2e-3
GATE_INTERP_E = 5e-3
GATE_INTERP_GAMMA = 1e-3
GATE_INTERP_NE = 1e-2
GATE_QUASINEUTRALITY = 1e-8
GATE_COLD_GAMMA = 0.01      # |gamma_eff - 1.4| for cold undissociated air

PROPERTIES = ["ln_rho", "e", "gamma_eff", "ln_ne"]
UNITS = ["ln(kg/m^3)", "J/kg", "1", "ln(1/m^3)"]

# Off-grid sample points for the Rust G6 gate: deliberately at cell midpoints
# (worst case for bilinear) and concentrated on the ionization onset.
N_SAMPLES_ONSET = 60
N_SAMPLES_BROAD = 40
ONSET_T_RANGE = (6_000.0, 18_000.0)


def _git_describe(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "describe", "--always", "--dirty"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class Equilibrium:
    """Equilibrium air properties from Mutation++, at one (T, p) at a time."""

    def __init__(self) -> None:
        import mutationpp as mpp

        opts = mpp.MixtureOptions(MIXTURE)
        opts.setThermodynamicDatabase(THERMO_DB)
        opts.setStateModel(STATE_MODEL)
        self.mix = mpp.Mixture(opts)
        ns = self.mix.nSpecies()
        self.names = [self.mix.speciesName(i) for i in range(ns)]
        self.mw = np.array([self.mix.speciesMw(i) for i in range(ns)])
        self.charge = np.array(
            [self.mix.speciesCharge(i) for i in range(ns)]) / QE
        self.i_electron = self.names.index("e-")
        if not np.any(self.charge > 0.5):
            raise SystemExit(f"mixture {MIXTURE} has no positive ions")
        if np.any(self.charge > 1.5):
            raise SystemExit(
                "mixture contains multiply charged ions -- the Z_bar == 1 "
                "assumption in this script and in plasmaprops.rs is void")

    def at(self, t: float, p: float):
        """(rho, e, gamma_eff, n_e, n_i) at temperature `t` (K), pressure `p` (Pa)."""
        self.mix.setState(np.array([p]), np.array([t]), SETSTATE_PT)
        number_density = np.array(self.mix.densities()) / self.mw * NA
        n_e = float(number_density[self.i_electron])
        n_i = float(np.sum(number_density[self.charge > 0.5]))
        return (self.mix.density(), self.mix.mixtureEnergyMass(),
                self.mix.mixtureEquilibriumGamma(), n_e, n_i)


def build_table(eq: Equilibrium) -> np.ndarray:
    table = np.zeros((len(PROPERTIES), len(T_AXIS), len(P_AXIS)))
    for i, t in enumerate(T_AXIS):
        for j, p in enumerate(P_AXIS):
            rho, e, gamma, n_e, _ = eq.at(t, p)
            table[0, i, j] = np.log(rho)
            table[1, i, j] = e
            table[2, i, j] = gamma
            # The cold tail underflows to zero in double precision; clamp to a
            # floor so the log is finite. Anything near it is far below
            # NE_FLOOR and carries no accuracy claim.
            table[3, i, j] = np.log(max(n_e, 1e-300))
    return table


def bilinear(table: np.ndarray, t: float, p: float) -> np.ndarray:
    """Interpolate exactly as src/plasmaprops.rs does, for the in-script gates."""
    ti = (t - T_MIN) / T_STEP
    pj = (np.log10(p) - P_MIN_LOG10) * P_PER_DECADE
    i0 = min(int(np.floor(ti)), len(T_AXIS) - 2)
    j0 = min(int(np.floor(pj)), len(P_AXIS) - 2)
    fi, fj = ti - i0, pj - j0
    return (table[:, i0, j0] * (1 - fi) * (1 - fj)
            + table[:, i0 + 1, j0] * fi * (1 - fj)
            + table[:, i0, j0 + 1] * (1 - fi) * fj
            + table[:, i0 + 1, j0 + 1] * fi * fj)


def gate_interpolation(eq: Equilibrium, table: np.ndarray) -> dict:
    """G6, in-script half: the frozen table vs direct Mutation++ at cell midpoints."""
    worst = np.zeros(4)
    worst_at = [None] * 4
    checked = 0
    for i in range(len(T_AXIS) - 1):
        t = 0.5 * (T_AXIS[i] + T_AXIS[i + 1])
        for j in range(len(P_AXIS) - 1):
            p = float(np.sqrt(P_AXIS[j] * P_AXIS[j + 1]))
            rho, e, gamma, n_e, _ = eq.at(t, p)
            if n_e < NE_FLOOR:
                continue
            got = bilinear(table, t, p)
            rel = np.array([
                abs(np.exp(got[0]) / rho - 1.0),
                abs(got[1] - e) / max(abs(e), 1e-30),
                abs(got[2] / gamma - 1.0),
                abs(np.exp(got[3]) / n_e - 1.0),
            ])
            for k in range(4):
                if rel[k] > worst[k]:
                    worst[k], worst_at[k] = rel[k], (t, p)
            checked += 1
    limits = [GATE_INTERP_RHO, GATE_INTERP_E, GATE_INTERP_GAMMA, GATE_INTERP_NE]
    names = ["rho", "e", "gamma_eff", "n_e"]
    print(f"  interpolation gate over {checked} midpoints (n_e > {NE_FLOOR:.0e}):")
    ok = True
    for k, name in enumerate(names):
        loc = worst_at[k]
        where = f"T={loc[0]:.0f} K, p={loc[1]:.3e} Pa" if loc else "n/a"
        print(f"    {name:10s} max rel {worst[k]:.3e} (limit {limits[k]:.0e}) at {where}")
        ok &= worst[k] < limits[k]
    if not ok:
        raise SystemExit("interpolation gate FAILED -- refusing to write the table")
    return {name: float(worst[k]) for k, name in enumerate(names)}


def gate_quasineutrality(eq: Equilibrium) -> float:
    """n_i == n_e, the justification for not storing n_i."""
    worst = 0.0
    for t in np.linspace(1_000.0, T_MAX, 120):
        for p in np.logspace(P_MIN_LOG10, P_MAX_LOG10, 17):
            _, _, _, n_e, n_i = eq.at(float(t), float(p))
            if n_e < NE_FLOOR:
                continue
            worst = max(worst, abs(n_i - n_e) / n_e)
    print(f"  quasi-neutrality  max |n_i/n_e - 1| = {worst:.3e} "
          f"(limit {GATE_QUASINEUTRALITY:.0e})")
    if worst >= GATE_QUASINEUTRALITY:
        raise SystemExit("quasi-neutrality gate FAILED -- n_i must be stored after all")
    return float(worst)


def gate_cold_limit(eq: Equilibrium) -> float:
    """Cold undissociated air must come back as a gamma = 1.4 ideal gas.

    An independent check that the mixture, database and state model are wired
    up as intended: it is the one point in the table where the answer is known
    without Mutation++.
    """
    _, _, gamma, _, _ = eq.at(300.0, 101_325.0)
    dev = abs(gamma - 1.4)
    print(f"  cold limit        gamma_eff(300 K, 1 atm) = {gamma:.5f} "
          f"(|dev| {dev:.4f}, limit {GATE_COLD_GAMMA})")
    if dev >= GATE_COLD_GAMMA:
        raise SystemExit("cold-limit gate FAILED")
    return float(gamma)


def write_reference_samples(eq: Equilibrium, path: Path) -> int:
    """Off-grid direct-Mutation++ samples for the Rust G6 gate.

    The Rust side cannot call Mutation++ (no runtime FFI, D8/P3), so the
    comparison it needs is frozen here at digitization time -- the same
    treatment tests/data/tt2012_*.csv gets.
    """
    rng = np.random.default_rng(20260726)
    rows = []

    def sample(t: float, p: float) -> None:
        rho, e, gamma, n_e, _ = eq.at(t, p)
        if n_e < NE_FLOOR:
            return
        rows.append((t, p, rho, e, gamma, n_e))

    # Concentrated on the ionization onset, where interpolation is worst.
    for _ in range(N_SAMPLES_ONSET):
        sample(float(rng.uniform(*ONSET_T_RANGE)),
               float(10 ** rng.uniform(P_MIN_LOG10, P_MAX_LOG10)))
    # Plus a broad sweep over the rest of the table.
    for _ in range(N_SAMPLES_BROAD):
        sample(float(rng.uniform(2_000.0, T_MAX)),
               float(10 ** rng.uniform(P_MIN_LOG10, P_MAX_LOG10)))
    rows.sort()

    lines = [
        "# Equilibrium air properties, direct Mutation++ evaluation.",
        "# The frozen-table reference for the M6c G6 gate: these points are",
        "# OFF the table grid, so interpolating data/plasma_properties.npy to",
        f"# them is a real test of the tabulation. Only n_e > {NE_FLOOR:.0e} m^-3 is",
        "# sampled; below that n_e is physically irrelevant and the uniform-T",
        "# grid interpolates its log badly (see scripts/make_plasma_table.py).",
        f"# mixture={MIXTURE} thermo={THERMO_DB} state={STATE_MODEL}",
        f"# generated={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
        f"# script=scripts/make_plasma_table.py git={_git_describe(str(Path(__file__).resolve().parents[1]))}",
        "# LIMITATION: Z_bar == 1; the RRHO database has no doubly ionized N/O,",
        f"# so samples above {SECOND_IONIZATION_K:.0f} K are a singly-ionized approximation.",
        "T_K,p_Pa,rho_kg_m3,e_J_kg,gamma_eff,n_e_m3",
    ]
    for t, p, rho, e, gamma, n_e in rows:
        lines.append(f"{t:.6e},{p:.6e},{rho:.10e},{e:.10e},{gamma:.10e},{n_e:.10e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--samples-out", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "tests" / "data" / "plasma_reference_samples.csv")
    args = parser.parse_args()

    print(f"building {MIXTURE} equilibrium ({THERMO_DB}, {STATE_MODEL}) ...")
    eq = Equilibrium()

    print(f"tabulating {len(T_AXIS)} x {len(P_AXIS)} states "
          f"(T {T_MIN:.0f}-{T_MAX:.0f} K, p 1e{P_MIN_LOG10:.0f}-1e{P_MAX_LOG10:.0f} Pa) ...")
    table = build_table(eq)

    print("gates:")
    gamma_cold = gate_cold_limit(eq)
    quasineutrality = gate_quasineutrality(eq)
    interp = gate_interpolation(eq, table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "plasma_properties.npy"
    np.save(npy_path, table)

    n_samples = write_reference_samples(eq, args.samples_out)

    try:
        import mutationpp
        mpp_prov = getattr(mutationpp, "__file__", "unknown")
    except Exception:
        mpp_prov = "unknown"

    sidecar = {
        "schema": "beamprop-plasma-table-v1",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "properties": PROPERTIES,
        "units": UNITS,
        "shape": list(table.shape),
        "T_axis_K": {"min": T_MIN, "max": T_MAX, "step": T_STEP,
                     "n": len(T_AXIS), "spacing": "uniform"},
        "p_axis_Pa": {"log10_min": P_MIN_LOG10, "log10_max": P_MAX_LOG10,
                      "per_decade": P_PER_DECADE, "n": len(P_AXIS),
                      "spacing": "uniform in log10"},
        "interpolation": "bilinear in (T, log10 p); rho and n_e are logs",
        "mixture": {"name": MIXTURE, "thermo_database": THERMO_DB,
                    "state_model": STATE_MODEL},
        "not_stored": {
            "n_i": "identical to n_e by quasi-neutrality (singly charged ions "
                   "only); gated to GATE_QUASINEUTRALITY",
            "Z_bar": "identically 1 -- the RRHO database has no doubly ionized "
                     "N or O, so air_11 cannot represent second ionization",
        },
        "limitations": {
            "second_ionization_absent": True,
            "trustworthy_below_K": SECOND_IONIZATION_K,
            "note": "above trustworthy_below_K the table understates n_e and "
                    "therefore alpha_IB; it is a singly-ionized approximation, "
                    "not a prediction",
            "n_e_accuracy_floor_m3": NE_FLOOR,
        },
        "gates": {
            "cold_limit_gamma_300K_1atm": gamma_cold,
            "quasineutrality_max_rel": quasineutrality,
            "interpolation_max_rel": interp,
            "reference_samples": n_samples,
        },
        "sources": {
            "all": "Mutation++ (VKI) equilibrium thermochemistry; trusted for "
                   "the physics per D8, with the TABULATION gated here and in "
                   "tests (G6)",
        },
        "provenance": {
            "script": "scripts/make_plasma_table.py",
            "beamprop_git": _git_describe(str(Path(__file__).resolve().parents[1])),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mutationpp": mpp_prov,
        },
    }
    json_path = args.out_dir / "plasma_properties.json"
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"wrote {npy_path} {table.shape}")
    print(f"wrote {json_path}")
    print(f"wrote {args.samples_out} ({n_samples} off-grid samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
