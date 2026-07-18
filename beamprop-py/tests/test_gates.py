"""M5 validation gates: the Python bindings must reproduce the Rust solver.

No new physics enters at M5, so the gates are parity and reproducibility:

1. CLI parity — `run_*()` arrays are bit-identical to the `.npy` files the
   Rust CLI writes for the same parameters and seed.
2. Closed form — a vacuum-propagated Gaussian matches the analytic width
   (the M1 anchor, re-run through the bindings).
3. Determinism — the turbulence Monte-Carlo is seed-reproducible.
4. Round-trip — `Field.u` set from numpy stays consistent.
5. Error mapping — solver validity errors surface as `ValueError`.
"""

import subprocess

import numpy as np
import pytest

import beamprop as bp

# Fixed parameter sets for the parity gate. Chosen small enough to run in
# seconds but on the validated grid geometry (beam resolved by >= 4 samples,
# well inside the boundary guard band).
PROP = dict(n=512, dx=1e-3, w0=1e-2, z=200.0, steps=50, frames=3, visibility=5000.0)
TURB = dict(n=256, dx=2e-3, w0=1e-2, z=1000.0, screens=5, cn2=1.5e-14, realizations=4, seed=7)
BLOOM = dict(n=512, dx=1e-3, w0=5e-2, power=2e4, wind=2.0, alpha_abs=1e-4, z=500.0, steps=50, frames=3)


def run_cli(binary, tmp_path, case, params, out):
    args = [str(binary), case, "--out", out, "--out-dir", str(tmp_path)]
    for key, val in params.items():
        flag = "--" + key.replace("_", "-")
        args += [flag, str(val)]
    subprocess.run(args, check=True, capture_output=True)


class TestCliParity:
    """Gate 1: bit-identical to the CLI (`np.array_equal`, no tolerance)."""

    def test_propagate(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "propagate", PROP, "par")
        r = bp.run_propagate(**PROP)
        assert np.array_equal(r["xz"], np.load(tmp_path / "par_xz.npy"))
        assert np.array_equal(r["frames"], np.load(tmp_path / "par_frames.npy"))
        assert np.array_equal(r["final"], np.load(tmp_path / "par_final.npy"))

    def test_turbulence(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "turbulence", TURB, "par")
        r = bp.run_turbulence(**TURB)
        assert np.array_equal(r["frames"], np.load(tmp_path / "par_frames.npy"))
        assert np.array_equal(r["xz_frames"], np.load(tmp_path / "par_xz_frames.npy"))
        assert np.array_equal(r["longexp"], np.load(tmp_path / "par_longexp.npy"))

    def test_blooming(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "blooming", BLOOM, "par")
        r = bp.run_blooming(**BLOOM)
        assert np.array_equal(r["xz"], np.load(tmp_path / "par_xz.npy"))
        assert np.array_equal(r["frames"], np.load(tmp_path / "par_frames.npy"))
        assert np.array_equal(r["final"], np.load(tmp_path / "par_final.npy"))


class TestClosedForm:
    """Gate 2: the M1 analytic Gaussian anchor through the bindings."""

    def test_vacuum_width_matches_analytic(self):
        g = bp.Grid(512, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 1e-2)
        beam = bp.GaussianBeam(1e-2, 1e-6)
        z = 2.0 * beam.rayleigh_range
        p = bp.Propagator(g, 1e-6)
        p.propagate(f, bp.Medium.vacuum(512), z / 200, 200)
        wx, wy = f.beam_width()
        w_ref = beam.width_at(z)
        assert abs(wx - w_ref) / w_ref < 0.01
        assert abs(wy - w_ref) / w_ref < 0.01

    def test_power_conserved_in_vacuum(self):
        g = bp.Grid(256, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)
        p0 = f.power
        bp.Propagator(g, 1e-6).propagate(f, bp.Medium.vacuum(256), 1.0, 50)
        assert abs(f.power - p0) / p0 < 1e-12

    def test_beer_lambert_transmission(self):
        r = bp.run_propagate(n=256, dx=1e-3, w0=5e-3, z=50.0, steps=10, alpha=1e-3)
        t_ref = np.exp(-1e-3 * 50.0)
        assert abs(r["transmission"] - t_ref) / t_ref < 1e-10


class TestDeterminism:
    """Gate 3: seed-reproducible Monte-Carlo across calls."""

    def test_same_seed_is_identical(self):
        kw = dict(n=256, dx=2e-3, w0=1e-2, z=500.0, screens=3, cn2=1e-14, realizations=2, seed=42)
        a = bp.run_turbulence(**kw)
        b = bp.run_turbulence(**kw)
        assert np.array_equal(a["frames"], b["frames"])
        assert np.array_equal(a["longexp"], b["longexp"])

    def test_different_seed_differs(self):
        kw = dict(n=256, dx=2e-3, w0=1e-2, z=500.0, screens=3, cn2=1e-14, realizations=2)
        a = bp.run_turbulence(seed=42, **kw)
        c = bp.run_turbulence(seed=43, **kw)
        assert not np.array_equal(a["frames"], c["frames"])

    def test_medium_turbulence_realization_selects_member(self):
        g = bp.Grid(256, 2e-3)
        outs = []
        for realization in (0, 1, 0):
            f = bp.Field.gaussian(g, 1e-6, 1e-2)
            m = bp.Medium.turbulence(g, 1e-6, 1e-14, 1e3, 500.0, 3, seed=5, realization=realization)
            bp.Propagator(g, 1e-6).propagate(f, m, 500.0 / 3, 3)
            outs.append(f.intensity)
        assert np.array_equal(outs[0], outs[2])
        assert not np.array_equal(outs[0], outs[1])


class TestRoundTrip:
    """Gate 4: numpy <-> Field consistency."""

    def test_u_round_trip(self):
        g = bp.Grid(64, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)
        u = f.u
        assert u.dtype == np.complex128
        np.testing.assert_allclose(np.abs(u) ** 2, f.intensity, rtol=1e-15)
        f2 = bp.Field.gaussian(g, 1e-6, 8e-3)
        f2.u = u
        assert np.array_equal(f2.intensity, f.intensity)
        assert f2.power == pytest.approx(f.power, rel=1e-15)

    def test_wrong_shape_raises(self):
        g = bp.Grid(64, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)
        with pytest.raises(ValueError, match="shape"):
            f.u = np.zeros((32, 32), dtype=np.complex128)

    def test_on_step_sees_every_step(self):
        g = bp.Grid(64, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)
        seen = []
        bp.Propagator(g, 1e-6).propagate(
            f, bp.Medium.vacuum(64), 0.5, 10, on_step=lambda i, fld: seen.append((i, fld.power))
        )
        assert [i for i, _ in seen] == list(range(10))

    def test_on_step_exception_propagates(self):
        g = bp.Grid(64, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)

        def boom(i, fld):
            raise RuntimeError("stop here")

        with pytest.raises(RuntimeError, match="stop here"):
            bp.Propagator(g, 1e-6).propagate(f, bp.Medium.vacuum(64), 0.5, 10, on_step=boom)


class TestErrorMapping:
    """Gate 5: Rust validity errors arrive as ValueError with the message."""

    def test_stagnant_air_peclet(self):
        g = bp.Grid(128, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 1e-2)
        with pytest.raises(ValueError, match="P.clet"):
            bp.Medium.thermal_blooming(f, 1e-2, 1e4, 1e-3, 1e-4)

    def test_bad_grid(self):
        with pytest.raises(ValueError, match="even"):
            bp.Grid(65, 1e-3)
        with pytest.raises(ValueError, match="spacing"):
            bp.Grid(64, 0.0)

    def test_bad_beam(self):
        g = bp.Grid(64, 1e-3)
        with pytest.raises(ValueError, match="waist"):
            bp.Field.gaussian(g, 1e-6, -1.0)

    def test_delta_t_ceiling(self):
        # Absurd power drives dT past 0.1 T0: the propagate call must fail
        # with the small-perturbation message, not return garbage.
        with pytest.raises(ValueError, match="small-perturbation"):
            bp.run_blooming(n=256, dx=1e-3, w0=2e-2, power=1e9, alpha_abs=1e-3, z=200.0, steps=10)
