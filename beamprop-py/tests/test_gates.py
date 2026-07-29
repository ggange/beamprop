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
# The M6 cases. Kept small: the LSD run needs its full 2500 cells to keep the
# absorption layer resolved (check_regime refuses coarser), so the frame count
# is what shrinks. The ignition sweep costs points x realizations propagations.
LSD = dict(frames=4)
BREAKDOWN = dict(points=5, steps=100)
IGNITION = dict(points=3, realizations=6, n=128, dx=4e-3)


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

    def test_lsd(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "lsd", LSD, "par")
        r = bp.run_lsd(**LSD)
        assert np.array_equal(r["profiles"], np.load(tmp_path / "par_profiles.npy"))

    def test_breakdown(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "breakdown", BREAKDOWN, "par")
        r = bp.run_breakdown(**BREAKDOWN)
        assert np.array_equal(r["ne_traces"], np.load(tmp_path / "par_ne_traces.npy"))

    def test_ignition(self, cli_binary, tmp_path):
        run_cli(cli_binary, tmp_path, "ignition", IGNITION, "par")
        r = bp.run_ignition(**IGNITION)
        # The CLI stacks [focal_ratio, strehl]; the binding returns them apart.
        cli = np.load(tmp_path / "par_realizations.npy")
        assert np.array_equal(r["focal_ratio"], cli[0])
        assert np.array_equal(r["strehl"], cli[1])


class TestClosedForm:
    """Gate 2: the M1 analytic Gaussian anchor through the bindings."""

    def test_vacuum_width_matches_analytic(self):
        g = bp.Grid(512, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 1e-2)
        beam = bp.GaussianBeam(1e-2, 1e-6)
        z = 2.0 * beam.rayleigh_range
        p = bp.Propagator(g, 1e-6)
        p.propagate(f, bp.Medium.vacuum(g), z / 200, 200)
        wx, wy = f.beam_width()
        w_ref = beam.width_at(z)
        assert abs(wx - w_ref) / w_ref < 0.01
        assert abs(wy - w_ref) / w_ref < 0.01

    def test_power_conserved_in_vacuum(self):
        g = bp.Grid(256, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)
        p0 = f.power
        bp.Propagator(g, 1e-6).propagate(f, bp.Medium.vacuum(g), 1.0, 50)
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
            f, bp.Medium.vacuum(g), 0.5, 10, on_step=lambda i, fld: seen.append((i, fld.power))
        )
        assert [i for i, _ in seen] == list(range(10))

    def test_on_step_exception_propagates(self):
        g = bp.Grid(64, 1e-3)
        f = bp.Field.gaussian(g, 1e-6, 5e-3)

        def boom(i, fld):
            raise RuntimeError("stop here")

        with pytest.raises(RuntimeError, match="stop here"):
            bp.Propagator(g, 1e-6).propagate(f, bp.Medium.vacuum(g), 0.5, 10, on_step=boom)


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

    def test_turbulence_underresolved_beam_raises_not_panics(self):
        # An under-resolved beam (width < 4 samples) is rejected inside the
        # parallel ensemble; it must surface as ValueError, not a Rust panic.
        with pytest.raises(ValueError, match="under-resolved|resolved"):
            bp.run_turbulence(n=128, dx=2e-3, w0=1e-3, z=500.0, screens=3, realizations=2)

    def test_propagate_past_medium_slabs_raises_not_panics(self):
        # A turbulence path has screens*substeps slabs; marching past it must
        # be a clean ValueError, not an out-of-bounds panic mid-loop.
        g = bp.Grid(256, 2e-3)
        f = bp.Field.gaussian(g, 1e-6, 1e-2)
        m = bp.Medium.turbulence(g, 1e-6, 1e-14, 1e3, 500.0, 3, seed=1)  # 3 slabs
        with pytest.raises(ValueError, match="exceeds the medium"):
            bp.Propagator(g, 1e-6).propagate(f, m, 500.0 / 3, 4)

    def test_medium_grid_mismatch_raises(self):
        # A linear medium sized to a different grid than the field is rejected
        # at propagate time with a shape-mismatch ValueError.
        g_field = bp.Grid(256, 1e-3)
        g_other = bp.Grid(128, 1e-3)
        f = bp.Field.gaussian(g_field, 1e-6, 5e-3)
        with pytest.raises(ValueError, match="shape|expected"):
            bp.Propagator(g_field, 1e-6).propagate(f, bp.Medium.vacuum(g_other), 1.0, 5)


class TestPropagatorReuse:
    """guard_frac must reflect only the most recent call (finding fix)."""

    def test_guard_frac_is_per_call_on_reuse(self):
        g = bp.Grid(256, 1e-3)
        # A tightly-contained beam barely touches the guard band, so a single
        # run's guard_frac is small; reusing the propagator must not let it
        # accumulate across calls.
        p = bp.Propagator(g, 1e-6)
        f1 = bp.Field.gaussian(g, 1e-6, 5e-3)
        p.propagate(f1, bp.Medium.vacuum(g), 1.0, 20)
        first = p.guard_frac
        f2 = bp.Field.gaussian(g, 1e-6, 5e-3)
        p.propagate(f2, bp.Medium.vacuum(g), 1.0, 20)
        second = p.guard_frac
        # Identical runs → identical per-call fraction (no accumulation).
        assert second == pytest.approx(first, abs=1e-18)

class TestM6Cases:
    """The M6 cases carry claims the propagation cases do not, and the ones
    that are *not* claims have to survive the FFI as clearly as the ones that
    are."""

    def test_lsd_reproduces_the_closed_form(self):
        r = bp.run_lsd(frames=4)
        assert r["ignited"] is True
        # G3's number, through the bindings. Verification, not validation:
        # Raizer's expression is the construction the model is built from.
        assert abs(r["d_measured"] / r["d_raizer"] - 1.0) < 0.01
        assert r["energy_residual"] < 1e-10
        assert r["boundaries_undisturbed"] is True
        assert r["profiles"].shape == (4, 5, 2500)
        assert r["quantities"] == [
            "p_Pa", "rho_kg_m3", "u_m_s", "alpha_1_m", "I_W_m2",
        ]

    def test_lsd_below_threshold_reports_cleanly(self):
        """Not an error and not a hang: the spec's failure-mode contract has to
        cross the FFI intact, or a caller learns about it as an exception."""
        r = bp.run_lsd(ignite_power=1.0)
        assert r["ignited"] is False
        assert r["profiles"].size == 0
        assert np.isnan(r["d_measured"])

    def test_lsd_headline_gap_survives_the_binding(self):
        """The sustaining drive sits orders of magnitude below the intensity
        that could light the wave. Pinned here too, so a binding that silently
        swapped the two intensities would fail rather than look plausible."""
        r = bp.run_lsd(frames=2)
        assert r["drive"] / r["i_threshold"] < 1e-4

    def test_ignition_carries_its_binomial_error(self):
        """p_ignite is a Bernoulli mean; a caller plotting it without the error
        bar is over-claiming, so the binding must hand both back together."""
        n = 6
        r = bp.run_ignition(points=3, realizations=n, n=128, dx=4e-3)
        p = np.asarray(r["p_ignite"])
        se = np.asarray(r["p_ignite_se"])
        assert p.shape == se.shape == (3,)
        assert np.allclose(se, np.sqrt(p * (1 - p) / n))
        assert r["focal_ratio"].shape == (3, n)
        # The two degradations stay distinct across the FFI. Note neither
        # bounds the other: they have different denominators — the Strehl
        # normalises against this beam's own pupil amplitude, the ratio against
        # the vacuum run — and turbulence flattens the pupil amplitude, which
        # raises the Strehl's denominator. What does hold is the triangle
        # inequality on the coherent sum, and that both fall as turbulence
        # strengthens.
        assert np.all(r["strehl"] > 0) and np.all(r["strehl"] <= 1.0 + 1e-12)
        assert np.all(r["focal_ratio"] > 0)
        assert r["focal_ratio"].mean(axis=1)[0] > r["focal_ratio"].mean(axis=1)[-1]
        assert r["strehl"].mean(axis=1)[0] > r["strehl"].mean(axis=1)[-1]

    def test_ignition_is_seed_reproducible(self):
        kw = dict(points=2, realizations=4, n=128, dx=4e-3)
        a = bp.run_ignition(seed=11, **kw)
        b = bp.run_ignition(seed=11, **kw)
        c = bp.run_ignition(seed=12, **kw)
        assert np.array_equal(a["focal_ratio"], b["focal_ratio"])
        assert not np.array_equal(a["focal_ratio"], c["focal_ratio"])

    def test_breakdown_shapes_and_units(self):
        r = bp.run_breakdown(points=5, steps=80)
        assert r["ne_traces"].shape == (5, 80)
        assert len(r["pressure_torr"]) == 5
        # W/cm^2, the unit the literature is quoted in, as the CLI writes.
        assert 1e10 < r["drive_intensity"] < 1e14
        assert r["n_seed"] < r["n_bd"]

    def test_m6_validity_errors_are_value_errors(self):
        """The refuse-do-not-mis-model guards must arrive as ValueError with
        their message intact, not as a panic across the FFI."""
        with pytest.raises(ValueError, match="unresolved"):
            bp.run_lsd(cells=200)          # absorption layer under-resolved
        with pytest.raises(ValueError, match="cross"):
            bp.run_lsd(cross=0.9)          # front would reach the boundary
        with pytest.raises(ValueError, match="aperture"):
            bp.run_ignition(aperture=50.0, points=2, realizations=2)
        with pytest.raises(ValueError):
            bp.run_ignition(points=1, realizations=2)
        with pytest.raises(ValueError):
            bp.run_breakdown(p_min_torr=2000.0, p_max_torr=300.0)
