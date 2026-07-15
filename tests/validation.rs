//! M1 validation gate.
//!
//! The contract is explicit that a Gaussian check alone under-tests the
//! propagator (it is the most forgiving input), so this suite is:
//! Gaussian-beam accuracy (<1% on width/Rayleigh evolution and far-field
//! divergence) **plus** power conservation (~1e-13), boundary wraparound,
//! order-of-accuracy (slope ≈ 2), medium-trait interchangeability, and the
//! long-throw Fresnel impulse-response path.

use ndarray::Array2;

use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::{ConstantDeltaN, Medium, Vacuum};
use beamprop::propagate::{DiffractionMethod, Propagator, beam_width, centroid};
use beamprop::validate::{GaussianBeam, observed_order};

/// A smooth defocusing Gaussian duct: `δn(r) = -A·exp(-r²/(2s²))`.
///
/// z-invariant, so the split-step error is pure operator-splitting error —
/// exactly what the order-of-accuracy test needs to isolate.
struct GaussianDuct {
    grid: Grid,
    amplitude: f64,
    sigma: f64,
}

impl Medium for GaussianDuct {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        let g = self.grid;
        Array2::from_shape_fn((g.n, g.n), |(iy, ix)| {
            let x = g.coord(ix);
            let y = g.coord(iy);
            -self.amplitude * (-(x * x + y * y) / (2.0 * self.sigma * self.sigma)).exp()
        })
    }
}

/// M1 headline check: free-space Gaussian evolution matches the analytic
/// `w(z) = w0·√(1 + (z/zR)²)` to <1% through the near field, and the
/// far-field expansion slope matches `θ = λ/(π·w0)` to <1%.
#[test]
fn gaussian_free_space_evolution() {
    let grid = Grid::new(512, 1e-3);
    let wavelength = 1e-6;
    let w0 = 8e-3;
    let beam = GaussianBeam { w0, wavelength };
    let zr = beam.rayleigh_range();

    let mut field = Field::gaussian(grid, wavelength, w0);
    let mut prop = Propagator::new(grid, wavelength).unwrap();
    let vacuum = Vacuum::new(grid.n);

    // Near field: 100 steps to 2·zR, checking at 0.5, 1, and 2 Rayleigh ranges.
    let dz = 2.0 * zr / 100.0;
    let mut checked = 0;
    prop.propagate(&mut field, &vacuum, dz, 0, 100, |i, f| {
        let step = i + 1;
        if step == 25 || step == 50 || step == 100 {
            let z = step as f64 * dz;
            let (wx, wy) = beam_width(f);
            let w_ref = beam.width_at(z);
            for w in [wx, wy] {
                let rel = (w - w_ref).abs() / w_ref;
                assert!(
                    rel < 0.01,
                    "width at z = {z:.1} m: {w:.6e} vs {w_ref:.6e} ({rel:.2e})"
                );
            }
            checked += 1;
        }
    })
    .unwrap();
    assert_eq!(checked, 3);

    // Centroid must not drift in free space.
    let (cx, cy) = centroid(&field);
    assert!(cx.abs() < grid.dx / 10.0 && cy.abs() < grid.dx / 10.0);

    // Far field: continue to 10·zR in 2·zR steps (still below z_c = 512 m).
    let dz_far = 2.0 * zr;
    assert_eq!(prop.method_for(dz_far), DiffractionMethod::AngularSpectrum);
    let (mut w6, mut w10) = (0.0, 0.0);
    prop.propagate(&mut field, &vacuum, dz_far, 100, 4, |i, f| {
        let step = i - 100 + 1; // 1..=4 → z = (2 + 2·step)·zR
        if step == 2 {
            w6 = beam_width(f).0;
        }
        if step == 4 {
            w10 = beam_width(f).0;
        }
    })
    .unwrap();
    let w10_ref = beam.width_at(10.0 * zr);
    assert!(
        (w10 - w10_ref).abs() / w10_ref < 0.01,
        "w(10·zR) = {w10:.6e} vs {w10_ref:.6e}"
    );
    // expansion slope between 6·zR and 10·zR vs the divergence angle
    let theta_num = (w10 - w6) / (4.0 * zr);
    let theta = beam.divergence();
    assert!(
        (theta_num - theta).abs() / theta < 0.01,
        "divergence {theta_num:.6e} vs {theta:.6e}"
    );
}

/// Lossless propagation must conserve power to near machine precision: the
/// angular-spectrum kernel has |H| = 1 and the boundary mask is exactly 1
/// where the beam lives.
#[test]
fn power_conservation_lossless() {
    let grid = Grid::new(256, 1e-3);
    let mut field = Field::gaussian(grid, 1e-6, 8e-3);
    let p0 = field.power();
    let mut prop = Propagator::new(grid, 1e-6).unwrap();
    let vacuum = Vacuum::new(grid.n);
    prop.propagate(&mut field, &vacuum, 1.0, 0, 50, |_, _| {})
        .unwrap();
    let drift = (field.power() - p0).abs() / p0;
    assert!(drift < 1e-13, "relative power drift {drift:.3e}");
}

/// The absorbing boundary must prevent FFT wraparound: a tilted beam walking
/// off the +x edge must be absorbed, not re-enter from -x. The same run
/// without the boundary demonstrates the failure mode being prevented.
#[test]
fn boundary_absorbs_instead_of_wrapping() {
    let grid = Grid::new(256, 1e-3);
    let wavelength = 1e-6;
    let tilt = 2e-4; // rad; center walks 120 mm over 600 m
    let k = 2.0 * std::f64::consts::PI / wavelength;

    let make_tilted = || {
        let mut f = Field::gaussian(grid, wavelength, 8e-3);
        for ((_, ix), u) in f.u.indexed_iter_mut() {
            *u *= num_complex::Complex64::from_polar(1.0, k * tilt * grid.coord(ix));
        }
        f
    };

    // Power that ends up in the strip x < -60 mm (opposite side of the walk).
    let wrapped_fraction = |f: &Field| {
        let inten = f.intensity();
        let mut wrapped = 0.0;
        let mut total = 0.0;
        for ((_, ix), &p) in inten.indexed_iter() {
            total += p;
            if grid.coord(ix) < -60e-3 {
                wrapped += p;
            }
        }
        wrapped / total
    };

    let vacuum = Vacuum::new(grid.n);

    let mut guarded = make_tilted();
    let p0 = guarded.power();
    let mut prop = Propagator::new(grid, wavelength).unwrap();
    prop.propagate(&mut guarded, &vacuum, 10.0, 0, 60, |_, _| {})
        .unwrap();

    let mut unguarded = make_tilted();
    let mut prop_raw = Propagator::new(grid, wavelength)
        .unwrap()
        .without_boundary();
    prop_raw
        .propagate(&mut unguarded, &vacuum, 10.0, 0, 60, |_, _| {})
        .unwrap();

    let frac_guarded = wrapped_fraction(&guarded);
    let frac_unguarded = wrapped_fraction(&unguarded);

    // The guard band absorbed real power (the beam did reach the edge)...
    assert!(
        guarded.power() < 0.99 * p0,
        "beam never reached the boundary; test is vacuous"
    );
    // ...nothing re-entered on the far side...
    assert!(
        frac_guarded < 1e-8,
        "guarded wrapped fraction {frac_guarded:.3e}"
    );
    // ...whereas the unguarded FFT wraps visibly.
    assert!(
        frac_unguarded > 1e-4,
        "unguarded wrapped fraction {frac_unguarded:.3e}"
    );
    assert!(frac_unguarded > 1e3 * frac_guarded);
}

/// Symmetric (Strang) splitting must converge at 2nd order in dz. Propagates
/// through a smooth defocusing duct at dz, dz/2, dz/4 against a dz/32
/// reference and checks the observed order on both refinement pairs.
#[test]
fn split_step_is_second_order() {
    let grid = Grid::new(128, 2e-3);
    let wavelength = 1e-6;
    let w0 = 20e-3;
    let z_total = 400.0;
    let duct = GaussianDuct {
        grid,
        amplitude: 5e-9,
        sigma: 30e-3,
    };

    let run = |n_steps: usize| {
        let mut f = Field::gaussian(grid, wavelength, w0);
        let mut prop = Propagator::new(grid, wavelength).unwrap();
        prop.propagate(
            &mut f,
            &duct,
            z_total / n_steps as f64,
            0,
            n_steps,
            |_, _| {},
        )
        .unwrap();
        f
    };

    let reference = run(8 * 32);
    let l2_err = |f: &Field| {
        f.u.iter()
            .zip(reference.u.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
    };

    let e1 = l2_err(&run(8));
    let e2 = l2_err(&run(16));
    let e3 = l2_err(&run(32));

    assert!(
        e1 > 1e-12,
        "error {e1:.3e} too close to noise floor to measure order"
    );
    let p12 = observed_order(e1, e2);
    let p23 = observed_order(e2, e3);
    assert!(
        (1.75..=2.25).contains(&p12),
        "observed order {p12:.3} (e1={e1:.3e}, e2={e2:.3e})"
    );
    assert!(
        (1.75..=2.25).contains(&p23),
        "observed order {p23:.3} (e2={e2:.3e}, e3={e3:.3e})"
    );
}

/// T2 verify: different `Medium` implementations flow through the same
/// propagator. `ConstantDeltaN(0)` must equal `Vacuum` exactly, and a uniform
/// nonzero δn is a pure global phase — identical intensity to vacuum.
#[test]
fn medium_trait_interchangeability() {
    let grid = Grid::new(128, 1e-3);
    let wavelength = 1e-6;
    let run = |medium: &dyn Medium| {
        let mut f = Field::gaussian(grid, wavelength, 10e-3);
        let mut prop = Propagator::new(grid, wavelength).unwrap();
        prop.propagate(&mut f, medium, 5.0, 0, 20, |_, _| {})
            .unwrap();
        f
    };

    let vac = run(&Vacuum::new(grid.n));
    let zero = run(&ConstantDeltaN::new(grid.n, 0.0));
    let uniform = run(&ConstantDeltaN::new(grid.n, 1e-6));

    // δn = 0 must be *identical* to vacuum
    let max_diff = vac
        .u
        .iter()
        .zip(zero.u.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        max_diff < 1e-14,
        "ConstantDeltaN(0) differs from Vacuum by {max_diff:.3e}"
    );

    // uniform δn: same intensity, global phase only
    let max_int_diff = vac
        .intensity()
        .iter()
        .zip(uniform.intensity().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        max_int_diff < 1e-12,
        "uniform δn changed intensity by {max_int_diff:.3e}"
    );
}

/// Long-throw sampling: beyond `z_c` the Fresnel impulse-response kernel is
/// selected and must still reproduce the analytic Gaussian to <1%.
///
/// The width is extracted by fitting `ln I` against `r²` over the beam core
/// (a Gaussian is a straight line there) rather than by second moment: the IR
/// kernel's sampled chirp leaves a faint wide-field halo whose `r²`-weighted
/// contribution corrupts a moment-based width while being physically
/// irrelevant to the beam itself. Peak intensity is checked independently.
#[test]
fn fresnel_impulse_response_long_throw() {
    let grid = Grid::new(512, 1e-3);
    let wavelength = 1e-6;
    let w0 = 8e-3;
    let beam = GaussianBeam { w0, wavelength };

    let mut prop = Propagator::new(grid, wavelength).unwrap();
    let z = 800.0; // z_c = 512 m for this grid
    assert_eq!(
        prop.method_for(z),
        DiffractionMethod::FresnelImpulseResponse
    );

    let mut field = Field::gaussian(grid, wavelength, w0);
    prop.diffract(&mut field, z).unwrap();

    let w_ref = beam.width_at(z);
    let inten = field.intensity();
    let mid = grid.n / 2;
    let peak = inten[[mid, mid]];

    // Peak: analytic on-axis intensity of a unit-amplitude Gaussian is (w0/w)².
    let peak_ref = (w0 / w_ref).powi(2);
    let peak_rel = (peak - peak_ref).abs() / peak_ref;
    assert!(
        peak_rel < 0.01,
        "IR peak {peak:.6e} vs {peak_ref:.6e} ({peak_rel:.2e})"
    );

    // Width: least-squares fit of ln I = ln I0 − 2·x²/w² along the central
    // row, over the core where I > 10% of peak.
    let fit_width = |samples: &[(f64, f64)]| {
        let m = samples.len() as f64;
        let (mut st, mut sy, mut stt, mut sty) = (0.0, 0.0, 0.0, 0.0);
        for &(x, i) in samples {
            let (t, y) = (x * x, i.ln());
            st += t;
            sy += y;
            stt += t * t;
            sty += t * y;
        }
        let slope = (sty - st * sy / m) / (stt - st * st / m);
        (-2.0 / slope).sqrt()
    };
    for (axis_is_x, w_axis) in [(true, "x"), (false, "y")] {
        let samples: Vec<(f64, f64)> = (0..grid.n)
            .filter_map(|i| {
                let v = if axis_is_x {
                    inten[[mid, i]]
                } else {
                    inten[[i, mid]]
                };
                (v > 0.1 * peak).then(|| (grid.coord(i), v))
            })
            .collect();
        assert!(samples.len() > 10);
        let w_fit = fit_width(&samples);
        let rel = (w_fit - w_ref).abs() / w_ref;
        assert!(
            rel < 0.01,
            "IR fitted width ({w_axis}) at z = {z} m: {w_fit:.6e} vs {w_ref:.6e} ({rel:.2e})"
        );
    }
}
