//! M1 + M2 validation gates.
//!
//! The contract is explicit that a Gaussian check alone under-tests the
//! propagator (it is the most forgiving input), so the M1 suite is:
//! Gaussian-beam accuracy (<1% on width/Rayleigh evolution and far-field
//! divergence) **plus** power conservation (~1e-13), boundary wraparound,
//! order-of-accuracy (slope ≈ 2), medium-trait interchangeability, and the
//! long-throw Fresnel impulse-response path.
//!
//! The M2 gate anchors Beer–Lambert extinction to its closed form: uniform
//! extinction must reproduce `T = exp(−α·z)` to near machine precision
//! without touching the beam shape, `α = 0` must be bit-identical to vacuum,
//! and a transversely varying absorber must remove exactly the power its
//! profile predicts.

use ndarray::Array2;

use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::{ConstantDeltaN, Medium, UniformExtinction, Vacuum};
use beamprop::montecarlo::seeded_ensemble;
use beamprop::propagate::{DiffractionMethod, Propagator, beam_width, centroid};
use beamprop::turbulence::{ScreenGenerator, TurbulentPath};
use beamprop::validate::{
    GaussianBeam, fried_r0, kolmogorov_structure_function, observed_order, rytov_variance,
};

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

// ------------------------------------------------------------------------
// M2 gate: Beer–Lambert attenuation.
// ------------------------------------------------------------------------

/// M2 headline check: uniform extinction reproduces the closed form
/// `P(z) = P0·exp(−α·z)` to near machine precision, and — because uniform
/// loss is a pure scalar factor — the beam still diffracts exactly like the
/// analytic free-space Gaussian.
#[test]
fn beer_lambert_matches_closed_form() {
    let grid = Grid::new(256, 1e-3);
    let wavelength = 1e-6;
    let w0 = 8e-3;
    let beam = GaussianBeam { w0, wavelength };
    let alpha = 0.02; // 1/m
    let z = 50.0; // α·z = 1 → T = 1/e

    let mut field = Field::gaussian(grid, wavelength, w0);
    let p0 = field.power();
    let mut prop = Propagator::new(grid, wavelength).unwrap();
    let medium = UniformExtinction::new(grid.n, alpha);
    prop.propagate(&mut field, &medium, 1.0, 0, 50, |_, _| {})
        .unwrap();

    let t_num = field.power() / p0;
    let t_ref = (-alpha * z).exp();
    let rel = (t_num - t_ref).abs() / t_ref;
    assert!(
        rel < 1e-12,
        "transmission {t_num:.15e} vs exp(−α·z) = {t_ref:.15e} ({rel:.2e})"
    );

    // Uniform loss must not touch the shape: width still analytic to <1%.
    let (wx, wy) = beam_width(&field);
    let w_ref = beam.width_at(z);
    for w in [wx, wy] {
        assert!(
            (w - w_ref).abs() / w_ref < 0.01,
            "width under uniform loss: {w:.6e} vs {w_ref:.6e}"
        );
    }
}

/// `α = 0` must be *identical* to vacuum — the lossless path is untouched by
/// the M2 change.
#[test]
fn zero_extinction_matches_vacuum() {
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
    let lossless = run(&UniformExtinction::new(grid.n, 0.0));
    let max_diff = vac
        .u
        .iter()
        .zip(lossless.u.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        max_diff == 0.0,
        "UniformExtinction(0) differs from Vacuum by {max_diff:.3e}"
    );
}

/// A transversely varying absorber column: `α(x, y) = α0·exp(−r²/(2s²))`.
struct GaussianAbsorber {
    grid: Grid,
    alpha0: f64,
    sigma: f64,
}

impl Medium for GaussianAbsorber {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::zeros((self.grid.n, self.grid.n))
    }

    fn extinction(&self, _z_slab: usize) -> Option<Array2<f64>> {
        let g = self.grid;
        Some(Array2::from_shape_fn((g.n, g.n), |(iy, ix)| {
            let x = g.coord(ix);
            let y = g.coord(iy);
            self.alpha0 * (-(x * x + y * y) / (2.0 * self.sigma * self.sigma)).exp()
        }))
    }
}

/// Spatially varying extinction removes exactly the power its profile
/// predicts: over one thin slab (dz ≪ zR, so diffraction barely moves the
/// intensity) the surviving power is `Σ I·exp(−α(x,y)·dz)·dx²`.
#[test]
fn transverse_extinction_removes_predicted_power() {
    let grid = Grid::new(256, 1e-3);
    let wavelength = 1e-6;
    let w0 = 8e-3; // zR ≈ 201 m ≫ dz
    let dz = 0.1;
    let absorber = GaussianAbsorber {
        grid,
        alpha0: 5.0, // e^{-0.5} ≈ 0.61 on-axis transmission over the slab
        sigma: 6e-3,
    };

    let mut field = Field::gaussian(grid, wavelength, w0);
    let dx2 = grid.dx * grid.dx;
    let alpha = absorber.extinction(0).unwrap();
    let p_expected: f64 = field
        .intensity()
        .iter()
        .zip(alpha.iter())
        .map(|(&i, &a)| i * (-a * dz).exp() * dx2)
        .sum();

    let mut prop = Propagator::new(grid, wavelength).unwrap();
    prop.propagate(&mut field, &absorber, dz, 0, 1, |_, _| {})
        .unwrap();

    let rel = (field.power() - p_expected).abs() / p_expected;
    assert!(
        rel < 1e-6,
        "power {:.9e} vs predicted {p_expected:.9e} ({rel:.2e})",
        field.power()
    );

    // The on-axis absorber is symmetric: the centroid must stay put.
    let (cx, cy) = centroid(&field);
    assert!(cx.abs() < grid.dx / 10.0 && cy.abs() < grid.dx / 10.0);
}

// ------------------------------------------------------------------------
// M3 gate: turbulence phase screens + Monte-Carlo.
// ------------------------------------------------------------------------

/// M3 screen check: the ensemble structure function of generated screens
/// matches the Kolmogorov `D_φ(r) = 6.88·(r/r0)^(5/3)` across more than a
/// decade of separations. The subharmonic compensation is what makes the
/// large-separation lags pass; FFT-only screens fall tens of percent short
/// there.
#[test]
fn phase_screen_structure_function_matches_kolmogorov() {
    let grid = Grid::new(256, 0.02); // extent 5.12 m
    let r0 = 0.1;
    let l0_outer = 1e4; // effectively infinite: Kolmogorov regime for all lags
    let n_screens = 160;

    use rand::SeedableRng;
    let mut generator = ScreenGenerator::new(grid, r0, l0_outer, true);
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(20260716);
    let screens: Vec<_> = (0..n_screens)
        .map(|_| generator.generate(&mut rng))
        .collect();

    // D(r) estimated over both axes, all screens, all non-wrapping pairs.
    let estimate = |lag: usize| -> f64 {
        let n = grid.n;
        let (mut sum, mut count) = (0.0, 0u64);
        for s in &screens {
            for iy in 0..n {
                for ix in 0..n - lag {
                    let d = s[[iy, ix + lag]] - s[[iy, ix]];
                    sum += d * d;
                    count += 1;
                }
            }
            for iy in 0..n - lag {
                for ix in 0..n {
                    let d = s[[iy + lag, ix]] - s[[iy, ix]];
                    sum += d * d;
                    count += 1;
                }
            }
        }
        sum / count as f64
    };

    for lag in [2usize, 4, 8, 16, 32] {
        let r = lag as f64 * grid.dx;
        let d_num = estimate(lag);
        let d_ref = kolmogorov_structure_function(r, r0);
        let rel = (d_num - d_ref).abs() / d_ref;
        println!("D(r={r:.2}) = {d_num:.2} vs {d_ref:.2} ({rel:.3})");
        assert!(
            rel < 0.10,
            "D(r = {r:.2} m): {d_num:.2} vs Kolmogorov {d_ref:.2} rad^2 ({rel:.3})"
        );
    }
}

/// M3 propagation check: the long-exposure (ensemble-mean) beam radius after
/// 1 km of moderate turbulence matches the Andrews–Phillips weak-fluctuation
/// prediction `W_LT = W(z)·sqrt(1 + 1.33·sigma_R^2·Lambda^(5/6))`.
#[test]
fn long_exposure_beam_spread_matches_theory() {
    let grid = Grid::new(256, 2e-3); // extent 0.512 m
    let wavelength = 1e-6;
    let w0 = 1e-2;
    let beam = GaussianBeam { w0, wavelength };
    let z = 1000.0;
    let n_screens = 10;
    let n_real = 64;
    // sigma_R^2 = 0.5: strong enough to measure, weak enough for the theory.
    let cn2 = 0.5 / rytov_variance(1.0, wavelength, z);
    let l0_outer = 1e4;

    let mean_intensity = seeded_ensemble(n_real, |i| {
        let path = TurbulentPath::new(grid, wavelength, cn2, l0_outer, z, n_screens, 71, i);
        let mut field = Field::gaussian(grid, wavelength, w0);
        let mut prop = Propagator::new(grid, wavelength).unwrap();
        prop.propagate(&mut field, &path, path.dz(), 0, n_screens, |_, _| {})
            .unwrap();
        field.intensity()
    })
    .into_iter()
    .fold(Array2::<f64>::zeros((grid.n, grid.n)), |acc, i| acc + i);

    // Long-exposure width = 1/e^2 radius of the ensemble-mean profile,
    // extracted by fitting ln I against r^2 over the beam core (> 10% of
    // peak). Theory quotes the Gaussian-equivalent radius of the mean
    // irradiance; a second-moment estimate would be inflated by the faint
    // wide-angle scattered halo the formula does not describe.
    let mid = grid.n / 2;
    let peak = mean_intensity[[mid, mid]];
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
    let mut widths = Vec::new();
    for axis_is_x in [true, false] {
        let samples: Vec<(f64, f64)> = (0..grid.n)
            .filter_map(|i| {
                let v = if axis_is_x {
                    mean_intensity[[mid, i]]
                } else {
                    mean_intensity[[i, mid]]
                };
                (v > 0.1 * peak).then(|| (grid.coord(i), v))
            })
            .collect();
        assert!(samples.len() > 10);
        widths.push(fit_width(&samples));
    }
    let w_num = 0.5 * (widths[0] + widths[1]);

    let w_ref = beam.long_exposure_width(z, cn2);
    let w_vac = beam.width_at(z);
    let rel = (w_num - w_ref).abs() / w_ref;
    println!("W_LT {w_num:.4e} vs theory {w_ref:.4e} ({rel:.3}); vacuum {w_vac:.4e}");
    assert!(
        rel < 0.05,
        "long-exposure width {w_num:.4e} vs theory {w_ref:.4e} ({rel:.3}); vacuum {w_vac:.4e}"
    );
    // The measured spread must be a real turbulence effect, not a pass by
    // proximity to the vacuum width.
    assert!(
        w_num > w_vac * (1.0 + 0.5 * (w_ref / w_vac - 1.0)),
        "turbulent spread too weak: {w_num:.4e} vs vacuum {w_vac:.4e}, theory {w_ref:.4e}"
    );
}

/// M3 scintillation check: on-axis (plane-wave) scintillation index in weak
/// fluctuation matches the Rytov variance sigma_I^2 ~ sigma_R^2.
///
/// Uses a periodic plane wave with FFT-only screens (no subharmonics) and no
/// absorbing boundary: scintillation is driven by Fresnel-scale eddies, which
/// the FFT grid covers, and periodicity keeps the plane wave statistically
/// homogeneous so every pixel samples the same statistics.
#[test]
fn scintillation_index_matches_rytov_weak_theory() {
    let grid = Grid::new(256, 5e-3); // extent 1.28 m >> Fresnel zone 4.5 cm
    let wavelength = 1e-6;
    let z = 2000.0;
    let n_screens = 16;
    let n_real = 64;
    let sigma_r2 = 0.2; // weak-fluctuation regime
    let cn2 = sigma_r2 / rytov_variance(1.0, wavelength, z);
    let dz = z / n_screens as f64;
    let r0_slab = fried_r0(cn2, wavelength, dz);

    let sums = seeded_ensemble(n_real, |i| {
        use rand::SeedableRng;
        let mut generator = ScreenGenerator::new(grid, r0_slab, 1e4, false);
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(2029);
        rng.set_stream(i);
        let screens: Vec<_> = (0..n_screens)
            .map(|_| generator.generate(&mut rng))
            .collect();
        let path = TurbulentPath::from_screens(screens, wavelength, dz);

        let mut field = Field {
            grid,
            wavelength,
            u: Array2::from_elem((grid.n, grid.n), num_complex::Complex64::new(1.0, 0.0)),
        };
        let mut prop = Propagator::new(grid, wavelength)
            .unwrap()
            .without_boundary();
        for slab in 0..n_screens {
            prop.step(&mut field, &path, slab, dz).unwrap();
        }
        let inten = field.intensity();
        let s1: f64 = inten.sum();
        let s2: f64 = inten.mapv(|v| v * v).sum();
        (s1, s2)
    });

    // Fixed-order reduction over realizations (T4 discipline).
    let (sum_i, sum_i2) = sums
        .iter()
        .fold((0.0, 0.0), |(a, b), &(s1, s2)| (a + s1, b + s2));
    let n_samples = (n_real * grid.n * grid.n) as f64;
    let mean_i = sum_i / n_samples;
    let scint = sum_i2 / n_samples / (mean_i * mean_i) - 1.0;

    let rel = (scint - sigma_r2).abs() / sigma_r2;
    println!("scintillation index {scint:.4} vs Rytov {sigma_r2:.4} ({rel:.3})");
    assert!(
        rel < 0.15,
        "scintillation index {scint:.4} vs Rytov {sigma_r2:.4} ({rel:.3})"
    );
}

/// T4 verify: the same master seed gives bitwise-identical Monte-Carlo
/// results on 1 and 4 threads — per-realization ChaCha streams plus a
/// fixed-order reduction make the ensemble independent of scheduling.
#[test]
fn monte_carlo_reproducible_across_thread_counts() {
    let grid = Grid::new(64, 1e-3);
    let wavelength = 1e-6;
    let w0 = 8e-3;
    let z = 200.0;
    let n_screens = 4;

    let run = |threads: usize| -> Array2<f64> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                seeded_ensemble(8, |i| {
                    let path =
                        TurbulentPath::new(grid, wavelength, 1e-15, 1e3, z, n_screens, 4242, i);
                    let mut field = Field::gaussian(grid, wavelength, w0);
                    let mut prop = Propagator::new(grid, wavelength).unwrap();
                    prop.propagate(&mut field, &path, path.dz(), 0, n_screens, |_, _| {})
                        .unwrap();
                    field.intensity()
                })
                .into_iter()
                .fold(Array2::<f64>::zeros((grid.n, grid.n)), |acc, i| acc + i)
            })
    };

    let a = run(1);
    let b = run(4);
    assert_eq!(a, b, "MC ensemble differs between 1 and 4 threads");
}
