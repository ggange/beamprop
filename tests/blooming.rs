//! M4 gate: coupled thermal blooming.
//!
//! Tiered against `docs/M4_SPEC.md`:
//! - **B1** closed-form single-slab blooming phase (tight, no propagation);
//! - **upwind sign** — the beam bends into the wind, not with it;
//! - **order of accuracy** — the slab-centre coupling stays 2nd order (a
//!   slab-entrance coupling would drop to 1st, which this catches);
//! - **B2** weak-blooming limit — coupled deficit is linear in N_φ with an
//!   O(N_φ²) residual, checked against an analytic first-order reference;
//! - **stability** — a strong-blooming run stays finite and dz-convergent;
//! - **B3** qualitative signatures (upwind shift, crescent, irradiance
//!   rollover); the quantitative Smith/Gebhardt curve is `#[ignore]`d pending
//!   the digitized figure.

use ndarray::Array2;

use beamprop::airprops::{AirProperties, AirTable};
use beamprop::blooming::ThermalBlooming;
use beamprop::field::Field;
use beamprop::grid::Grid;
use beamprop::medium::Medium;
use beamprop::propagate::{Propagator, centroid};
use beamprop::turbulence::TurbulentPath;
use beamprop::validate::{BloomingCase, GaussianBeam, observed_order};

const T0: f64 = 288.15;
const P0: f64 = 101_325.0;
const WAVELENGTH: f64 = 1e-6;
const ALPHA_ABS: f64 = 1e-4;

fn standard_air() -> AirProperties {
    AirTable::load().unwrap().at(T0, P0, WAVELENGTH).unwrap()
}

/// Build a `BloomingCase` (the analytic reference bundle) matching a run.
fn case(air: AirProperties, power: f64, w: f64, wind: f64) -> BloomingCase {
    BloomingCase {
        alpha_abs: ALPHA_ABS,
        power,
        w,
        wind,
        rho: air.rho,
        cp: air.cp,
        n_minus_1: air.n_minus_1,
        t0: T0,
        wavelength: WAVELENGTH,
    }
}

/// B1: the medium's single-slab δn reproduces the closed-form crosswind
/// blooming phase to < 0.5 % wherever the beam carries power. Pure quadrature
/// + interpolation, so this is a correctness gate, not a physics tolerance.
#[test]
fn b1_closed_form_blooming_phase() {
    let grid = Grid::new(512, 0.25e-3); // extent 128 mm
    let w = 10e-3;
    let wind = 1.0;
    let power = 1e4;
    let path_len = 500.0;
    let air = standard_air();

    let field = Field::gaussian(grid, WAVELENGTH, w);
    let bloom = ThermalBlooming::new(
        grid,
        air,
        ALPHA_ABS,
        wind,
        power,
        field.power(),
        w,
        T0,
    )
    .unwrap();
    // dz = 0: no midpoint absorption factor — a pure quadrature gate on the
    // medium's temperature integral against the closed form.
    let dn = bloom.index_response(0, &field.intensity(), 0.0);

    let c = case(air, power, w, wind);
    let k = 2.0 * std::f64::consts::PI / WAVELENGTH;
    let inten = field.intensity();
    let i0 = inten[[grid.n / 2, grid.n / 2]];

    let mut max_rel = 0.0_f64;
    let mut checked = 0u64;
    for iy in 0..grid.n {
        let y = grid.coord(iy);
        for ix in 0..grid.n {
            if inten[[iy, ix]] <= 1e-6 * i0 {
                continue;
            }
            let x = grid.coord(ix);
            let phase_num = k * dn[[iy, ix]] * path_len;
            let phase_ref = c.phase_ref(x, y, path_len);
            let rel = (phase_num - phase_ref).abs() / phase_ref.abs().max(1e-30);
            max_rel = max_rel.max(rel);
            checked += 1;
        }
    }
    println!("B1 max rel error {max_rel:.2e} over {checked} points");
    assert!(checked > 1000, "B1 sampled too few points ({checked})");
    assert!(
        max_rel < 5e-3,
        "B1 blooming phase max relative error {max_rel:.2e} over {checked} points"
    );
}

/// Shared collimated-beam setup for the coupled gates: w0 = 5 cm, z = 500 m,
/// v = 2 m/s. zR ≈ 7.9 km ≫ z, so the beam is effectively collimated and stays
/// well inside the guard interior.
struct Setup {
    grid: Grid,
    w0: f64,
    wind: f64,
    z: f64,
    air: AirProperties,
}

fn coupled_setup() -> Setup {
    Setup {
        grid: Grid::new(512, 1e-3), // extent 512 mm, interior ±205 mm
        w0: 5e-2,
        wind: 2.0,
        z: 500.0,
        air: standard_air(),
    }
}

impl Setup {
    /// Beam power giving distortion number `n_phi` over the path.
    fn power_for(&self, n_phi: f64) -> f64 {
        case(self.air, 1.0, self.w0, self.wind).power_for_distortion(n_phi, self.z)
    }

    /// Propagate a collimated Gaussian through blooming at `power` in `steps`
    /// slabs; return the receiver field.
    fn run(&self, power: f64, steps: usize) -> Field {
        let mut field = Field::gaussian(self.grid, WAVELENGTH, self.w0);
        let bloom = ThermalBlooming::new(
            self.grid,
            self.air,
            ALPHA_ABS,
            self.wind,
            power,
            field.power(),
            self.w0,
            T0,
        )
        .unwrap();
        let mut prop = Propagator::new(self.grid, WAVELENGTH).unwrap();
        prop.propagate(&mut field, &bloom, self.z / steps as f64, 0, steps, |_, _| {})
            .unwrap();
        field
    }
}

/// Sign gate: the beam bends **into** the wind. Wind is +x, so the cool,
/// dense upwind side has higher index and the centroid shifts to −x.
#[test]
fn beam_bends_upwind() {
    let s = coupled_setup();
    let power = s.power_for(2.0);
    let field = s.run(power, 100);
    let (cx, _) = centroid(&field);
    assert!(
        cx < -1e-4,
        "beam did not bend upwind: cx = {cx:.3e} m (expected negative)"
    );
}

/// The coupling stays 2nd-order in dz: a slab-entrance evaluation would be
/// 1st-order and fail this.
///
/// Measured by **self-convergence** — Cauchy differences between successive
/// resolutions, `e(n) = ‖u_n − u_2n‖` — rather than against a fixed fine
/// reference, whose finite fineness corrupts the order estimate at the finer
/// test points. For a p-th order scheme `e(n) ∝ dz^p`, so `e(n)/e(2n) → 2^p`.
#[test]
fn coupling_is_second_order() {
    let s = coupled_setup();
    let power = s.power_for(1.0);
    let l2_diff = |a: &Field, b: &Field| {
        a.u.iter()
            .zip(b.u.iter())
            .map(|(x, y)| (x - y).norm_sqr())
            .sum::<f64>()
            .sqrt()
    };
    // Successive-resolution fields; e_k = ‖u_{n_k} − u_{n_{k+1}}‖. N_φ = 1 and
    // these resolutions keep every Cauchy difference well above the ~1e-6
    // numerical floor (FFT/boundary roundoff) where the order estimate decays.
    let fields: Vec<Field> = [16, 32, 64, 128].iter().map(|&n| s.run(power, n)).collect();
    let e: Vec<f64> = fields.windows(2).map(|w| l2_diff(&w[0], &w[1])).collect();
    let p12 = observed_order(e[0], e[1]);
    let p23 = observed_order(e[1], e[2]);
    println!("self-conv errors {e:?}; orders {p12:.3}, {p23:.3}");
    assert!(e[2] > 1e-5, "error {:.3e} at noise floor", e[2]);
    assert!(
        (1.75..=2.25).contains(&p12),
        "order {p12:.3} (e0={:.3e}, e1={:.3e})",
        e[0],
        e[1]
    );
    assert!(
        (1.75..=2.25).contains(&p23),
        "order {p23:.3} (e1={:.3e}, e2={:.3e})",
        e[1],
        e[2]
    );
}

/// On-axis receiver intensity deficit `1 − I/(I_vac·T)` for a given power,
/// normalized by the Beer–Lambert transmission `T = e^(−α·z)` so the deficit
/// isolates the blooming distortion from plain absorption (which the lossless
/// first-order reference below does not carry).
fn on_axis_deficit(s: &Setup, power: f64, steps: usize, vac_peak: f64) -> f64 {
    let f = s.run(power, steps);
    let mid = s.grid.n / 2;
    let transmission = (-ALPHA_ABS * s.z).exp();
    1.0 - f.intensity()[[mid, mid]] / (vac_peak * transmission)
}

/// First-order (no back-reaction) reference: apply blooming phase screens
/// built from the *analytic vacuum* beam, via the linear TurbulentPath medium.
fn first_order_deficit(s: &Setup, power: f64, steps: usize, vac_peak: f64) -> f64 {
    let dz = s.z / steps as f64;
    let k = 2.0 * std::f64::consts::PI / WAVELENGTH;
    let beam = GaussianBeam {
        w0: s.w0,
        wavelength: WAVELENGTH,
    };
    // One phase screen per slab, from the undisturbed beam at the slab centre.
    let screens: Vec<Array2<f64>> = (0..steps)
        .map(|j| {
            let z_mid = (j as f64 + 0.5) * dz;
            let w = beam.width_at(z_mid);
            let p = power * (-ALPHA_ABS * z_mid).exp();
            let c = case(s.air, p, w, s.wind);
            Array2::from_shape_fn((s.grid.n, s.grid.n), |(iy, ix)| {
                let x = s.grid.coord(ix);
                let y = s.grid.coord(iy);
                // δn = −(n0−1)/T0·ΔT; slab phase φ = k·δn·dz.
                let delta_n = -s.air.n_minus_1 / T0 * c.delta_t_ref(x, y);
                k * delta_n * dz
            })
        })
        .collect();
    let path = TurbulentPath::from_screens(screens, WAVELENGTH, dz);
    let mut field = Field::gaussian(s.grid, WAVELENGTH, s.w0);
    let mut prop = Propagator::new(s.grid, WAVELENGTH).unwrap();
    prop.propagate(&mut field, &path, dz, 0, steps, |_, _| {})
        .unwrap();
    let mid = s.grid.n / 2;
    1.0 - field.intensity()[[mid, mid]] / vac_peak
}

/// B2: in the weak limit the coupled deficit is linear in N_φ and matches the
/// first-order reference to 1 %; the coupled−first-order gap is O(N_φ²), so
/// doubling N_φ roughly quadruples it.
#[test]
fn b2_weak_blooming_linear_limit() {
    let s = coupled_setup();
    let steps = 64;
    let mut vac = Field::gaussian(s.grid, WAVELENGTH, s.w0);
    let mut prop = Propagator::new(s.grid, WAVELENGTH).unwrap();
    prop.propagate(
        &mut vac,
        &beamprop::medium::Vacuum::new(s.grid.n),
        s.z / steps as f64,
        0,
        steps,
        |_, _| {},
    )
    .unwrap();
    let vac_peak = vac.intensity()[[s.grid.n / 2, s.grid.n / 2]];

    let p_lo = s.power_for(0.1);
    let p_hi = s.power_for(0.2);

    let d_lo = on_axis_deficit(&s, p_lo, steps, vac_peak);
    let f_lo = first_order_deficit(&s, p_lo, steps, vac_peak);
    let d_hi = on_axis_deficit(&s, p_hi, steps, vac_peak);
    let f_hi = first_order_deficit(&s, p_hi, steps, vac_peak);

    // Both deficits are positive (blooming lowers the peak).
    assert!(d_lo > 0.0 && f_lo > 0.0, "d_lo={d_lo:.3e}, f_lo={f_lo:.3e}");
    // Coupled ≈ first-order in the weak limit.
    let rel_lo = (d_lo - f_lo).abs() / f_lo;
    println!("B2 weak-limit deviation {rel_lo:.2e}; deficits d={d_lo:.3e} f={f_lo:.3e}");
    assert!(rel_lo < 0.01, "weak-limit coupled vs first-order {rel_lo:.3e}");
    // The back-reaction gap grows quadratically: gap(0.2)/gap(0.1) ≈ 4.
    let gap_lo = (d_lo - f_lo).abs();
    let gap_hi = (d_hi - f_hi).abs();
    let ratio = gap_hi / gap_lo;
    println!("B2 back-reaction gap ratio {ratio:.2}");
    assert!(
        (2.5..=6.0).contains(&ratio),
        "back-reaction gap ratio {ratio:.2} (gap_lo={gap_lo:.3e}, gap_hi={gap_hi:.3e})"
    );
}

/// Stability: a strong-blooming run (N_φ ≈ 20) stays finite, conserves power
/// to the guard+Beer–Lambert budget, and agrees under dz refinement.
#[test]
fn strong_blooming_is_stable() {
    let s = coupled_setup();
    let power = s.power_for(20.0);
    let steps = 400;

    let mut field = Field::gaussian(s.grid, WAVELENGTH, s.w0);
    let p0 = field.power();
    let bloom = ThermalBlooming::new(
        s.grid, s.air, ALPHA_ABS, s.wind, power, p0, s.w0, T0,
    )
    .unwrap();
    let mut prop = Propagator::new(s.grid, WAVELENGTH).unwrap();
    prop.propagate(&mut field, &bloom, s.z / steps as f64, 0, steps, |_, _| {})
        .unwrap();

    assert!(field.u.iter().all(|c| c.norm().is_finite()), "non-finite field");
    // Power budget: initial = final + Beer–Lambert absorbed + guard-band.
    let transmitted = (-ALPHA_ABS * s.z).exp();
    let expected = p0 * transmitted;
    let accounted = field.power() + prop.guard_absorbed();
    assert!(
        (accounted - expected).abs() / expected < 0.02,
        "power budget: accounted {accounted:.4e} vs expected {expected:.4e}"
    );

    // dz refinement: receiver intensity at 400 vs 800 steps within 10 %.
    let coarse = field.intensity();
    let fine = s.run(power, 800).intensity();
    let mid = s.grid.n / 2;
    let rel = (coarse[[mid, mid]] - fine[[mid, mid]]).abs() / fine[[mid, mid]];
    assert!(rel < 0.10, "dz refinement peak change {rel:.3e}");
}

/// B3 qualitative signatures: upwind shift of the peak, crescent asymmetry
/// along the wind axis, and monotonically falling peak irradiance with power.
#[test]
fn b3_qualitative_signatures() {
    let s = coupled_setup();
    let steps = 150;

    // Peak irradiance falls monotonically as N_φ climbs.
    let mut vac = Field::gaussian(s.grid, WAVELENGTH, s.w0);
    let mut prop = Propagator::new(s.grid, WAVELENGTH).unwrap();
    prop.propagate(
        &mut vac,
        &beamprop::medium::Vacuum::new(s.grid.n),
        s.z / steps as f64,
        0,
        steps,
        |_, _| {},
    )
    .unwrap();
    let mid = s.grid.n / 2;
    let vac_peak = vac.intensity()[[mid, mid]];

    let mut last = vac_peak;
    for n_phi in [1.0, 3.0, 6.0] {
        let f = s.run(s.power_for(n_phi), steps);
        let peak = f.intensity().iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            peak < last,
            "peak irradiance not decreasing at N_φ = {n_phi}: {peak:.3e} vs {last:.3e}"
        );
        last = peak;
    }

    // At N_φ ≈ 3: peak shifted upwind and a downwind crescent.
    let f = s.run(s.power_for(3.0), steps);
    let inten = f.intensity();
    // Location of the global peak.
    let (mut pmax, mut pix, mut piy) = (0.0, 0, 0);
    for ((iy, ix), &v) in inten.indexed_iter() {
        if v > pmax {
            pmax = v;
            pix = ix;
            piy = iy;
        }
    }
    assert!(
        s.grid.coord(pix) < 0.0,
        "peak not shifted upwind: x = {:.3e} m",
        s.grid.coord(pix)
    );
    // Crescent: along the wind axis through the peak, the downwind half-width
    // at half-max exceeds the upwind one (the beam smears downwind).
    let half = 0.5 * pmax;
    let mut up = 0usize;
    let mut ix = pix;
    while ix > 0 && inten[[piy, ix]] > half {
        ix -= 1;
        up += 1;
    }
    let mut down = 0usize;
    let mut ix = pix;
    while ix < s.grid.n - 1 && inten[[piy, ix]] > half {
        ix += 1;
        down += 1;
    }
    assert!(
        down > up,
        "no downwind crescent: down HWHM {down} samples vs up {up}"
    );
}

/// Which Smith (1977) finite-`F₀` peak-irradiance curve the quantitative gate
/// compares against. Smith's `F₀ = k·a²/z` is the collimated-beam Fresnel
/// number (`a = w₀/√2`, the 1/e amplitude radius). We reproduce it by choosing
/// the run geometry so `F₀` lands exactly on a digitized curve, rather than by
/// converting our N_φ — Smith's N_c is a *geometrical-optics* number and shares
/// no wavenumber with N_φ, so it is computed directly per run.
const B3_F0: f64 = 5.0;
const B3_Z: f64 = 500.0;

/// Waist `w₀` (1/e² intensity radius) giving the target Fresnel number `B3_F0`
/// at range `B3_Z`: `F₀ = k·a²/z`, `a = w₀/√2` ⟹ `w₀ = √(2·F₀·z/k)`.
fn b3_waist() -> f64 {
    let k = 2.0 * std::f64::consts::PI / WAVELENGTH;
    (2.0 * B3_F0 * B3_Z / k).sqrt()
}

/// A digitized (N_c, I_REL) sample of Smith's published curve.
struct SmithPoint {
    n_c: f64,
    i_rel: f64,
}

/// Load `tests/data/smith1977_F<F0>.csv` — two columns `N_c,I_rel`, one header
/// line, ascending in N_c. Returns `None` if the file is absent (gate skips).
fn load_smith_curve() -> Option<Vec<SmithPoint>> {
    let path = format!(
        "{}/tests/data/smith1977_F{}.csv",
        env!("CARGO_MANIFEST_DIR"),
        B3_F0 as u32
    );
    let text = std::fs::read_to_string(path).ok()?;
    let pts: Vec<SmithPoint> = text
        .lines()
        .map(str::trim)
        // Drop `#` comments, blanks, and any header row (non-numeric first field).
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut c = l.split(',');
            let n_c = c.next()?.trim().parse().ok()?;
            let i_rel = c.next()?.trim().parse().ok()?;
            Some(SmithPoint { n_c, i_rel })
        })
        .collect();
    (pts.len() >= 2).then_some(pts)
}

/// Linear interpolation of the digitized curve at `n_c` (curve ascending in N_c).
fn interp_i_rel(curve: &[SmithPoint], n_c: f64) -> f64 {
    let w = curve
        .windows(2)
        .find(|w| n_c >= w[0].n_c && n_c <= w[1].n_c)
        .expect("N_c outside digitized curve range");
    let t = (n_c - w[0].n_c) / (w[1].n_c - w[0].n_c);
    w[0].i_rel + t * (w[1].i_rel - w[0].i_rel)
}

/// B3 quantitative: the coupled solver reproduces Smith's (1977) whole-beam
/// steady-state peak-irradiance rollover, `I_REL(N)`, to ±15 % along the
/// `F₀ = B3_F0` curve.
///
/// **x-axis (N).** Smith plots against his effective whole-beam number
/// `N = N_c·{ (2/z²)∫₀ᶻ Q(z')⁻¹ ∫₀^{z'} e^(−αz'')/(Ω(z'')Q²(z'')) dz'' dz' }`,
/// where the braces carry absorption (`e^(−αz'')`) and diffractive spreading
/// (`Q`, `Ω`). We deliberately pick a **sub-Rayleigh** geometry: `F₀ = 5` gives
/// `z_R = k·a² = 5·z`, so `Q(z) = √(1+(z/z_R)²) ≤ 1.02` across the whole path
/// and `αz = 0.05`. The brace factor is then within a few percent of the pure
/// absorption bracket already inside `smith_distortion_number`, so `N ≈ N_c` to
/// ≲4 % — well under the ±15 % gate. We therefore read the run's N straight
/// from `BloomingCase::smith_distortion_number`.
///
/// **y-axis (I_REL).** Smith's `I_REL = I_bloomed/I_unbloomed` cancels the
/// common Beer–Lambert loss, so it → 1 at `N → 0`. Our vacuum reference carries
/// no absorption, so we divide the measured peak ratio by the transmission
/// `e^(−αz)` to match Smith's normalization (identical to the B2 fix).
///
/// Ignored until the digitized figure is supplied as
/// `tests/data/smith1977_F<F0>.csv`; enabling it is the final M4 step.
#[test]
fn b3_smith1977_curve_quantitative() {
    let curve = load_smith_curve()
        .expect("digitize Smith 1977 F₀ curve into tests/data/smith1977_F<F0>.csv");

    let grid = Grid::new(512, 1e-3);
    let w0 = b3_waist();
    let wind = 2.0;
    let air = standard_air();
    let steps = 200;
    let transmission = (-ALPHA_ABS * B3_Z).exp();

    // Bloom-free (vacuum-diffracted) reference peak at the receiver.
    let launch = Field::gaussian(grid, WAVELENGTH, w0);
    let p0 = launch.power();
    let mut vac = launch.clone();
    let mut prop = Propagator::new(grid, WAVELENGTH).unwrap();
    prop.propagate(
        &mut vac,
        &beamprop::medium::Vacuum::new(grid.n),
        B3_Z / steps as f64,
        0,
        steps,
        |_, _| {},
    )
    .unwrap();
    let mid = grid.n / 2;
    let vac_peak = vac.intensity()[[mid, mid]];

    let unit = case(air, 1.0, w0, wind);
    let mut worst = 0.0_f64;
    // Sample the rollover; all interior to the digitized curve (which ends at
    // N ≈ 1.87) so interpolation is always bracketed — no extrapolation.
    for &n in &[0.5, 1.0, 1.5, 1.8] {
        if n < curve[0].n_c || n > curve[curve.len() - 1].n_c {
            continue;
        }
        let power = unit.power_for_smith_number(n, B3_Z);
        let bloom = ThermalBlooming::new(grid, air, ALPHA_ABS, wind, power, p0, w0, T0).unwrap();
        let mut field = launch.clone();
        let mut prop = Propagator::new(grid, WAVELENGTH).unwrap();
        prop.propagate(&mut field, &bloom, B3_Z / steps as f64, 0, steps, |_, _| {})
            .unwrap();
        let peak = field.intensity().iter().cloned().fold(0.0_f64, f64::max);
        // Divide out the common absorption so I_REL isolates blooming (→1 at N→0).
        let i_rel = peak / (vac_peak * transmission);
        let reference = interp_i_rel(&curve, n);
        let rel = (i_rel - reference).abs() / reference;
        let pct = 100.0 * rel;
        println!("B3 N={n:.2}: I_REL solver {i_rel:.3} vs Smith {reference:.3} ({pct:.1}%)");
        worst = worst.max(rel);
        assert!(
            rel < 0.15,
            "B3 I_REL at N={n:.2}: {i_rel:.3} vs Smith {reference:.3} (off {pct:.1}%)"
        );
    }
    println!("B3 quantitative worst deviation {:.1}% (F₀={B3_F0})", 100.0 * worst);
}
