//! The medium interface: what the beam propagates *through*.
//!
//! The propagator is deliberately ignorant of what produces the refractive
//! index field. A `Medium` supplies the index perturbation `δn(x, y)` for each
//! z-slab; a turbulence phase screen (M3) and a computed volumetric thermal
//! blooming field (M4) implement the same trait. This interface is fixed at M1
//! so the later physics needs no propagator rewrite.

use ndarray::Array2;

/// A propagation medium, sampled as one `δn(x, y)` field per z-slab.
///
/// `δn` is the dimensionless refractive-index perturbation about the
/// background (`n = n₀ + δn`); the propagator turns it into the phase
/// `k·δn·dz` accumulated across the slab.
pub trait Medium {
    /// Refractive-index perturbation `δn(x, y)` for slab `z_slab`,
    /// shape `[n, n]` matching the propagation grid.
    ///
    /// For a **nonlinear** medium whose index depends on the beam itself (M4
    /// thermal blooming), this is not enough on its own — see
    /// [`needs_intensity`](Self::needs_intensity) and
    /// [`index_response`](Self::index_response). Such a medium may leave this
    /// method as an unreachable stub.
    fn index_perturbation(&self, z_slab: usize) -> Array2<f64>;

    /// Whether this medium's index depends on the propagating field, so the
    /// propagator must route through [`index_response`](Self::index_response)
    /// with the slab-centre intensity rather than call
    /// [`index_perturbation`](Self::index_perturbation). Linear media (vacuum,
    /// attenuation, a fixed turbulence screen) leave this `false`.
    fn needs_intensity(&self) -> bool {
        false
    }

    /// Refractive-index perturbation `δn(x, y)` for slab `z_slab` given the
    /// beam `intensity` (`|u|²`) at the slab centre and the slab thickness
    /// `dz` (m).
    ///
    /// The default ignores both and defers to
    /// [`index_perturbation`](Self::index_perturbation), so linear media need
    /// not implement it. A field-coupled medium (thermal blooming) overrides
    /// this and sets [`needs_intensity`](Self::needs_intensity) to `true`; the
    /// propagator supplies the field *after* the leading half-step of
    /// diffraction — diffractively the slab-centre field — which keeps the
    /// symmetric split second-order. The field has **not** yet seen the slab's
    /// own extinction; a lossy nonlinear medium must apply its own
    /// half-slab decay (e.g. `e^(−α·dz/2)` on intensity) to stay a midpoint
    /// rule in the absorbed power — skipping this demotes the coupling to 1st
    /// order, which the M4 order gate catches.
    fn index_response(&self, z_slab: usize, intensity: &Array2<f64>, dz: f64) -> Array2<f64> {
        let _ = (intensity, dz);
        self.index_perturbation(z_slab)
    }

    /// Power extinction coefficient `α(x, y)` (1/m) for slab `z_slab`, or
    /// `None` for a lossless slab (the default).
    ///
    /// Beer–Lambert: intensity decays as `exp(−α·dz)` across the slab, so the
    /// propagator multiplies the field amplitude by `exp(−α·dz/2)`. `α` is the
    /// total extinction — molecular plus aerosol, absorption plus scattering
    /// out of the beam.
    fn extinction(&self, z_slab: usize) -> Option<Array2<f64>> {
        let _ = z_slab;
        None
    }
}

/// Vacuum (or unperturbed air): `δn = 0` everywhere.
#[derive(Debug, Clone, Copy)]
pub struct Vacuum {
    n: usize,
}

impl Vacuum {
    /// Vacuum sampled on an `n × n` grid.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Medium for Vacuum {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::zeros((self.n, self.n))
    }
}

/// A uniform index offset, constant across the grid and along z.
///
/// Physically this only adds a global phase, which makes it a useful
/// validation medium: intensity must be identical to vacuum propagation.
#[derive(Debug, Clone, Copy)]
pub struct ConstantDeltaN {
    n: usize,
    delta_n: f64,
}

impl ConstantDeltaN {
    /// Uniform perturbation `delta_n` on an `n × n` grid.
    pub fn new(n: usize, delta_n: f64) -> Self {
        Self { n, delta_n }
    }
}

impl Medium for ConstantDeltaN {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::from_elem((self.n, self.n), self.delta_n)
    }
}

/// A homogeneous absorbing/scattering atmosphere: `δn = 0`, uniform power
/// extinction coefficient `α` (1/m), constant along z.
///
/// The M2 workhorse. Transmission over a path `z` is the closed-form
/// Beer–Lambert `T = exp(−α·z)`, which is exactly what the validation gate
/// checks the propagator against.
#[derive(Debug, Clone, Copy)]
pub struct UniformExtinction {
    n: usize,
    alpha: f64,
}

impl UniformExtinction {
    /// Uniform extinction `alpha` (1/m, ≥ 0) on an `n × n` grid.
    pub fn new(n: usize, alpha: f64) -> Self {
        assert!(
            alpha >= 0.0 && alpha.is_finite(),
            "extinction coefficient must be non-negative and finite, got {alpha}"
        );
        Self { n, alpha }
    }
}

impl Medium for UniformExtinction {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        Array2::zeros((self.n, self.n))
    }

    fn extinction(&self, _z_slab: usize) -> Option<Array2<f64>> {
        if self.alpha == 0.0 {
            None
        } else {
            Some(Array2::from_elem((self.n, self.n), self.alpha))
        }
    }
}

/// Aerosol extinction coefficient (1/m) from meteorological visibility via the
/// Kruse model.
///
/// `α(λ) = (3.912 / V) · (λ / 550 nm)^(−q)` with the Kruse size-distribution
/// exponent `q`: 1.6 for V > 50 km, 1.3 for 6 km < V ≤ 50 km, and
/// `0.585·V_km^(1/3)` for V ≤ 6 km (Kruse et al. 1962; the 3.912 factor is
/// Koschmieder's 2 % contrast threshold, so at λ = 550 nm this reduces to
/// `α = 3.912/V` exactly).
///
/// `wavelength` and `visibility` in metres; valid in the visible/near-IR
/// window. This is aerosol scattering only — molecular absorption lines
/// (HITRAN-class data) are deliberately out of scope until a validated table
/// is imported.
pub fn kruse_extinction(wavelength: f64, visibility: f64) -> f64 {
    assert!(
        visibility > 0.0 && visibility.is_finite(),
        "visibility must be positive and finite, got {visibility}"
    );
    assert!(
        wavelength > 0.0 && wavelength.is_finite(),
        "wavelength must be positive and finite, got {wavelength}"
    );
    let v_km = visibility / 1e3;
    let q = if v_km > 50.0 {
        1.6
    } else if v_km > 6.0 {
        1.3
    } else {
        0.585 * v_km.cbrt()
    };
    (3.912 / visibility) * (wavelength / 550e-9).powf(-q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuum_is_zero() {
        let m = Vacuum::new(16);
        let dn = m.index_perturbation(3);
        assert_eq!(dn.dim(), (16, 16));
        assert!(dn.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_is_constant_across_slabs() {
        let m = ConstantDeltaN::new(8, 1e-6);
        assert_eq!(m.index_perturbation(0), m.index_perturbation(99));
        assert!(m.index_perturbation(0).iter().all(|&v| v == 1e-6));
    }

    #[test]
    fn lossless_media_report_no_extinction() {
        assert!(Vacuum::new(8).extinction(0).is_none());
        assert!(ConstantDeltaN::new(8, 1e-6).extinction(0).is_none());
        assert!(UniformExtinction::new(8, 0.0).extinction(0).is_none());
    }

    #[test]
    fn uniform_extinction_is_uniform() {
        let m = UniformExtinction::new(8, 2e-4);
        let a = m.extinction(5).unwrap();
        assert_eq!(a.dim(), (8, 8));
        assert!(a.iter().all(|&v| v == 2e-4));
        assert!(m.index_perturbation(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn kruse_reduces_to_koschmieder_at_550nm() {
        // At the visibility reference wavelength the spectral factor is 1
        // regardless of q, so α = 3.912/V exactly.
        for v in [1e3, 10e3, 80e3] {
            let alpha = kruse_extinction(550e-9, v);
            assert!((alpha - 3.912 / v).abs() < 1e-15 * alpha);
        }
    }

    #[test]
    fn kruse_extinction_decreases_with_wavelength_and_visibility() {
        // Longer wavelengths scatter less; clearer air extinguishes less.
        let a_vis = kruse_extinction(550e-9, 10e3);
        let a_ir = kruse_extinction(1.55e-6, 10e3);
        assert!(a_ir < a_vis);
        assert!(kruse_extinction(1e-6, 50e3) < kruse_extinction(1e-6, 5e3));
    }
}
