//! Frozen dry-air property table for thermal blooming (M4).
//!
//! The solver never calls Mutation++ at runtime: `scripts/make_air_table.py`
//! generates `data/air_properties.npy` offline (see `docs/M4_SPEC.md`), and it
//! is embedded here as data and bilinearly interpolated. This preserves
//! bit-reproducibility and keeps the LGPL property library out of the build.
//!
//! Table layout (asserted at load): `float64`, shape `(4, n_T, n_p)`, property
//! axis ordered `[ρ, c_p, κ_t, (n−1)]`, on a uniform `(T, p)` grid.

use std::io::Cursor;

use anyhow::{Result, bail};
use ndarray::{Array3, Ix3};
use ndarray_npy::ReadNpyExt;

/// Temperature axis (K): `T_MIN + i·T_STEP`, `N_T` points.
const T_MIN: f64 = 200.0;
const T_STEP: f64 = 2.0;
const N_T: usize = 101;
/// Pressure axis (Pa): `P_MIN + j·P_STEP`, `N_P` points.
const P_MIN: f64 = 40_000.0;
const P_STEP: f64 = 2_500.0;
const N_P: usize = 29;
/// Wavelength the tabulated refractivity `(n−1)` is referenced to (µm); other
/// wavelengths are rescaled by the Ciddor dispersion ratio.
const LAMBDA_REF_UM: f64 = 1.0;

const TABLE_NPY: &[u8] = include_bytes!("../data/air_properties.npy");

/// Air properties at one `(T, p)` and wavelength, from the frozen table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirProperties {
    /// Density (kg/m³).
    pub rho: f64,
    /// Isobaric specific heat (J/(kg·K)).
    pub cp: f64,
    /// Thermal conductivity (W/(m·K)).
    pub kappa_t: f64,
    /// Refractivity `n − 1` at the requested wavelength (dimensionless).
    pub n_minus_1: f64,
}

/// The embedded table, parsed once.
pub struct AirTable {
    data: Array3<f64>,
}

impl AirTable {
    /// Parse and validate the embedded property table.
    pub fn load() -> Result<Self> {
        let data = Array3::<f64>::read_npy(Cursor::new(TABLE_NPY))
            .map_err(|e| anyhow::anyhow!("parsing embedded air table: {e}"))?;
        if data.raw_dim() != Ix3(4, N_T, N_P) {
            bail!(
                "air table shape {:?}, expected (4, {N_T}, {N_P})",
                data.dim()
            );
        }
        Ok(Self { data })
    }

    /// Bilinearly interpolate the four properties at `(t, p)` and rescale the
    /// refractivity to `wavelength` (m) via Ciddor dispersion.
    ///
    /// Errors if `(t, p)` is outside the tabulated range.
    pub fn at(&self, t: f64, p: f64, wavelength: f64) -> Result<AirProperties> {
        let ti = (t - T_MIN) / T_STEP;
        let pj = (p - P_MIN) / P_STEP;
        if !(0.0..=(N_T - 1) as f64).contains(&ti) {
            bail!(
                "temperature {t} K outside air table [{T_MIN}, {}]",
                T_MIN + (N_T - 1) as f64 * T_STEP
            );
        }
        if !(0.0..=(N_P - 1) as f64).contains(&pj) {
            bail!(
                "pressure {p} Pa outside air table [{P_MIN}, {}]",
                P_MIN + (N_P - 1) as f64 * P_STEP
            );
        }
        let i0 = (ti.floor() as usize).min(N_T - 2);
        let j0 = (pj.floor() as usize).min(N_P - 2);
        let fi = ti - i0 as f64;
        let fj = pj - j0 as f64;
        let bilinear = |prop: usize| {
            let a = self.data.index_axis(ndarray::Axis(0), prop);
            a[[i0, j0]] * (1.0 - fi) * (1.0 - fj)
                + a[[i0 + 1, j0]] * fi * (1.0 - fj)
                + a[[i0, j0 + 1]] * (1.0 - fi) * fj
                + a[[i0 + 1, j0 + 1]] * fi * fj
        };
        let dispersion = dispersion_ratio(wavelength);
        Ok(AirProperties {
            rho: bilinear(0),
            cp: bilinear(1),
            kappa_t: bilinear(2),
            n_minus_1: bilinear(3) * dispersion,
        })
    }
}

/// Ciddor (1996) standard dry-air dispersion group `D(λ)` in units of the
/// paper's `10⁻⁸` refractivity coefficient; only the ratio is used.
fn ciddor_group(lambda_um: f64) -> f64 {
    let sigma2 = (1.0 / lambda_um).powi(2);
    5_792_105.0 / (238.0185 - sigma2) + 167_917.0 / (57.362 - sigma2)
}

/// Dispersion rescale factor `D(λ)/D(λ_ref)` taking tabulated `(n−1)` at
/// `LAMBDA_REF_UM` to the requested wavelength (m).
fn dispersion_ratio(wavelength_m: f64) -> f64 {
    let lambda_um = wavelength_m * 1e6;
    ciddor_group(lambda_um) / ciddor_group(LAMBDA_REF_UM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_with_expected_shape() {
        let table = AirTable::load().unwrap();
        assert_eq!(table.data.dim(), (4, N_T, N_P));
    }

    #[test]
    fn grid_point_matches_sidecar_spot_value() {
        // 300 K, 100 kPa are exact grid points; the JSON sidecar documents
        // rho ≈ 1.1610, cp ≈ 1004.3 there. At λ = 1 µm the refractivity is the
        // tabulated value (dispersion ratio 1).
        let table = AirTable::load().unwrap();
        let a = table.at(300.0, 100_000.0, 1e-6).unwrap();
        assert!((a.rho - 1.16101714).abs() < 1e-6, "rho {}", a.rho);
        assert!((a.cp - 1004.28695).abs() < 1e-2, "cp {}", a.cp);
        assert!((a.kappa_t - 0.02624708).abs() < 1e-6, "kappa {}", a.kappa_t);
    }

    #[test]
    fn interpolated_density_near_ideal_gas() {
        // Midpoint between grid nodes: bilinear rho within 0.1 % of p·M/(R·T).
        let table = AirTable::load().unwrap();
        let (t, p) = (301.0, 101_250.0);
        let a = table.at(t, p, 1e-6).unwrap();
        let mw = 0.7811 * 28.014e-3 + 0.2096 * 31.998e-3 + 0.0093 * 39.948e-3;
        let rho_ideal = p * mw / (8.31446261815324 * t);
        assert!(
            (a.rho - rho_ideal).abs() / rho_ideal < 1e-3,
            "rho {} vs ideal {rho_ideal}",
            a.rho
        );
    }

    #[test]
    fn refractivity_rescales_with_wavelength() {
        // Shorter wavelength → larger refractivity (normal dispersion).
        let table = AirTable::load().unwrap();
        let n1_1um = table.at(300.0, 100_000.0, 1e-6).unwrap().n_minus_1;
        let n1_633 = table.at(300.0, 100_000.0, 633e-9).unwrap().n_minus_1;
        assert!(n1_633 > n1_1um, "{n1_633} !> {n1_1um}");
        assert!((dispersion_ratio(1e-6) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn out_of_range_errors() {
        let table = AirTable::load().unwrap();
        assert!(table.at(150.0, 100_000.0, 1e-6).is_err());
        assert!(table.at(300.0, 200_000.0, 1e-6).is_err());
    }
}
