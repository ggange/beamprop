//! Frozen plasma-range air property table for the LSD wave (M6c, D8).
//!
//! The solver never calls Mutation++ at runtime: `scripts/make_plasma_table.py`
//! generates `data/plasma_properties.npy` offline (see `docs/M6C_SPEC.md`), and
//! it is embedded here as data and bilinearly interpolated. Same discipline as
//! [`airprops`](crate::airprops), and for the same two reasons —
//! bit-reproducibility, and keeping an LGPL property library out of the build
//! and out of the M5 wheels.
//!
//! This is a **separate file** from M4's `data/air_properties.npy`, which M6c
//! never touches (gate G0b). M4's table covers 200–400 K for thermal blooming;
//! this one covers 200–30,000 K because the Chapman–Jouguet state behind an LSD
//! front sits at ~1.5×10⁷ Pa and 1.5–2.5×10⁴ K.
//!
//! Table layout (asserted at load): `float64`, shape `(4, n_T, n_p)`, property
//! axis ordered `[ln ρ, e, γ_eff, ln n_e]`, on a grid uniform in `T` and
//! uniform in `log₁₀ p`.
//!
//! **`ρ` and `n_e` are stored as logarithms.** Both cross dozens of decades
//! through the ionization onset, where interpolating the raw values is
//! hopeless; the log is interpolated and exponentiated on read, which holds the
//! error near 10⁻³ (G6). `e` changes sign around 800 K so no log is available,
//! and neither it nor `γ_eff` needs one.
//!
//! # Two things this table does not contain
//!
//! **`n_i` is not stored.** The mixture has only singly charged ions, so
//! quasi-neutrality makes `n_i` identical to `n_e` — verified to 6.2×10⁻¹¹
//! wherever `n_e` is physically relevant, and gated in the generator. Storing
//! it would be a second copy of the same array dressed up as data;
//! [`PlasmaProperties::n_i`] returns `n_e` and says so.
//!
//! **`Z̄` is not stored, and is not a prediction.** It is identically 1 and
//! *cannot be anything else here*: the RRHO thermodynamic database ships no
//! doubly ionized nitrogen or oxygen (He⁺⁺ is the only such species in it), so
//! `air_11` structurally cannot represent second ionization.
//!
//! # Limitation: no second ionization above ~20,000 K
//!
//! Real equilibrium air begins to doubly ionize above roughly 20,000–25,000 K.
//! This table cannot follow it, so toward the top of its range it **understates
//! `n_e`**, and therefore understates the inverse-bremsstrahlung absorption
//! `α_IB ∝ n_e·n_i·Z̄²` that the LSD front runs on. The table is still built to
//! 30,000 K because the CJ state needs the range, but above
//! [`SECOND_IONIZATION_K`] it is a singly-ionized approximation rather than a
//! prediction — [`PlasmaTable::is_singly_ionized_approximation`] is the
//! predicate callers should route through when it matters.

use std::io::Cursor;

use anyhow::{Result, bail};
use ndarray::{Array3, Ix3};
use ndarray_npy::ReadNpyExt;

/// Temperature axis (K): `T_MIN + i·T_STEP`, `N_T` points.
const T_MIN: f64 = 200.0;
const T_STEP: f64 = 50.0;
const N_T: usize = 597;
/// Pressure axis (Pa): `10^(P_LOG10_MIN + j/P_PER_DECADE)`, `N_P` points.
const P_LOG10_MIN: f64 = 4.0;
const P_PER_DECADE: f64 = 8.0;
const N_P: usize = 33;

/// Mean ion charge. Not tabulated because it cannot vary — see the module note.
pub const Z_BAR: f64 = 1.0;

/// Above this temperature (K) the singly-ionized species set stops being a fair
/// description of equilibrium air, and the table understates `n_e`.
pub const SECOND_IONIZATION_K: f64 = 20_000.0;

/// `n_e` (1/m³) below which the table makes **no accuracy claim**.
///
/// For scale this is still eight orders below the ~10²³ of an LSD plasma, and
/// the cold tail runs down to 10⁻⁹⁶. Below it, `ln n_e` is nearly linear in
/// `1/T` rather than `T`, so the uniform-`T` grid interpolates it badly; the
/// generator's gates and G6 both exclude that region deliberately, since gating
/// a 100× error on a thousand electrons per cubic metre would be theatre.
pub const NE_ACCURACY_FLOOR: f64 = 1e15;

const TABLE_NPY: &[u8] = include_bytes!("../data/plasma_properties.npy");

/// Equilibrium air properties at one `(T, p)`, from the frozen table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasmaProperties {
    /// Density (kg/m³).
    pub rho: f64,
    /// Specific internal energy (J/kg).
    pub e: f64,
    /// Equilibrium ratio of specific heats (dimensionless).
    pub gamma_eff: f64,
    /// Electron number density (1/m³).
    pub n_e: f64,
}

impl PlasmaProperties {
    /// Ion number density (1/m³).
    ///
    /// Equal to `n_e` by quasi-neutrality: every ion in the mixture is singly
    /// charged, so `Σ Z_k n_k = n_e` collapses to `n_i = n_e`. Not a stored
    /// array — see the module note.
    pub fn n_i(&self) -> f64 {
        self.n_e / Z_BAR
    }

    /// Mean ion charge `Z̄`, identically [`Z_BAR`].
    pub fn z_bar(&self) -> f64 {
        Z_BAR
    }
}

/// The embedded plasma table, parsed once.
pub struct PlasmaTable {
    data: Array3<f64>,
}

impl PlasmaTable {
    /// Highest tabulated temperature (K).
    pub fn t_max() -> f64 {
        T_MIN + (N_T - 1) as f64 * T_STEP
    }

    /// Lowest tabulated temperature (K).
    pub fn t_min() -> f64 {
        T_MIN
    }

    /// Lowest tabulated pressure (Pa).
    pub fn p_min() -> f64 {
        10f64.powf(P_LOG10_MIN)
    }

    /// Highest tabulated pressure (Pa).
    pub fn p_max() -> f64 {
        10f64.powf(P_LOG10_MIN + (N_P - 1) as f64 / P_PER_DECADE)
    }

    /// Whether `t` (K) is in the range where the table is only a singly-ionized
    /// approximation. See the module-level limitation note.
    pub fn is_singly_ionized_approximation(t: f64) -> bool {
        t > SECOND_IONIZATION_K
    }

    /// Parse and validate the embedded property table.
    pub fn load() -> Result<Self> {
        let data = Array3::<f64>::read_npy(Cursor::new(TABLE_NPY))
            .map_err(|e| anyhow::anyhow!("parsing embedded plasma table: {e}"))?;
        if data.raw_dim() != Ix3(4, N_T, N_P) {
            bail!(
                "plasma table shape {:?}, expected (4, {N_T}, {N_P})",
                data.dim()
            );
        }
        Ok(Self { data })
    }

    /// Bilinearly interpolate the properties at temperature `t` (K) and
    /// pressure `p` (Pa).
    ///
    /// `ρ` and `n_e` are interpolated in the log and exponentiated, which is
    /// what makes the ionization onset tractable.
    ///
    /// Errors if `(t, p)` is outside the tabulated range — the `airprops`
    /// precedent, and the spec's failure-mode table: a query off the end of the
    /// table is a modelling error, not something to clamp.
    pub fn at(&self, t: f64, p: f64) -> Result<PlasmaProperties> {
        if !t.is_finite() || !p.is_finite() {
            bail!("plasma table queried at non-finite (T = {t}, p = {p})");
        }
        if p <= 0.0 {
            bail!("plasma table queried at non-positive pressure {p} Pa");
        }
        let ti = (t - T_MIN) / T_STEP;
        let pj = (p.log10() - P_LOG10_MIN) * P_PER_DECADE;
        if !(0.0..=(N_T - 1) as f64).contains(&ti) {
            bail!(
                "temperature {t} K outside plasma table [{T_MIN}, {}]",
                Self::t_max()
            );
        }
        if !(0.0..=(N_P - 1) as f64).contains(&pj) {
            bail!(
                "pressure {p:.4e} Pa outside plasma table [{:.4e}, {:.4e}]",
                Self::p_min(),
                Self::p_max()
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
        Ok(PlasmaProperties {
            rho: bilinear(0).exp(),
            e: bilinear(1),
            gamma_eff: bilinear(2),
            n_e: bilinear(3).exp(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_with_expected_shape() {
        let table = PlasmaTable::load().unwrap();
        assert_eq!(table.data.dim(), (4, N_T, N_P));
        assert!((PlasmaTable::t_max() - 30_000.0).abs() < 1e-9);
        assert!((PlasmaTable::p_max() - 1e8).abs() / 1e8 < 1e-12);
    }

    #[test]
    fn cold_air_is_an_ideal_diatomic_gas() {
        // The one point in the table whose answer is known without Mutation++:
        // undissociated air at STP is γ = 1.4 and ρ = p·M/(R·T) ≈ 1.18 kg/m³.
        let table = PlasmaTable::load().unwrap();
        let a = table.at(300.0, 101_325.0).unwrap();
        assert!((a.gamma_eff - 1.4).abs() < 0.01, "γ_eff = {}", a.gamma_eff);
        assert!(
            (a.rho - 1.177).abs() / 1.177 < 0.02,
            "ρ = {} kg/m³, expected ≈1.177",
            a.rho
        );
        // Cold air is un-ionized to an absurd degree.
        assert!(a.n_e < 1.0, "n_e = {:.3e} in cold air", a.n_e);
    }

    #[test]
    fn ionization_rises_monotonically_through_the_onset() {
        let table = PlasmaTable::load().unwrap();
        let mut prev = 0.0;
        for step in 0..40 {
            let t = 5_000.0 + step as f64 * 250.0;
            let n_e = table.at(t, 101_325.0).unwrap().n_e;
            assert!(
                n_e > prev,
                "n_e fell from {prev:.4e} to {n_e:.4e} at T = {t} K"
            );
            prev = n_e;
        }
        // By 15,000 K at 1 atm the gas is substantially ionized.
        assert!(prev > 1e22, "n_e only reached {prev:.4e} by 15,000 K");
    }

    #[test]
    fn quasineutrality_and_charge_state_are_explicit() {
        let table = PlasmaTable::load().unwrap();
        let a = table.at(15_000.0, 101_325.0).unwrap();
        assert_eq!(a.n_i(), a.n_e);
        assert_eq!(a.z_bar(), 1.0);
    }

    #[test]
    fn second_ionization_limit_is_flagged() {
        assert!(!PlasmaTable::is_singly_ionized_approximation(15_000.0));
        assert!(PlasmaTable::is_singly_ionized_approximation(25_000.0));
    }

    #[test]
    fn out_of_range_queries_error() {
        let table = PlasmaTable::load().unwrap();
        assert!(table.at(100.0, 101_325.0).is_err(), "below T range");
        assert!(table.at(40_000.0, 101_325.0).is_err(), "above T range");
        assert!(table.at(5_000.0, 1e3).is_err(), "below p range");
        assert!(table.at(5_000.0, 1e9).is_err(), "above p range");
        assert!(table.at(5_000.0, -1.0).is_err(), "negative pressure");
        assert!(table.at(f64::NAN, 101_325.0).is_err(), "NaN temperature");
    }

    #[test]
    fn interpolation_is_continuous_across_a_cell_boundary() {
        // Approaching a grid node from both sides must agree: catches an
        // off-by-one in the index or a mismatched axis convention.
        let table = PlasmaTable::load().unwrap();
        let t_node = T_MIN + 100.0 * T_STEP; // an exact node
        let eps = 1e-6;
        let below = table.at(t_node - eps, 101_325.0).unwrap();
        let above = table.at(t_node + eps, 101_325.0).unwrap();
        assert!((below.rho / above.rho - 1.0).abs() < 1e-6);
        assert!((below.gamma_eff / above.gamma_eff - 1.0).abs() < 1e-6);
        assert!((below.n_e / above.n_e - 1.0).abs() < 1e-5);
    }
}
