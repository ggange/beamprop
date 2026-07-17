//! Steady-state thermal blooming as a field-coupled [`Medium`] (M4).
//!
//! A high-power beam heats the air it passes through; the warmed air thins,
//! lowering the refractive index, and the beam self-distorts — defocusing and
//! bending into the wind. In the convection-dominated CW steady state (see
//! `docs/M4_SPEC.md`) the temperature rise at a point is an upwind line
//! integral of the absorbed intensity, evaluated slab-locally:
//!
//! ```text
//! ΔT(x, y) = (α_abs / (ρ·c_p·v)) · ∫_{-∞}^{x} I(x', y) dx'
//! δn(x, y) = −(n₀ − 1)/T₀ · ΔT(x, y)
//! ```
//!
//! with the wind along `+x`. This is a nonlinear medium: `δn` depends on the
//! beam's own intensity, so it drives the propagator through the field-aware
//! [`Medium::index_response`] path with the slab-centre intensity.

use anyhow::{Result, bail};
use ndarray::Array2;

use crate::airprops::AirProperties;
use crate::grid::Grid;
use crate::medium::Medium;

/// Minimum Péclet number for the convection-dominated model to hold; below
/// this, conduction across the beam is not negligible and v1 refuses the case
/// rather than mis-modelling it (M4_SPEC § Fluid model).
const MIN_PECLET: f64 = 100.0;
/// Small-perturbation ceiling: the isobaric/linearized-index model requires
/// `ΔT ≪ T₀`; beyond this fraction the run is rejected.
const MAX_DELTA_T_FRAC: f64 = 0.1;

/// Steady-state thermal-blooming medium for a collimated beam in uniform
/// crosswind.
pub struct ThermalBlooming {
    n: usize,
    dx: f64,
    /// Physical-intensity scale: `I_phys = intensity_scale · |u|²`, fixed from
    /// the initial field so `|u|²` (arbitrary units) becomes W/m².
    intensity_scale: f64,
    /// `α_abs / (ρ·c_p·v)`: intensity line-integral → temperature rise.
    heat_coeff: f64,
    /// `(n₀ − 1)/T₀`: temperature rise → index drop.
    index_coeff: f64,
    /// Ambient temperature (K), for the small-perturbation guard.
    t0: f64,
    /// Absorbed-power extinction coefficient (1/m).
    alpha_abs: f64,
}

impl ThermalBlooming {
    /// Build a blooming medium.
    ///
    /// - `air`: properties at the ambient state and beam wavelength.
    /// - `alpha_abs`: absorbed-power coefficient (1/m).
    /// - `wind_speed`: crosswind `v` (m/s, along +x).
    /// - `beam_power`: total beam power `P` (W).
    /// - `initial_field_power`: [`Field::power`](crate::field::Field::power) of
    ///   the launch field, so its arbitrary `|u|²` units map to W/m².
    /// - `beam_width`: characteristic `1/e²` radius `w` (m), used only for the
    ///   Péclet validity check.
    /// - `t0`: ambient temperature `T₀` (K).
    ///
    /// Errors on non-positive wind/power, or a Péclet number below
    /// [`MIN_PECLET`] (stagnant-air regime, out of scope for v1).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: Grid,
        air: AirProperties,
        alpha_abs: f64,
        wind_speed: f64,
        beam_power: f64,
        initial_field_power: f64,
        beam_width: f64,
        t0: f64,
    ) -> Result<Self> {
        if !(wind_speed > 0.0 && wind_speed.is_finite()) {
            bail!("wind speed must be positive and finite, got {wind_speed}");
        }
        if !(beam_power > 0.0 && beam_power.is_finite()) {
            bail!("beam power must be positive and finite, got {beam_power}");
        }
        if initial_field_power <= 0.0 {
            bail!("initial field power must be positive, got {initial_field_power}");
        }
        if !(alpha_abs >= 0.0 && alpha_abs.is_finite()) {
            bail!("absorption coefficient must be non-negative and finite, got {alpha_abs}");
        }
        let peclet = air.rho * air.cp * wind_speed * beam_width / air.kappa_t;
        if peclet < MIN_PECLET {
            bail!(
                "Péclet number {peclet:.1} < {MIN_PECLET}: conduction not negligible; \
                 the convection-dominated model does not apply (raise wind or beam size)"
            );
        }
        Ok(Self {
            n: grid.n,
            dx: grid.dx,
            intensity_scale: beam_power / initial_field_power,
            heat_coeff: alpha_abs / (air.rho * air.cp * wind_speed),
            index_coeff: air.n_minus_1 / t0,
            t0,
            alpha_abs,
        })
    }
}

impl Medium for ThermalBlooming {
    fn index_perturbation(&self, _z_slab: usize) -> Array2<f64> {
        panic!(
            "ThermalBlooming is a field-coupled medium; the propagator must use \
             the index_response path (needs_intensity = true)"
        );
    }

    fn needs_intensity(&self) -> bool {
        true
    }

    fn index_response(&self, _z_slab: usize, intensity: &Array2<f64>, dz: f64) -> Array2<f64> {
        let (ny, nx) = intensity.dim();
        // The propagator hands over the field before the slab's own
        // Beer–Lambert decay; the physical intensity at the slab midpoint
        // carries half a slab of absorption. Without this factor the heating
        // is a rectangle rule in absorbed power and the coupling drops to
        // 1st order (caught by the M4 order gate).
        let midpoint_decay = (-0.5 * self.alpha_abs * dz).exp();
        // δn coefficient combining physical-intensity scaling, the heat
        // integral, and the index drop: δn = −(index·heat·scale)·∫I_arb dx.
        let coeff = -self.index_coeff * self.heat_coeff * self.intensity_scale * midpoint_decay;
        let mut dn = Array2::zeros((ny, nx));
        let mut max_delta_t = 0.0_f64;
        for iy in 0..ny {
            // Cumulative trapezoid integral of intensity along +x (the wind
            // direction): cum[j] = ∫_{x₀}^{x_j} I dx.
            let mut cum = 0.0;
            let mut prev = intensity[[iy, 0]];
            dn[[iy, 0]] = 0.0; // no upwind path yet
            for ix in 1..nx {
                let cur = intensity[[iy, ix]];
                cum += 0.5 * (prev + cur) * self.dx;
                dn[[iy, ix]] = coeff * cum;
                // ΔT = |δn| / index_coeff; track the largest for the guard.
                let delta_t = self.heat_coeff * self.intensity_scale * cum;
                if delta_t > max_delta_t {
                    max_delta_t = delta_t;
                }
                prev = cur;
            }
        }
        assert!(
            max_delta_t < MAX_DELTA_T_FRAC * self.t0,
            "thermal-blooming ΔT_max = {max_delta_t:.1} K exceeds {MAX_DELTA_T_FRAC}·T₀ \
             = {:.1} K: outside the small-perturbation model (lower power or raise wind)",
            MAX_DELTA_T_FRAC * self.t0
        );
        dn
    }

    fn extinction(&self, _z_slab: usize) -> Option<Array2<f64>> {
        // The absorbed power that heats the air also leaves the beam.
        if self.alpha_abs == 0.0 {
            None
        } else {
            Some(Array2::from_elem((self.n, self.n), self.alpha_abs))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;

    fn standard_air() -> AirProperties {
        crate::airprops::AirTable::load()
            .unwrap()
            .at(288.15, 101_325.0, 1e-6)
            .unwrap()
    }

    #[test]
    fn rejects_stagnant_air() {
        let grid = Grid::new(64, 1e-3);
        let field = Field::gaussian(grid, 1e-6, 1e-2);
        let r = ThermalBlooming::new(
            grid,
            standard_air(),
            1e-4,
            1e-3, // v = 1 mm/s → Pe ≪ 100
            1e4,
            field.power(),
            1e-2,
            288.15,
        );
        assert!(r.is_err());
    }

    #[test]
    fn index_is_negative_and_grows_downwind() {
        let grid = Grid::new(128, 1e-3);
        let field = Field::gaussian(grid, 1e-6, 1e-2);
        let bloom = ThermalBlooming::new(
            grid,
            standard_air(),
            1e-4,
            2.0,
            1e4,
            field.power(),
            1e-2,
            288.15,
        )
        .unwrap();
        let dn = bloom.index_response(0, &field.intensity(), 0.0);
        let mid = grid.n / 2;
        // Heated air thins: δn ≤ 0, and |δn| accumulates toward +x (downwind).
        assert!(dn.iter().all(|&v| v <= 0.0));
        assert!(dn[[mid, grid.n - 1]] < dn[[mid, 0]]);
        assert!(dn[[mid, grid.n - 1]] < dn[[mid, mid]]);
    }
}
