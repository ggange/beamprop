//! Python bindings for `beamprop` (M5).
//!
//! Exposes the core objects (`Grid`, `Field`, `Medium`, `Propagator`), the
//! analytic references used by the validation suite (`GaussianBeam`,
//! `fried_r0`, `rytov_variance`, `kruse_extinction`), and high-level
//! `run_*` helpers that mirror the CLI cases and return numpy arrays plus
//! derived diagnostics.
//!
//! The helpers cover every case the CLI has: the three propagation cases
//! (`run_propagate`, `run_turbulence`, `run_blooming`) and the M6 cases
//! (`run_breakdown`, `run_lsd`, `run_ignition`). Rendering deliberately stays
//! in `scripts/render*.py` — these return data only, which is the same
//! data/image split the solver keeps.
//!
//! Conventions across the boundary:
//! - SI units and `float64`/`complex128` everywhere, matching the Rust core.
//! - Solver errors (`anyhow::Result`) surface as Python `ValueError`.
//! - Arguments that the Rust constructors would `assert!` on are pre-validated
//!   here so invalid input raises `ValueError`, not a Rust panic.

use anyhow::Error;
use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use beamprop::cases::{
    BloomingParams, BreakdownParams, IgnitionParams, IgnitionSweepParams, LsdParams,
    PropagateParams, TurbulenceParams, run_blooming, run_breakdown, run_ignition_sweep, run_lsd,
    run_propagate, run_turbulence,
};

/// Map a solver error onto a Python `ValueError`, preserving the message.
fn to_py(e: Error) -> PyErr {
    PyValueError::new_err(format!("{e:#}"))
}

/// Validate the `Grid::new` invariants (which are `assert!`s on the Rust side)
/// so bad input raises `ValueError` instead of panicking across the FFI.
fn checked_grid(n: usize, dx: f64) -> PyResult<beamprop::grid::Grid> {
    if n == 0 || n % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "grid size must be positive and even (origin sits at n/2), got {n}"
        )));
    }
    if !(dx > 0.0 && dx.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "grid spacing must be positive and finite, got {dx}"
        )));
    }
    Ok(beamprop::grid::Grid::new(n, dx))
}

fn checked_beam(wavelength: f64, w0: f64) -> PyResult<()> {
    if !(wavelength > 0.0 && wavelength.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "wavelength must be positive and finite, got {wavelength}"
        )));
    }
    if !(w0 > 0.0 && w0.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "waist must be positive and finite, got {w0}"
        )));
    }
    Ok(())
}

/// A square n × n transverse grid with uniform spacing dx (metres),
/// coordinates centred on zero (sample n/2 sits at the origin).
#[pyclass(name = "Grid", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyGrid {
    inner: beamprop::grid::Grid,
}

#[pymethods]
impl PyGrid {
    #[new]
    fn new(n: usize, dx: f64) -> PyResult<Self> {
        Ok(Self {
            inner: checked_grid(n, dx)?,
        })
    }

    /// Number of samples per side.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Sample spacing in metres.
    #[getter]
    fn dx(&self) -> f64 {
        self.inner.dx
    }

    /// Physical side length in metres (n · dx).
    #[getter]
    fn extent(&self) -> f64 {
        self.inner.extent()
    }

    /// Coordinate of sample `i` along one axis, centred on zero (metres).
    fn coord(&self, i: usize) -> PyResult<f64> {
        if i >= self.inner.n {
            return Err(PyValueError::new_err(format!(
                "sample index {i} out of range for grid size {}",
                self.inner.n
            )));
        }
        Ok(self.inner.coord(i))
    }

    fn __repr__(&self) -> String {
        format!("Grid(n={}, dx={})", self.inner.n, self.inner.dx)
    }
}

/// A monochromatic scalar optical field u(x, y) sampled on a Grid.
#[pyclass(name = "Field")]
struct PyField {
    inner: beamprop::field::Field,
}

#[pymethods]
impl PyField {
    /// A circular Gaussian beam of 1/e²-intensity waist radius `w0` (m) with
    /// unit on-axis amplitude and flat phase, centred on the grid.
    #[staticmethod]
    fn gaussian(grid: PyRef<'_, PyGrid>, wavelength: f64, w0: f64) -> PyResult<Self> {
        checked_beam(wavelength, w0)?;
        Ok(Self {
            inner: beamprop::field::Field::gaussian(grid.inner, wavelength, w0),
        })
    }

    #[getter]
    fn grid(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.grid,
        }
    }

    /// Vacuum wavelength in metres.
    #[getter]
    fn wavelength(&self) -> f64 {
        self.inner.wavelength
    }

    /// Complex amplitude u(x, y), indexed [iy, ix] (complex128 copy).
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
        self.inner.u.clone().into_pyarray(py)
    }

    /// Replace the complex amplitude; shape must match the grid.
    #[setter]
    fn set_u(&mut self, u: PyReadonlyArray2<'_, Complex64>) -> PyResult<()> {
        let arr = u.as_array();
        let n = self.inner.grid.n;
        if arr.dim() != (n, n) {
            return Err(PyValueError::new_err(format!(
                "amplitude shape {:?} does not match the {n} x {n} grid",
                arr.dim()
            )));
        }
        self.inner.u = arr.to_owned();
        Ok(())
    }

    /// Intensity |u|² as a float64 array, indexed [iy, ix].
    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.intensity().into_pyarray(py)
    }

    /// Total power Σ |u|² · dx² in the field's (arbitrary) amplitude units.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power()
    }

    /// 1/e² intensity half-widths (wx, wy) of the field (m).
    fn beam_width(&self) -> (f64, f64) {
        beamprop::propagate::beam_width(&self.inner)
    }

    /// Intensity centroid (x, y) of the field (m).
    fn centroid(&self) -> (f64, f64) {
        beamprop::propagate::centroid(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Field(n={}, dx={}, wavelength={})",
            self.inner.grid.n, self.inner.grid.dx, self.inner.wavelength
        )
    }
}

/// A propagation medium (what the beam travels through), built by one of the
/// static constructors: `vacuum`, `constant_delta_n`, `uniform_extinction`,
/// `turbulence`, `thermal_blooming`.
#[pyclass(name = "Medium", frozen)]
struct PyMedium {
    inner: Box<dyn beamprop::medium::Medium + Send + Sync>,
    describe: String,
}

#[pymethods]
impl PyMedium {
    /// Vacuum (or unperturbed air): δn = 0 everywhere. Sized to `grid` so it
    /// matches the field it will propagate.
    #[staticmethod]
    fn vacuum(grid: PyRef<'_, PyGrid>) -> Self {
        Self {
            inner: Box::new(beamprop::medium::Vacuum::new(grid.inner.n)),
            describe: "vacuum".into(),
        }
    }

    /// A uniform refractive-index offset δn (pure phase, no loss).
    #[staticmethod]
    fn constant_delta_n(grid: PyRef<'_, PyGrid>, delta_n: f64) -> Self {
        Self {
            inner: Box::new(beamprop::medium::ConstantDeltaN::new(grid.inner.n, delta_n)),
            describe: format!("constant_delta_n({delta_n})"),
        }
    }

    /// Uniform Beer–Lambert power extinction `alpha` (1/m).
    #[staticmethod]
    fn uniform_extinction(grid: PyRef<'_, PyGrid>, alpha: f64) -> Self {
        Self {
            inner: Box::new(beamprop::medium::UniformExtinction::new(
                grid.inner.n,
                alpha,
            )),
            describe: format!("uniform_extinction({alpha})"),
        }
    }

    /// A reproducible von Kármán turbulent path: `screens` phase screens over
    /// `z` metres with structure constant `cn2` (m^(-2/3)) and outer scale
    /// `l0` (m). `realization` selects the Monte-Carlo member of `seed`'s
    /// ensemble; `substeps` subdivides each slab for smooth side views.
    /// Propagate with `dz = z / (screens * substeps)` for `screens * substeps`
    /// steps.
    #[staticmethod]
    #[pyo3(signature = (grid, wavelength, cn2, l0, z, screens, seed=1, realization=0, substeps=1))]
    #[allow(clippy::too_many_arguments)]
    fn turbulence(
        grid: PyRef<'_, PyGrid>,
        wavelength: f64,
        cn2: f64,
        l0: f64,
        z: f64,
        screens: usize,
        seed: u64,
        realization: u64,
        substeps: usize,
    ) -> PyResult<Self> {
        if screens == 0 {
            return Err(PyValueError::new_err("need at least one screen"));
        }
        if substeps == 0 {
            return Err(PyValueError::new_err(
                "need at least one substep per screen",
            ));
        }
        if !(cn2 > 0.0 && z > 0.0 && l0 > 0.0) {
            return Err(PyValueError::new_err(format!(
                "cn2, z, and l0 must all be positive, got {cn2}, {z}, {l0}"
            )));
        }
        checked_beam(wavelength, 1.0)?;
        let path = beamprop::turbulence::TurbulentPath::new(
            grid.inner,
            wavelength,
            cn2,
            l0,
            z,
            screens,
            seed,
            realization,
        )
        .with_substeps(substeps);
        Ok(Self {
            inner: Box::new(path),
            describe: format!(
                "turbulence(cn2={cn2}, z={z}, screens={screens}, seed={seed}, \
                 realization={realization})"
            ),
        })
    }

    /// Steady-state thermal blooming for `field` carrying `power` watts in a
    /// `wind` m/s crosswind (+x), absorbing `alpha_abs` 1/m, at ambient
    /// `t0` K / `p0` Pa. `w0` is the beam's 1/e² radius (Péclet validity
    /// check). Raises ValueError outside the model's validity (stagnant air).
    #[staticmethod]
    #[pyo3(signature = (field, w0, power, wind, alpha_abs, t0=288.15, p0=101_325.0))]
    fn thermal_blooming(
        field: PyRef<'_, PyField>,
        w0: f64,
        power: f64,
        wind: f64,
        alpha_abs: f64,
        t0: f64,
        p0: f64,
    ) -> PyResult<Self> {
        let f = &field.inner;
        let air = beamprop::airprops::AirTable::load()
            .and_then(|t| t.at(t0, p0, f.wavelength))
            .map_err(to_py)?;
        let medium = beamprop::blooming::ThermalBlooming::new(
            f.grid,
            air,
            alpha_abs,
            wind,
            power,
            f.power(),
            w0,
            t0,
        )
        .map_err(to_py)?;
        Ok(Self {
            inner: Box::new(medium),
            describe: format!(
                "thermal_blooming(power={power}, wind={wind}, alpha_abs={alpha_abs})"
            ),
        })
    }

    fn __repr__(&self) -> String {
        format!("Medium.{}", self.describe)
    }
}

/// Symmetric split-step propagator advancing a Field through a Medium.
#[pyclass(name = "Propagator")]
struct PyPropagator {
    inner: beamprop::propagate::Propagator,
    /// Guard-band absorbed fraction of the most recent `propagate` call — the
    /// per-call delta of the (lifetime-accumulated) `guard_absorbed`, divided
    /// by that call's initial power. `None` before the first call.
    last_guard_frac: Option<f64>,
}

#[pymethods]
impl PyPropagator {
    #[new]
    fn new(grid: PyRef<'_, PyGrid>, wavelength: f64) -> PyResult<Self> {
        checked_beam(wavelength, 1.0)?;
        Ok(Self {
            inner: beamprop::propagate::Propagator::new(grid.inner, wavelength).map_err(to_py)?,
            last_guard_frac: None,
        })
    }

    /// Longest single-step distance the angular-spectrum method can take
    /// without frequency aliasing (m); longer steps switch to Fresnel.
    #[getter]
    fn critical_distance(&self) -> f64 {
        self.inner.critical_distance()
    }

    /// Power absorbed by the boundary guard band during the most recent
    /// `propagate` call, as a fraction of that call's initial power (grid-edge
    /// artifact unless ≈ 0). Computed per call, so it is correct even when the
    /// propagator is reused. `None` before the first call.
    #[getter]
    fn guard_frac(&self) -> Option<f64> {
        self.last_guard_frac
    }

    /// Advance `field` in place through `medium` by `steps` slabs of `dz`
    /// metres. `on_step(i, field)`, if given, is called after each step with
    /// the step index and a snapshot copy of the field.
    #[pyo3(signature = (field, medium, dz, steps, on_step=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        field: Bound<'_, PyField>,
        medium: PyRef<'_, PyMedium>,
        dz: f64,
        steps: usize,
        on_step: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let mut f = field.borrow_mut();
        let p0 = f.inner.power();
        // guard_absorbed accumulates over the propagator's lifetime; snapshot
        // it here so guard_frac reflects only this call (see the getter).
        let guard_before = self.inner.guard_absorbed();
        let mut cb_err: Option<PyErr> = None;
        let result = self.inner.propagate(
            &mut f.inner,
            medium.inner.as_ref(),
            dz,
            0,
            steps,
            |i, fld| {
                if cb_err.is_some() {
                    return;
                }
                if let Some(cb) = &on_step {
                    let snapshot = PyField { inner: fld.clone() };
                    if let Err(e) = cb.call1(py, (i, snapshot)) {
                        cb_err = Some(e);
                    }
                }
            },
        );
        if let Some(e) = cb_err {
            return Err(e);
        }
        result.map_err(to_py)?;
        self.last_guard_frac = Some((self.inner.guard_absorbed() - guard_before) / p0);
        Ok(())
    }
}

/// Analytic Gaussian-beam reference (the M1 closed form).
#[pyclass(name = "GaussianBeam", frozen)]
struct PyGaussianBeam {
    inner: beamprop::validate::GaussianBeam,
}

#[pymethods]
impl PyGaussianBeam {
    #[new]
    fn new(w0: f64, wavelength: f64) -> PyResult<Self> {
        checked_beam(wavelength, w0)?;
        Ok(Self {
            inner: beamprop::validate::GaussianBeam { w0, wavelength },
        })
    }

    /// Rayleigh range z_R = π w0² / λ (m).
    #[getter]
    fn rayleigh_range(&self) -> f64 {
        self.inner.rayleigh_range()
    }

    /// 1/e² radius w(z) after distance z (m).
    fn width_at(&self, z: f64) -> f64 {
        self.inner.width_at(z)
    }

    /// Far-field half-angle divergence (rad).
    #[getter]
    fn divergence(&self) -> f64 {
        self.inner.divergence()
    }

    /// Long-exposure (turbulence-broadened) 1/e² radius at z for a uniform
    /// Cn² path (Andrews–Phillips weak-fluctuation form).
    fn long_exposure_width(&self, z: f64, cn2: f64) -> f64 {
        self.inner.long_exposure_width(z, cn2)
    }
}

/// Fried parameter r0 (m) for a uniform-Cn² path of length z (m).
#[pyfunction]
fn fried_r0(cn2: f64, wavelength: f64, z: f64) -> f64 {
    beamprop::validate::fried_r0(cn2, wavelength, z)
}

/// Plane-wave Rytov variance σ_R² for a uniform-Cn² path of length z (m).
#[pyfunction]
fn rytov_variance(cn2: f64, wavelength: f64, z: f64) -> f64 {
    beamprop::validate::rytov_variance(cn2, wavelength, z)
}

/// Kruse-model aerosol extinction (1/m) from meteorological visibility (m).
#[pyfunction]
fn kruse_extinction(wavelength: f64, visibility: f64) -> f64 {
    beamprop::medium::kruse_extinction(wavelength, visibility)
}

fn set_array2(d: &Bound<'_, PyDict>, key: &str, a: Array2<f64>) -> PyResult<()> {
    d.set_item(key, a.into_pyarray(d.py()))
}

/// Propagate a Gaussian beam through vacuum or uniform Beer–Lambert
/// extinction (the CLI `propagate` case). Returns a dict of numpy arrays
/// (`xz`, `frames`, `final`) and scalar diagnostics.
#[pyfunction(name = "run_propagate")]
#[pyo3(signature = (n=512, dx=1e-3, wavelength=1e-6, w0=1e-2, z=None, steps=200,
                    frames=5, alpha=None, visibility=None))]
#[allow(clippy::too_many_arguments)]
fn py_run_propagate<'py>(
    py: Python<'py>,
    n: usize,
    dx: f64,
    wavelength: f64,
    w0: f64,
    z: Option<f64>,
    steps: usize,
    frames: usize,
    alpha: Option<f64>,
    visibility: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    checked_grid(n, dx)?;
    checked_beam(wavelength, w0)?;
    let r = run_propagate(&PropagateParams {
        n,
        dx,
        wavelength,
        w0,
        z,
        steps,
        frames,
        alpha,
        visibility,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    set_array2(&d, "xz", r.xz)?;
    d.set_item("frames", r.snapshots.into_pyarray(py))?;
    d.set_item("snapshot_z", r.snapshot_z)?;
    set_array2(&d, "final", r.final_intensity)?;
    d.set_item("z", r.z_total)?;
    d.set_item("dz", r.dz)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("width_x", r.width_x)?;
    d.set_item("transmission", r.transmission)?;
    d.set_item("guard_frac", r.guard_frac)?;
    Ok(d)
}

/// Monte-Carlo propagation through von Kármán turbulence (the CLI
/// `turbulence` case). Returns a dict with the per-realization receiver and
/// side-view stacks, the long-exposure mean, and path statistics.
#[pyfunction(name = "run_turbulence")]
#[pyo3(signature = (n=512, dx=1e-3, wavelength=1e-6, w0=1e-2, z=1000.0, screens=10,
                    cn2=1.5e-14, l0=1e3, realizations=48, seed=1))]
#[allow(clippy::too_many_arguments)]
fn py_run_turbulence<'py>(
    py: Python<'py>,
    n: usize,
    dx: f64,
    wavelength: f64,
    w0: f64,
    z: f64,
    screens: usize,
    cn2: f64,
    l0: f64,
    realizations: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    checked_grid(n, dx)?;
    checked_beam(wavelength, w0)?;
    if screens == 0 || realizations == 0 {
        return Err(PyValueError::new_err(
            "screens and realizations must both be at least 1",
        ));
    }
    let r = run_turbulence(&TurbulenceParams {
        n,
        dx,
        wavelength,
        w0,
        z,
        screens,
        cn2,
        l0,
        realizations,
        seed,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("frames", r.frames.into_pyarray(py))?;
    d.set_item("xz_frames", r.xz_frames.into_pyarray(py))?;
    set_array2(&d, "longexp", r.longexp)?;
    d.set_item("guard_frac_mean", r.guard_frac_mean)?;
    d.set_item("substeps", r.substeps)?;
    d.set_item("r0", beamprop::validate::fried_r0(cn2, wavelength, z))?;
    d.set_item(
        "rytov",
        beamprop::validate::rytov_variance(cn2, wavelength, z),
    )?;
    Ok(d)
}

/// Coupled steady-state thermal blooming (the CLI `blooming` case). Returns a
/// dict of numpy arrays and the blooming diagnostics (N_φ, Péclet, centroid).
#[pyfunction(name = "run_blooming")]
#[pyo3(signature = (n=512, dx=1e-3, wavelength=1e-6, w0=1e-2, power=1e4, wind=2.0,
                    alpha_abs=1e-5, t0=288.15, p0=101_325.0, z=500.0, steps=200, frames=5))]
#[allow(clippy::too_many_arguments)]
fn py_run_blooming<'py>(
    py: Python<'py>,
    n: usize,
    dx: f64,
    wavelength: f64,
    w0: f64,
    power: f64,
    wind: f64,
    alpha_abs: f64,
    t0: f64,
    p0: f64,
    z: f64,
    steps: usize,
    frames: usize,
) -> PyResult<Bound<'py, PyDict>> {
    checked_grid(n, dx)?;
    checked_beam(wavelength, w0)?;
    let r = run_blooming(&BloomingParams {
        n,
        dx,
        wavelength,
        w0,
        power,
        wind,
        alpha_abs,
        t0,
        p0,
        z,
        steps,
        frames,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    set_array2(&d, "xz", r.xz)?;
    d.set_item("frames", r.snapshots.into_pyarray(py))?;
    d.set_item("snapshot_z", r.snapshot_z)?;
    set_array2(&d, "final", r.final_intensity)?;
    d.set_item("dz", r.dz)?;
    d.set_item("n_phi", r.n_phi)?;
    d.set_item("peclet", r.peclet)?;
    d.set_item("delta_t_sat", r.delta_t_sat)?;
    d.set_item("centroid_x", r.centroid_x)?;
    d.set_item("transmission", r.transmission)?;
    d.set_item("guard_frac", r.guard_frac)?;
    Ok(d)
}

/// Sweep the 0-D optical-breakdown threshold against pressure and record the
/// electron avalanche (the CLI `breakdown` case, M6a). Returns a dict of numpy
/// arrays and scalar diagnostics.
///
/// Pure rate physics: no grid, no propagator, so none of the beam arguments
/// apply. `threshold*` and `drive_intensity` come back in **W/cm²**, the unit
/// the breakdown literature is quoted in, matching the CLI's CSV.
#[pyfunction(name = "run_breakdown")]
#[pyo3(signature = (wavelength=1064e-9, fwhm=6e-9, p_min_torr=300.0, p_max_torr=2000.0,
                    points=48, steps=400, drive=1.08))]
#[allow(clippy::too_many_arguments)]
fn py_run_breakdown<'py>(
    py: Python<'py>,
    wavelength: f64,
    fwhm: f64,
    p_min_torr: f64,
    p_max_torr: f64,
    points: usize,
    steps: usize,
    drive: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = run_breakdown(&BreakdownParams {
        wavelength,
        fwhm,
        p_min_torr,
        p_max_torr,
        points,
        steps,
        drive,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("pressure_torr", r.pressure_torr)?;
    d.set_item("threshold", r.threshold)?;
    d.set_item("threshold_lo_slope", r.threshold_lo_slope)?;
    d.set_item("threshold_hi_slope", r.threshold_hi_slope)?;
    d.set_item("cascade_theory", r.cascade_theory)?;
    d.set_item("neutral_density", r.neutral_density)?;
    d.set_item("ne_traces", r.ne_traces.into_pyarray(py))?;
    d.set_item("trace_time", r.trace_time)?;
    d.set_item("slope", r.slope)?;
    d.set_item("slope_envelope", r.slope_envelope)?;
    d.set_item("drive_intensity", r.drive_intensity)?;
    d.set_item("n_bd", r.n_bd)?;
    d.set_item("n_seed", r.n_seed)?;
    Ok(d)
}

/// Ignite a spark at the M6a threshold and run the laser-supported detonation
/// wave the sustaining beam drives back up itself (the CLI `lsd` case, M6c).
///
/// `profiles` is `[frame, quantity, cell]` with quantities ordered
/// `p (Pa), rho (kg/m^3), u (m/s), alpha (1/m), I (W/m^2)`; `quantities` in the
/// returned dict names them, so a caller never has to hard-code the order.
///
/// If the igniting pulse is below threshold the run returns with
/// `ignited=False` and empty arrays — a clean report rather than an error, the
/// same contract the CLI has.
#[pyfunction(name = "run_lsd")]
#[pyo3(signature = (wavelength=1064e-9, fwhm=6e-9, ignite_power=1.5e7, w_focus=20e-6,
                    drive=1e11, p0=101_325.0, t0=288.0, length=2.5e-2, cells=2500,
                    alpha=2e4, cross=0.5, frames=48, ignition_steps=400))]
#[allow(clippy::too_many_arguments)]
fn py_run_lsd<'py>(
    py: Python<'py>,
    wavelength: f64,
    fwhm: f64,
    ignite_power: f64,
    w_focus: f64,
    drive: f64,
    p0: f64,
    t0: f64,
    length: f64,
    cells: usize,
    alpha: f64,
    cross: f64,
    frames: usize,
    ignition_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = run_lsd(&LsdParams {
        wavelength,
        fwhm,
        ignite_power,
        w_focus,
        drive,
        p0,
        t0,
        length,
        cells,
        alpha,
        cross_fraction: cross,
        frames,
        ignition_steps,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("ignited", r.ignited)?;
    d.set_item("i_peak", r.i_peak)?;
    d.set_item("i_threshold", r.i_threshold)?;
    d.set_item("drive", r.drive)?;
    d.set_item("rho_0", r.rho_0)?;
    d.set_item("d_measured", r.d_measured)?;
    d.set_item("d_raizer", r.d_raizer)?;
    d.set_item("x", r.x)?;
    d.set_item("frame_time", r.frame_time)?;
    d.set_item("profiles", r.profiles.into_pyarray(py))?;
    d.set_item(
        "quantities",
        vec!["p_Pa", "rho_kg_m3", "u_m_s", "alpha_1_m", "I_W_m2"],
    )?;
    d.set_item("front_x", r.front_x)?;
    d.set_item("optical_depth", r.optical_depth)?;
    d.set_item("log10_transmission", r.log10_transmission)?;
    d.set_item("deposited_energy", r.deposited_energy)?;
    d.set_item("energy_residual", r.energy_residual)?;
    d.set_item("boundaries_undisturbed", r.boundaries_undisturbed)?;
    d.set_item("ib_alpha", r.ib_alpha)?;
    d.set_item("ib_optical_depth", r.ib_optical_depth)?;
    d.set_item("ib_alpha_co2", r.ib_alpha_co2)?;
    d.set_item("ib_optical_depth_co2", r.ib_optical_depth_co2)?;
    Ok(d)
}

/// Sweep turbulence strength and report how often a focused beam still ignites
/// the air, and where the spark lands (the CLI `ignition` case, M6a.2).
///
/// **`p_ignite`'s position on the `cn2` axis is not a validated claim.** It is
/// set by M6a's absolute breakdown threshold, which is explicitly ungated, so
/// the shape of the curve is the result and its position is not. `p_ignite_se`
/// is the binomial standard error and any plot must carry it — a Bernoulli mean
/// over `realizations` samples has no tighter error than that.
///
/// `realizations` is per sweep point, so the run costs `points × realizations`
/// propagations.
#[pyfunction(name = "run_ignition")]
#[pyo3(signature = (n=256, dx=2e-3, wavelength=1064e-9, w0=0.05, z=1000.0, screens=10,
                    l0=100.0, aperture=0.3, focal_length=10.0, power=3e9, fwhm=6e-9,
                    p0=101_325.0, ignition_steps=200, realizations=48, seed=7,
                    cn2_min=1e-16, cn2_max=1e-13, points=13))]
#[allow(clippy::too_many_arguments)]
fn py_run_ignition<'py>(
    py: Python<'py>,
    n: usize,
    dx: f64,
    wavelength: f64,
    w0: f64,
    z: f64,
    screens: usize,
    l0: f64,
    aperture: f64,
    focal_length: f64,
    power: f64,
    fwhm: f64,
    p0: f64,
    ignition_steps: usize,
    realizations: usize,
    seed: u64,
    cn2_min: f64,
    cn2_max: f64,
    points: usize,
) -> PyResult<Bound<'py, PyDict>> {
    checked_grid(n, dx)?;
    checked_beam(wavelength, w0)?;
    let r = run_ignition_sweep(&IgnitionSweepParams {
        base: IgnitionParams {
            n,
            dx,
            wavelength,
            w0,
            z,
            screens,
            cn2: cn2_min,
            l0,
            aperture,
            focal_length,
            power,
            fwhm,
            p0,
            ignition_steps,
            realizations,
            seed,
        },
        cn2_min,
        cn2_max,
        points,
    })
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("cn2", r.cn2)?;
    d.set_item("r0", r.r0)?;
    d.set_item("d_over_r0", r.d_over_r0)?;
    d.set_item("p_ignite", r.p_ignite)?;
    d.set_item("p_ignite_se", r.p_ignite_se)?;
    d.set_item("mean_focal_ratio", r.mean_ratio)?;
    d.set_item("median_focal_ratio", r.median_ratio)?;
    d.set_item("wander_rms", r.wander_rms)?;
    set_array2(&d, "focal_ratio", r.focal_ratio)?;
    set_array2(&d, "strehl", r.strehl)?;
    d.set_item("i_focus_vacuum", r.i_focus_vacuum)?;
    d.set_item("i_threshold", r.i_threshold)?;
    d.set_item("transition_decades", r.transition_decades)?;
    Ok(d)
}

#[pymodule(name = "beamprop")]
fn beamprop_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrid>()?;
    m.add_class::<PyField>()?;
    m.add_class::<PyMedium>()?;
    m.add_class::<PyPropagator>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_function(wrap_pyfunction!(fried_r0, m)?)?;
    m.add_function(wrap_pyfunction!(rytov_variance, m)?)?;
    m.add_function(wrap_pyfunction!(kruse_extinction, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_turbulence, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_blooming, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_breakdown, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_lsd, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_ignition, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
