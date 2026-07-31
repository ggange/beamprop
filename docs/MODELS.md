# Physical models and references

Every physical model in `beamprop`, with its governing equation, where it is
implemented, the validation gate that pins it, and the literature it comes
from. This file is the citation record for the solver: if a formula is in the
code, it is in this table.

Conventions: `λ` vacuum wavelength (m), `k = 2π/λ` (rad/m), `z` propagation
distance (m), `κ` transverse spatial frequency (rad/m), intensity `I = |u|²`.

`D`- and `T`-numbered tags (`D5`, `T4`, …) label the project's design decisions
and implementation tasks. They are **shorthand, not citations**: wherever one
carries weight the substance is stated in full alongside it, so nothing here
depends on looking a number up.

## Claims ledger

**Read this before the test count.** `cargo test` runs 219 tests. That number is
not a measure of how much of this solver is validated against the world, and
reading it as one would be a mistake: most of those tests check that the code
solves the equations it was given, several deliberately assert a *known
disagreement* so it cannot drift silently, and some of the constants that move
the answer most are not gated at all. This section says which is which.

Every claim below carries exactly one of four statuses:

- **verified** — reproduces a closed form or analytic result the model does not
  itself assume (an exact Riemann solution, Noll's Zernike coefficients, the
  Rytov weak-fluctuation variance, the Keldysh limits), or a numerical property
  the model must have (observed order of accuracy, step/window/seed
  independence, thread-count determinism). Verification says the code is right
  about the equations. It says nothing about whether the equations are right
  about air.
- **validated** — compared against something external and measured: a digitized
  published dataset, or an independent third-party code. This is the only status
  that puts the model against the world.
- **pinned** — a **known disagreement**, asserted green so that its size is
  fixed in CI and any change to it has to be argued for. A pinned gate passing
  means the gap is still exactly as big as we said it was. It does **not** mean
  the model agrees with anything.
- **ungated** — asserted from literature or from geometry, with no test behind
  it. These are where the model is most exposed.

Two rows carry an extra flag in the number column:

- **circular** — the reference is the construction the model is built from, so
  agreement is arithmetic, not evidence (`lsd_front_speed_matches_the_raizer_closed_form`).
- **same lineage** — the reference is a closed form from the same paper whose
  data is the comparison target, so it establishes agreement with *that theory*,
  not with the measurement (`tt2012_cascade_theory_reference`,
  `tt2012_wavelength_scaling_matches_cascade_theory`).

**Census of the 118 rows below: 80 verified, 10 validated, 16 pinned, 12 ungated.**
Every `site` names a test function or a `src/` symbol; a claim with neither does
not belong in this table. The remaining unit tests in `src/` are code-level
verification (constructors, guards, closed-form limits of individual rate terms)
and are counted here only where they carry a physics claim in their own right.

### M1 — Diffraction

| claim | site | status | number |
|---|---|---|---|
| Gaussian beam evolution matches the closed form | `gaussian_free_space_evolution` | verified | width and divergence < 1 % |
| Lossless propagation conserves power | `power_conservation_lossless` | verified | ~1e-14 |
| The boundary absorbs rather than wrapping around | `boundary_absorbs_instead_of_wrapping` | verified | wraparound suppressed |
| Split-step is 2nd order in `dz` | `split_step_is_second_order` | verified | observed order 2 |
| Long-throw path matches the Fresnel impulse response | `fresnel_impulse_response_long_throw` | verified | closed form |
| Phase bends the beam toward higher index | `medium_phase_bends_toward_higher_index` | verified | sign of the deflection |
| A positive-index duct focuses | `positive_index_duct_focuses` | verified | focusing recovered |
| `Medium` implementations are interchangeable | `medium_trait_interchangeability` | verified | identical results |

### M2 — Attenuation

| claim | site | status | number |
|---|---|---|---|
| Uniform extinction matches `exp(−α z)` | `beer_lambert_matches_closed_form` | verified | ~1e-13 |
| `α = 0` is bit-identical to vacuum | `zero_extinction_matches_vacuum` | verified | bitwise |
| A transverse absorber removes exactly the predicted power | `transverse_extinction_removes_predicted_power` | verified | exact |

### M3 — Turbulence

| claim | site | status | number |
|---|---|---|---|
| Phase-screen structure function is Kolmogorov | `phase_screen_structure_function_matches_kolmogorov` | verified | < 10 % over a decade of lags |
| Long-exposure spread matches Andrews–Phillips | `long_exposure_beam_spread_matches_theory` | verified | 0.5 % |
| Scintillation index matches Rytov weak theory | `scintillation_index_matches_rytov_weak_theory` | verified | 1.6 % |
| Monte-Carlo is thread-count reproducible | `monte_carlo_reproducible_across_thread_counts` | verified | bitwise |

### M4 — Thermal blooming

| claim | site | status | number |
|---|---|---|---|
| Closed-form erf blooming phase | `b1_closed_form_blooming_phase` | verified | 0.39 % max |
| Weak-blooming first-order limit, quadratic back-reaction | `b2_weak_blooming_linear_limit` | verified | 0.008 %; ratio 3.65 vs 4 |
| Slab coupling is 2nd order by self-convergence | `coupling_is_second_order` | verified | slope 2.000 |
| Stable at `N_φ` = 20 with a closed power budget | `strong_blooming_is_stable` | verified | budget closes |
| The beam bends upwind | `beam_bends_upwind` | verified | sign of the centroid shift |
| Crescent + irradiance-rollover signatures | `b3_qualitative_signatures` | verified | qualitative |
| **Smith (1977) whole-beam `I_REL(N)` curve** | `b3_smith1977_curve_quantitative` | **validated** | 7.2 % over `N ∈ [0.5, 1.8]`, `F₀ = 5` |
| `IntensityScale` extraction is arithmetically identical | `g0_intensity_scale_extraction_is_arithmetically_identical` | verified | bitwise |
| The M4 air table is untouched by T4 | `g0_m4_air_table_is_untouched` | verified | bitwise |

### M6a — Optical breakdown threshold

The milestone with the most pins, and the reason this ledger exists. Its
external status in one line: **`K_m` is validated, the digitizations are
validated, the shape is not** — the kernel's `I_thr(p)` disagrees with both
measured datasets, and the disagreement is pinned rather than papered over.

| claim | site | status | number |
|---|---|---|---|
| Electron-neutral collision frequency `K_m` vs T&T's `E_eff/E_B` | `tt2012_collision_frequency_matches_literature` | **validated** | ratio 1.05×, flat to < 0.15 over 46–1858 Torr |
| `E_eff` rises with pressure (the sign is the physics) | `tt2012_effective_field_rises_with_pressure` | **validated** | `p^+0.695` vs predicted `p^+0.642` |
| Chylek Fig. 3 air trace reproduces the printed exponent | `chylek1990_digitization_reproduces_the_published_slope` | **validated** | α = 0.43–0.46 vs printed 0.45 ± 0.01 |
| Chylek Fig. 2 He/Ar/Xe traces reproduce six printed exponents | `chylek1990_fig2_digitization_reproduces_the_published_slopes` | **validated** | all six inside their printed tolerances |
| Level vs T&T's cascade closed form (Eq. 4) | `tt2012_cascade_theory_reference` | verified *(same lineage)* | 4.1–5.1× (climb), 1.3–3.2× (fixed `⟨ε⟩`) |
| `I_thr ∝ λ⁻²`, and the plateau is Eq. 4's `λ⁻²` coefficient | `tt2012_wavelength_scaling_matches_cascade_theory` | verified *(same lineage)* | −2.000 over 0.53–10.6 µm; coefficient 1.01× |
| Keldysh `γ → ∞` recovers the multiphoton photon order | `keldysh_multiphoton_limit_recovers_the_photon_order` | verified | `K = U_i/ħω` |
| Keldysh `γ → 0` recovers the static-field tunnelling exponent | `keldysh_tunnelling_limit_matches_the_static_field_exponent` | verified | closed form |
| The small-`γ` series joins the direct form | `keldysh_exponent_series_matches_direct_form` | verified | join pinned at γ = 0.1 |
| Dawson's integral against its closed forms and its ODE | `breakdown0d::tests::dawson_matches_its_closed_forms` | verified | max at 0.5410442246; `Φ' = 1 − 2xΦ` to 1e-5 |
| The two Dawson branches join continuously | `breakdown0d::tests::dawson_series_join_is_continuous` | verified | 3e-7 at the `x` = 4 seam |
| `ln Γ` against factorials, `Γ(½)`, and the reflection formula | `breakdown0d::tests::ln_gamma_matches_its_closed_forms` | verified | 1e-12 |
| The PPT above-threshold sum is converged at every `γ` it is evaluated at | `breakdown0d::tests::ppt_ati_sum_is_converged` | verified | adaptive vs 2²¹ terms, 1e-6, down to the γ = 0.1 cutover |
| PPT reduces to the ADK closed form as `γ → 0` | `breakdown0d::tests::ppt_reduces_to_adk_in_the_tunnelling_limit` | verified | residual `O(γ²)` and falling, 3e-2 → 1e-3 |
| The tunnelling branch joins the sum without a step | `breakdown0d::tests::ppt_tunnelling_branch_joins_the_sum` | verified | converged `A₀` = 0.994 vs the limit's 1 at the cutover |
| The PPT rate is monotonic across the bisection bracket | `breakdown0d::tests::ppt_rate_is_monotonic_across_the_bisection_bracket` | verified | 201 samples over 10¹²–10²² W/m², both λ |
| PPT gives an **integer** photon order `K = ⌈ν⌉` | `ppt_multiphoton_order_is_the_integer_photon_count` | verified | 7.998 / 10.998 / 6.000 at 800 / 1064 / 532 nm |
| PPT and Keldysh share one exponent | `ppt_and_keldysh_share_the_same_exponent` | verified | ratio is `I^0.66` while the rates are `I^9.9` |
| **PPT's absolute rate vs a measured O₂ cross-section** | `ppt_rate_matches_the_measured_o2_cross_section` | **validated** | 1.99× of `σ₈` = 3.3e-130 W⁻⁸m¹⁶s⁻¹ (Sci. Rep. **8**, 2874 (2018)), nothing fitted; theory high, the direction the paper reports |
| **PPT does not close the wavelength gap either** | `ppt_does_not_close_the_wavelength_gap_either` | **pinned** | derived prefactor lands at 2.947 vs measured 0.80 — 16 % of the gap; `2n* − 3/2 < 0` for `Z_eff` = 0.53, so the Coulomb correction is order-unity, not orders |
| **The two anchor experiments sit either side of the MPI seeding threshold** | `ppt_seeding_thresholds_separate_the_two_experiments` | **validated** | at each paper's own measured `I_th`: `N_seed` = 5.4e-9 (1064 nm) vs 3.15 (532 nm); seeding threshold 5.73× above measured at 1064 nm, 0.83× at 532 nm |
| Threshold is independent of the integration window | `breakdown0d::tests::threshold_is_window_independent` | verified | invariant in `w` |
| The high-pressure slope lies between the model's analytic limits | `breakdown0d::tests::high_pressure_threshold_slope_lies_between_analytic_limits` | verified | 0.095 … 0.468 |
| The literature-range inelastic envelope brackets the slope | `breakdown0d::tests::inelastic_loss_envelope_brackets_the_slope` | verified | over `δ_eff` 0.01–0.05, `⟨ε⟩` 2–5 eV |
| Per-slice integrator is exact and step-size independent | `breakdown0d::tests::{pure_cascade_is_exponential_growth, pure_loss_is_exponential_decay, mpi_only_seeding_is_linear, balance_point_is_linear_from_seed, slice_refinement_is_consistent}` | verified | 1e-9 relative |
| **Measured `I_thr(p)` slope, against the `δ_eff` literature envelope** | `tt2012_threshold_slope_matches_measurement` | **validated** | measured 0.329 inside `[0.174, 0.382]`; centre 0.264. Red and `#[ignore]`d 2026-07-25 → 2026-07-30, retired by the closure change, not by re-banding |
| **Chylek's low-pressure branch: the kernel is still far too steep** | `chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not` | **pinned** | 10–100 Torr: kernel 1.293 vs measured 0.428, down from 1.954 once the free-molecular escape landed. The high-pressure window *agrees* (0.431 vs 0.468), so the failure is one-sided |
| **The residual against Chylek is localised to 70–350 Torr** | `chylek1990_residual_is_localised_to_mid_pressure` | **pinned** | six matched bands: kernel tracks to <0.25 outside, 2.0–2.3× too steep inside; the measurement is *not* locally flat (0.31–0.55) |
| The mid-pressure residual is not a level artifact | `the_mid_pressure_residual_is_not_a_level_artifact` | verified | walking `δ_eff` from a 15.8× level to 3.4× makes the peak local exponent *rise*, 1.04 → 1.42 |
| **The wavelength ratio is falsified against measurement, in sign** | `chylek1990_tt2012_wavelength_ratio_falsifies_cascade_lambda_squared` | **pinned** | kernel 3.39 vs measured ≈ 0.80; overshoot ≈ 4.24× (was 3.99 / 4.99× before the closure change) |
| **Keldysh MPI does not close the wavelength gap** | `keldysh_mpi_does_not_close_the_wavelength_gap` | **pinned** | order-unity prefactor lands at 2.89 vs measured 0.80; the 18 % of the gap it closes is a smaller denominator, not a better rate |
| **T&T's own MPI calibration undershoots their own measurement** | `breakdown0d::tests::tt2012_mpi_calibration_undershoots_the_data` | **pinned** | 37× below |
| Level offset stays inside the inter-lab scatter, with a drift pin | `tt2012_level_ratio_is_bounded_within_scatter` | **pinned** | 3.90–4.69× high; drift 1.48× → 1.20× as the slope error shrank |
| The two cascade limits bracket the measurement | `breakdown0d::tests::the_two_cascade_models_bracket_the_measurement` | **pinned** | 0.468 / 0.095 straddle 0.329 — but this is a **one-parameter sensitivity**, not two independent limits |
| **The continuum diffusion loss is invalid at low pressure** | `the_diffusion_approximation_is_invalid_at_low_pressure` | verified | `Kn` = 0.013 at 760 Torr → **0.96 at 10 Torr**; `Kn ∝ 1/p` exactly |
| Escape rate recovers the continuum and free-molecular limits | `escape_rate_recovers_both_limits` | verified | 0.9375 / 0.9757 / 0.9951 of `D_e/Λ²` at 760 / 2000 / 10⁴ Torr; saturates at `v̄/ℓ` |
| The escape correction adds no constant | `the_escape_correction_adds_no_constant` | verified | `v̄` reproduces `D_e` to 1e-12; 6.740 eV, the same energy `D_e` implies |
| `Λ`, the Cauchy chord and `V` are separate scales from one geometry | `focus_geometry_separates_its_three_length_scales` | verified | `Λ` = 7.74 µm, `ℓ = 4V/S` = 30.72 µm, ratio 3.97 |
| **Free-molecular escape halves the low-pressure slope error, and no more** | `free_molecular_escape_flattens_the_low_pressure_branch` | **pinned** | 10–100 Torr: 1.954 → **1.293** vs measured 0.428 — still 2.6× steep; the high-pressure window is undisturbed (0.431 vs 0.468) |
| First-passage rate reduces to the mean-trajectory closed form as `D_ε → 0` | `first_passage_reduces_to_the_mean_energy_climb` | verified | ratio 0.821 → 0.990 as `ħω` falls 1.166 → 0.05 eV; grid-converged at each point |
| First-passage quadrature is 2nd order and the shipped `N` is converged | `first_passage_quadrature_is_converged` | verified | order ≈2; `N` = 512 within 2e-4 at 1064 and 532 nm |
| The `ε_∞ = U_i` bifurcation is gone | `distribution_resolved_has_no_bifurcation` | verified | `ν_i > 0` below the old cutoff, continuous through it to 2 % |
| The distribution-resolved rate adds no constant | `first_passage_rate_depends_only_on_two_dimensionless_groups` | verified | function of `(ε_∞/U_i, ħω/U_i)` alone; `ν_i ∝ p` preserved |
| **High-pressure slope, resolving the electron energy distribution** | `distribution_resolved_cascade_fixes_the_high_pressure_slope` | **validated** | T&T 0.0859 → **0.2636** vs 0.329; Chylek 0.1509 → **0.4313** vs 0.468; the one-free-constant envelope moves from excluding the measurement to containing it |
| **It does not fix the low-pressure branch** | `distribution_resolved_does_not_fix_the_low_pressure_branch` | **pinned** | 1.289 → 1.293 vs measured 0.428 — localises that failure to the loss term, not the cascade closure |
| **It does not close the wavelength gap** | `distribution_resolved_does_not_close_the_wavelength_gap` | **pinned** | 4.00 → 3.39 vs ≈0.80 — right sign at last, 15 % of a 5× gap |
| The hard plateau floor stops being a bound | `distribution_resolved_softens_the_plateau_floor` | verified | threshold/floor 1.09 → 0.75 → 0.63 at 300 / 760 / 2000 Torr |
| The cascade plateau floor is free of every transport constant | `cascade_plateau_floor_is_independent_of_the_transport_constants` | verified | invariant under `D_e` ×0.01…×100; `ν_i ≡ 0` below it at every pressure |
| **Parameter-free noble-gas floor: ordering right, spacing wrong** | `chylek1990_noble_gas_plateau_floors_are_unequally_tight` | **pinned** | He/Ar predicted 15.6 vs measured ≈2.5; Ar/Xe 4.27 vs ≈3.0. Headroom 1.85×/7.8×/13.2× is a bound for the **mean-trajectory closure only** — see the plateau-softening row |
| Noble-gas `K_m` and `D_e` | `Gas::from_monatomic` | **ungated** | required arguments, not defaults — no citable momentum-transfer table; Ar/Xe cross sections swing 100× across the Ramsauer minimum |
| Chylek's focal geometry (`Λ`, focal volume) | — | **ungated** | the paper gives the lens and spot but not the beam diameter, so the divergence-limited depth of focus cannot be reconstructed as T&T's Eq. 5 was |
| `D_e,ref` = 0.2 m²/s is consistent with the gated `K_m` | `d_e_ref_implies_a_stated_electron_energy` | verified | ⟺ `ε` = 6.740 eV at `p_ref`, inside the (2, `U_i`] eV band the cascade occupies |
| **Diffusion and inelastic loss assume different electron energies** | `d_e_ref_implies_a_stated_electron_energy` | **pinned** | 6.74 eV vs `⟨ε⟩` = 3 eV — **2.25×**, same population, two terms of one balance |
| **`D_e` cannot explain the slope gap** | `d_e_sensitivity_is_pinned_across_the_kinetic_band` | **pinned** | a 6.0× band in `D_e` moves `n` by 0.078 (0.0523 → 0.1305) against a 0.243 shortfall to the measured 0.329 |
| Absolute threshold **level** | — | **ungated** | published thresholds scatter 3–10× across labs |
| `D_e,ref` against a **measurement** | — | **ungated** | swarm data sits at 0.1–2 eV and reaches 6.7 eV only through the same formula, so it would re-validate `K_m`, not `D_e`; needs a measurement at the cascade's own energy |
| `δ_eff` = 0.02 | `AirBreakdown::new` | **ungated** | free within ≈0.01–0.05; sets the plateau level |
| `⟨ε⟩` = 3 eV (`FixedMeanEnergy` only) | `AirBreakdown::new` | **ungated** | free within ≈2–5 eV |
| `n_bd` = 10²³ m⁻³ | `AirBreakdown::new` | **ungated** | asserted; audit says the slope is insensitive to ×0.1/×10 |
| The seed is the attachment/ionization equilibrium | `breakdown0d::tests::background_electron_density_is_the_attachment_equilibrium` | verified | `n_e0·ν_att = q` to 1e-12, against the same `ν_att` the loss term uses |
| The focus holds essentially no free electrons | `breakdown0d::tests::the_focus_holds_essentially_no_free_electrons` | verified | `n_e0` = 0.149 m⁻³ at 1 atm → **1.2×10⁻¹⁴** electrons in the focus |
| The ionization background is not load-bearing | `breakdown0d::tests::ionization_background_is_not_load_bearing` | verified | threshold **bit-identical** over 12 decades of seed (10⁻⁶–10⁶ m⁻³) |
| The seed floor applies to an explicit seed only | `breakdown0d::tests::seed_floor_applies_only_to_an_explicit_seed` | verified | floored vs free peak differ by >10³ at 8×10¹⁵ W/m² |
| `Λ` = 7.74 µm and `ℓ` = 30.72 µm | `Focus::cylinder` | **ungated** | both pinned from T&T's Eq. 5 geometry, never fit; `Λ` matches the 8 µm the paper states |
| The `ε_∞ → U_i` margin at threshold | — | **ungated** | `ε_∞/U_i` = 1.032 at 760 Torr, 1.011 at 1500 — the model sits at the bifurcation that *is* its plateau |

### M6a.2 — Aperture optics and ignition statistics

| claim | site | status | number |
|---|---|---|---|
| Tip/tilt-removed residual variance vs Noll (1976) | `noll_tip_tilt_removed_variance_matches_the_closed_form` | verified | 0.1407 vs 0.134, banded at ±12 % by the ensemble spread |
| Piston-removed variance converges to Kolmogorov | `noll_piston_removed_variance_converges_to_kolmogorov` | verified | 0.34 → 0.99 over `L₀/D` = 10 → 2000 |
| RMS focal-spot wander `∝ Cn²^(1/2)` | `wander_follows_the_square_root_of_cn2` | verified | +0.4953 / +0.4977 / +0.4987 |
| The ignition ensemble converges | `ignition_ensemble_converges` | verified | numerical hygiene |
| The ignition ensemble is thread-count reproducible | `ignition_ensemble_is_reproducible_across_thread_counts` | verified | bitwise |
| Where the ignition curve sits on the `Cn²` axis | — | **ungated** | rides M6a's absolute threshold; the *shape* is the result, the position is not — the figure says so in-panel |

Three gates here were **landed and then withdrawn**, which the ledger records
because a retired gate is a claim that was made and taken back: an aperture-dependence
exponent (seed-dependent — it was measuring the draw, swinging −0.10 to −0.32
across seeds), a `(D/r₀)^(5/3)` exponent (a tautology — the generator scales the
screen linearly in `r₀`), and a width gate (no independent anchor).

### M6c — 1-D gas dynamics and the LSD wave

| claim | site | status | number |
|---|---|---|---|
| G1 — Sod vs the exact Riemann solution | `sod_shock_tube_matches_exact_riemann_solution` | verified | L1(ρ) 6.55e-3 → 6.55e-4, rate 0.79–0.88 |
| G2 — 2nd order on smooth flow | `euler_muscl_hancock_is_second_order_on_smooth_flow` | verified | 1.86 → 1.94 |
| G2b — coupled hydro↔source is 2nd order | `lsd_source_coupling_is_second_order` | verified | 1.99/2.03/1.99 vs a 1st-order contrast at 0.88/1.02/1.07 |
| G3 — LSD velocity vs Raizer's closed form | `lsd_front_speed_matches_the_raizer_closed_form` | verified *(**circular**)* | 5402 vs 5392 m/s, +0.19 % — it is the Chapman–Jouguet construction the model is built from |
| G3b — residual falls as the absorption layer thins | `lsd_front_speed_converges_as_the_absorption_layer_thins` | verified | −8.26 % → +0.19 % |
| G3c — front speed is seed-independent | `lsd_front_speed_is_seed_independent` | verified | 1.1e-3 |
| **G4 — parameter-free `D ∝ S^(1/3)`, `ρ₀^(−1/3)`** | `lsd_velocity_follows_the_parameter_free_one_third_scaling` | verified | `S^+0.33190` over 1.52 decades, `ρ₀^−0.33020` over 1.50, gated inside ±0.01 |
| The level rides the EOS coefficient while the exponents do not | `lsd_velocity_level_tracks_the_eos_coefficient` | verified | level moves 59 % under `γ`; exponents move 0.001 |
| G5 — energy budget closes | `lsd_energy_budget_closes` | verified | 2.1e-16 |
| **G6 — frozen plasma table vs direct Mutation++ off-grid** | `plasma_table_matches_direct_mutationpp_off_grid` | **validated** | worst 1.48e-3 in `n_e` (independent third-party code) |
| G8 — a real beam through the plasma column is Beer–Lambert, `δn ≡ 0` | `plasma_column_absorbs_as_beer_lambert` | verified | 1.7e-13 at τ = 339 |
| The table's charge-state ceiling | `plasma_table_charge_state_ceiling_is_pinned` | **pinned** | regression pin on the table's extrapolation limit |
| G7 — absolute LSD velocity vs measurement | — | **ungated** | **Amended by M6d.** Radial relief is no longer the excuse — it is modelled and pinned at δ = 0.23 (R_b·α = 3.2), i.e. ~23 % of the front speed, which covers part but not all of the ~2× gap to measurement. G7 stays ungated for exactly one reason now: there is no anchored measured dataset (the M6a-D5-style debt, inherited by M6d). Remaining candidates for the rest: radiation losses, incomplete absorption, the production EOS, and the un-refracted beam |

### M6d — Axisymmetric gas dynamics and radial relief

| claim | site | status | number |
|---|---|---|---|
| G9 — the planar 2-D solver reproduces `Euler1d` bit for bit | `euler2d_planar_limit_reproduces_euler1d_bit_for_bit` | verified | bit-identical over 240 cells x 40 steps, with both non-vacuity legs |
| **G10 — Sedov–Taylor point blast** | `sedov_blast_matches_the_self_similar_solution` | verified | exponent 0.38628 vs 2/5; level 1.0842x falling to 1.0587 under refinement; peak compression 2.09 → 2.62 climbing toward 6 |
| The Sedov reference reproduces the published `ξ₀` | `sedov_xi_0_matches_the_published_value` | verified | `ξ₀` = 1.03278 **derived** from the energy integral vs the published ≈1.033 |
| The Sedov profile solves the Euler equations it was derived from | `sedov_profile_satisfies_the_euler_equations` | verified | worst residual 6.9e-5, falling as the finite-difference step squared |
| G11 — 2nd order on smooth axisymmetric flow | `euler2d_is_second_order_on_smooth_axisymmetric_flow` | verified | 1.861 / 1.964 against a split-source contrast at 1.030 / 1.155 |
| G12 — conservation in the `r`-weighted measure | `euler2d_conserves_mass_and_energy_in_the_r_weighted_measure` | verified | mass and energy < 1e-13 closed box; escape term needed and sufficient when the wall is brought in |
| G13 — the axis is not a wall | `the_axis_boundary_does_not_heat_or_starve_the_on_axis_cells` | verified | on-axis entropy defect 2.99e-7 → 3.12e-8 under refinement, vs 3.02e-6 for an even-parity contrast |
| G13(i) — a radially uniform state is a fixed point | `a_radially_uniform_state_is_a_fixed_point_of_the_axisymmetric_operator` | verified | bit-identical in mass and both momenta; energy drift < 1e-13 |
| **G14 — the wide-beam limit reproduces the 1-D column** | `lsd2d_with_a_full_width_beam_reproduces_the_one_dimensional_column` | verified | 3.1e-13 while the front is smooth |
| **The modelled LSD front is transversely unstable** | `lsd2d_with_a_full_width_beam_reproduces_the_one_dimensional_column` | **pinned** | grows out of round-off, amplitude-proportional (10⁶× seed → 10⁶× response), saturating at `\|u_r\|` ≈ 200–400 m/s. Identical in planar and axisymmetric geometry, so it is not the geometric source. A planar solver structurally cannot show it |
| **G15 — radial relief lowers the front speed** | `radial_relief_lowers_the_lsd_front_speed_by_a_pinned_amount` | **pinned** | δ = 0.230 at `R_b·α` = 3.2 and 0.305 at 1.6, monotone in `R_b`; banded at ±13 %, which is the measured spread over grid (+6 %), seed (−7 %) and ignition threshold (±8 %) |
| The relief deficit is not a boundary effect | `src/lsd2d.rs` (`rim_undisturbed`) | verified | 21.1 / 21.3 / 21.3 % at domain radii of 3 / 5 / 8 beam radii |
| G16 — the one-third scaling survives relief | `the_one_third_scaling_survives_radial_relief` | verified | `S^0.34666` at finite `R_b` against the parameter-free 1/3, while the level moves 23 % |
| The beam is not refracted by the plasma | `src/lsd2d.rs` (`BeamProfile`) | **ungated** | independent parallel pencils, by assumption; a two-way beam↔plasma loop is a later milestone |

### What this table says, in one paragraph

Eight claims in this solver are checked against something external and measured,
and two of those eight are M6a data-integrity checks — they establish that a
digitization reproduces its own published figure, not that the model reproduces
the gas. Of the rest, `K_m` and the sign of `E_eff(p)` are agreements about
model *inputs* rather than outputs. **M6a's threshold curve has exactly one
validated agreement, and it is recent and partial**: resolving the electron
energy distribution brought the high-pressure slope from outside the model's
literature envelope to inside it, on both measured datasets, which retired a gate
that had been failing on purpose for five days. It is envelope containment over a
constant still free within a 5× literature range, not a point agreement. The
low-pressure branch, the wavelength ratio, and the noble-gas spacing remain
pinned disagreements — and the fact that one change moved the first and left the
others untouched is what now separates M6a's failures into distinct mechanisms.
M1–M4 are in a different position: their physics is
diffraction, extinction and turbulence, where closed forms are exact and
verification is close to the whole job, and M4 additionally reproduces a
published experimental curve. M6c's core is verified to high order but its one
headline agreement (Raizer) is circular by construction, which is why G4 — a
parameter-free scaling exponent — is the milestone's real physics gate.

M6d changes two things about that picture and neither is a validation. First, it
adds the repo's **first multidimensional verification anchor**: the Sedov–Taylor
blast is a self-similar solution the model is not built from, and its
coefficient `ξ₀` is derived here from the energy integral rather than quoted, so
agreeing with the published value to 0.03 % is evidence rather than bookkeeping.
Second, it retires an *excuse*. M6c's G7 was ungated on the grounds that a
planar solver structurally cannot show radial relief; relief is now modelled and
pinned at 23 % of the front speed, which is a real effect and not the whole ~2×
gap to measurement. G7 remains ungated, but only because no anchored measured
dataset has been acquired — a smaller and more actionable claim than the one it
replaces. M6d also produced something nobody asked it for: the modelled front is
**transversely unstable**, growing cellular structure out of round-off, which a
1-D solver has no way to exhibit and which had to be separated from relief before
the relief number meant anything.

## M1 — Diffraction

### Scalar paraxial propagation, split-step spectral method

The field `u(x, y, z)` obeys the paraxial Helmholtz equation; each slab `dz`
is advanced by the symmetric (Strang) splitting

```text
u(z + dz) = D(dz/2) · M(dz) · D(dz/2) · u(z)
```

with `D` the free-space (vacuum) spectral propagator and `M` the medium
operator applied at the slab centre — second-order accurate in `dz`
(verified: observed order ≈ 2).

- `D` uses the **angular-spectrum transfer function**
  `H(κ) = exp(−i·κ²·dz/(2k))` when the grid resolves it, switching to the
  **Fresnel impulse-response** form for long throws
  (criterion `z_c = N·dx²/λ`).
- Implemented in `src/propagate.rs`; gates in `tests/validation.rs`
  (Gaussian width/divergence < 1 %, power conservation ~1e-14, second-order
  convergence, long-throw Fresnel path).

References:
- J. A. Fleck, J. R. Morris, M. D. Feit, *Time-dependent propagation of high
  energy laser beams through the atmosphere*, Appl. Phys. **10**, 129–160
  (1976) — the original split-step beam-propagation method.
- G. Strang, *On the construction and comparison of difference schemes*,
  SIAM J. Numer. Anal. **5**, 506–517 (1968) — symmetric operator splitting.
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Co.
  (2005) — angular spectrum and Fresnel propagators.
- J. D. Schmidt, *Numerical Simulation of Optical Wave Propagation with
  Examples in MATLAB*, SPIE Press (2010) — sampling criteria, TF vs IR
  propagator selection.

### Gaussian beam evolution (validation reference)

```text
w(z) = w0·√(1 + (z/zR)²),   zR = π·w0²/λ,   θ = λ/(π·w0)
```

Implemented in `src/validate.rs` (`GaussianBeam`).

Reference: A. E. Siegman, *Lasers*, University Science Books (1986), ch. 17.

## M2 — Attenuation

### Beer–Lambert extinction

Power extinction coefficient `α` (1/m) applied as amplitude decay inside the
medium operator: `u ← u·exp(−α·dz/2)`, giving transmission
`T(z) = exp(−α·z)`. Supports transversely varying `α(x, y)` per slab.

Implemented in `src/medium.rs` (`Medium::extinction`, `UniformExtinction`) and
`src/propagate.rs`; gates: uniform extinction matches `exp(−α·z)` to ~1e-13,
transverse absorber removes exactly the predicted power, `α = 0` bit-identical
to vacuum.

Reference: standard radiative transfer (Bouguer–Lambert–Beer); see e.g.
E. J. McCartney, *Optics of the Atmosphere*, Wiley (1976).

### Kruse visibility model (aerosol extinction)

```text
α = (3.912 / V) · (λ / 550 nm)^(−q)
q = 1.6 (V > 50 km),  1.3 (6–50 km),  0.585·V_km^(1/3) (V ≤ 6 km)
```

with `V` the meteorological visibility (Koschmieder 2 % contrast). Implemented
in `src/medium.rs` (`kruse_extinction`).

References:
- P. W. Kruse, L. D. McGlauchlin, R. B. McQuistan, *Elements of Infrared
  Technology*, Wiley (1962).
- I. I. Kim, B. McArthur, E. Korevaar, *Comparison of laser beam propagation
  at 785 nm and 1550 nm in fog and haze*, Proc. SPIE **4214**, 26–37 (2001) —
  the q-exponent branches.

## M3 — Turbulence

### Von Kármán / Kolmogorov phase spectrum

Refractive-index fluctuations with structure constant `Cn²` (m^(−2/3))
integrated over a slab give a phase screen with power spectral density

```text
Φ_φ(κ) = 0.4896 · r0^(−5/3) · (κ² + κ0²)^(−11/6),   κ0 = 2π/L0
```

(`L0` outer scale; the pure Kolmogorov `κ^(−11/3)` limit for `κ ≫ κ0`), with
the plane-wave Fried parameter of the slab

```text
r0 = (0.423 · k² · Cn² · dz)^(−3/5)
```

Implemented in `src/turbulence.rs`, `src/validate.rs` (`fried_r0`).

References:
- A. N. Kolmogorov, Dokl. Akad. Nauk SSSR **30**, 301 (1941) — the −11/3
  inertial-range spectrum.
- D. L. Fried, *Optical resolution through a randomly inhomogeneous medium
  for very long and very short exposures*, J. Opt. Soc. Am. **56**, 1372
  (1966) — `r0`.
- L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through Random
  Media*, 2nd ed., SPIE Press (2005) — von Kármán form, coefficient values.

### FFT phase-screen synthesis with subharmonic compensation

Screens are drawn as `φ = N²·Re(IFFT(a))` with complex-Gaussian mode
amplitudes `a(κ) = (g₁ + i·g₂)·√Φ_φ(κ)·Δκ`, plus 6 levels of Lane-style
subharmonics (3×3 modes at spacing `Δκ/3^p`) to restore the large-scale power
the FFT grid cannot represent. Subharmonic modes use a cell-averaged PSD
(5×5 midpoint rule); the FFT modes use the point value — see the quadrature
note in `src/turbulence.rs::cell_mean_psd`.

Gate: Kolmogorov structure function `D_φ(r) = 6.88·(r/r0)^(5/3)` reproduced
to < 10 % over a decade of separations; screen variance vs the von Kármán
total `σ² ≈ 0.0863·(L0/r0)^(5/3)` within 15 %.

References:
- B. L. McGlamery, *Computer simulation studies of compensation of turbulence
  degraded images*, Proc. SPIE **74**, 225–233 (1976) — FFT screen method.
- R. G. Lane, A. Glindemann, J. C. Dainty, *Simulation of a Kolmogorov phase
  screen*, Waves in Random Media **2**, 209–224 (1992) — subharmonic
  compensation.

### Weak-fluctuation propagation statistics (validation references)

```text
Rytov variance (plane wave):  σ_R² = 1.23·Cn²·k^(7/6)·z^(11/6)
Scintillation index:          σ_I² = ⟨I²⟩/⟨I⟩² − 1 ≈ σ_R²   (σ_R² ≲ 0.3)
Long-exposure beam radius:    W_LT = W(z)·√(1 + 1.33·σ_R²·Λ^(5/6)),
                              Λ = 2z/(k·W(z)²)
```

Implemented in `src/validate.rs`; gates: long-exposure spread 0.5 % off
theory, scintillation index 1.6 % off Rytov.

Reference: L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through
Random Media*, 2nd ed., SPIE Press (2005), chs. 6–8.

### Monte-Carlo reproducibility

Realization `i` draws from `ChaCha12Rng::seed_from_u64(master).set_stream(i)`
with fixed draw order, so ensembles are bitwise reproducible across thread
counts (gated).

Reference: D. J. Bernstein, *ChaCha, a variant of Salsa20* (2008); rand /
rand_chacha crates.

## M4 — Thermal blooming

### Steady-state convection-dominated heating (isobaric)

A high-power beam deposits `α_abs·I` (W/m³) into the air; with crosswind `v`
along `+x` and Péclet number `Pe = ρ·c_p·v·w/κ_t ≫ 1` (asserted > 100 at
construction) the CW steady state is the upwind line integral

```text
ΔT(x, y, z) = (α_abs / (ρ·c_p·v)) · ∫_{−∞}^{x} I(x', y, z) dx'
δn(x, y, z) = −(n₀ − 1)/T₀ · ΔT     (isobaric: Δρ/ρ₀ = −ΔT/T₀, Gladstone–Dale)
```

evaluated slab-locally as a trapezoid cumulative sum. Implemented in
`src/blooming.rs` (`ThermalBlooming`) through the field-aware
`Medium::index_response` path: the propagator hands over the intensity after
the leading half-step of diffraction (the slab-centre field), and the medium
applies the half-slab Beer–Lambert factor `e^(−α_abs·dz/2)` so the heating is
a midpoint rule in absorbed power — without that factor the coupling
measurably degrades to 1st order. Ambient `ρ, c_p, κ_t, n₀−1` come from the
frozen air table (`src/airprops.rs`, embedded `data/air_properties.npy`,
bilinear in `(T, p)`, refractivity rescaled to the run wavelength by Ciddor
dispersion). The absorbed power also leaves the beam (`extinction = α_abs`).

The absolute intensity the heating needs comes from
[`IntensityScale`](../src/field.rs) — `I_phys = (P_beam/P_field)·|u|²`, pinned
from the **launch** field, since propagation conserves `P_field` and any
extinction is already carried by `|u|²`. Blooming computed this inline until
M6c's T4 extraction moved it to `src/field.rs` for the LSD driver to share;
the move is gated as an exact no-op (G0a below).

Gates (`tests/blooming.rs`):
- **B1** closed-form crosswind phase (erf profile) reproduced to 0.39 % max
  over all points with `I > 10⁻⁶·I₀`;
- coupling order by self-convergence: observed slope **2.000 / 2.000**;
- **B2** weak-blooming limit vs an analytic first-order (no back-reaction)
  screen reference, transmission-normalized: 0.008 % agreement at `N_φ = 0.1`,
  back-reaction residual scaling ratio 3.65 (theory: 4, quadratic);
- upwind bend sign, strong-blooming (`N_φ = 20`) boundedness with a closed
  power budget, and the qualitative Smith/Gebhardt signatures (upwind peak
  shift, downwind crescent, peak-irradiance rollover);
- **B3 quantitative** — the coupled solver reproduces Smith's (1977)
  whole-beam steady-state peak-irradiance curve `I_REL(N)` along the `F₀ = 5`
  branch to **7.2 % max** over `N ∈ [0.5, 1.8]` (rollover minimum at `N ≈ 1`
  matched to 0.7 %), against the WebPlotDigitizer trace in
  `tests/data/smith1977_F5.csv`. The largest deviation is at the high-N end,
  where the wave solver shows a mild diffractive recovery (`I_REL` rising
  0.757 → 0.807) that Smith's flat `F₀ = 5` curve does not — it behaves like a
  marginally higher effective Fresnel number there; still inside the ±15 %
  gate.

Two distortion numbers appear. The **phase** number (spec convention, reported
in run notes) measures peak on-axis blooming phase in radians:

```text
N_φ = √(2/π) · k·(n₀−1)·α_abs·P·L / (T₀·ρ·c_p·v·w)
```

Smith's **geometrical-optics** number `N_c` (the x-axis of his curves) carries
no wavenumber — it is a ray-bending measure — with `a = w/√2` the 1/e amplitude
radius and an absorption path factor that → 1 as `α·z → 0`:

```text
N_c = (n₀−1)/T₀ · I₀·α·z² / (n₀·ρ·c_p·v·a) · [2/(αz) − 2/(αz)²·(1 − e^(−αz))]
```

Smith plots against an effective `N = N_c·{path factor with diffraction Q, Ω}`;
for the B3 sub-Rayleigh geometry (`F₀ = 5` ⟹ `z_R = 5z`, `Q ≤ 1.02`; `αz =
0.05`) that factor is within a few percent of unity, so `N ≈ N_c` to ≲4 %.

Both implemented in `src/validate.rs` (`BloomingCase`: closed-form ΔT/phase
references, `distortion_number`, `smith_distortion_number`, A&S 7.1.26 `erf`).

References:
- D. C. Smith, *High-power laser propagation: Thermal blooming*, Proc. IEEE
  **65**, 1679–1714 (1977) — steady-state crosswind theory, the `N_c`
  distortion number and the whole-beam `I_REL(N)` curves (B3, F₀ = 5 branch
  digitized in `tests/data/smith1977_F5.csv`).
- F. G. Gebhardt, *High power laser propagation*, Appl. Opt. **15**,
  1479–1493 (1976) — scaling laws, distortion-number phenomenology.
- R. M. Manning, NASA/TM—2012-217634 (2012) — forced-convection heat budget
  and the closed-form erf thin-screen phase behind B1.
- P. E. Ciddor, Appl. Opt. **35**, 1566–1573 (1996) — air refractivity and
  dispersion.
- E. W. Lemmon et al., J. Phys. Chem. Ref. Data **33**, 111 (2004) — NIST air
  data gating the property table's `κ_t` (see `docs/M4_SPEC.md`).

## M6a — Optical breakdown threshold (0-D kernel)

### Electron-avalanche rate balance

At a point in dry air the electron density obeys the avalanche balance

```text
dn_e/dt = (ν_i(I, p) − ν_att(p) − ν_diff(p))·n_e + S_mpi(I, p)
```

with cascade ionization driven by the **net** power — inverse-bremsstrahlung
heating minus the inelastic excitation losses the electron pays climbing to
`U_i`:

```text
ν_i(I, p)     = max(0, heating − L)/U_i
heating(I, p) = (e²·I)/(m_e·c·ε₀) · ν_m/(ν_m² + ω²),   ν_m = K_m·p,   ω = 2πc/λ
L(p)          = δ_eff·ν_m·⟨ε⟩ ≡ L′·p
```

Both terms scale `∝ p`, so their difference gives `I_thr(p) = L′/h + …/p` — a
constant high-pressure **plateau** on top of the `1/p` avalanche term. Without
`L` the model's flattest reachable slope is exactly `p^-1`; the plateau is what
lets it approach the measured trend at all. `L′ = δ_eff·K_m·⟨ε⟩` is one lumped
constant taken from the centre of its literature range and never tuned.

The loss terms are attachment from measured rate coefficients (dissociative
`k₂·n_O₂ ∝ p` plus three-body `k₃·n_O₂·n ∝ p²`; the two-body channel leads at
1 atm, `5.4×10⁷` against `1.4×10⁷ s⁻¹`, with three-body overtaking it only above
`n = k₂/k₃ = 10²⁶ m⁻³` ≈ 4 atm) and free-electron diffusion,
free-electron **escape** from the focal volume. Attachment is negligible against
escape throughout the gate window — `6.7×10⁷` vs `3.3×10⁹ s⁻¹` at 1 atm.

Escape is *not* `D_e/Λ²`. That is a continuum random-walk result and assumes the
electron collides many times while crossing the focus, which fails badly at low
pressure: the Knudsen number `Kn = λ_mfp/ℓ` runs 0.013 at 760 Torr to **0.96 at
10 Torr** (3.8× against the diffusion length `Λ`), so at the bottom of Chylek's
range the electron leaves ballistically without colliding. An electron cannot
cross the region faster than it can travel across it, so the escape *time* is the
diffusive time plus the ballistic transit time:

```text
ν_esc = 1/(τ_diff + τ_ballistic),   τ_diff = Λ²/D_e,   τ_ballistic = ℓ/v̄
```

reducing to `D_e/Λ²` for `Kn ≪ 1` and saturating at `v̄/ℓ` — pressure-independent
— for `Kn ≫ 1`. Site: `breakdown0d::AirBreakdown::escape_rate`,
`knudsen_number`. **No new constant**: `v̄ = √(3·D_e,ref·p_ref·K_m)` follows from
`D_e = v̄²/(3ν_m)` with the pressure cancelling, giving 1.540e6 m/s — the same
6.740 eV that `d_e_ref_implies_a_stated_electron_energy` reads out of `D_e`.

The three focal length scales are distinct and now come from one geometry
(`breakdown0d::Focus`): `Λ` = 7.74 µm is a diffusion **eigenvalue**, the Cauchy
mean chord `ℓ = 4V/S` = **30.7 µm** is a **distance** (the mean free path of an
isotropically-directed particle leaving a convex body — a theorem, not a model
choice), and `V` sets the seed density. Using `Λ` as the ballistic transit
distance understates the correction by 4.0×.

`D_e,ref` used to be M6a's largest ungated number. It is no longer free: kinetic
theory ties it to the externally-gated `K_m` by `D_e = 2ε/(3 m_e K_m p)`, so
`D_e,ref = 0.2 m²/s` **is** the statement `ε = 6.740 eV`
(`d_e_ref_implies_a_stated_electron_energy`). Sweeping the whole band that
formula admits — `ε` from 2 eV to `U_i`, a 6.0× range in `D_e` — moves the fitted
slope by only 0.101 (0.0532 → 0.1545), against a 0.234 shortfall to the measured
0.329, so `D_e` **cannot** account for the slope gap
(`d_e_sensitivity_is_pinned_across_the_kinetic_band`). Two debts remain, both
recorded rather than papered over: diffusion assumes 6.74 eV while the
`FixedMeanEnergy` loss term assumes `⟨ε⟩` = 3 eV (a 2.25× internal
inconsistency), and there is still no measurement of `D_e` at the cascade's own
energy — swarm data reaches only 0.1–2 eV. See `docs/M6A_SPEC.md`.

`Λ` is the diffusion length of T&T's **divergence-limited** focus, from their
Eq. 5: `(1/Λ)² = (π/l₀)² + (2.405/r₀)²` with `r₀ = f·α/2 = 20 µm` and
`l₀ = 0.414·(α/d)·f² = 66 µm`, giving **`Λ = 7.74 µm`** — matching the 8 µm the
paper states. **Pinned from geometry, never fit.** (Two earlier guesses were
wrong in opposite directions: a *sphere*, `Λ = r₀/π = 6.37 µm`, overstated
`ν_diff` by 1.48×; a diffraction-limited *filament* put the depth of focus at
2.4 mm instead of 66 µm.) Because the focus is set by the beam's 1 mrad
divergence rather than by diffraction, `Λ` and the focal volume are
wavelength-independent — which is what makes the wavelength gate below a clean
one-variable test.

Finally a swappable multiphoton seed `S_mpi = σ_K·I^K·N` (off by default; the
seed is one electron in the focal volume). Breakdown at `n_e ≥ n_bd = 10²³ m⁻³`.

The per-slice ODE is advanced by its exact solution. With `σ_K = 0` the slice
is Bernoulli — growth is **logistic**, since ionization depletes the neutrals it
feeds on — and is evaluated as `n_e' = n_e/(e^{−βdt} + b·n_e·(1 − e^{−βdt})/β)`
with `β = ν_i − ν_loss`, `b = ν_i/N`; that form underflows harmlessly instead of
overflowing to `NaN` far above threshold, and `n_e` saturates at full ionization
rather than running away. Threshold intensity is found by log-bisection; a
pressure sweep gives `I_thr(p)`.

Implemented in `src/breakdown0d.rs`.

Gate (internal, model self-consistency): solving the avalanche criterion gives

```text
I_thr(p) = L′/h + U_i·(ν_att + ν_diff + G)/(h·p),   G = ln(n_bd/n_seed)/τ
```

Attachment is negligible here, so the exponent runs between two **exact**
limits — plateau-dominated `n → 0` and diffusion-dominated `n → 2` — with the
growth-limited `p^-1` in between. Fitted over the pinned range **300–2000
Torr** (8 log-spaced points, 6 ns FWHM) the model gives **n = 0.095**, and the
gate asserts `n ∈ (0, 1)`: the plateau must be doing work, since without `L`
the model is stuck at `n ≥ 1`. Sweeping the literature ranges of `L′`
(δ_eff ≈ 0.01–0.05) gives an envelope `n ∈ [0.023, 0.231]`, separately pinned so
it cannot drift. The level at 760 Torr is **1.18×10¹² W/cm²**, and the
`FixedMeanEnergy` variant gives `n = 0.468` (level 4.58×10¹¹) as the other end
of the bracket. Absolute threshold level is **not** gated (3–10× inter-lab
scatter). Integrator sub-gates are unit tests of the exact per-slice solver,
not physics validation.

External gates (Thiyagarajan & Thompson 2012, digitized into
`tests/data/tt2012_*.csv`; the paper's setup — 1064 nm, 6 ns FWHM, 20 µm radius
focus, 10–2000 Torr — is exactly what the kernel assumes, so nothing is fitted):

- **`K_m` collision frequency — passes.** `E_eff/E_B = ν_m/√(ν_m²+ω²)`, so the
  paper's two curves measure `ν_m` independently of this crate. Implied
  `K_m = 4.21×10⁷` vs the kernel's `3.90×10⁷ s⁻¹Pa⁻¹`; ratio flat at
  1.05 ± 0.01 over 46–1858 Torr. Non-circular anchor.
- **`E_eff(p)` slope — passes.** Predicted `p^+0.642`, measured `p^+0.695`;
  the positive sign confirms the `ν_m ≪ ω` branch.
- **`I_thr(p)` slope vs the MEASURED curve — RED, and now known to be the
  wrong target.** Measured `p^-0.329`; kernel `p^-0.095`. The measured curve
  is not cascade-only (88 % cascade / 12 % MPI at 760 Torr per the paper), so
  no cascade-only kernel can match it. This gate was previously green at `p^-0.356` (an "8 % match"),
  but that rested on an integration artifact: the seed decayed by `e^-60`
  before the pulse arrived, so an arbitrary integration bound supplied most of
  the threshold requirement and, being pressure-dependent, manufactured slope.
  Corrected route: `p^-1.74` → `p^-0.468` (inelastic loss) → `p^-0.095`
  (`⟨ε⟩` eliminated). What survives is a **bracket** — the two cascade limits
  give 0.095 and 0.468 and straddle the measurement — gated by
  `the_two_cascade_models_bracket_the_measurement`. Read narrowly: the two
  variants differ only in where the cascade cuts off (`⟨ε⟩` = 3 eV vs
  `U_i` = 12.06 eV), so the bracket is a *one-parameter sensitivity*, not two
  independent limits, and not a bound — `⟨ε⟩` ≈ 5 eV gives `n` = 0.346 on its
  own, and `⟨ε⟩ = U_i` gives 0.192, inside the interval.
- **"Too flat" is window-specific — the real error is CURVATURE.** The
  `n = 0.095` above is fitted over 300–2000 Torr. Chylek's 532 nm curve extends
  the comparison two decades lower and shows the kernel is not a power law at
  all: local exponents **1.951** (10–100 Torr), **1.047** (100–300), **0.170**
  (300–786), against a measurement that holds **0.41–0.47** throughout on 1.5–6 %
  scatter. The kernel is 4.6× too steep at the bottom, 2.8× too flat at the top,
  and crosses the data near 250 Torr — an 11.5× swing where the measurement
  varies by 1.13×. Same behaviour at 1064 nm, so this is shape, not level or
  wavelength. Gated as
  `chylek1990_air_is_a_power_law_and_the_cascade_kernel_is_not`. Any statement
  that the kernel is simply "too flat" holds only above ~250 Torr.
- **Cascade theory (T&T Eq. 4) — PASSES, and is the apples-to-apples
  reference.** `I_B(CC) = 1.44×10⁶(p_atm² + 2.2×10⁵λ_µm⁻²)` W/cm², implemented
  as `validate::tt2012_cascade_threshold`. Flat at 1064 nm (`n = −0.00002`)
  because `λ⁻²` dominates `p²` by 10⁵ — so the kernel's flatness *agrees* with
  cascade theory. Level: `SelfConsistentClimb` 4.1–5.1× high, `FixedMeanEnergy`
  1.3–3.2×.
- **Wavelength scaling vs Eq. 4 — PASSES, and is the strongest shape check in
  M6a.** Both terms of `I_thr` carry `1/h ∝ ω²`, so the kernel predicts
  `I_thr ∝ λ⁻²`, the same exponent as Eq. 4's dominant term. Over
  **0.53–10.6 µm** (a 20× span, geometry frozen — legitimate, since the focus is
  divergence- not diffraction-limited) both give **−2.000**, with the ratio
  between them constant to `2×10⁻⁵`; the residual is the `(ν_m/ω)²` correction
  at 10.6 µm. Level offsets are flat at 1.64× (`FixedMeanEnergy`) and 4.22×
  (`SelfConsistentClimb`). Sharper still: the plateau `L′/h` and Eq. 4's `λ⁻²`
  coefficient are the *same physical quantity* — `ω²` times the inelastic energy
  loss per collision — and agree to **1.01×** at the literature centre
  (`δ_eff` = 0.02, `⟨ε⟩` = 3 eV). This is a shape agreement on an axis where
  nothing is tunable: `δ_eff·⟨ε⟩` sets the level and cannot produce a `λ`
  exponent. It does *not* independently discover the scaling (the `λ⁻²` is
  analytic in the `ν_m ≪ ω` limit) — it establishes that the two theories share
  it exactly, and fails loudly if that limit is left. Gated by
  `tt2012_wavelength_scaling_matches_cascade_theory`. **Not a pin:**
  `δ_eff·⟨ε⟩` stays asserted from its literature range; re-pinning it *from*
  Eq. 4 would make the level assertions here and in
  `tt2012_cascade_theory_reference` circular, and both would have to be retired
  in the same change, leaving only the exponent.
- **Level vs the measured curve — bounded, drifting, ungated.** The model sits
  4.84× above the data at 380 Torr and 6.97× at 1896 Torr, inside the ungated
  3–10× inter-lab scatter. The 1.48× drift is the residual slope error in
  absolute clothing; an earlier "flat within 1.16×" claim was withdrawn with the
  artifact that produced it. Converting `E_B` to intensity uses
  `I = ε₀cE_rms²`, since the `E_eff` ratio establishes `E_B` is an RMS
  amplitude.
- **MPI calibrated to the paper's own estimate — implemented, left OFF.**
  Anchoring a rate to their `I_B(MPI) = 4.42×10⁹ W/cm²` collapses the threshold
  to 5.5×10⁹, 37× below their own measurement, and contradicts the paper's own
  88 %-cascade accounting. Their number is an order-of-magnitude significance
  indicator (Nelson's flux-density criterion, whose constant the paper never
  states), not a rate anchor. A real `σ_K` from multiphoton cross-section data
  is the open item.

**D5 debt — DISCHARGED 2026-07-30, in the negative.** With the slope gate red,
the plan's fallback clause required an anchor independent of the kernel's own
coefficients. Eq. 4 supplied that for the **exponent** (`λ⁻²` is untouched by
any coefficient choice) but not for the **level**: it is the same paper, and its
`λ⁻²` coefficient implies the same `δ_eff·⟨ε⟩ = 0.060 eV` the kernel already
assumes, so the 1.01× agreement is intra-lineage consistency, not corroboration.

The second dataset the clause asked for is now in the suite — **Chylek et al.
1990, clean air at 532 nm** (`tests/data/chylek1990_air_threshold_vs_pressure.csv`,
digitized programmatically by `scripts/digitize_chylek1990.py`). It is the anchor
D5 specified: different group, different apparatus, and a different wavelength —
exactly half T&T's 1064 nm, at a nearly identical pulse length (6.5 vs 6 ± 1 ns)
and focal radius (16.5 vs 20 µm). That last point is what makes it usable: the
paper's own Sec. II names pulse duration and focal spot as the reason literature
values of `α` contradict one another, and here they are matched, so the 532/1064
comparison measures the *wavelength* scaling rather than two different benches.

It does not corroborate the model — it falsifies the `λ⁻²` prediction against
measurement:

```text
cascade / kernel:  I_th(532)/I_th(1064) = 3.99   (= λ⁻²; shorter λ costs more)
measured:          I_th(532)/I_th(1064) ≈ 0.80   (532 nm breaks down EASIER)
```

Wrong by ~5×, and wrong in **sign**. At 532 nm the multiphoton order falls from
`K = ⌈12.06/1.166⌉ = 11` photons to `⌈12.06/2.33⌉ = 6`, so the MPI channel the
kernel leaves OFF is enormously stronger exactly where the measurement drops —
the same missing channel the pressure-slope gates indict, seen on a second axis.
Gated as `chylek1990_tt2012_wavelength_ratio_falsifies_cascade_lambda_squared`.

**Consequence for how M6a is described.** `λ⁻²` remains a correct statement about
the kernel's internal structure and its gate stays — it fails loudly if the IB
Lorentzian limit is ever left. It may **not** be called external agreement with
measurement, and it is no longer M6a's headline. M6a's honest status: verified
against cascade theory, falsified against air on both the pressure axis and the
wavelength axis. Does not block M6c (gated separately on Chapman–Jouguet
velocity). See `docs/M6A_SPEC.md` § Fallback.

Open question M6a hands forward: the measured `n` = 0.329 is unreachable by any
cascade-only model, since accepted cascade theory is flat at this wavelength.
Closing it means the MPI contribution the paper itself invokes (12 % at
760 Torr, dominant below 100 Torr), not a flatter cascade — and separately, a
distribution-resolved cascade rate, since the default variant's near-flatness
comes from putting every electron on the mean trajectory (at threshold it runs
within 0.8 % of the `ε_∞ = U_i` pole at 2000 Torr, where that idealization is
least defensible).

**The obvious MPI candidate has been tried and it fails — 2026-07-30.** Keldysh
photoionization is implemented (`breakdown0d::keldysh_rate`) and **verified**
against both closed-form limits of its own exponent: the multiphoton branch
recovers the photon order `U_i/ħω` to better than 0.2 % (10.35 at 1064 nm, 5.17
at 532 nm) and the tunnelling branch reproduces the static-field exponent
`4√(2m)U_i^{3/2}/(3ħeE)` to 1 part in 10⁶. There is nothing tunable in that
exponent, which is what makes it a legitimate test rather than a fit.

It does not close the wavelength gap:

| prefactor × ω | `I_th(532)/I_th(1064)` |
|---|---|
| 0 (cascade only) | 3.99 |
| **1 (order unity)** | **3.87** |
| 10³ | 1.84 |
| 10⁶ | 0.48 |
| **measured** | **0.80** |

At an order-unity prefactor MPI closes 3 % of the gap. Reaching the measurement
needs `~10⁵·ω`, i.e. an ionization rate faster than the optical frequency, which
is not a rate. Gated as `keldysh_mpi_does_not_close_the_wavelength_gap`.

Two by-products worth keeping:

- **The seed density is unphysical, and it is a latent defect.** `n_e0 = 1/V_focal
  = 1.2×10¹³ m⁻³` is ~10⁴ above the cosmic-ray background (10⁹–10¹⁰ m⁻³), which
  puts ~10⁻⁴ electrons in an `8.3×10⁻¹⁴ m³` focus — the focus essentially never
  holds one. Seeding *should* therefore be MPI's job, importing the photon-order
  asymmetry (at prefactor 1, MPI makes 295 electrons per pulse in the focus at
  532 nm and 2×10⁻⁸ at 1064 nm — ten orders of magnitude). Removing the seed
  changes the ratio only 3.99 → 3.85, because the model's threshold is already
  5.7–28× too high and MPI is copious there at both wavelengths. So the defect is
  real but masked by the level error. Exposed as
  `AirBreakdown::with_seed_density`.
- **An earlier claim of mine, withdrawn.** The wavelength ratio is *not*
  prefactor-insensitive. The `x^(1/K)` suppression argument only holds once MPI
  dominates at both wavelengths; the ratio then scales as `x^(−0.097)`, so three
  decades of prefactor still move it 1.9×, and across the transition it runs 3.99
  → 0.48. Any future prefactor claim has to be justified to ~2 orders, not waved
  through.

The open item is therefore narrower than "add MPI": either a PPT-corrected rate
for molecular O₂ (Coulomb corrections can lift the prefactor by orders of
magnitude — checkable against published `σ_K`), or a systematic in the two-paper
comparison. It is *not* a missing channel at these intensities.

**Both branches were settled on 2026-07-31 — see the PPT section below.** The
prefactor branch is closed by measurement: PPT's prefactor is *derived* once
`Z_eff` is given, `Z_eff` = 0.53 for O₂ is published, and the resulting absolute
rate reproduces a measured cross-section within 2×. It is not orders above
unity, and it moves the ratio only to 2.947. The systematic branch turns out to
be real: the two anchor experiments sit on opposite sides of the multiphoton
*seeding* threshold.

Chylek's 532 nm data sharpens the remaining question into a quantitative target
rather than a direction. Any candidate MPI channel now has to do three things at
once:
lift the 532 nm threshold's *ratio* to 1064 nm from 3.99 down to ≈0.80, flatten
the low-pressure branch from 1.951 to ≈0.43, and steepen the high-pressure
branch from 0.170 to ≈0.47 — with `K = 6` photons at 532 nm against `K = 11` at
1064 nm supplying most of the wavelength leverage for free. The three
`chylek1990_*` gates pin all three numbers, so a channel that fixes one while
breaking another cannot land quietly.

### Distribution-resolved cascade (`CascadeModel::DistributionResolved`)

The mean-trajectory closure ionizes only once `ε_∞ > U_i`, a hard bifurcation
that the model is evaluated on top of (`ε_∞/U_i` = 1.032 at 760 Torr). Resolving
the distribution removes it. An electron absorbs inverse-bremsstrahlung quanta of
size `ħω`, so its energy is an Ornstein–Uhlenbeck process rather than a
trajectory:

```text
dε = δ_eff·ν_m·(ε_∞ − ε)·dt + √(2·D_ε)·dW,     D_ε = ½·P_heat·ħω
```

The drift is unchanged; `D_ε` is photon shot noise and adds **no new constant**.
Ionization is first passage to `U_i` with a reflecting wall at `ε = 0`, from
Siegert's formula:

```text
ν_i = 1/T,   T = (1/D_ε)·∫₀^{U_i} dy e^{+φ(y)} ∫₀^y dz e^{−φ(z)},
φ(ε) = (ε² − 2·ε_∞·ε)/(ε_∞·ħω)
```

Site: `breakdown0d::first_passage_ionization_rate`, `first_passage_integral`.
Evaluated by an `O(N)` recurrence in the *differences* of `φ` — the naive form
splits a bounded product into factors of `10^±274`.

**Verification:** exact reduction to the mean-trajectory closed form as
`D_ε → 0`, ratio 0.821 → 0.990 as `ħω` falls 1.166 → 0.05 eV
(`first_passage_reduces_to_the_mean_energy_climb`, self-refining); quadrature 2nd
order with the shipped `N` = 512 inside 2e-4 at both photon energies
(`first_passage_quadrature_is_converged`); the bifurcation is gone and the rate
continuous through `ε_∞ = U_i` (`distribution_resolved_has_no_bifurcation`); and
the rate is a function of `(ε_∞/U_i, ħω/U_i)` alone, preserving `ν_i ∝ p`
(`first_passage_rate_depends_only_on_two_dimensionless_groups`).

**Validated — the high-pressure slope** (`distribution_resolved_cascade_fixes_the_high_pressure_slope`).
At the untouched literature centre `δ_eff` = 0.02: T&T 300–2000 Torr goes
0.0951 → **0.2793** against a measured 0.329, and Chylek 300–786 Torr goes
0.1717 → **0.4665** against 0.468. Over the literature range of that single free
constant the envelope moves from `[0.023, 0.231]`, which *excludes* 0.329, to
`[0.183, 0.407]`, which contains it — and on Chylek's window from `[0.039,
0.414]` to `[0.307, 0.657]` containing 0.468.

**Pinned — what it does not fix.** The low-pressure branch is unmoved,
1.952 → 1.954 against a measured 0.428, which localises that failure to diffusion
loss rather than to the cascade closure
(`distribution_resolved_does_not_fix_the_low_pressure_branch`). The wavelength
ratio moves 4.00 → 3.39 against ≈0.80 — the right sign at last, since `D_ε ∝ ħω`
makes a shorter wavelength take bigger energy steps, but a 15 % move against a 5×
gap (`distribution_resolved_does_not_close_the_wavelength_gap`). And the hard
plateau floor no longer bounds anything: the threshold slides under it,
1.09× → 0.75× → 0.63× at 300 / 760 / 2000 Torr
(`distribution_resolved_softens_the_plateau_floor`), so the noble-gas headroom
figures below are a bound for the mean-trajectory closure only.

**The default since 2026-07-30.** It landed as a variant first so that its
effect on every published number was measured rather than asserted, then was
promoted. The promotion retired M6a's long-standing red gate:
`tt2012_threshold_slope_matches_measurement` had been `#[ignore]`d and failing
since 2026-07-25 and now passes — with no tolerance moved and no constant
touched, because the model changed rather than the test. Numbers that moved with
it: the level-ratio drift 1.48× → 1.20× (a smaller drift is a smaller residual
*slope* error), the λ-ratio baseline 3.99 → 3.39, and the M6c pulse-length floor,
which is now asymptotic rather than flat — 8.510e15 at 6 ns converging to
6.797e15 by ~10 µs, a bounded 1.25 % … 1.25× fall rather than the fluence
criterion that would break M6c's two-stage argument.

Reference: A. J. F. Siegert, *On the first passage time probability problem*,
Phys. Rev. **81**, 617 (1951) — the mean-first-passage quadrature; the
energy-space diffusion picture of cascade breakdown is standard, see Raizer
(above) and Zel'dovich & Raizer, *Physics of Shock Waves and High-Temperature
Hydrodynamic Phenomena*, ch. VI.

### General-gas kernel and the parameter-free plateau floor

The gas-dependent constants live in `breakdown0d::Gas`; the laser and the focal
geometry stay on `AirBreakdown`. `Gas::dry_air()` is a re-packaging that changes
no number — the `breakdown` case is bit-identical across the split.

Writing out the equilibrium energy,
`ε_∞ = (e²I/(m_e c ε₀))·ν_m/((ν_m²+ω²)·δ_eff·ν_m)`, the collision frequency
**cancels exactly** in the optical regime `ν_m ≪ ω`, because heating and
inelastic loss both scale `∝ ν_m`. Ionization needs `ε_∞ > U_i`, so the cascade
has a hard floor with no transport constant in it at all:

```text
I_plateau = δ_eff · U_i · m_e·c·ε₀·ω² / e²
```

Site: `breakdown0d::cascade_plateau_intensity`, `AirBreakdown::plateau_intensity`.
Gated as `cascade_plateau_floor_is_independent_of_the_transport_constants`,
which perturbs `D_e` by 100× either way and demands the floor not move, and
checks `cascade_rate` is identically zero just below it at every pressure.

For a **monatomic** gas this is a prediction with nothing to choose:
`δ = 2m_e/M` is the atomic mass, `U_i` is spectroscopy. `breakdown0d::MonatomicGas`
carries only those exactly-known constants:

| gas | `U_i` (eV) | first excitation (eV) | `M` (u) | `δ = 2m_e/M` | `K` @ 532 nm | floor (W/cm²) |
|---|---|---|---|---|---|---|
| He | 24.587 | 19.82 | 4.0026 | 2.741e-4 | 11 | 1.275e11 |
| Ar | 15.760 | 11.55 | 39.948 | 2.747e-5 | 7 | 8.190e9 |
| Xe | 12.130 | 8.32 | 131.293 | 8.357e-6 | 6 | 1.918e9 |

Against Chylek's Fig. 2 measurements
(`chylek1990_noble_gas_plateau_floors_are_unequally_tight`): every curve sits
above its own floor, and the *ordering* He > Ar > Xe is right. The **spacing** is
not — predicted floor ratios He/Ar = 15.6 and Ar/Xe = 4.27 against measured
threshold ratios ≈2.5 and ≈3.0, so He/Ar is over by 6.3× and He/Xe by 8.8×, with
no constant left to turn. The headroom above the floors is 1.85× (He), 7.8× (Ar),
13.2× (Xe) — monotone in atomic mass, which a cascade-only kernel gives no reason
for. And `δ_elastic` is a **lower bound**: the last leg of every climb runs above
the first excitation threshold (19 % of the ascent in He, 27 % Ar, 31 % Xe),
where inelastic loss dwarfs elastic recoil, so the true floors are higher and He
— with 1.85× of room — is the gas that breaks first.

Three gases at **one** wavelength and one bench span `K` = 11/7/6, which is the
only way in this repository to separate photon order from wavelength; the air
data confounds them. `Gas::from_monatomic` takes `K_m` and `D_e` as **required
arguments** rather than defaults, because no citable momentum-transfer table was
landed and for Ar and Xe those cross sections swing two orders of magnitude
across the Ramsauer minimum. Full noble-gas *threshold curves* are therefore not
computed here; the plateau gate needs neither constant.

Full model and constants: `docs/M6A_SPEC.md`.

References:
- Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991) — cascade ionization
  and the inverse-bremsstrahlung heating rate.
- N. Kroll, K. M. Watson, *Theoretical study of ionization of air by intense
  laser pulses*, Phys. Rev. A **5**, 1883 (1972).
- C. G. Morgan, *Laser-induced breakdown of gases*, Rep. Prog. Phys. **38**,
  621 (1975) — regime map and threshold scaling.
- A. Thiyagarajan, J. B. Thompson, *Optical breakdown threshold investigation
  of 1064 nm laser induced air plasmas*, J. Appl. Phys. **111**, 073302 (2012)
  — the external anchor: `E_B` and `E_eff` curves digitized into
  `tests/data/tt2012_*.csv`, focal geometry from their Eq. 5, and the cascade
  closed form from their Eq. 4.
- P. Chylek, M. A. Jarzembski, V. Srivastava, R. G. Pinnick, *Pressure dependence
  of the laser-induced breakdown thresholds of gases and droplets*, Appl. Opt.
  **29**, 2303 (1990) — the independent D5 anchor: the clean-air threshold at
  532 nm (their Fig. 3, `α = 0.45 ± 0.01`), digitized into
  `tests/data/chylek1990_air_threshold_vs_pressure.csv` by
  `scripts/digitize_chylek1990.py`. Second group, second apparatus, second
  wavelength, matched pulse and focus. Their **Fig. 2** additionally gives clean
  He, Ar and Xe thresholds on the same bench, hand-traced into
  `tests/data/chylek1990_{he,ar,xe}_threshold_vs_pressure.csv` and cross-checked
  against an independent programmatic trace to 0.3 % on Ar and Xe. Those three
  gases span `U_i` = 24.59 / 15.76 / 12.13 eV, i.e. multiphoton order
  **K = 11 / 7 / 6 at a single wavelength and a single apparatus** — the one
  dataset here that separates photon order from `λ` — and having no attachment
  channel they isolate cascade + diffusion + MPI. Consumed since 2026-07-30 by
  `chylek1990_noble_gas_plateau_floors_are_unequally_tight`, which tests the
  parameter-free plateau floor `δ·U_i·m_e c ε₀ ω²/e²` against them — `δ = 2m_e/M`
  is the atomic mass, so for these gases the cascade has no free constant at all.
  Their own integrity gate
  (`chylek1990_fig2_digitization_reproduces_the_published_slopes`) stays.

- A. Sharma, M. N. Slipchenko, M. N. Shneider, X. Wang, K. A. Rahman,
  A. Shashurin, *Counting the electrons in a multiphoton ionization by elastic
  scattering of microwaves*, Sci. Rep. **8**, 2874 (2018); arXiv:1710.03361 —
  the **absolute** eight-photon ionization cross-section of O₂ at 800 nm,
  `σ₈ = (3.3 ± 0.3)×10⁻¹³⁰ W⁻⁸m¹⁶s⁻¹`, from direct electron counting by Rayleigh
  microwave scattering calibrated against dielectric scatterers of known
  properties. Consumed by `ppt_rate_matches_the_measured_o2_cross_section`. This
  is the only anchor in this file that constrains an ionization **rate** rather
  than a breakdown threshold, which is why it can test the MPI channel without
  the circularity that sank the earlier attempts.

- A. Talebpour, C.-Y. Chien, S. L. Chin, *The effects of dissociative
  recombination in multiphoton ionization of O₂*, J. Phys. B **32**, 1229
  (1999) — the source of `Z_eff` = 0.53 for molecular O₂ (`breakdown0d::Z_EFF_O2`), from
  their PPT fit to measured O₂ ionization at 800 nm, and of an independent rate
  point (3×10⁹ s⁻¹ at 3×10¹³ W/cm²) that sits 7× below `σ₈·I⁸` at the same
  intensity. The two published anchors disagree by more than either disagrees
  with PPT, which is what bounds the prefactor to about an order of magnitude.

### PPT photoionization for molecular O₂ (`breakdown0d::ppt_rate`)

Added 2026-07-31, to settle the prefactor branch left open above. **Site:**
`breakdown0d::ppt_rate`, off by default, enabled by
`AirBreakdown::with_ppt_mpi(Z_EFF_O2)`.

```text
W = |C_n*|²·√(6/π)·U_i·(2F₀/(F√(1+γ²)))^{2n*−3/2}·A₀(ω,γ)·exp[−(2U_i/ħω)·f(γ)]
```

in atomic units, with `κ = √(2U_i)`, `F₀ = κ³`, `n* = Z_eff/κ`,
`|C_n*|² = 2^{2n*}/(n*Γ(n*+1)Γ(n*))`, and the above-threshold sum

```text
A₀(ω,γ) = (4/√(3π))·(γ²/(1+γ²))·Σ_{n≥⌈ν⌉} e^{−α(n−ν)}·Φ(√(β(n−ν)))
ν = (U_i/ħω)(1 + 1/2γ²)     α = 2[asinh γ − γ/√(1+γ²)]     β = 2γ/√(1+γ²)
```

`Φ` is Dawson's integral (`breakdown0d::dawson`). The sum's truncation is
**derived, not fixed**: `α → ⅔γ³` as `γ → 0`, so the number of terms needed to
reach a stated depth diverges as `1/γ³` — 10 terms at `γ` = 10, tens of
thousands by `γ` = 0.1. `ppt_ati_terms` sizes it from `α`, and below
`γ` = 0.1 the tunnelling limit `A₀ → 1` takes over, which makes the whole
expression ADK. A **fixed** 64-term truncation was tried first and is wrong by
29 % at `γ` = 0.2 in a way that looks like physics; the gates that catch it are
`ppt_reduces_to_adk_in_the_tunnelling_limit` and
`ppt_rate_is_monotonic_across_the_bisection_bracket`, the latter because an
un-converged sum makes the rate *fall* with intensity inside the bracket that
`threshold_intensity` bisects on.

The exponent is **the same
object** as Keldysh's — PPT's `(2F₀/3F)g(γ)` is algebraically `(2U_i/ħω)f(γ)` —
so `keldysh_tunnel_exponent` is reused rather than re-derived, and
`ppt_and_keldysh_share_the_same_exponent` gates that the two have not drifted.

**Why this is a test and not a fit.** Keldysh's prefactor is an order-unity
function no derivation pins, which is why `keldysh_rate` exposes it as an
argument. PPT's is fully determined once `Z_eff` is given, and `Z_eff` = 0.53
for O₂ is published (Talebpour 1999). `ppt_rate` therefore takes **no prefactor
argument**, and its absolute magnitude is a prediction.

**It passes an absolute validation.** Against the measured `σ₈` at 800 nm, with
nothing fitted:

| `I` (W/cm²) | `W_PPT / σ₈I⁸` |
|---|---|
| 10¹⁰ | 1.995 |
| 10¹¹ | 1.974 |
| 10¹² | 1.768 |

Within a factor of 2 of an absolutely calibrated measurement, and high — the
direction the source paper reports for purely theoretical predictions. The
comparison is made below 10¹² W/cm² because PPT's fitted order softens as `γ`
falls, and a magnitude ratio between two different powers of `I` means nothing.

**A structural result found by a gate that expected something else.** PPT
returns an **integer** photon order, `K = ⌈ν⌉` — 7.998 at 800 nm, 10.998 at
1064 nm, 6.000 at 532 nm — where the bare Keldysh exponential gives the
fractional `U_i/ħω`. The leading ATI term carries `e^{−α(γ)(⌈ν⌉−ν)}` and
`dα/d ln I = −1`, which contributes exactly `⌈ν⌉ − ν` to the log-log slope. You
cannot absorb 10.34 photons; the fractional order is an artifact of dropping the
sum, and restoring it is what makes reading `σ₈` as an eight-photon
cross-section legitimate.

**And it does not close the wavelength gap.**

| source | `I_th(532)/I_th(1064)` | gap closed |
|---|---|---|
| cascade only | 3.349 | — |
| Keldysh, prefactor 1 | 2.89 | 18 % |
| **PPT, `Z_eff` = 0.53** | **2.947** | **16 %** |
| PPT + a physical seed (10⁹ m⁻³) | 2.835 | 20 % |
| **measured** | **0.80** | |

The reason is worth stating because it inverts the expectation that motivated
the work: `n* = Z_eff/κ` = 0.563 makes the Coulomb exponent `2n* − 3/2`
**negative**, so the correction that lifts an atomic rate by orders of magnitude
at `Z` = 1 is order-unity for a molecule at `Z_eff` = 0.53. "Coulomb corrections
can lift the prefactor by orders of magnitude" is true of atoms and false here.

**The finding that does explain something.** Evaluated at each paper's *own
measured* threshold — no model threshold anywhere in the calculation, so the
kernel's pinned 3.90–4.69× level offset cannot contaminate it:

| | 1064 nm (T&T) | 532 nm (Chylek) |
|---|---|---|
| measured `I_th` (W/cm²) | 2.06×10¹¹ | 1.56×10¹¹ |
| `N_seed = W·N·V·τ` there | **5.4×10⁻⁹** | **3.15** |
| `I` where `N_seed` = 1 | 1.18×10¹² | 1.30×10¹¹ |
| that, over measured `I_th` | 5.73× | **0.83×** |

At 532 nm the measured breakdown threshold **is** the multiphoton seeding
threshold, to 17 %. At 1064 nm multiphoton ionization is 5.7× short of making a
single electron, so breakdown there is seeded by something else — background
ionization, impurities, dust — as the classical picture of ns IR breakdown has
it. The two experiments are on opposite sides of that transition, so their
threshold *ratio* is not a measurement of one mechanism's wavelength scaling.
That is the "systematic in the two-paper comparison" branch, made specific.

Gated by `ppt_multiphoton_order_is_the_integer_photon_count`,
`ppt_and_keldysh_share_the_same_exponent`,
`ppt_rate_matches_the_measured_o2_cross_section`,
`ppt_does_not_close_the_wavelength_gap_either`,
`ppt_seeding_thresholds_separate_the_two_experiments`, and the four
special-function checks in `breakdown0d::tests`.

**References.** V. S. Popov, *Tunnel and multiphoton ionization of atoms and
ions in a strong laser field (Keldysh theory)*, Phys.-Usp. **47**, 855 (2004),
for the PPT rate in the form used here; the two anchors above.

### Seed production (`AirBreakdown::background_electron_density`)

Added 2026-07-31. The last knowingly-false assumption on M6a's default path.

The kernel used to start every pulse from `n_e0 = 1/V_focal` = 1.2×10¹³ m⁻³ —
one electron sitting in the focus — and to clamp `n_e` at that value throughout
the integration. Both are now gone. The initial condition is the physical
ambient density and the pulse produces its own electrons.

```text
n_e0(p) = q(p)/ν_att(p),      q(p) = q_ref·(p/p_ref),      q_ref = 10⁷ m⁻³s⁻¹
```

`ν_att` is the **same** expression the loss term uses (`Gas::attachment_rate`,
factored out so the two cannot drift), and `q_ref` ≈ 10 ion pairs cm⁻³ s⁻¹ is a
standard atmospheric-electricity value (AFRL *Handbook of Geophysics and the
Space Environment*, ch. 20, Sagalyn & Burke, 1985). Multiphoton production is
`ppt_rate`, on by default — without it a physical background would never break
down.

**The retired assumption was wrong by ~14 orders, not the ~10⁴ this file used to
claim.** That claim compared `1/V_focal` against the cosmic-ray **ion** density,
10⁹–10¹⁰ m⁻³. Air is electronegative: `ν_att` = 6.7×10⁷ s⁻¹ at 1 atm, so a free
electron survives ~15 ns and the free-*electron* background is
`q/ν_att` = **0.149 m⁻³** — about **1.2×10⁻¹⁴** electrons in an 8.3×10⁻¹⁴ m³
focus. The lower atmosphere holds essentially no free electrons, and a tight
focus cannot expect to find one waiting.

**The new constant is not load-bearing, and that is the argument for it.**
Sweeping the seed over twelve decades (10⁻⁶ → 10⁶ m⁻³) leaves the threshold
*bit-identical*; it takes ~10 orders before it moves 0.04 %. The retired
`1/V_focal` sat in the range where the seed *does* matter (2.6 % at 10¹²), so
the old constant was load-bearing and wrong while the new one is neither.
Gated as `ionization_background_is_not_load_bearing`.

**An explicit seed still behaves as a floor; the derived one does not.** Setting
`with_seed_density` is a modelling assumption — "this many electrons are
available" — and holding it constant is what keeps a source-free run independent
of the integration window. The derived background is a physical initial
condition and must be free to deplete. That distinction is what lets the gates
that **isolate** a cascade closure or a loss term (`seeding_suppressed` in
`tests/validation.rs`) keep their published baselines unchanged, and it is gated
by `seed_floor_applies_only_to_an_explicit_seed`.

**What it does to the measurements.** The low-pressure branch — this milestone's
worst residual for its whole life, and the one *both* source papers attribute to
multiphoton ionization — is essentially repaired:

| Chylek window (Torr) | seeding off | **default** | measured |
|---|---|---|---|
| 10–100 | 1.292 | **0.501** | 0.428 |
| 100–300 | 0.947 | 0.857 | 0.413 |
| 300–786 | 0.431 | 0.386 | 0.468 |

3.0× too steep → **1.17×**. It is the largest single improvement M6a has had,
and it came from deleting an assumption rather than adding a term. The
wavelength ratio also moves, 3.39 → **2.854** against a measured 0.80 (overshoot
4.24× → 3.57×), because multiphoton production is `I⁶` at 532 nm against `I¹¹`
at 1064.

**What it costs, recorded rather than smoothed over.** The 300–786 Torr window
slips from 0.431 to 0.386 against 0.468, absolute thresholds rise 3.3–4.2 %, and
a **mid-pressure residual** survives — see below, where it is diagnosed
properly.

### The mid-pressure residual (`chylek1990_residual_is_localised_to_mid_pressure`)

The three wide windows above are the right shape for asking *is the kernel a
power law* and the wrong shape for asking *where is it wrong*: they compare the
model's local behaviour against a measured window average. Redone like-for-like
on six narrow bands, both fitted over the same measured abscissae:

| band (Torr) | measured | model | cascade only |
|---|---|---|---|
| 4–12 | 0.308 | 0.238 | 2.203 |
| 12–30 | 0.485 | 0.316 | 1.275 |
| 30–70 | 0.553 | **0.580** | 1.300 |
| 70–150 | 0.455 | **1.045** | 1.191 |
| 150–350 | 0.339 | **0.690** | 0.769 |
| 350–800 | 0.458 | 0.350 | 0.389 |

The measurement is **not** locally flat — it runs 0.31–0.55 — and the kernel
tracks it to better than 0.25 everywhere except **70–350 Torr**, where it is
2.0–2.3× too steep. That is much narrower than "the kernel is not a power law",
and it is exactly the band nothing masks: below ~30 Torr free-molecular escape
and multiphoton seeding both bite, above ~350 Torr diffusion is sub-dominant to
the cascade plateau, and in between `D_e/Λ²` carries the pressure dependence
alone, at `Kn` = 0.03–0.14 where the free-molecular correction is a few per cent.

Two candidate explanations have been tested and **both fail**:

- **It is not the absolute level.** The tidy story would be that a model running
  14–33× high overstates an `I⁶` source by ~10⁶, with a drift because the offset
  drifts. Sweeping `δ_eff` walks the level from 15.8× to 3.4× and the bump gets
  *worse*, peak local exponent 1.04 → 1.42. Gated as
  `the_mid_pressure_residual_is_not_a_level_artifact`, which also rules out
  fitting `δ_eff` — the tempting move, since it is the milestone's one remaining
  free constant.
- **It is not space-charge screening.** Free diffusion is only valid while the
  plasma is tenuous; above `n_e` ≈ `ε₀ε_e/(e²Λ²)` = 6.2×10¹⁸ m⁻³ — four decades
  below `n_bd` — diffusion should become ambipolar and ~130× weaker. Prototyped
  with a cited ion mobility: it **redistributes** the error rather than removing
  it (70–150 Torr 1.022 → 0.339, but 4–12 Torr 0.237 → 0.702 against a measured
  0.308), for a net improvement of ~13 % in total absolute error at the cost of
  a new constant. Not landed: a marginal gain bought with a new constant is what
  this project's rules exist to refuse.

So the residual is a genuine shape defect in the continuum diffusion loss, it is
not reachable by any constant the model already has, and the obvious missing
mechanism does not account for it. That is M6a's sharpest open question.

**Window independence now holds for a better reason.** It used to hold because
the seed was clamped, which patched a symptom; with production replacing the
initial condition there is nothing left to decay, and the spread over
`w ∈ [1,4]` is 5×10⁻⁵ against the 1 % the gate tolerates. The threshold stays an
intensity floor rather than a fluence criterion — 8.815×10¹⁵ at 6 ns converging
to 6.745×10¹⁵ by ~10 µs, a bounded 1.31× fall (was 1.25×), so M6c's two-stage
argument is untouched.

## M6a.2 — Aperture optics and pupil phase statistics

### On-axis focal intensity from the pupil field

In the Fraunhofer regime the focal field is the Fourier transform of the
aperture field, so the on-axis focal amplitude is its DC component:

```text
U_focus(0) = (1/(λf))·∫∫ U(x, y) dA
I_focus    = |∫ U dA|² / (λf)²
```

Site: `src/aperture.rs` (`Aperture`). **Exact** given Fraunhofer and a thin
lens — not a small-aberration approximation; the Maréchal form `S ≈ exp(−σ_φ²)`
is a weak-aberration limit of it, and is gated as a limit rather than used as
the definition.

This is what lets M6a.2 exist at all. Turbulence is resolved on a centimetre
grid over a kilometre path while the focal spot that ignites a spark is
micrometres across; resolving `λf/D` while spanning `D` needs `N ≳ 10⁴` per
side. **There is no focal grid anywhere** — every quantity is a pupil integral
on the grid the propagator already produced.

Two degradations are reported and deliberately never conflated:

- `focal_intensity_ratio` — against the same beam through vacuum. Total
  degradation, wavefront **and** amplitude scintillation, because the pupil
  field carries both. This is the quantity that feeds an ignition test.
- `phase_only_strehl` = `|∫U dA|²/(∫|U| dA)²` — normalised against the beam's
  own amplitude, so scintillation divides out and the wavefront contribution is
  isolated. Diagnostic only. Calling the first one "the Strehl ratio" would be
  wrong, which is why both exist.

Gates (`src/aperture.rs` unit tests): a flat wavefront gives exactly `S = 1` and
nothing exceeds it (the triangle inequality on the coherent sum); a pure tilt
steers the spot without dimming it — `S` unchanged, wander `= f·θ` — which is
the sharpest statement of why the two quantities differ; an amplitude-only
perturbation leaves `phase_only_strehl` at 1 while costing focal intensity; and
`S → exp(−σ_φ²)` as the aberration shrinks, with the residual gated to fall.

Reference: J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts &
Co. (2005), § 5.2. V. N. Mahajan, J. Opt. Soc. Am. **73**, 860 (1983) — the
Maréchal limit.

### Residual pupil phase variance (Noll coefficients)

Kolmogorov phase over a circular pupil, with the low-order Zernike terms
projected out:

```text
piston removed        σ_φ² = 1.0299·(D/r₀)^(5/3)     (Noll 1976, Δ₁)
piston + tilts removed σ_φ² = 0.134 ·(D/r₀)^(5/3)     (Noll 1976, Δ₃)
```

Site: `src/aperture.rs` (`Aperture::residual_phase_variance`, `TiltRemoval`).
Both coefficients are parameter-free. Taken on **phase screens, not propagated
fields**: `arg(u)` is only recoverable modulo 2π and wraps many times at these
`D/r₀`, so a variance from a propagated field would be measuring the wrapping.

These are an independent projection of the statistics M3 already gates through
the structure function — a pupil integral in the Zernike basis versus `D_φ(r)`
in the plane. Passing one does not imply the other.

- **N1** (`noll_tip_tilt_removed_variance_matches_the_closed_form`) — measured
  **0.1407** at the pinned seed, +5.0 % on Noll, banded at ±12 %. The band is
  set by the ensemble spread, not the central value: across three seed sets at
  64 and 128 screens the coefficient runs 0.129–0.143, and that spread does
  **not** shrink with screen count because it is dominated by how much low-order
  power an ensemble happened to draw. A tighter band would gate the draw.
- **N2** (`noll_piston_removed_variance_converges_to_kolmogorov`) — Noll assumes
  an infinite outer scale; the screens are von Kármán. Piston-removed variance
  is dominated by the largest scales, so it is strongly `L₀`-dependent:
  `L₀/D` = 10 → 0.345, 40 → 0.541, 200 → 0.842, 2000 → 1.001, against Noll's
  1.0299. Gated as the convergence. Runs 32 screens, not N1's 128: the trend is
  identical at either count because the noise is common-mode across the sweep,
  so the extra screens buy nothing and cost 4× the runtime. A trend gate is the *stronger* choice here — the
  absolute coefficient swings 1.02–1.23 between seed sets, and that noise is
  common-mode across an `L₀` sweep on the same seeds, so it cancels in the trend
  while dominating any level.

**A `(D/r₀)^(5/3)` exponent gate was specified and withdrawn**, and the reason is
recorded because it is the M6a "D5" trap in new costume. Sweeping `r₀` at fixed
screens is a *tautology*: `phase_psd` takes `r₀` only through the multiplicative
`0.4896·r₀^(−5/3)`, so identical draws scale the screen as `r₀^(−5/6)` and the
variance as `r₀^(−5/3)` by construction. Measured that way the exponent came back
**1.66667 for both modes** — five decimals that establish nothing but correct
multiplication. Sweeping the *aperture* is a real geometric change but is
Monte-Carlo limited (deviation from 5/3 up to 0.09 at 24 screens, 0.05 at 96,
0.007 at 256), and a coefficient constant across apertures *is* a 5/3 exponent,
so N1 is the better-conditioned form of the same claim.

Reference: R. J. Noll, *Zernike polynomials and atmospheric turbulence*,
J. Opt. Soc. Am. **66**, 207 (1976).

### Turbulence-degraded ignition statistics (the driver)

`cases::run_ignition`. Per realization: propagate a launch beam through a
`TurbulentPath`, take the pupil integral at the receiver, turn it into W/m²
through `IntensityScale` (the T4 helper's third consumer), and hand that one
number to `AirBreakdown`. Reductions over the ensemble give the ignition
probability, the focal-intensity ratio distribution, and the focal-spot wander.
`seeded_ensemble` supplies the parallelism; realizations derive all randomness
from their index and come back in index order, so every reduction is bitwise
thread-count independent (**E2**).

**The position of `P_ig` on the `Cn²` axis is not a claim about the world.** It
carries `AirBreakdown`'s absolute threshold, which is M6a's explicitly ungated
quantity, and must be labelled so wherever it is plotted. Everything upstream of
that one boolean is independent of it and is gated.

- **E1** (`ignition_ensemble_converges`) — the spec asked for `P_ig` within
  ±0.02 on a realization doubling; that is **not achievable** and the gate says
  so. `P_ig` is a Bernoulli mean whose binomial standard error is 0.030 at
  n = 256 and 0.022 at n = 512, so ±0.02 at any affordable `n` would gate the
  draw. Gated instead: the continuous reductions converge (`wander_rms` under
  5 % on doubling, measured 1.15 → 1.11 ×10⁻⁴ m over n = 32 → 512) and `P_ig`
  moves within two binomial standard errors (measured 1.8σ, 0.0σ, 0.6σ, 0.7σ),
  plus a non-vacuity check that `P_ig` is not saturated.
- **W1** (`wander_follows_the_square_root_of_cn2`) — **PHYSICS.** RMS wander
  `∝ Cn²^(1/2)`; measured **0.4953 / 0.4977 / 0.4987** over two decades and
  three seeds, gated at ±0.02. Parameter-free: path length, aperture, outer
  scale and beam all enter as coefficients, and none can produce a 1/2.
- **W2 — retired, not gated.** An aperture-dependence gate was landed and then
  withdrawn as seed-dependent. The observation stands: the textbook
  `σ_α² ∝ D^(−1/3)` implies `D^(−1/6)` = −0.167 for the RMS and the default
  geometry does not show it (−0.003), because the tilt estimator is
  intensity-weighted and a 5 cm beam in a 15–40 cm pupil is weighted by its own
  footprint while the closed form assumes *uniform* illumination. The
  *measurement* cannot carry a gate: it fits a slope across four nested
  apertures on shared screens, and across three seeds the overfilled exponent
  runs −0.183/−0.249/+0.004 at 16 realizations and −0.102/−0.318/−0.143 at 32,
  a spread that does not shrink with ensemble size. The gate passed only at the
  seed it pinned. Documented observation, not a validated claim.

The `ignition` CLI case (`cases::run_ignition_sweep`, rendered by
`scripts/render_ignition.py`) sweeps `Cn²` and reports the ignition probability
with its **binomial** error bars, the focal-intensity distribution behind it,
and the wander law. The figure carries the shape/position caveat inside the
panel rather than in a caption.

It also reports the transition width — the span from `P_ig` = 0.9 to 0.1,
measured **1.42 decades**, and roughly invariant (1.49 at 2× drive, 1.39 at
0.5×, 1.44 at a 1.33× larger pupil) while a 4× drive change slides the curve
0.6 decades sideways. **Deliberately not gated**: no closed form for the width
has been derived, so gating it would check the measured number against itself —
the same trap that retired the `(D/r₀)^(5/3)` exponent gate above.

Reference: L. C. Andrews, R. L. Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press (2005) — angle of arrival and beam wander.

## M6c — 1-D gas dynamics (the LSD substrate)

### Compressible Euler equations, HLLC + MUSCL-Hancock

```text
∂U/∂t + ∂F(U)/∂x = Ṡ

U = (ρ, ρu, E)ᵀ,   F = (ρu, ρu² + p, (E + p)u)ᵀ,   E = p/(γ−1) + ½ρu²
```

Site: `src/euler1d.rs`. Ideal gas at constant `γ` (the verification EOS of the
two `docs/M6C_SPEC.md` pins; the plasma-range table EOS attaches later without
touching the flux routines). HLLC approximate Riemann solver with
Einfeldt/Davis wave speeds, MUSCL-Hancock reconstruction under a minmod
limiter, CFL ≤ 0.8 recomputed per step from the current wave speeds, and a
positivity guard that **bails with the cell and step** rather than clamping.

**No laser physics is in this module**, deliberately (spec gate decision 4).
`Ṡ` is exposed only as `step_with_source`, the seam the LSD driver attaches to;
deposition, ignition, and the plasma column live one layer up. That is what
keeps the two gates below independent of the model that will use them.

Gate **G0** guards the T4 extraction that this milestone required
(`tests/blooming.rs`): **G0a** reproduces the pre-extraction `δn` arithmetic
from scratch and demands bit-for-bit equality with `ThermalBlooming`'s output —
a tolerance would not see a reassociation — and **G0b** pins the size and hash
of `data/air_properties.npy`, since M4's numbers are calibrated to that table
and M6c's plasma properties belong in a separate file (D8). G0a deliberately
avoids FFTs so it stays deterministic on every platform the M5 wheels target;
a whole-run field fingerprint would look stronger and be flakier, as FFT
results are not bit-portable across libraries.

The G1/G2 gates are **verification** — "the code solves the equations written
down" — not validation. Nothing here is yet a claim about the world; the physics
gate for M6c is the parameter-free `D ∝ S^(1/3)`, `ρ₀^(−1/3)` scaling (G4),
which arrives with the deposition layer.

- **G1 — Sod shock tube vs the exact Riemann solution**
  (`sod_shock_tube_matches_exact_riemann_solution`). Exact solution from the
  Newton-iterated star-state solver in `src/validate.rs` (`RiemannProblem`,
  which shares only the plain `(ρ, u, p)` struct with the solver under test).
  Measured L1(ρ) = 6.55e-3 at n = 100 falling to 6.55e-4 at n = 1600, observed
  rate 0.79–0.88. First order is the ceiling on a solution containing a shock
  and a contact; ~0.8 is the textbook minmod value.
- **G2 — observed order on smooth flow**
  (`euler_muscl_hancock_is_second_order_on_smooth_flow`). Isentropic advection
  of `ρ = 1 + 0.2·sin(2πx)` at uniform `u`, `p` over one period. L1(ρ) falls
  7.50e-4 → 3.78e-6 over n = 128 → 2048, observed order rising monotonically
  1.86 → 1.94. It approaches 2 **from below** because minmod clips the two
  smooth extrema, degrading the scheme to 1st order in a shrinking region — so
  the gate is on the finest pair (> 1.85) plus the monotone climb, not on
  hitting 2 exactly.

### Equilibrium plasma-range air properties (frozen table)

Equilibrium air from 200 K to 30,000 K and 10⁴–10⁸ Pa, tabulated offline by
`scripts/make_plasma_table.py` from Mutation++ (`air_11` mixture, RRHO thermo
database, equilibrium state model) into `data/plasma_properties.npy`, and read
through `src/plasmaprops.rs`. Shape `(4, 597, 33)`, property axis
`[ln ρ, e, γ_eff, ln n_e]`, on a grid uniform in `T` and uniform in `log₁₀ p`,
bilinearly interpolated. The pressure ceiling is set by the CJ state behind an
LSD front (~1.5×10⁷ Pa) and the temperature ceiling by its post-front
temperature. No runtime FFI, no LGPL in the build or the M5 wheels — the
`airprops.rs` discipline. `data/air_properties.npy` (M4) is a separate file and
is untouched (G0b).

`ρ` and `n_e` are stored as **logarithms**: both cross dozens of decades
through the ionization onset, where interpolating raw values fails badly.
`e` changes sign near 800 K so no log is available, and neither it nor `γ_eff`
needs one.

Two quantities are deliberately absent. **`n_i` is not stored** — every ion in
the mixture is singly charged, so quasi-neutrality makes it identical to `n_e`
(verified to 6.2×10⁻¹¹ wherever `n_e` matters); it is returned as `n_e`.
**`Z̄` is not stored and is not a prediction**: it is identically 1 and cannot
be otherwise, because the RRHO database ships no doubly ionized N or O (He⁺⁺ is
the only `++` species in it).

**Limitation — no second ionization.** Real equilibrium air begins to doubly
ionize above ~20,000–25,000 K; `air_11` structurally cannot. Toward the top of
the range the table therefore understates `n_e`, and so understates the
inverse-bremsstrahlung absorption `α_IB ∝ n_e·n_i·Z̄²` the LSD front runs on.
The range is still built to 30,000 K because the CJ state needs it, but above
`SECOND_IONIZATION_K` = 20,000 K the table is a singly-ionized approximation,
flagged by `PlasmaTable::is_singly_ionized_approximation`.

Per D8, Mutation++ is **trusted for the physics** and the gate is on the
**tabulation**:
- **G6** (`plasma_table_matches_direct_mutationpp_off_grid`) interpolates the
  frozen table to 99 points deliberately **off** its grid — 75 of them in the
  6,000–18,000 K ionization onset, where bilinear interpolation is worst — and
  compares against direct Mutation++ evaluations frozen into
  `tests/data/plasma_reference_samples.csv` at generation time (the
  `tt2012_*.csv` precedent, since the solver cannot call Mutation++). Measured
  max relative error: **ρ 4.10e-4, e 8.61e-4, γ_eff 1.68e-4, n_e 1.48e-3**.
- The generator runs its own harsher cell-midpoint sweep over 17,683 points
  before it will write anything (ρ 4.53e-4, e 1.05e-3, γ_eff 2.40e-4,
  n_e 3.61e-3), plus a quasi-neutrality gate and a cold-limit check that
  `γ_eff(300 K, 1 atm)` = 1.39883 — the one point whose answer is known without
  Mutation++.
- Neither gate makes an accuracy claim below `NE_ACCURACY_FLOOR` = 10¹⁵ m⁻³.
  There `ln n_e` is nearly linear in `1/T` rather than `T`, so the uniform-`T`
  grid interpolates it poorly — but the values are ~10³ m⁻³ against ~10²³ in an
  LSD plasma, and gating that error would be theatre.

### Laser-supported detonation (the coupled wave)

The beam ignites a plasma and the resulting absorption wave runs **back up the
beam** toward the laser as a detonation. `src/lsd.rs`, driven by
`LsdColumn::advance`.

The column's `x` axis is the beam axis: the laser sits beyond `x_min`, so the
beam travels in `+x` and the front travels in `−x`. Everything upstream of the
front is cold transparent air, which is why the front sees the full incident
intensity `S`. Beam attenuation is Beer–Lambert in the direction of travel,
`dI/dx = −α_pl·I`, and the source term in the energy equation is the absorbed
power density:

```text
∂U/∂t + ∂F(U)/∂x = (0, 0, q)ᵀ,     I_{k+1} = I_k·exp(−α_k·dx),
q_k = (I_k − I_{k+1})/dx
```

The deposition is discretised **conservatively** — each cell takes exactly what
it removes from the beam, so `Σ q·dx ≡ I_in − I_out` with no quadrature error.
Hydro and source are coupled by **Strang splitting**
(`source(dt/2) → hydro(dt) → recompute α, I → source(dt/2)`), matching the
propagator's own splitting discipline and for the same reason. The step is sized
from the *post-deposition* wave speeds: the leading half-step raises `p` and so
`c` before the flux update sees it, and sizing from the pre-deposition state
overshoots the CFL bound.

Two absorption closures:
- `Absorption::GreyThreshold` — verification. A fixed `α` wherever the specific
  internal energy exceeds a threshold, zero below. Nothing that can drift
  between refinements. Keying on internal energy rather than temperature keeps
  it well defined under the constant-`γ` EOS, which has no gas constant.
- `Absorption::InverseBremsstrahlung` — production. Thermal free-free
  absorption, `α_IB = C·Z̄² n_e n_i T^(−1/2) ν^(−3) (1 − e^(−hν/kT))·ḡ`, with
  `n_e` from the plasma table above (`n_i = n_e`, `Z̄ ≡ 1`, both structural).
  `C = 3.7×10⁻²` in SI, converted from the CGS `3.7×10⁸` of Rybicki & Lightman
  Eq. 5.18b. The Gaunt factor `ḡ` is a dimensionless multiplier of order unity,
  set to 1: a proper Gaunt table is out of scope, and it need not be in scope,
  because `ḡ` enters as a *coefficient* and no coefficient can shift the `1/3`
  exponent the physics gate measures. One honest inconsistency: the hydro
  carries a constant-`γ` ideal gas while the ionization comes from the
  equilibrium table, so `T` is the table's inversion of the hydro's `(ρ, p)`
  rather than a self-consistent EOS.

Per D7 the plasma couples to the beam through **absorption only**. `PlasmaColumn`
is the read-only `Medium` the propagator sees: `extinction(z_slab)` from the
hydro state, and `δn ≡ 0` — no Drude index, which is what keeps M6c clear of the
near-critical failure a Drude plasma column would hit in a paraxial envelope.

In the M4 Péclet spirit, `check_regime` refuses rather than mis-models: the
absorption length must be resolved by ≥ 4 cells and be under a quarter of the
domain (thicker is the LSC regime, out of scope), ≥ 90 % of the beam must reach
the front, and the front must be a strong detonation (`p₁ ≥ 10·p₀`).

"Reaching the front" is measured at the leading edge of the **strongly
absorbing** region (first cell with `α ≥ ½·α_max`), not at the pressure
half-maximum that `front_position` reports. The two genuinely differ — in a
detonation the reaction zone leads the pressure peak — and using the pressure
front here would score normal front structure as upstream extinction. Under the
grey model this check is ~1 by construction, since cold gas is exactly
transparent; it earns its keep under inverse bremsstrahlung, where a long
weakly ionized precursor really can eat the beam before it arrives.

**`SECOND_IONIZATION_K` is enforced, not just documented.** The CJ state behind
a strong LSD front sits at 1.5–2.5×10⁴ K and so legitimately crosses the table's
singly-ionized ceiling. `IonizationCeiling::Refuse` (the default) bails naming
the temperature; `::Flag` proceeds and records it on the run
(`used_singly_ionized_approximation`). Extending the table is not currently
possible from shipped data — no Mutation++ thermodynamic database contains
doubly ionized N or O — so the bias is converted into an explicit boundary
instead of being carried silently.

Gates (`tests/validation.rs`):
- **G3** (`lsd_front_speed_matches_the_raizer_closed_form`) — **VERIFICATION,
  not validation.** At `S = 10¹¹ W/m²`, `ρ₀ = 1.225 kg/m³`, `γ = 1.4`, and an
  absorption length `1/α = 50 µm`, the measured front speed is **5402 m/s**
  against Raizer's `D = [2(γ²−1)S/ρ₀]^(1/3)` = **5392 m/s**, `+0.19 %`;
  gated below 1 %. Refining `dx` from 10 µm to 5 µm moves it by 5×10⁻⁵, so the
  answer is grid-converged and the residual is physical, not numerical. Both
  boundaries are asserted undisturbed alongside `check_regime`: once the wave
  runs off the laser-side end, `front_position` degrades to the first cell
  centre and would report a plausible speed for a front that no longer exists.
  G3b carries the same assertion.

  The label matters. Raizer's expression is **not** an independent check on this
  model — it is the Chapman–Jouguet construction the deposition model is built
  from, with the chemical heat release replaced by `q = S/(ρ₀D)`. Reproducing it
  establishes that HLLC, the Strang-split source, and the EOS together solve a
  nontrivial self-similar problem correctly. It establishes nothing about
  whether that problem describes the world. `raizer_lsd_velocity` therefore
  lives in `src/lsd.rs` beside the model and *not* in `src/validate.rs` among
  the independent reference solutions — filing it there would quietly assert
  otherwise, which is precisely the M6a "D5" trap.
- **G3b** (`lsd_front_speed_converges_as_the_absorption_layer_thins`) — the
  refinement that actually bites. Over `1/α = 400 → 200 → 100 → 50 µm` at
  `dx = 10 µm` the residual runs **−8.26 % → −2.66 % → −0.43 % → +0.19 %**,
  monotone, each halving taking at least 2.3× off the error.

  The residual is a **relaxation transient**, not a permanent thick-layer
  deficit: held at `1/α = 400 µm` and given longer to settle (0.15 → 0.30 → 0.50
  of the domain) it runs `−8.3 % → −3.7 % → −1.5 %` and does not plateau. A
  thicker deposition zone relaxes onto the self-sustaining speed more slowly, so
  at a fixed settle it sits further from it; given long enough they all reach the
  same CJ speed. That is the textbook result — a CJ velocity depends on total
  heat release, not reaction-zone length — and it is worth stating because an
  earlier version of this entry claimed a steady-state deficit instead, which
  contradicted the theory the gate checks against.
- **G3c** (`lsd_front_speed_is_seed_independent`) — the answer must not depend
  on how the wave was lit. A seeded detonation starts overdriven and relaxes
  onto the CJ speed slowly, and G3's 1 % tolerance is the same order as that
  transient, so the seed is a free parameter sitting directly under the headline
  number unless it is checked. Between a 1× and a 2× CJ-pressure seed the
  results differ by `5.3e-3` at a 1.0 µs settle, `2.7e-3` at 1.4 µs and
  `1.1e-3` at the 1.8 µs used, both converging on `≈ +0.2 %`. Gated below
  `2e-3`, with both boundaries asserted undisturbed.
- **G2b** (`lsd_source_coupling_is_second_order`) — the hydro↔source coupling is
  2nd order, the M6c counterpart of M1's `split_step_is_second_order` and M4's
  `coupling_is_second_order`. Refining `dx` and `dt` **together at fixed CFL**
  (the limit the claim is stated in; MUSCL-Hancock degenerates to forward Euler
  in time if `dt→0` at fixed `dx`), the Strang cadence gives observed order
  **1.99 / 2.03 / 1.99**. The gate also runs a deliberate 1st-order contrast —
  folding the source into the update gives **0.88 / 1.02 / 1.07** — so it
  demonstrably resolves the difference rather than passing vacuously.

  This is why `Euler1d::step_with_source` carries a warning: used on its own it
  *is* that 1st-order contrast. Nothing about its output looks wrong; it simply
  converges half as fast. The driver's four-call Strang sandwich is the
  supported path.
- **G5** (`lsd_energy_budget_closes`) — absorbed laser energy versus the
  domain's energy gain. Measured relative residual **2.1×10⁻¹⁶ to 4.6×10⁻¹⁵**,
  five orders inside the 10⁻¹⁰ the spec asks. Exact rather than approximate for
  two reasons: the conservative deposition above, and a boundary flux that is
  *verified* zero rather than estimated — with transmissive ends and undisturbed
  ambient gas at both, `(E + p)u` vanishes identically, and
  `boundaries_undisturbed` asserts that premise rather than assuming it.

- **G4** (`lsd_velocity_follows_the_parameter_free_one_third_scaling`) — **THE
  PHYSICS GATE.** `D ∝ S^(1/3)` over 1.52 decades of absorbed intensity and
  `D ∝ ρ₀^(−1/3)` over 1.50 decades of ambient density, exponents gated inside
  `±0.01`. Measured at `γ = 1.4`: **+0.33190** and **−0.33020**.

  Everything above this line is verification — it establishes that the code
  solves the equations it was given, and G3 in particular is checked against a
  closed form the model is *derived from*. This gate is different in kind. Every
  quantity uncertain about the *level* of `D` — `γ_eff`, the absorbed fraction,
  radial relief, radiation losses, the Gaunt factor — enters as a coefficient,
  and no coefficient can produce a `1/3` exponent. The exponent is what the
  model predicts independently of the coefficient soup, and it is what measured
  LSD velocities are reported to follow.

  The EOS-independence leg is done by moving `γ`, since the table EOS is not
  wired into the hydro. `2(γ²−1)` runs 0.88 → 3.56 from `γ = 1.2` to `5/3`,
  shifting the level of `D` by 1.59×, while the fitted exponents move by 0.001
  and 0.002:

  | γ | 2(γ²−1) | D at S = 10¹¹ | S exponent | ρ₀ exponent |
  |---|---|---|---|---|
  | 1.2 | 0.88 | 4169 m/s | +0.33127 | −0.32895 |
  | 1.3 | 1.38 | 4842 m/s | +0.33176 | −0.32977 |
  | 1.4 | 1.92 | 5400 m/s | +0.33190 | −0.33020 |
  | 5/3 | 3.56 | 6632 m/s | +0.33213 | −0.33096 |

  This is **not** a demonstration that a real equilibrium EOS leaves the exponent
  alone — a `γ_eff` varying with local state is not a different constant `γ` —
  and the gate does not claim it is.

  The density sweep holds ambient **temperature** fixed (`p₀ ∝ ρ₀`), not
  pressure: at fixed `p₀` a decade of `ρ₀` moves the ambient internal energy by
  a decade, and at the thin end the undisturbed gas would cross the ignition
  threshold and the whole column would absorb. The threshold itself is 5×
  ambient `e₀`, not G3's fixed 2 MJ/kg — at the sweep corner (`γ = 1.2`,
  `ρ₀ = 12.25`) the post-shock state is only 11× ambient, so a 10× threshold
  there starts *controlling* the front rather than enabling it and drives the
  fitted exponent to −0.459. That the exponents agree to 1e-3 between 3× and 5×
  thresholds is the evidence the threshold is out of the loop.
- **`lsd_velocity_level_tracks_the_eos_coefficient`** — the counterpart to G4,
  pinning what the *level* is worth. Moving `γ` 1.4 → 1.2 must scale `D` by
  `(0.88/1.92)^(1/3) = 0.772`; measured **0.7722**. The solver tracks the
  coefficient exactly where the coefficient is knowable, which is the sharpest
  statement of why agreement on the level would not be evidence about the
  physics.

- **G8** (`plasma_column_absorbs_as_beer_lambert`) — the D7 coupling itself,
  end to end. A real beam marched through a `PlasmaColumn` built from G3's
  settled hydro state, against `exp(−τ)`, `τ = Σ α_k·dx`. The M2 twin
  (`beer_lambert_matches_closed_form`) does this for a constant absorber; this
  is its M6c counterpart with the absorber coming from gas dynamics. Added at
  step 6: until then D7's claim was carried by `PlasmaColumn`'s unit tests,
  which exercise its `Medium` methods in isolation, and no field had ever been
  marched through one. `δn ≡ 0` is asserted at every slab rather than assumed —
  a Drude index appearing there is the near-critical failure D7 avoids.

  Measured at `τ = 339`: **1.7e-13 at 500 slabs, 8.4e-14 at 100**, across 500
  successive amplitude multiplications against a single exponential. Two slab
  resolutions because `PlasmaColumn::from_column_resampled` — mean `α` over each
  bin, so `α_slab·dz = Σ α_k·dx` exactly — is what makes marching a 2500-cell
  hydro state through an FFT propagator affordable.

Not yet gated: **G7**, absolute velocity against measurement, which is expected
to land high and is documented-but-ungated because a planar 1-D solver has no
radial relief. See `docs/M6C_SPEC.md`.

### The `lsd` demonstration run (CLI case)

`beamprop lsd` (`src/cases.rs::run_lsd`, written by `src/main.rs`, rendered by
`scripts/render_lsd.py`) is the case that puts M6a and M6c in the same run: a
spark is lit at M6a's breakdown threshold and the absorption wave it launches is
tracked back up the beam. The igniting pulse's peak intensity comes from its
power and focal radius through `IntensityScale` — the T4 extraction's second
consumer, and the reason it was extracted.

**Its headline is a result, not a demonstration.** The case takes a short
*igniting* pulse and a separate long *sustaining* drive, and the two models
together say the second could never have produced the first:

- M6a's threshold in air at 1 atm saturates at **≈1.14×10¹⁶ W/m² and does not
  fall with pulse length** (6 ns → 1.18×10¹⁶, 1 ms → 1.14×10¹⁶). It is an
  intensity floor, not a fluence one: below it the inelastic losses paid
  climbing to the ionization potential exceed the inverse-bremsstrahlung
  heating, the net cascade rate is negative, and no exposure time rescues it.
  Widening the focus moves it by 4 % over a 500× range of spot radius.
- The sustaining LSD drive is ~10¹¹ W/m² — five orders of magnitude below.

So the detonation must be *initiated* by something far brighter than what
*sustains* it, which is the known experimental situation: LSD waves in clean air
are started on a target, on an aerosol, or by a separate spike. Pinned by
`the_sustaining_drive_is_far_below_the_breakdown_threshold`, so a future change
to either model that closes the gap fails rather than quietly invalidating the
write-up.

**What each half is worth.** *When and where* the spark lights inherits M6a's
explicitly ungated absolute level (4.8–7.0× above the measured T&T curve). The
front speed does not: it depends on the absorbed intensity at the front and on
`ρ₀`, not on where the spark was lit — which is why G3/G3b/G3c and the G4
physics gate all use seeded ignition and never touch `AirBreakdown`. The gap
above is likewise untouched by that uncertainty: 10⁵ against ~7×. Default run:
`D` = 5401 m/s against Raizer's 5391 (+0.19 %), energy budget closing to 1.3e-16,
final column optical depth 374.

**Why the grey closure drives it.** `GreyThreshold` is what G3–G5 gate and it
introduces nothing that can drift. The run *evaluates* the production
inverse-bremsstrahlung closure at its own measured post-front state rather than
asserting a reason for not using it, and the answer is informative: `α ≈ 6.8 1/m`
at 1064 nm, making the whole 2.5 cm column **0.17 optical depths** — nearly
transparent to the beam driving it, with no front, and `check_regime` would
correctly refuse it as volumetric — against `α ≈ 1.1×10³ 1/m` at 10.6 µm, an
absorption length of 0.92 mm that is 92 cells on the demo grid and 3.7 % of the
domain. Free-free absorption falls steeply toward short wavelengths, so this is
the model reproducing **why LSD experiments are done with CO₂ lasers**. What
blocks running that closure coupled is cost, and specifically the table
inversion: `PlasmaTable::temperature` bisects ~45 times per cell per deposition
call, three deposition calls per step. A faster inversion, not a finer grid, and
a separate change with its own gate.

References:
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer (2009) — HLLC (§10.4–10.6), MUSCL-Hancock (§14.4), and the
  exact Riemann solver and Test 1 tabulation (§4.3, §6.4) the gates use.
- Yu. P. Raizer, *Laser-Induced Discharge Phenomena*, Consultants Bureau (1977)
  — LSD wave theory, the detonation analogy, and the velocity closed form.
- G. B. Rybicki, A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley
  (1979), Eq. 5.18b — the thermal free-free absorption coefficient.
- G. Strang, SIAM J. Numer. Anal. **5**, 506 (1968) — the operator splitting.
- J. B. Scoggins et al., *Mutation++: multicomponent thermodynamic and
  transport properties for ionized gases*, SoftwareX **12**, 100575 (2020) —
  the equilibrium thermochemistry behind the plasma table.
- B. Einfeldt, *On Godunov-type methods for gas dynamics*, SIAM J. Numer. Anal.
  **25**, 294 (1988) — the wave-speed estimates.
- G. A. Sod, *A survey of several finite difference methods…*, J. Comput. Phys.
  **27**, 1 (1978) — the shock-tube problem.

Full model, the remaining gates (G3–G7), and which of them are verification
versus validation: `docs/M6C_SPEC.md`.

## Rendering (not physics)

The solver writes data only (`.npy` arrays + `_meta.json`/`_notes.md`
sidecars; collection helpers in `src/viz.rs`). All images come from
`scripts/render.py` (matplotlib): the perceptually-uniform **magma** colormap
applied to `t = (I/I_max)^γ` with `γ = 0.5` to lift the dim wings; `I_max` is
the global peak (across all frames of a GIF), so brightness differences
between frames are physical. Colorbars are labeled in `I/I_max`, axes in
metres.

Reference: S. van der Walt, N. Smith, *matplotlib colormaps* (magma),
<https://bids.github.io/colormap/>.

## M6d — Axisymmetric gas dynamics and radial relief

### Axisymmetric Euler, area-weighted finite volume

```text
∂U/∂t + (1/r)·∂(r·F_r)/∂r + ∂F_x/∂x = Ṡ_geom + Ṡ_laser

U   = (ρ, ρu_r, ρu_x, E)ᵀ
F_r = (ρu_r, ρu_r² + p, ρu_r u_x, (E + p)u_r)ᵀ
F_x = (ρu_x, ρu_r u_x, ρu_x² + p, (E + p)u_x)ᵀ
Ṡ_geom = (0, p/r, 0, 0)ᵀ
```

Site: `src/euler2d.rs`. HLLC + MUSCL-Hancock with minmod, Strang-split
dimensionally as `R(dt/2) → X(dt) → R(dt/2)`, CFL asserted per step in both
directions, positivity guard that bails with the cell **and stage** rather than
clamping. **No laser physics is in this module**, deliberately — M6c gate
decision 4, carried forward.

Two implementation facts carry the milestone and both are recorded in the code:

- **The `1/r` never appears.** Cells are annuli, interfaces carry area `A ∝ r`,
  and the axis interface has zero area, so nothing crosses `r = 0` by
  construction. `(A_+ − A_−)/V` *is* `1/r_j` analytically, finite in the
  innermost ring without a floor.
- **The geometric source is written as the same floating-point expression as
  the pressure part of the flux difference**, so a radially uniform state is a
  bit-exact fixed point. That is what lets G9, G13(i) and G14 assert equality
  rather than a tolerance.

The 1-D Riemann solver is **reused, not reimplemented**: a sweep packs
`(ρ, ρu_∥, E − ½ρv_t²)` into `euler1d`'s `Conserved`, calls its `hllc_flux`, and
recovers the transverse flux as `F_ρ·v_t` with the upwind side read off
`sign(F_ρ)`. Both identities are exact.

- **G9 — the planar limit reproduces `Euler1d` bit for bit**
  (`euler2d_planar_limit_reproduces_euler1d_bit_for_bit`). Not a tolerance: the
  same floating-point operations. Two non-vacuity legs.
- **G10 — Sedov–Taylor** (`sedov_blast_matches_the_self_similar_solution`).
  Exponent 0.38628 vs the exact 2/5; level and peak compression both gated as
  *trends* under refinement, because a spherical blast's density spike is one or
  two cells wide at any affordable resolution.
- **G11 — 2nd order on smooth axisymmetric flow** (1.861 / 1.964 against a
  split-source contrast at 1.030 / 1.155). This gate found a real defect: the
  Hancock predictor originally built the geometric source from the reconstructed
  *face* pressures, which is well-balanced and wrong — the pressure terms then
  cancel identically against the flux difference and the gradient disappears
  from the predictor. It cost an order.
- **G12 — conservation in the `r dr dx` measure**, with radial momentum
  deliberately *not* conserved, and an escape-flux leg that closes the budget
  when relief reaches the wall.
- **G13 — the axis is not a wall**, against a deliberate even-parity contrast.

### The Sedov–Taylor reference

Site: `src/validate.rs` (`SedovBlast`). The self-similar ODEs are integrated
inward from the strong-shock Rankine–Hugoniot state and `ξ₀` follows from the
energy integral, so **nothing external is quoted**: the published ≈1.033 for
`γ` = 1.4 is an independent cross-check that the derived 1.03278 passes to
0.03 %. Its own unit tests include putting the profile back into the Euler PDEs,
where the residual is 6.9e-5 and falls as the finite-difference step squared —
the check that verifies the hand derivation rather than the arithmetic.

### Radial relief and the transverse instability

Site: `src/lsd2d.rs`. M6c's `LsdColumn` with a beam of finite radius: one
independent Beer–Lambert march per ring, no refraction, no diffraction.
`Absorption`, `IonizationCeiling` and `raizer_lsd_velocity` are reused
unchanged — the closures depend on `(ρ, p)` alone.

**The transverse instability, found rather than sought.** A radially uniform
run diverges exponentially from the 1-D column, out of round-off, reaching 3 %
by M6c's settle. Three measurements identify it: it is bit-identical in planar
and axisymmetric geometry (so not the geometric source), amplitude-proportional
(a 10⁶× larger seed gives a 10⁶× larger early response), and it saturates at
`|u_r|` ≈ 200–400 m/s. That is a linear instability going nonlinear — the
mechanism behind the cellular structure real detonations have. It is pinned, not
validated: no measurement has been compared to.

**The relief deficit** (G15, `pinned`). Because the instability is present at
*every* beam radius including infinite, the deficit is measured against the
**wide-beam 2-D run**, not the 1-D column — otherwise the instability would be
reported as relief. Measured `δ = 1 − D/D_wide` = 0.305 at `R_b·α` = 1.6 and
0.230 at 3.2, monotone in `R_b`, with the wide-beam limit itself within 1 % of
Raizer. The pinned claim is a **band** of ±13 %, and that width is measured
rather than chosen: grid (+6 % on halving `Δx`), seed (−7 % at a 1× rather than
2× CJ-pressure seed), and ignition threshold (±8 % over a 4× sweep). Pinning a
third digit would assert a precision three separate knobs say is not there.

A **failure radius** is predicted and was not reached: `check_regime` requires
eight cells across `R_b`, so the smallest beam affordable at this `Δr` still
carries a healthy wave. Recorded as an open item rather than asserted.

- **G16 — the one-third scaling survives relief**
  (`the_one_third_scaling_survives_radial_relief`): `S^0.34666` against the
  parameter-free 1/3 while the level moves 23 %. M6c's G4 argues that relief can
  only enter as a coefficient; this measures it.

### The `lsd2d` demonstration run (CLI case)

`cargo run --release -- lsd2d` seeds a wave, drives it with a 160 µm top-hat
beam at `R_b·α` = 3.2, and reports the deficit against a matching wide-beam run.
Writes `_fields.npy` `[frame, quantity, ring, cell]`, a front-track CSV carrying
both the axis and the beam edge, `_meta.json` and `_notes.md`; images come from
`scripts/render_lsd2d.py`. No M6a ignition stage, deliberately — the `lsd` case
owns that story, and seeding keeps this one from re-inheriting M6a's ungated
absolute threshold.
