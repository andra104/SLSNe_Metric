# SLSN Pipeline Audit — How Everything Works

---

## 1. The Big Picture

This pipeline asks a single astrophysical question: given a proposed ten-year survey strategy for the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), how many Type I Superluminous Supernovae (SLSNe-I) would be detected, characterized, and flagged for spectroscopic follow-up? SLSNe-I are hydrogen-free stellar explosions powered by a millisecond magnetar — a rapidly spinning, highly magnetized neutron star born in the core collapse — that are ten to one hundred times more luminous than ordinary core-collapse supernovae. Their extreme brightness makes them detectable to redshifts beyond z = 2 (when the universe was roughly three billion years old), making them unique probes of star formation in metal-poor galaxies at epochs inaccessible to most transients.

The primary scientific input is a set of historical SLSN light curves drawn from the Gómez et al. 2024 catalog (arXiv:2407.07946), which provides multi-band photometry for roughly one hundred spectroscopically confirmed SLSNe-I. Each observed event is treated as a template: the code fits a two-dimensional Gaussian Process (GP) regression — a non-parametric probabilistic model that interpolates across both time and observed frequency simultaneously — to the flux measurements, then stores the resulting absolute-magnitude light curve as a representative SLSN. These templates are the empirical backbone of the simulation. The secondary input is a tabulated volumetric rate model provided by Ben Margalit, encoding how often SLSNe occur per unit comoving volume per unit time as a function of redshift, conditional on different assumptions about the host-galaxy metallicity threshold below which SLSNe can form.

Given the templates and a rate model, the pipeline uses rubin_sim's Metrics Analysis Framework (MAF) to inject a synthetic population of SLSNe at random positions on the sky and at redshifts drawn from the volumetric rate, then plays each injected event through the actual planned observation sequence of a given cadence simulation. The cadence simulations are stored in OpSim databases — SQLite files in `cadences/` — which record every planned telescope pointing over ten simulated years. For each injected event, the pipeline asks whether the observation sequence happens to catch the event with sufficient signal-to-noise ratio (SNR, the ratio of source flux to measurement noise) and sufficient cadence to satisfy the detection, characterization, or spectroscopic-trigger criteria. The fraction of injected events that pass each criterion is the detection efficiency; multiplied by the total expected event rate integrated over the survey volume, this gives detections per year.

The pipeline has five main stages. First, raw archival photometry is ingested and normalized into a standard format with per-observation central wavelengths. Second, a 2D GP is fitted to each event's flux time series to create smooth, band-interpolated absolute-magnitude templates stored in `templates.pkl`. Third — optionally and as a planned upgrade — physically motivated spectral energy distribution (SED) templates are computed from Gómez+2024's magnetar model fits, covering the ultraviolet through near-infrared for use at redshifts where the GP templates lose wavelength coverage. Fourth, a synthetic SLSN population is drawn from the volumetric rate model and placed on a HEALPix sky grid with realistic Galactic dust extinction. Fifth, the MAF metrics evaluate each injected event against the chosen OpSim cadence, producing per-event pass/fail arrays and aggregate summary tables for comparison across the three rate models and four cadence strategies under study.

---

## 2. Pipeline Stages — In Execution Order

### Stage 0 — Configuration and Paths (`paths.py`, `constants.py`)

All path resolution derives from `paths.py`'s own location on disk, so the repository works on any machine without hardcoded paths or environment flags. `constants.py` builds a shared cosmological lookup table once per process using Planck18 cosmology and caches LSST bandpasses.

**Responsible files:** [py_files/slsn_metrics/paths.py](py_files/slsn_metrics/paths.py), [py_files/slsn_metrics/constants.py](py_files/slsn_metrics/constants.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `get_repo_root()` | none | `Path` | Returns absolute path two levels above `paths.py` |
| `get_shared_output_dir(science_case)` | str | `Path` | Returns `output/SLSNe/shared/`, creating it if absent |
| `get_rate_csv_path(filename)` | str | `Path` | Returns path to Ben's CSV in `shared/` |
| `get_cadence_path(cadence_name)` | str or list | `Path` or list | Full path to OpSim `.db` file |
| `dm_from_z(z)` | float or array | float or array | Distance modulus from redshift via cached table |
| `get_lsst_bands()` | none | dict | Loads and caches all six LSST bandpass throughput curves |
| `phase_bucket_vec(phase_rest, step)` | array, float | int array | Quantizes rest-frame phases to stable bin indices for cache keys |

**Reads from disk:** `$RUBIN_SIM_DATA_DIR/throughputs/baseline/total_{ugrizy}.dat`  
**Writes to disk:** Nothing (paths created on demand by `get_output_dir`)  
**Next stage needs:** All functions in `__init__.py` import from these two modules first.

---

### Stage 1 — Raw Data Ingestion (`export_slsne_photometry.py`)

Reads per-object photometry from the Gómez+2024 catalog format (whitespace-separated `.txt` files with BOM-stripped headers), canonicalizes filter names, flags upper limits, and writes clean CSVs or Parquet files. A second pass attaches central wavelength (Cenwave, in Å) to each photometric point by reading per-event `*_model.txt` and `*_rest.txt` files.

**Responsible file:** [py_files/slsn_metrics/export_slsne_photometry.py](py_files/slsn_metrics/export_slsne_photometry.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `canonical_filter(s)` | str | str | Strips `swift_`, `ps1_`, `-AB` etc. from filter names |
| `read_supernova_table_txt(path)` | Path | DataFrame | Reads whitespace-delimited `.txt` with BOM cleaning |
| `to_export(df_raw)` | DataFrame | DataFrame | Converts to standard 7-column format; sets `UL=True` rows to `detected=0` |
| `process_one(event_dir, event_name, out_perevent)` | Path, str, Path | DataFrame | Processes a single event and writes Parquet/CSV |
| `process_all_events(supernovae_dir, out_perevent, out_allevent)` | Path, Path, Path | (index_df, csv, parq) | Batch processes all events and writes combined catalog |
| `per_filter_cenwave(event_root, name)` | Path, str | dict | Returns `{filter: Cenwave_Å}` using mode-clustering on `_model.txt` then `_rest.txt` |
| `attach_cenwave_to_perevent_csv(csv_path, cen_map)` | Path, dict | Path | Adds `Cenwave` column to per-event CSV; writes `{name}_cenwave.csv` |
| `load_allparams_robust(path)` | Path | DataFrame | Tolerant reader for `all_parameters.txt` with flexible delimiters |

**Reads from disk:** `{supernovae_dir}/{event}/{event}.txt`, `{event}_model.txt`, `{event}_rest.txt`  
**Writes to disk:** `output/per_event_files/{event}.csv`, `{event}.parquet`, `{event}_cenwave.csv`, `_index.csv`; optionally `output/all_events/all_objects.csv`  
**Next stage needs:** `{event}_cenwave.csv` files with `mjd`, `mag`, `mag_err`, `filter`, `Cenwave` columns.

---

### Stage 2 — GP Template Construction (`gp_build.py`, `model.py` — `LC.from_catalog()`)

For each event in the parameter table, `LC.from_catalog()` reads the cenwave CSV, converts magnitudes to flux (Jy), and fits a 2D Matérn-3/2 GP in the (time [days], frequency [Hz]) plane. The GP peak defines the reference epoch t0, which is cross-checked against the catalog peak MJD via `pick_t0_hybrid()`. The GP surface is then evaluated at a dense time grid and converted to rest-frame absolute magnitudes. Simultaneously, a per-event SED grid is built by evaluating the GP at a rest-frame wavelength grid and scaling to absolute flux at 10 pc.

**Responsible files:** [py_files/slsn_metrics/gp_build.py](py_files/slsn_metrics/gp_build.py), [py_files/slsn_metrics/model.py](py_files/slsn_metrics/model.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `default_gp_kernel(scale_guess)` | float | `george.Kernel` | Matérn-3/2 in (time, freq); frequency metric frozen for sparse-color stability |
| `fit_2d_gp(time, freq, flux, fluxerr, kernel)` | arrays, kernel | `partial` | Fits GP via L-BFGS-B; returns callable `gp_predict(X_new)` |
| `gp_predict_surface(gp_predict, t_eval, target_freq_hz)` | partial, array, dict | dict | Predicts `{band: (flux_mean, flux_sigma)}` on time grid at specified frequencies |
| `pick_t0_hybrid(mjd, band, lamA, z, gp_predict, t0_catalog, ...)` | arrays, float, partial, float | (float, dict) | Selects reference epoch: GP peak in rest-frame g/r bands, then checks agreement with catalog peak |
| `LC.from_catalog(inputs, ...)` | `CatalogInputs` | `LC` | Main builder: loops events, GP fits, builds per-band templates and SED grids, saves pickle |
| `synthesize_mag_at_z(sed_grid, phase_rest, z, filt)` | dict, float, float, str | float | Computes apparent AB magnitude from rest-frame SED grid via full bandpass integration |
| `atomic_save_pickle(obj, path)` | any, Path | None | Atomic write via `os.replace` to prevent corruption |

**Reads from disk:** `output/per_event_files/{event}_cenwave.csv` or `.parquet`; `all_parameters.txt` (for redshifts and peak MJDs via `CatalogInputs`)  
**Writes to disk:** `output/SLSNe/shared/templates.pkl` (joblib+zstd; contains `lightcurves`, `t_grid`, `names`, `sed_grid`, `median_cenwave_by_band`, `meta`)  
**Next stage needs:** `templates.pkl` with `sed_grid` list for mag-grid building; `lightcurves` list for metric evaluation.

---

### Stage 3 — Physical SED Templates (`gomez_models.py`, `mosfit_interface.py`)

An alternative to Stage 2 for events where GP wavelength coverage is insufficient. Reads MOSFiT posterior median parameters from `all_parameters.txt` and computes a physically motivated SED for each event using the Gómez+2024 magnetar+ejecta model. The output `sed_grid` format is identical to the GP-based SED grid, so all downstream code is unaffected.

**Responsible files:** [py_files/slsn_metrics/gomez_models.py](py_files/slsn_metrics/gomez_models.py), [py_files/slsn_metrics/mosfit_interface.py](py_files/slsn_metrics/mosfit_interface.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `total_luminosity(times, fnickel, mejecta, Pspin, Bfield, Mns, thetaPB, rest_t_explosion)` | arrays, floats | array | Nickel-cobalt + magnetar spin-down input luminosity [erg/s] |
| `diffusion(times, lum_in, kappa, kappa_gamma, mejecta, v_ejecta, rest_t_explosion)` | arrays, floats | array | Radiation diffusion through ejecta [erg/s] |
| `photosphere(times, luminosities, v_ejecta, temperature, rest_t_explosion)` | arrays, floats | (rphot, Tphot) | Photospheric radius [cm] and temperature [K] from bolometric luminosity |
| `blackbody_supressed(times, lum, rphot, Tphot, cutoff_wavelength, alpha, sample_wavelengths, redshift)` | arrays, floats | array-of-arrays | Modified blackbody SED with UV power-law suppression blueward of `cutoff_wavelength` |
| `_call_slsnni_safe(phases, Pspin, Bfield, ...)` | arrays, floats | 2D array [N_phase, N_lam] | Calls the four-step physical chain with T=0 phase masking to avoid NaN propagation |
| `build_physical_sed_grid(params_file, phase_grid, wave_grid_A)` | Path, arrays | (list of dicts, list of str) | Loops all events in `all_parameters.txt`, calls `_call_slsnni_safe`, converts F_λ → F_ν at 10 pc |
| `build_physical_templates(params_file, save_to, ...)` | Path, Path | `LC` | Wraps `build_physical_sed_grid`, returns `LC` instance with `sed_grid` set; same interface as `LC.from_catalog()` |

**Reads from disk:** `all_parameters.txt` (Gómez+2024 MOSFiT posteriors, columns: `Pspin_med`, `log(Bfield)_med`, `Mns_med`, `thetaPB_med`, `kappa_med`, `kappagamma_med`, `mejecta_med`, `fnickel_med`, `vejecta_med`, `temperature_med`, `cutoff_wavelength_med`, `alpha_med`, `texplosion_med`)  
**Writes to disk:** Optionally `physical_templates.pkl`; optionally a joblib cache file  
**Next stage needs:** Same `sed_grid` format as Stage 2 output.

---

### Stage 4 — Magnitude Grid Pre-computation (`model.py` — `build_magnitude_grid()`)

Pre-computes a four-dimensional array of apparent magnitudes indexed by (template, z, phase, filter). Each cell calls `synthesize_mag_at_z()`, which redshifts the rest-frame SED, scales to the luminosity distance, and integrates against the LSST throughput curve. The computation is parallelized across templates using `ProcessPoolExecutor` with checkpointing every N templates. The finished grid is stored as `RegularGridInterpolator` objects for O(1) lookup.

**Responsible file:** [py_files/slsn_metrics/model.py](py_files/slsn_metrics/model.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `build_magnitude_grid(self, z_grid, phase_grid, filters, save_to, checkpoint_every)` | LC instance (method), arrays, list, Path, int | `LC` | Parallel grid computation over templates with checkpoint/resume; builds `_interps` dict |
| `_compute_grid_slice(i_tpl, sed_grid, z_grid, phase_grid, filters, DMs)` | int, list, arrays, list, array | (int, dict) | Per-template worker: calls `synthesize_mag_at_z` for all (z, phase, filter) cells |
| `load_magnitude_grid(self, grid_file)` | LC instance (method), Path | `LC` | Loads grid pickle and rebuilds interpolators |
| `_build_interpolators(self)` | LC instance (method) | None | Creates `RegularGridInterpolator` per (filter, template) from loaded grid |

**Reads from disk:** `templates.pkl` or `physical_templates.pkl` (via prior `LC` construction); LSST throughputs (via `get_lsst_bands()`)  
**Writes to disk:** `output/SLSNe/shared/mag_grid.pkl` (contains `mag_grid` dict, `mag_grid_axes`, `filters`); `.checkpoint.pkl` during build  
**Next stage needs:** `mag_grid.pkl` is intended for fast metric evaluation (see Gap §8.3).

---

### Stage 5 — Population Generation (`population.py`)

Draws a synthetic all-sky SLSN population consistent with the chosen rate model, survey duration, and redshift range. The three tabulated rate models read from `fiducial_models.csv`. Each event is assigned a random sky position (uniform HEALPix, nside=64), redshift importance-sampled from the volumetric rate, template index, peak time, and Galactic E(B−V) from the SFD dust map. The result is a MAF `UserPointsSlicer` whose `slice_points` dictionary carries all per-event properties needed by the metrics.

**Responsible file:** [py_files/slsn_metrics/population.py](py_files/slsn_metrics/population.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `load_tabulated_rate(csv_path, model_name)` | Path, str | callable | Returns `R(z)` interpolator [Mpc⁻³ yr⁻¹]; anchors to Frohmaier+2021 at z=0.17 |
| `sample_events_from_tabulated_rate(rate_interp, z_min, z_max, t_start, t_end)` | callable, floats | int | Poisson-samples total event count from ∫R(z)·dV/dz·dz·Δt |
| `sample_redshifts_from_tabulated_rate(n_events, z_min, z_max, rate_interp)` | int, floats, callable | array | Importance-samples redshifts from P(z) ∝ R(z)·dV/dz via inverse-CDF |
| `inject_uniform_healpix(nside, n_events, seed)` | int, int, int | (ra, dec) arrays | Draws uniform sky positions from HEALPix pixel centers |
| `generate_SLSN_PopSlicer(lc_model, ...)` | `LC`, floats, str, Path, ... | `UserPointsSlicer` | Master population builder: samples N events, sky positions, redshifts, templates, peak times, EBV; saves pickle |
| `cosmic_sfr_density_MD14(z)` | float or array | float or array | Madau & Dickinson 2014 CSFRD Ψ(z) [M☉ yr⁻¹ Mpc⁻³] |
| `metallicity_fraction(z, OH_max, n_mass_bins)` | float, float, int | float | Fraction f(z) of SF in galaxies below metallicity threshold via MZR integral |

**Reads from disk:** `output/SLSNe/shared/fiducial_models.csv`; SFD dust maps (via `dustmaps`); optionally an existing population pickle  
**Writes to disk:** `output/SLSNe/shared/population_{model_name}.pkl`; debug diagnostic plots in shared dir  
**Next stage needs:** The returned `UserPointsSlicer` with `slice_points` keys: `z`, `distance`, `distance_modulus`, `peak_time`, `file_indx`, `ebv`, `A_{ugrizy}`, `ra`, `dec`, `sid`.

---

### Stage 6 — MAF Metric Evaluation (`metrics.py`, `runners.py`)

Each metric is a subclass of `SLSN_Base_Metric` (which extends rubin_sim's `BaseMetric`). MAF calls `metric.run(dataSlice, slice_point)` for every event × every cadence pointing window. `evaluate_slsn()` retrieves the injected event's template, maps catalog photometric bands (B, V, R) to the nearest LSST band, interpolates the absolute magnitude from the template, applies the distance modulus and Galactic extinction, and computes SNR using the 5σ depth reported by the OpSim database. The detection function is then applied to the resulting SNR time series. `run_slsn_multi_metrics()` runs all five metrics in a single MAF pass per cadence, saving per-event `.npy` arrays and an incremental summary CSV after each cadence.

**Responsible files:** [py_files/slsn_metrics/metrics.py](py_files/slsn_metrics/metrics.py), [py_files/slsn_metrics/runners.py](py_files/slsn_metrics/runners.py)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `evaluate_slsn(self, dataSlice, slice_point, return_full_obs)` | metric instance, structured array, dict, bool | (snr, filters, mjds, obs_record) | Core evaluator: maps catalog bands → LSST, interpolates M_abs, applies DM + extinction, computes SNR |
| `detect_slsn(filters, snr, times, mags, obs_record)` | arrays | bool | Firth+2015: ≥2 filters at SNR≥5, rising LC (dm<−0.1 over 0.5–30 d), ≥15-day baseline |
| `SLSN_Detect_Metric.run(dataSlice, slice_point)` | metric instance (method), structured array, dict | float | Returns 1.0 if detected, 0.0 otherwise; stores metadata in `obs_records` |
| `SLSN_CharacterizeMetric.run(...)` | same | float | Requires detection + ≥5 epochs, ≥3 filters, ≥2 near peak (±10 d), ≥1 post-peak (>+30 d) |
| `SLSN_VillarMetric.run(...)` | same | float | Three Villar+2018 sub-criteria: >10 total detections (M1), >20 near peak (M2), measurable duration in r-band (M3) |
| `SLSN_ELAsTiCC_Metric.run(...)` | same | float | PLAsTiCC/ELAsTiCC alert trigger: ≥2 observations with |SNR|>3 separated by ≥30 min |
| `SLSN_SpecTriggerMetric.run(...)` | same | float | Detection + ≥2 detections within ±20 d of peak, ≥2 filters near peak, peak brighter than 23.0, Δmag<1.0 over 30 d |
| `run_slsn_multi_metrics(templates, population, cadences, ...)` | LC, UserPointsSlicer, list | DataFrame | Runs all five metrics in one MAF pass per cadence; saves `.npy` and summary CSV per cadence |
| `run_slsn_multi_metrics_parallel(...)` | same + `n_workers` | DataFrame | Parallel wrapper: one worker process per cadence via `ProcessPoolExecutor` |
| `build_filenames(rate_density, z_min, z_max, ...)` | floats, str | (str, str, str, str, str) | Constructs canonical filenames for templates, population, results, summary |

**Reads from disk:** `templates.pkl` (via `LC` instance); OpSim `.db` files in `cadences/`  
**Writes to disk:** `output/SLSNe/{model}/metric_values_{short}_{run_tag}.npy` (per-event 0/1, float32); `output/SLSNe/{model}/summary_{run_tag}.csv` (incremental per cadence)  
**Next stage needs:** `.npy` arrays and population `.pkl` files for analysis notebook; summary CSVs for aggregate tables.

---

### Stage 7 — Results and Diagnostics (`diagnostics.py`, analysis notebooks)

`diagnostics.py` provides template QA (residual analysis, band coverage), population diagnostics (rate vs redshift, sky maps), and the Monte Carlo rate uncertainty calculation. The `analysis_multimodel_comparison.ipynb` notebook loads all three summary CSVs, the three population pickles, and the `.npy` metric arrays, then produces the final science figures: detection-per-year tables, bar charts, efficiency-vs-redshift plots, HEALPix sky maps, and three-panel MC uncertainty figures.

**Responsible files:** [py_files/slsn_metrics/diagnostics.py](py_files/slsn_metrics/diagnostics.py), [notebooks/analysis_multimodel_comparison.ipynb](notebooks/analysis_multimodel_comparison.ipynb)

**Key functions:**

| Function | Input | Returns | Does |
|---|---|---|---|
| `diagnose_abs_from_templates(event_name, per_event_dir, templates_file, template_idx, z, t0)` | str, Path, Path, int, float, float | DataFrame | Computes m_obs − (M_template + DM) residuals per band |
| `characterize_template_coverage(templates_file)` | Path | DataFrame | Reports n_bands, phase_span, z per template |
| `plot_population_diagnostics(ra_rad, dec_rad, peak_times, distances_mpc, z_vals, ...)` | arrays | None | Generates sky map, redshift histogram, EBV map, distance distribution |
| `plot_rate_evolution_comparison(save_path)` | Path | None | Plots R(z) curves for all three rate models + observed measurements |
| `compare_simulated_vs_observed_rates(population_slicer)` | UserPointsSlicer | DataFrame | Bins simulated events in redshift; computes simulated rate and compares to literature |
| `_sample_R_ref(n_samples, R_mode, sig_hi, sig_lo, seed)` | int, floats, int | array | Draws from Frohmaier+2021 split-normal: 35 +25/−13 Gpc⁻³ yr⁻¹ |
| `plot_mc_rate_uncertainty(detect_vals, peak_times, N_injected_nominal, R_ref_nominal, ...)` | arrays, int, float | None | Single-model MC uncertainty: cumulative N(SLSNe) ± 68% CI and optional significance vs comparison model |
| `plot_mc_rate_uncertainty_panel(pop_data, cadence, metric_key, ...)` | dict, str, str | None | Three-panel figure: N(SLSNe) vs year, significance vs year, ratio significance for all three models |
| `plot_healpix_efficiency(...)` | arrays | None | HEALPix Mollweide projection of detection efficiency |

**Reads from disk:** Templates pickle; population pickles; `.npy` metric arrays; summary CSVs  
**Writes to disk:** PNG figures to `output/SLSNe/{model}/` and `output/SLSNe/`

---

## 3. How Ben's Rate Model Data Works

### What is in `fiducial_models.csv`?

The file at `output/SLSNe/shared/fiducial_models.csv` has three columns with no named header row. Column 0 is the redshift `z`, running from 0.0 to approximately 6.0 in steps of 0.06 (101 rows). Column 1 is `f_OH`, the fraction of cosmic star formation occurring in galaxies whose oxygen abundance falls below a threshold (Z_cutoff_OH = 0.6 Z☉, or equivalently 12+log(O/H) ≈ 8.5), as a function of redshift. Column 2 is `f_Fe_mixed`, the analogous fraction for an iron-abundance threshold (Z_cutoff_Fe = 0.2 Z☉). These fractions were computed by Ben Margalit by integrating the Tremonti+2004 mass-metallicity relation convolved with the Leja+2020 stellar mass function and the Leja+2022 star-forming main sequence, a procedure that determines what fraction of all star formation at each epoch occurs in galaxies metal-poor enough to produce a SLSN. Both fractions are dimensionless numbers between 0 and 1; at z=0 they are roughly 0.10 (O-dependent) and 0.05 (Fe-dependent), rising monotonically to near unity at z ≳ 4 as the universe was predominantly metal-poor.

A companion file, `fiducial_models_uncertainty.csv`, contains five columns: `z`, `f_OH+0.2`, `f_OH-0.2`, `f_Fe_mixed+0.1`, `f_Fe_mixed-0.1`. These are the metallicity fractions recomputed after shifting the threshold metallicity up and down by the amounts in the column names, encoding systematic uncertainty in the metallicity cutoff. **This file is not currently read by any module in the package** — it exists on disk but has no code path that consumes it.

### The Three Rate Models

The function `load_tabulated_rate()` in `population.py` (lines 524–594) reads the CSV and constructs one of three `R(z)` interpolators:

**`naive`**: R(z) = R_ref × Ψ(z)/Ψ(z_ref). The SLSN rate simply tracks the cosmic star formation rate density (CSFRD) Ψ(z) following Madau & Dickinson 2014, with no metallicity correction. SLSNe occur at the same fraction of star formation at all epochs.

**`fe_dependent`**: R(z) = R_ref × [Ψ(z)/Ψ(z_ref)] × [f_Fe(z)/f_Fe(z_ref)]. The SLSN rate is additionally weighted by the iron-abundance metallicity fraction, rising more steeply with redshift as an increasing fraction of star formation occurs in iron-poor galaxies. This is the primary science result model.

**`o_dependent`**: R(z) = R_ref × [Ψ(z)/Ψ(z_ref)] × [f_OH(z)/f_OH(z_ref)]. Identical in structure to `fe_dependent` but using the oxygen-abundance fraction, which has a weaker redshift dependence at low z because the oxygen threshold is less restrictive. This serves as a completeness check.

All three are anchored to R_ref = 35 Gpc⁻³ yr⁻¹ at z_ref = 0.17, the Frohmaier+2021 spectroscopic completeness-corrected measurement. The normalization constant `RATE_REF_GPC3 = 35.0` and `RATE_REF_MPC3 = 35.0 / 1e9` appear in `population.py` at line 71.

### How `generate_SLSN_PopSlicer()` Uses the CSV

When called with `rate_model='tabulated'` and a `model_name`, the function proceeds in seven steps. (1) `load_tabulated_rate(tabulated_csv, model_name)` reads the CSV with `np.genfromtxt`, extracts the appropriate column, normalizes by the CSFRD ratio at each redshift, converts from Gpc⁻³ to Mpc⁻³, and returns a cubic interpolating function `rate_interp(z)`. (2) `sample_events_from_tabulated_rate()` integrates ∫R(z)·(dV/dz)·dz over the redshift range, multiplies by the survey duration in years, and Poisson-samples the total event count. (3) `sample_redshifts_from_tabulated_rate()` builds a CDF of R(z)·dV/dz on a 1000-point grid and draws `n_events` redshifts by inverse-CDF sampling, correctly weighting the redshift distribution by where events actually occur. (4) Uniform HEALPix sky positions are drawn and cached. (5) A Galactic latitude cut (|b| > 15° by default) is applied jointly to both the sky positions and the redshift array, keeping them synchronized. (6) Each event is assigned a template index drawn uniformly from templates with wavelength coverage > 2000 Å, a peak time drawn uniformly across the survey window, and an E(B−V) from the SFD dust map. (7) All per-event properties are packed into the MAF `UserPointsSlicer.slice_points` dictionary and saved to pickle.

The output population is an array of N events, where N for the `fe_dependent` model at z=0.1–2.0 over ten years is approximately 2.24 million events (Cell 3 output). The difference in N between models directly reflects the different R(z) integrals: `fe_dependent` gives roughly 2× more events than `naive` because the iron-abundance metallicity fraction rises faster with redshift, producing proportionally more SLSN hosts at z > 1.

### Why the Rate Model Choice Matters

The detection count scales approximately linearly with the total volumetric rate (more events means more caught). At baseline cadence, `fe_dependent` yields ~642 detections/yr vs. ~330/yr for `naive` (Table from Cell 3 of the notebook) — a factor of ~1.95×. This factor is scientifically meaningful: if the true SLSN rate follows `fe_dependent` physics, LSST will detect roughly twice as many SLSNe as a rate-agnostic calculation suggests. The significance calculation in `plot_mc_rate_uncertainty_panel()` asks how many years of Rubin data are needed before `fe_dependent` and `naive` are distinguishable beyond the Poisson noise floor plus R_ref uncertainty. By year 10, the models produce detectably different cumulative counts if the measurement is purely Poisson-limited, though the R_ref uncertainty from Frohmaier+2021 contributes additional scatter that widens the confidence bands.

---

## 4. The K-Correction Problem and How the Physical Templates Solve It

### What is a K-correction?

A K-correction converts a flux measurement in a fixed observed-frame photometric bandpass to what the same source would have been measured at in a reference rest-frame bandpass. It is necessary because a telescope always observes at the same detector wavelength regardless of target redshift, but the photons it catches were emitted at shorter rest-frame wavelengths. For a source at redshift z, LSST's r-band (peak throughput ~620 nm) samples rest-frame ~620/(1+z) nm. At z=0.5 this is ~413 nm (blue-UV), at z=1.0 it is ~310 nm (far-UV). Without correcting for the different SED shape in these wavelength windows, the measured brightness cannot be compared directly to the known rest-frame absolute magnitude.

For a survey like LSST, which observes SLSNe over a vast redshift range (z=0.1–2.0), ignoring K-corrections introduces systematic errors in the apparent magnitude model that grow with redshift. An SLSN whose spectrum falls steeply blueward would appear much fainter in the LSST blue bands at high z than a naive distance-modulus calculation suggests, biasing the detection efficiency.

### What Goes Wrong with the GP Templates at z > 1.2

The GP templates in `templates.pkl` are constructed from observations of SLSNe that were actually detected by telescopes, primarily at z < 1. The cenwave files record which wavelengths were observed in the catalog photometry. `LC.from_catalog()` builds the SED grid restricted to the observed wavelength range: `lam_rest_min = lam_obs_min / (1+z)` and `lam_rest_max = lam_obs_max / (1+z)` (model.py lines 514–518). For a z=0.5 SLSN observed with optical bands, this covers roughly 2000–8000 Å rest-frame — adequate for prediction in LSST optical bands at similar redshifts.

The breakdown occurs in `synthesize_mag_at_z()` (model.py lines 106–161). After redshifting the rest-frame SED to the observer frame, the code checks how many wavelength samples overlap the LSST bandpass: `overlap_mask = (lam_obs_nm >= bp.wavelen.min()) & (lam_obs_nm <= bp.wavelen.max())`. If `np.sum(overlap_mask) < 20`, the function returns `np.nan`. For a template built from z=0.5 observations (covering 2000–8000 Å rest), when evaluated at z=1.5 in the LSST u-band (~320–400 nm observed, corresponding to 128–160 Å rest), there are zero template wavelengths that overlap the bandpass. The function returns NaN, the event registers as undetectable in that band, and effectively drops out of the detection statistics. This threshold of z ≈ 1.2 is where LSST optical bands begin sampling wavelengths blueward of the template's rest-frame UV coverage.

**However**, this drop path through `synthesize_mag_at_z()` is only triggered if that function is actually called. As described in Gap §8.3, the current production metric evaluator (`evaluate_slsn()` in metrics.py) does **not** call `synthesize_mag_at_z()`. It uses direct catalog-band interpolation with static color offsets. So the GP templates do not silently NaN at z>1.2 in the current production code — instead they silently give incorrectly computed apparent magnitudes derived from extrapolated color offsets.

### What the Physical Templates Provide

`gomez_models.py` computes a physically motivated SED from first principles for each SLSN, using the best-fit MOSFiT magnetar+ejecta parameters. The physical chain proceeds as follows:

**`total_luminosity()`** (gomez_models.py line 131): Sums the nickel-cobalt radioactive decay luminosity (Nadyozhin 1994) and the Ostriker & Gunn 1971 magnetar spin-down power to produce the total energy injection rate L_in(t) [erg/s]. This gives the bolometric input power as a function of rest-frame time.

**`diffusion()`** (line 157): Propagates L_in(t) through the expanding supernova ejecta using a radiation diffusion integral on a log-spaced time grid. The diffusion timescale τ_diff = sqrt(κ·M_ej / v_ej) sets the characteristic photon escape time. The output L_out(t) is the bolometric luminosity actually radiated from the photosphere.

**`photosphere()`** (line 214): Given L_out(t) and the expanding ejecta velocity, computes the photospheric radius R_phot(t) and temperature T_phot(t). When the ejecta are optically thick enough to confine the radiation, the photosphere is at the ejecta surface; when they thin out, T_phot is set to the `temperature` parameter (recombination floor).

**`blackbody_supressed()`** (line 288): Evaluates a modified blackbody SED — a Planck function scaled to R_phot(t) — at all wavelengths in the rest-frame grid (500–12000 Å), then applies a power-law suppression `(λ/λ_cutoff)^α` blueward of `cutoff_wavelength`. This suppression mimics the broad-line absorption features seen in SLSN spectra. The output is F_λ(λ, t) [erg/s/Å] over a 3000-point wavelength grid covering Swift UVW2 through LSST y at z=2.

Because the wavelength grid covers 500–12000 Å rest-frame by construction, `synthesize_mag_at_z()` will always find sufficient overlap with any LSST bandpass at any redshift z ≤ 2 (at z=2, LSST y samples ~3600 Å rest, well within the grid). This is the core advantage over the GP templates.

### How `mosfit_interface.py` Connects the Parameters to the Grid

`build_physical_sed_grid()` (mosfit_interface.py line 197) reads `all_parameters.txt` row by row. For each event, it extracts 13 posterior median parameters. The magnetic field convention requires care: the column `log(Bfield)_med` stores log10(B/Gauss), while `gomez_models.magnetar()` expects B in units of 10¹⁴ G, so `Bfield = 10^col / 1e14`. The `texplosion` parameter is the explosion epoch relative to the observed peak MJD, typically negative (−20 to −80 days). `_call_slsnni_safe()` runs the four-step physical chain with the phase grid shifted so that phase=0 is the explosion epoch (not peak), masks all phases where T_phot=0 (pre-explosion), calls `blackbody_supressed()` only on valid phases, then reconstructs the full phase array by setting pre-explosion rows to zero flux. `_flam_to_fnu_jy_at_10pc()` converts F_λ [erg/s/Å] (total luminosity, not flux) to F_ν [Jy] at 10 pc by dividing by 4π·D_10pc² and applying the unit conversion F_ν = F_λ·λ²/c.

The resulting `sed_entry` dict — containing `phase` (array of rest-frame days from +1 to +400), `lam_rest_A` (3000 wavelength samples), `Fnu_abs` (2D float32 array [N_phase, N_lam]), and `coverage` (bool mask) — is byte-for-byte identical in format to the SED grids produced by `LC.from_catalog()` (model.py lines 551–556). No other module knows which path produced the SED grid.

### How the Physical SED Grid Slots into the Rest of the Pipeline

`build_physical_templates()` (mosfit_interface.py line 371) wraps `build_physical_sed_grid()`, constructs an `LC` instance with empty `lightcurves` (required by the `LC` constructor but unused in the SED-synthesis path), attaches the `sed_grid` list, and optionally saves to `physical_templates.pkl`. This object is then passed to `generate_SLSN_PopSlicer()` and `build_magnitude_grid()` identically to a GP-derived template object. The magnitude grid build (`build_magnitude_grid()`) then calls `_compute_grid_slice()` → `synthesize_mag_at_z()` for each (template, z, phase, filter) cell using the physical SED grids, producing a `physical_mag_grid.pkl` that can be loaded at metric time.

What changes when switching from GP to physical templates: the SED grids (broader wavelength coverage, physically constrained shape). What stays the same: the metric classes, the population generator, the MAF runner, all paths downstream of the LC object.

---

## 5. How the MC Rate Uncertainty Works

### What is Monte Carlo Rate Uncertainty?

The detection count predicted by the pipeline scales as N_det = ε × R × V × T, where ε is the detection efficiency (from MAF), R is the volumetric rate, V is the effective survey volume, and T is the survey duration. The efficiency ε is computed precisely from the simulation. But R is known only from observations: Frohmaier+2021 measured it spectroscopically as 35 +25/−13 Gpc⁻³ yr⁻¹ at z=0.17, with large and asymmetric uncertainties. Propagating this uncertainty into the predicted detection count requires sampling R from its error distribution and computing how the distribution of predicted N_det changes — a Monte Carlo propagation.

### R_ref and Its Origin

R_ref = 35.0 Gpc⁻³ yr⁻¹ is the Frohmaier et al. 2021 (MNRAS, staa3607) spectroscopic volumetric rate at z=0.17. This measurement corrected for survey completeness and spectroscopic selection bias. Its asymmetric error (+25/−13 Gpc⁻³ yr⁻¹) reflects that the Poisson upper bound is looser than the lower bound given the small observed sample. In the code, `RATE_REF_GPC3 = 35.0` and `Z_REF = 0.17` in `population.py` (lines 71–72).

### What `fiducial_models_uncertainty.csv` Contains

This file, present at `output/SLSNe/shared/fiducial_models_uncertainty.csv`, encodes a different kind of uncertainty from R_ref: it stores the sensitivity of the metallicity fractions f_OH and f_Fe to the choice of metallicity threshold. The five columns are z, f_OH+0.2, f_OH-0.2, f_Fe_mixed+0.1, and f_Fe_mixed-0.1, representing the fractions recomputed after shifting the metallicity threshold up or down by the indicated amounts. This allows the rate model to be bracketed by its sensitivity to where the metallicity cutoff is drawn. **This file is not currently loaded or used by any Python code in the package.** It exists on disk but has no code path connecting it to the MC uncertainty analysis or the rate model loader.

### Where the MC Uncertainty Is Implemented

The MC rate uncertainty is implemented in **`diagnostics.py`**, not only in the notebook. Two functions handle it:

`_sample_R_ref(n_samples, R_mode=35.0, sig_hi=25.0, sig_lo=13.0, seed)` (diagnostics.py line 2291) draws from a split-normal distribution: each sample `u ~ N(0,1)` maps to `R_mode + sig_hi * u` if `u ≥ 0`, else `R_mode + sig_lo * u`. Samples are clipped to a minimum of 1.0 to prevent unphysical negatives.

`plot_mc_rate_uncertainty_panel(pop_data, cadence, metric_key, R_ref_nominal=35.0, n_realizations=1000, ...)` (diagnostics.py line 2465) is the production function called from Cell 10 of the notebook. For each of the three rate models, it:
1. Draws 1000 R_ref samples.
2. Computes the "efficiency at each survey year" as `base_cumulative[t] / n_injected`.
3. Constructs the MC matrix: `mc_mat = efficiency_t[None, :] * R_samples[:, None] * V_ref`, where `V_ref = n_injected / R_ref_nominal`.
4. Takes the 16th and 84th percentiles across realizations for the 68% CI bands.
5. Plots a three-panel figure: top panel (N(SLSNe) vs year), middle panel (significance between model pairs), bottom panel (ratio significance where R_ref cancels).

The significance formula in the middle panel is: `sig = |N_m1 - N_m2| / sqrt(σ_m1² + σ_m2² + N_m1 + N_m2)`, where σ is the MC standard deviation and the `N` terms add Poisson noise. The ratio significance panel uses the delta-method on ln(N_m1/N_m2): `sig_ratio = |ln(N1/N2)| / sqrt(1/N1 + 1/N2)`, which eliminates R_ref uncertainty entirely since the ratio `N_m1/N_m2 = ε_m1·V_m1 / ε_m2·V_m2` is independent of the common normalization.

The notebook Cell 10 calls `plot_mc_rate_uncertainty_panel` in a loop over cadences. The function lives in `diagnostics.py` and is fully importable from the module — it is not notebook-only. The simpler `plot_mc_rate_uncertainty()` (single model, single vs comparison) is also in `diagnostics.py` at line 2307.

### When Do `fe_dependent` and `naive` Separate at 3σ?

From the notebook's production run (Cell 3 outputs), the cumulative detections after 10 years at baseline cadence are ~6420 (fe_dependent) vs ~3300 (naive). The Poisson uncertainty on fe_dependent alone is √6420 ≈ 80, and the R_ref spread (±25/−13 Gpc⁻³ yr⁻¹, or ~40–70% fractional) is much larger. The significance depends on whether one is comparing absolute counts (R_ref uncertainty dominates and the two models cannot be separated) or the ratio (R_ref cancels and separation appears within the first few years). The notebook Cell 10 produces these curves numerically for each cadence; the exact crossing of 3σ in the ratio panel depends on the survey cadence but is generically reachable by year 3–5 given current efficiency estimates.

---

## 6. What Lives Where — The Cross-Reference Table

| Concept | Implemented in | Function name | Called by | Writes to disk? |
|---|---|---|---|---|
| GP template building | `model.py` | `LC.from_catalog()` | Notebook / manual | `templates.pkl` |
| Physical SED template building | `mosfit_interface.py` | `build_physical_templates()` | Notebook / manual | `physical_templates.pkl` |
| Magnitude grid building | `model.py` | `LC.build_magnitude_grid()` | Notebook / manual | `mag_grid.pkl` (+ `.checkpoint.pkl`) |
| GP template loading | `model.py` | `LC(load_from=...)` | `run_slsn_pipeline.py`, notebook | No |
| Physical template loading | `mosfit_interface.py` | `build_physical_templates()` (with cache) | Notebook | No (cache file optional) |
| Population generation (fe_dependent) | `population.py` | `generate_SLSN_PopSlicer(..., model_name='fe_dependent')` | `run_slsn_pipeline.py` | `population_fe_dependent.pkl` |
| Population generation (naive) | `population.py` | `generate_SLSN_PopSlicer(..., model_name='naive')` | `run_slsn_pipeline.py` | `population_naive.pkl` |
| Population generation (o_dependent) | `population.py` | `generate_SLSN_PopSlicer(..., model_name='o_dependent')` | `run_slsn_pipeline.py` | `population_o_dependent.pkl` |
| Detect metric | `metrics.py` | `SLSN_Detect_Metric.run()` | `run_slsn_multi_metrics()` | `metric_values_detect_{tag}.npy` |
| Characterize metric | `metrics.py` | `SLSN_CharacterizeMetric.run()` | `run_slsn_multi_metrics()` | `metric_values_characterize_{tag}.npy` |
| SpecTrigger metric | `metrics.py` | `SLSN_SpecTriggerMetric.run()` | `run_slsn_multi_metrics()` | `metric_values_spectrigger_{tag}.npy` |
| Villar metric | `metrics.py` | `SLSN_VillarMetric.run()` | `run_slsn_multi_metrics()` | `metric_values_villar_{tag}.npy` |
| ELAsTiCC metric | `metrics.py` | `SLSN_ELAsTiCC_Metric.run()` | `run_slsn_multi_metrics()` | `metric_values_elasticc_{tag}.npy` |
| MC rate uncertainty | `diagnostics.py` | `plot_mc_rate_uncertainty_panel()` | Notebook Cell 10 | PNG figures |
| Redshift-binned efficiency | Notebook Cell 7 | inline code | — | PNG figures |
| Summary CSV writing | `runners.py` | `run_slsn_multi_metrics()` | `run_slsn_pipeline.py` | `summary_{run_tag}.csv` |
| Per-event .npy writing | `runners.py` | `run_slsn_multi_metrics()` and `_run_cadence_worker()` | `run_slsn_pipeline.py` | `metric_values_{short}_{run_tag}.npy` |

---

## 7. Reload Decision Guide

| Artifact | What triggers a rebuild | Rebuild function call | Downstream artifacts that also need rebuilding |
|---|---|---|---|
| `templates.pkl` (GP) | New events added to catalog; cenwave CSVs updated; GP kernel changed; `t0` logic changed | `LC.from_catalog(inputs, save_to=Path("output/SLSNe/shared/templates.pkl"))` in a notebook | `mag_grid.pkl`, all three `population_{model}.pkl`, all `.npy` metric files |
| `physical_templates.pkl` (MOSFiT) | `all_parameters.txt` updated; wave/phase grid changed; `gomez_models.py` physics changed | `build_physical_templates(params_file=..., save_to=Path("output/SLSNe/shared/physical_templates.pkl"))` in a notebook | `physical_mag_grid.pkl`, all three `population_{model}.pkl`, all `.npy` metric files |
| `mag_grid.pkl` (GP) | `templates.pkl` rebuilt; z-grid or phase-grid bounds changed; filter list changed | `templates.build_magnitude_grid(save_to=Path("output/SLSNe/shared/mag_grid.pkl"))` on a loaded `LC` instance | All three `population_{model}.pkl`, all `.npy` metric files (only if/when mag_grid is used in evaluation; see Gap §8.3) |
| `physical_mag_grid.pkl` (MOSFiT) | `physical_templates.pkl` rebuilt; z-grid or phase-grid bounds changed | `phys_templates.build_magnitude_grid(save_to=Path("output/SLSNe/shared/physical_mag_grid.pkl"))` | All three `population_{model}.pkl`, all `.npy` metric files |
| `population_{model}.pkl` | Rate CSV (`fiducial_models.csv`) updated; z_min/z_max changed; `gal_lat_cut` changed; random seed changed; template set changed | `generate_SLSN_PopSlicer(lc_model=templates, rate_model='tabulated', model_name='{model}', tabulated_csv=..., save_to=..., ...)` | All `.npy` metric files for that model |
| `metric_values_{metric}_{model}_{cadence}.npy` | Population pickle changed; metric criterion changed (thresholds, detection logic); OpSim database updated; `evaluate_slsn()` logic changed | `run_slsn_pipeline.py --model {model} --cadences {cadence} --regen-population` (or `run_slsn_multi_metrics()` directly) | Summary CSVs; notebook figures |

---

## 8. Gaps and Missing Pieces

| # | Type | Location | What is missing | Fix |
|---|---|---|---|---|
| 1 | **Broken** | `launch_all.py` line 103 | Passes `--cadence {cadence}` (singular) to `run_slsn_pipeline.py`, which defines the argument as `--cadences` (plural, `nargs='+'`); this causes an argparse error on every launch | Change `'--cadence'` to `'--cadences'` in the `cmd` list in `launch_all.py` |
| 2 | **Unimplemented** | `run_slsn_pipeline.py` docstring lines 33–35; `mosfit_interface.py` docstring line 40 | `--regen-templates` and `--regen-mag-grid` CLI flags are documented as not yet existing; template and mag-grid rebuilds must be done manually in a notebook | Add these flags to `parse_args()` and the corresponding rebuild logic in `main()` in `run_slsn_pipeline.py` |
| 3 | **Gap (critical)** | `metrics.py` `evaluate_slsn()` lines 141–259 | `evaluate_slsn()` uses `self.lc_model.interp(t, catalog_band, tpl_idx)` — direct catalog-band interpolation with static color offsets — and never calls `synthesize_mag_at_z()` or uses the `_interps` RegularGridInterpolators from `build_magnitude_grid()`; the magnitude grid infrastructure is built but bypassed at evaluation time | Replace `evaluate_slsn()` with a path that calls `synthesize_mag_at_z_cached()` (already defined in metrics.py at line 118) using `lc_model.sed_grid[tpl_idx]` when `sed_grid` is available, falling back to interpolation only when it is not |
| 4 | **Undocumented** | `output/SLSNe/shared/fiducial_models_uncertainty.csv` | File exists on disk with metallicity-threshold sensitivity data (f_OH±0.2, f_Fe±0.1) but is not read by any module; no code path loads or uses it | Add a `load_tabulated_rate_uncertainty(csv_path, model_name, direction)` function in `population.py` that reads this file and add a `--rate-csv-uncertainty` flag to `run_slsn_pipeline.py` |
| 5 | **Notebook-only** | `analysis_multimodel_comparison.ipynb` Cell 7 | Redshift-binned efficiency calculation (efficiency vs z for all models × cadences × metrics) is implemented as inline cell code rather than a module function | Move the z-binning loop into a function in `diagnostics.py`, e.g. `compute_efficiency_vs_z(pop_data, z_edges, cadence, metric_key)` |
| 6 | **Notebook-only** | `analysis_multimodel_comparison.ipynb` Cell 8 | Adam's fixed-z-bin analysis (z=1.2–1.8 and z=1.3–1.7 detection tables) is inline notebook code with no module counterpart | Extract into `diagnostics.py` as `compute_efficiency_fixed_bin(pop_data, z_lo, z_hi, metric_key)` |
| 7 | **Undocumented** | `metrics.py` lines 92–113 | A commented-out `detect_slsn()` function using simplified criteria (≥2 observations at SNR≥5, ≥2 filters, no baseline or rising requirement) is dead code inside the active module; it is confusing and may be mistaken for an alternative production path | Delete the commented-out block or move it to a clearly labelled `_detect_simple_for_testing()` helper |
| 8 | **Undocumented** | `metrics.py` `SLSN_SpecTriggerMetric` docstring lines 738–743 | The spectroscopic magnitude limit of 23.0 is flagged as "pending confirmation from instrumentation collaborators" and should not be treated as a fixed science result | Track the confirmation status; add a `mag_limit_source` metadata field to `SLSN_SpecTriggerMetric` and record the sensitivity result in the notebook once confirmed |
| 9 | **Undocumented** | `population.py` lines 1056–1058 | `generate_SLSN_PopSlicer_with_rate_evolution` is a backward-compatibility alias that silently maps to `generate_SLSN_PopSlicer`; its existence is not mentioned anywhere, and callers may not realize they are calling the same function | Add a deprecation warning via `warnings.warn()` inside the alias or remove it and update any call sites |
| 10 | **Broken (latent)** | `mosfit_interface.py` default `params_file` (line 409) | `build_physical_templates()` defaults to `repo_root / "SLSNe" / "slsne" / "ref_data" / "all_parameters.txt"` — a path that does not exist in the checked-in repository layout (`paths.py` shows no `SLSNe/slsne` directory); the function only works if `params_file` is passed explicitly | Document the required `params_file` path and update the default to the actual location, or remove the default and require explicit specification |
| 11 | **Gap** | `population.py` `metallicity_fraction()` lines 374–379 | Uses `np.trapezoid` (NumPy 2.0 spelling) but the `cosmic_sfr_density_MD14()` function is called inside a Python `for` loop over z values instead of vectorized; for 200 mass bins × 1000 z values this is slow | Replace the loop with fully vectorized array operations using `np.trapezoid` on a 2D array |
