# GAP3\_AUDIT.md — Physical Template Integration Audit

**Branch:** `unified-template-interface`  
**Date:** 2026-05-08  
**Purpose:** Complete audit of the GP pipeline to ensure the physical template
implementation mirrors existing style exactly.  All line numbers refer to
the branch-current versions of the files as read.

---

## Section 1 — The GP Data Flow: Template to Figure

Each arrow in the chain below shows exact function signatures, data structures,
and assumptions.

---

### Step 1 — `LC(load_from=templates.pkl)`
**File:** `py_files/slsn_metrics/model.py` lines 260–272

**Signature:**
```python
class LC:
    def __init__(self, num_lightcurves=None, load_from=None, lightcurves=None, t_grid=None, names=None)
```

**What happens:** Opens the pickle file and unpacks into instance attributes.
```python
self.data                  = obj["lightcurves"]     # list of band dicts
self.t_grid                = obj.get("t_grid", None)
self.names                 = obj.get("names", [...])
self.template_file         = load_from              # string path
self.sed_grid              = obj.get("sed_grid", None)
self.median_cenwave_by_band = obj.get("median_cenwave_by_band", None)
```

**Returns:** `LC` instance.

**Next step expects:** `.data`, `.names`, `.sed_grid` all populated.

---

### Step 2 — `generate_SLSN_PopSlicer(lc_model=templates, ...)`
**File:** `py_files/slsn_metrics/population.py` lines 676–1048

**Signature:**
```python
def generate_SLSN_PopSlicer(lc_model, t_start=1, t_end=3652, z_min=0.1, z_max=2.0,
    rate_model='constant', rate_density=1e-7, R_ref=1e-7, z_ref=0.17, OH_max=8.3,
    tabulated_csv=None, model_name=None, peak_t_min=None, peak_t_max=None,
    gal_lat_cut=None, seed=42, healpix_cache_file=None, save_to=None,
    load_from=None, make_debug_plots=True, max_events=None)
```

**What it does with `lc_model`:**

1. Line 816: iterates `lc_model.sed_grid` to find templates with sufficient wavelength coverage (`lam_max - lam_min > 2000 Å`). Stores indices in `valid_templates`.
2. Line 954: draws per-event template assignments: `file_indx = rng.choice(valid_templates, size=n_events)`.
3. Lines 1000–1016: for each event, accesses `lc_model.data[idx][f]['mag']` to compute per-filter peak magnitude summaries (`peak_mag_abs_f`, `peak_app_mag_noebv_f`, `peak_app_mag_ebv_f`).

**What it writes to `slicer.slice_points`:**
```
sid, z, distance, distance_modulus, peak_time, file_indx, ebv,
gall, galb, ra, dec, rate_model, A_u/g/r/i/z/y,
peak_mag_abs_{f}, peak_app_mag_noebv_{f}, peak_app_mag_ebv_{f}  (for f in ugrizy)
```

**Returns:** `UserPointsSlicer` with populated `slice_points`.

**Next step expects:** `slice_points['file_indx']`, `slice_points['z']`, `slice_points['peak_time']`, `slice_points['distance_modulus']`, `slice_points['ebv']`, `slice_points['A_{f}']`, `slice_points['sid']`.

---

### Step 3 — `run_slsn_multi_metrics(templates, population, cadences, ...)`
**File:** `py_files/slsn_metrics/runners.py` lines 902–1145

**Signature:**
```python
def run_slsn_multi_metrics(templates, population, cadences, db_dir=None,
    output_dir=None, *, metrics_list=None, mjd0=60980.5, ignore_triples=True,
    save_summary=True, make_plots=False, verbose=True, store_obs_mode='none',
    model_name=None, z_min=0.1, z_max=2.0, only_metrics=None)
```

**What it does:**
1. Instantiates all five metric objects with `lc_model=templates` (lines 968–978).
2. Builds `MetricBundle` per metric, one MAF pass per cadence (lines 1040–1046).
3. Calls `group.run_all()` — MAF iterates over `population` slice points and calls each metric's `run()` which calls `evaluate_slsn()`.
4. For each metric × cadence, saves `bundle.metric_values` as:
   - `metric_values_{short}_{model}_{cadence}_{YYMMDD_HHMM}.npy` — shape `(N_events,)`, dtype `float32`, values 0.0 or 1.0 (lines 1083–1091)
5. Saves `summary_{run_tag}.csv` with columns `cadence, metric, n_events, n_success, efficiency` (lines 1119–1127).

**Next step expects:** `metric_values_{short}_*.npy` files with exactly `N_events` rows.

---

### Step 4 — `evaluate_slsn(self, dataSlice, slice_point, return_full_obs=True)`
**File:** `py_files/slsn_metrics/metrics.py` lines 140–259

Called inside each metric's `run()` method (e.g. `SLSN_Detect_Metric.run()` line 345).

**Inputs from slice_point used:** `file_indx`, `z`, `peak_time`, `distance_modulus`, `ebv`, `A_{filt}`, `sid`.

**Returns:** `(snr, filters, mjds, obs_record)` where arrays have length = number of OpSim observations in this sky pixel during the survey.

See Section 2 for the complete internal trace.

---

### Step 5 — `detect_slsn()` / characterize criteria / spectrigger criteria
**File:** `py_files/slsn_metrics/metrics.py` lines 18–90

**Signature:**
```python
def detect_slsn(filters, snr, times, mags, obs_record) -> bool
```

**Input:** Per-observation arrays of filter names, SNR, MJD times, apparent magnitudes.

**Criteria:**
1. ≥2 filters with SNR≥5.
2. Rising light curve seen in at least one filter (Δmag < −0.1 over 0.5–30 days).
3. Temporal baseline ≥15 days among SNR≥5 detections.

**Returns:** `bool` — 1.0 from `run()` if True, 0.0 if False.

---

### Step 6 — `metric_values_*.npy` written to disk
**File:** `py_files/slsn_metrics/runners.py` lines 1083–1091

```python
np.save(npy_file, bundle.metric_values.filled(0).astype(np.float32))
```

Shape: `(N_events,)`, one row per injected event, value 0.0 or 1.0.
Filename: `metric_values_{short}_{model_tag}_{cadence}_z{z_min}-{z_max}_{YYMMDD_HHMM}.npy`

---

### Step 7 — Notebook loads `.npy` files
**File:** `notebooks/analysis_multimodel_comparison.ipynb` Cell 3

```python
def find_npy(model_dir, model, cadence, metric_short):
    pattern = str(model_dir / f'metric_values_{metric_short}_{model}_{cadence}_*.npy')
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1])
...
metric_arrays[metric][cadence] = np.load(npy_path).astype(int)
```

Also loads population pickle for `z`, `ra`, `dec`, `peak_time` via `generate_SLSN_PopSlicer`.

Stores results in `pop_data[model]` dict (see Section 5).

---

### Step 8 — `plot_mc_rate_uncertainty_panel()` in diagnostics.py
**File:** `py_files/slsn_metrics/diagnostics.py` lines 2465–2663

**Signature:**
```python
def plot_mc_rate_uncertainty_panel(pop_data, cadence, metric_key='detect',
    metric_label='Detections', R_ref_nominal=35.0, n_realizations=1000,
    survey_years=None, seed=42, save_dir=None)
```

**Input:** `pop_data[model][metric_key][cadence]` = 0/1 integer array from `.npy`.
Also uses `pop_data[model]['peak_time']` (days, 1–3652 from survey start).

**What it computes:**
1. For each model: MC-scales cumulative detections by drawing `R_ref` from Frohmaier+2021 split-normal.
2. Plots cumulative N(SLSNe) vs survey year with 68% CI band.
3. Plots model-separation significance curves (fe vs naive, o vs naive, fe vs o).
4. Plots ratio significance with R_ref cancellation.

**Output:** PNG figures saved to `save_dir`.

---

## Section 2 — The `evaluate_slsn()` Function in Detail

**Location:** `py_files/slsn_metrics/metrics.py` lines 140–259

### What `self.lc_model` contains (GP templates loaded)

When `LC(load_from=templates.pkl)` is called:
- `.data` — `list[dict]`: each element is `{band_name: {"ph": ndarray, "mag": ndarray}}` where `band_name` is the catalog filter string (e.g. `'B'`, `'V'`, `'R'`, `'g'`, `'r'`). Magnitudes are **absolute** rest-frame AB mags.
- `.t_grid` — phase array (rest-frame days from peak, first template's grid).
- `.names` — `list[str]`: event name strings matching catalog.
- `.sed_grid` — `list[dict]`: per-template SED grids (see Section 3).
- `.median_cenwave_by_band` — `dict[name → dict[band → Cenwave_A]]`.
- `.template_file` — string path to pkl.
- `._interps` — NOT set at load time; set after `build_magnitude_grid()` or `load_magnitude_grid()` is called. Built per-filter as `list[RegularGridInterpolator]` over `(z, phase)`.

### Template index per injected event

**Line 172:**
```python
tpl_idx = int(slice_point['file_indx'])
```
`file_indx` was set in `generate_SLSN_PopSlicer` (population.py line 954):
```python
file_indx = rng.choice(valid_templates, size=n_events)
```
where `valid_templates` are indices of templates passing the wavelength coverage check (sed_grid `lam_max − lam_min > 2000 Å`).

### Rest-frame phase computation

**Line 196:**
```python
time_rel = (mjds - self.mjd0 - peak_time) / (1.0 + z)
```
- `self.mjd0 = 60980.5` (survey start MJD, set in `__init__` line 313)
- `peak_time` = days since survey start when this event peaks (from `slice_points['peak_time']`)
- Division by `(1+z)` converts observer-frame days to rest-frame days
- Result: **signed days from peak in rest frame** (negative = pre-peak, positive = post-peak)

### Band mapping (catalog → LSST)

**Lines 200–219:**
For each OpSim observation at LSST filter `filt`, the code searches `available_bands`
(catalog bands present in the template) for the best match:
```python
for cat_b in available_bands:
    lsst_b = map_catalog_to_lsst_band(cat_b)
    if lsst_b == filt:
        catalog_band = cat_b
        break
```
`map_catalog_to_lsst_band` is defined in `model.py` lines 37–59:
```
B,b → g    V,v → r    R,r → i    g → g    i → i    z → z    u → u    y → y
```
If no exact match, falls back to any valid mapping (lines 211–215).

### Absolute magnitude from GP template

**Line 222:**
```python
M_abs = self.lc_model.interp(t, catalog_band, tpl_idx)
```
`LC.interp()` defined at `model.py` lines 279–308. Does linear `np.interp` on
`self.data[tpl_idx][catalog_band]["ph"]` and `["mag"]`. Phase `t = time_rel`.
Returns `np.nan` if `t` is outside the template's phase support.

### Distance modulus and extinction

**Lines 175, 229–241:**
```python
dm     = float(slice_point.get('distance_modulus', 0.0))    # line 175
color_offset = get_color_offset(catalog_band, filt) if lsst_mapped != filt else 0.0  # line 229
m_app  = M_abs + dm + color_offset                          # line 232
A_filt = slice_point.get(f'A_{filt}', 0.0)                 # line 235
if A_filt == 0.0:                                           # lines 237–239
    dust_model = DustValues()
    A_filt = dust_model.ax1[filt] * ebv
m_app += A_filt                                             # line 241
```
`distance_modulus` and `A_{filt}` were pre-computed and stored in `slice_points`
during `generate_SLSN_PopSlicer` (population.py lines 973, 996–997).

### SNR computation

**Lines 133–138, 246:**
```python
def _m52snr(mag, m5):
    snr = np.full_like(mag, np.nan, dtype=float)
    finite = np.isfinite(mag) & np.isfinite(m5)
    snr[finite] = 5.0 * (10.0 ** (0.4 * (m5[finite] - mag[finite])))
    return snr
...
snr = _m52snr(mags, m5)
```
`m5 = dataSlice[self.m5Col]` = OpSim `fiveSigmaDepth` column (apparent AB mag of a 5σ source).

### Return value

**Lines 249–257:**
```python
obs_record = {
    'mjd_obs':        mjds,
    'mag_obs':        mags,
    'snr_obs':        snr,
    'filter':         filts,
    'available_bands': ','.join(available_bands)
}
return snr, filts, mjds, obs_record
```
Each calling metric's `run()` passes `snr, filters, times, obs_record` to
`detect_slsn()` and uses the obs_record to evaluate further criteria.

---

## Section 3 — LC Attribute Comparison: GP vs Physical Templates

| Attribute | GP templates | Physical templates | Where set (GP) | Where set (Physical) |
|---|---|---|---|---|
| `.data` | `list[{band: {"ph": arr, "mag": arr}}]` — per-band absolute mag curves, one dict per event | `list[{}]` — list of **empty dicts** | `model.py:375` (built in `from_catalog`) | `mosfit_interface.py:432` `lightcurves = [{} for _ in names]` |
| `.t_grid` | Phase array from first template (days from peak, pre- and post-peak) | `PHASE_GRID.tolist()` = [1, 2, …, 400] (post-peak only ⚠) | `model.py:563` | `mosfit_interface.py:438` |
| `.names` | Event name strings from catalog | Event name strings from `all_parameters.txt` | `model.py:566` | `mosfit_interface.py:435` |
| `.sed_grid` | `list[dict]` with `'phase'`, `'lam_rest_A'`, `'Fnu_abs'`, `'coverage'`; phase = days from peak; Fnu\_abs in Jy at 10 pc; coverage from GP spectral support | Same format, but `'phase'` = PHASE_GRID (1–400), physical SEDs from magnetar model; coverage = `Fnu_2d > 0` | `model.py:559` | `mosfit_interface.py:336–341` |
| `._interps` | Built after `build_magnitude_grid()` / `load_magnitude_grid()`. `dict[filt → list[RegularGridInterpolator]]` | Same, built after `build_magnitude_grid()` | `model.py:805–824` | Same method |
| `.template_file` | String path to pkl | `None` (not set by `build_physical_templates`) | `model.py:270` | Not set |
| `.median_cenwave_by_band` | `dict[name → dict[band → Cenwave_A]]` | Not set (no attribute) | `model.py:568` | Not set |
| `.mag_grid` | `dict[filt → ndarray (n_tpl, n_z, n_phase)]` — only after grid build | Same — only after grid build | `model.py:712–713` | Same |
| `.mag_grid_axes` | `{'z': arr, 'phase': arr}` | Same | `model.py:772` | Same |

### What `evaluate_slsn()` expects from each attribute

- `self.lc_model.data[tpl_idx]` → dict with band keys, each containing `'ph'` and `'mag'` arrays. ⚠ Physical has empty dict — triggers early return.
- `self.lc_model.interp(t, catalog_band, tpl_idx)` → accesses `data[tpl_idx][catalog_band]`. ⚠ Physical: catalog_band not in empty dict → `KeyError` or NaN.
- `self.lc_model.sed_grid` → needed by the physical path to call `synthesize_mag_at_z`. ✓ Present in both.

### What happens if physical templates are passed to current `evaluate_slsn()` (unmodified)

1. Line 179: `template = self.lc_model.data[tpl_idx]` → `{}` (empty dict)
2. Lines 182–184: `available_bands = []` (no keys pass the filter)
3. Lines 185–188: immediate early return:
   ```python
   if not available_bands:
       return np.array([]), np.array([]), np.array([]), None
   ```
4. Each metric's `run()` sees `obs_record is None` → returns `self.badval` or `0.0`
5. **Result: every injected event returns 0 — detection rate = 0% for all metrics.** ⚠

---

## Section 4 — `synthesize_mag_at_z()` in Detail

**Location:** `py_files/slsn_metrics/model.py` lines 106–161

### Exact function signature
```python
def synthesize_mag_at_z(sed_grid: dict, phase_rest: float, z: float, filt: str) -> float
```

### What `sed_entry` must contain

| Key | Shape | Units |
|---|---|---|
| `'phase'` | 1D array, N\_phase | Rest-frame days (see phase convention note below) |
| `'lam_rest_A'` | 1D array, N\_lambda | Ångströms, rest frame |
| `'Fnu_abs'` | 2D array [N\_phase, N\_lambda] | Jansky at 10 pc (absolute flux) |

The `'coverage'` key is present in both GP and physical sed\_grids but is not
read by `synthesize_mag_at_z`. It is only used in `_compute_grid_slice`
(implicitly, since out-of-coverage entries have `Fnu_abs ≈ 0`).

### `phase_rest` convention

`phase_rest` must be in the **same frame as `sed_grid['phase']`**.

For GP templates: `sed_grid['phase']` = rest-frame days from peak (built at
`model.py:485`: `phase = (t_eval_obs - t0_used) / (1.0 + z)`). Phase 0 = peak.
Negative phases are pre-peak.

For physical templates: `sed_grid['phase']` = `PHASE_GRID` = [1, 2, …, 400].
The docstring in `mosfit_interface.py` line 76 states "relative to peak," meaning
phase 1 = 1 day post-peak, phase 400 = 400 days post-peak. Pre-peak phases are
absent. **Phase convention is consistent with `evaluate_slsn()`'s `time_rel`** — both
measure days from peak — but physical templates have no pre-peak support. ⚠

⚠ **Critical ambiguity:** The model function `_call_slsnni_safe` receives
`phases=PHASE_GRID` and `rest_t_explosion=texplosion` (typically −20 to −80
days). If PHASE\_GRID truly starts at +1 day post-peak, then `texplosion ≈ −50`
would place the explosion 50 days before phase=0 (peak), and all PHASE\_GRID
values (1–400) represent times after peak. Pre-explosion T=0 phases would be at
phase < 0, which are absent from the grid. This is self-consistent.
However, it means all pre-peak observations (time\_rel < 0) will return
`np.nan` from `_interp_Fnu_abs_at_phase` because `phase_rest < ph.min()=1`.

### `z` usage

Lines 143–146:
```python
lam_obs_A  = lam_rest_A * (1.0 + z)
DL_pc      = cosmo.luminosity_distance(float(z)).to_value(u.pc)
scale      = (DL_pc / 10.0)**2 * (1.0 + z)
Fnu_obs_Jy = Fnu_abs_10pc / scale
```
`z` shifts rest-frame wavelengths to observer frame and scales flux from
10 pc to luminosity distance. The function already applies the full distance
modulus. **Do not add DM again when using `synthesize_mag_at_z` output.**

### `filt` parameter

LSST filter name string: one of `'u'`, `'g'`, `'r'`, `'i'`, `'z'`, `'y'`.
The function loads LSST bandpasses from `get_lsst_bands()` (constants.py line 136).
Returns `np.nan` if `filt not in bands`.

### Return value

`float` — **apparent AB magnitude** in the requested LSST filter at redshift `z`.
Returns `np.nan` on any failure (out of phase range, out of wavelength coverage,
fewer than 20 overlap points, non-finite `calcMag` result).

### The 20-point overlap check

**Lines 150–153:**
```python
overlap_mask = (lam_obs_nm >= bp.wavelen.min()) & (lam_obs_nm <= bp.wavelen.max())
if np.sum(overlap_mask) < 20:
    return np.nan
```
This is the main failure mode for high-z events (z > ~3) in blue bands where
the observed wavelength range no longer overlaps the LSST bandpass.

### How it differs from what `evaluate_slsn()` currently calls

`evaluate_slsn()` currently calls **`self.lc_model.interp(t, catalog_band, tpl_idx)`**
(line 222), which does 1D linear interpolation on per-band absolute mag curves
(days from peak → absolute mag). It does NOT call `synthesize_mag_at_z`.

`synthesize_mag_at_z` is only called in:
- `synthesize_mag_at_z_cached()` (metrics.py line 129) — which is defined but never
  called by any current code path (it exists as infrastructure but is unused)
- `_compute_grid_slice()` (model.py line 202) — during `build_magnitude_grid()`

For the physical path, `evaluate_slsn()` must call `synthesize_mag_at_z` directly,
replacing the `lc_model.interp` call.

---

## Section 5 — Analysis Notebook Data Contract

**File:** `notebooks/analysis_multimodel_comparison.ipynb`

### Summary CSV columns

Cell 5 of the notebook reads `summary_*.csv` and expects exactly:
```
cadence, metric, n_events, n_success, efficiency
```
Computed columns added in-notebook:
```
model     (added by notebook from file lookup)
per_year  = n_success / SURVEY_YEARS
cadence_short, model_label  (display labels)
```
Column `metric` contains class names: `'SLSN_Detect_Metric'`, `'SLSN_CharacterizeMetric'`, etc. ✓ Unchanged for physical runs.

### `.npy` file shape and structure

Cell 3: `np.load(npy_path).astype(int)` — expects 1D array of length `N_events`,
values 0 or 1 (coerced to int). The notebook uses `vals[mask_z].sum()` to count
detections in redshift bins. ✓ Identical requirement for physical runs.

### Population pickle keys used by notebook

Cell 3:
```python
z_vals   = np.asarray(pop.slice_points['z'])
ra_rad   = np.asarray(pop.slice_points['ra'])
dec_rad  = np.asarray(pop.slice_points['dec'])
peak_times = np.asarray(pop.slice_points['peak_time'])
```
Cell 4 also uses `distance`, `ebv`, `gall`, `galb`.

These keys are all set in `generate_SLSN_PopSlicer` regardless of template type.
The physical population pickle would need to use `population_{model}_physical.pkl`
naming, and the notebook `SHARED_DIR` and file lookup would need updating.

### `plot_mc_rate_uncertainty_panel()` inputs

Reads from `pop_data` dict (built in Cell 3):
```python
pop_data[model] = {
    'z':           z_vals,            # ndarray shape (N,)
    'ra':          ra_rad,            # ndarray shape (N,)
    'dec':         dec_rad,           # ndarray shape (N,)
    'peak_time':   peak_times,        # ndarray shape (N,), days 1–3652
    'detect':      {cadence: ndarray or None},
    'characterize': ...,
    'spectrigger': ...,
    'villar':      ...,
    'elasticc':    ...,
}
```
No GP-specific template attributes are accessed — it only uses `.npy` arrays
and `peak_time` from the population pickle.

### Would the notebook work unchanged with physical results?

**Yes, IF:**
1. Physical `.npy` files follow the same naming pattern (`metric_values_{short}_{model}_{cadence}_*.npy`).
2. Physical summary CSVs have the same columns (`cadence, metric, n_events, n_success, efficiency`).
3. The physical population pickle has the same `slice_points` keys.
4. Cell 1 `MODELS` and `SHARED_DIR` are updated to point to physical model names and pkl paths.

The only notebook changes needed are in Cell 1 (config block): update `MODELS`,
`SUMMARY_FILES` glob patterns, and population pkl names. All analysis logic is
format-agnostic.

---

## Section 6 — The Minimum Change Required for Gap #3

### What breaks with unmodified `evaluate_slsn()`

When physical templates are passed:
1. `template = self.lc_model.data[tpl_idx]` → `{}` (empty dict)
2. `available_bands = []` → early return with empty arrays
3. All metrics return 0.0 for every event
4. Detection rate: 0%

### Minimum `evaluate_slsn()` change

The physical path must be inserted after the `available_bands` check. The GP path
is left entirely unchanged. The branch condition is: **`available_bands` is empty
and `self.lc_model.sed_grid` is populated.**

**Code sketch — not full implementation, branch logic only:**

```python
def evaluate_slsn(self, dataSlice, slice_point, return_full_obs=True):
    tpl_idx  = int(slice_point['file_indx'])
    z        = float(slice_point['z'])
    peak_time = float(slice_point['peak_time'])
    dm       = float(slice_point.get('distance_modulus', 0.0))
    ebv      = float(slice_point['ebv'])

    template       = self.lc_model.data[tpl_idx]
    available_bands = [b for b in template.keys()
                       if isinstance(template[b], dict) and 'ph' in template[b]]

    mjds  = dataSlice[self.mjdCol]
    filts = dataSlice[self.filterCol]
    m5    = dataSlice[self.m5Col]
    time_rel = (mjds - self.mjd0 - peak_time) / (1.0 + z)
    mags  = np.full_like(mjds, np.nan, dtype=float)

    # ── PHYSICAL path (new) ──────────────────────────────────────────────────
    is_physical = (len(available_bands) == 0
                   and hasattr(self.lc_model, 'sed_grid')
                   and self.lc_model.sed_grid)

    if is_physical:
        sed_entry = self.lc_model.sed_grid[tpl_idx]
        for i, (t, filt) in enumerate(zip(time_rel, filts)):
            m_app = synthesize_mag_at_z(sed_entry, t, z, filt)
            # synthesize_mag_at_z already includes DL — no separate DM
            if not np.isfinite(m_app):
                continue
            A_filt = slice_point.get(f'A_{filt}', 0.0)
            if A_filt == 0.0:
                A_filt = DustValues().ax1[filt] * ebv
            mags[i] = m_app + A_filt

    # ── GP path (existing, completely unchanged) ──────────────────────────────
    else:
        if not available_bands:
            if return_full_obs:
                return np.array([]), np.array([]), np.array([]), None
            return np.array([]), np.array([]), np.array([])

        for i, (t, filt) in enumerate(zip(time_rel, filts)):
            # ... existing GP interpolation code unchanged ...

    snr = _m52snr(mags, m5)
    # ... rest of function unchanged ...
```

### File change count

| File | Lines touched | Nature |
|---|---|---|
| `py_files/slsn_metrics/metrics.py` | ~15–20 | New `if is_physical:` block in `evaluate_slsn()` |
| Any other file | 0 | None required |

The GP path is not restructured — it is only reached by the `else` branch.
`runners.py`, `population.py`, `model.py`, `mosfit_interface.py` require zero changes.

### What the physical path needs that the GP path does not

| Need | GP path | Physical path | Notes |
|---|---|---|---|
| Band lookup | catalog band name from `.data` dict | Not needed — `synthesize_mag_at_z` takes LSST filter directly | ✓ simpler |
| DM application | explicit `M_abs + dm` | `synthesize_mag_at_z` applies DL internally — **do not add DM** | ⚠ key difference |
| Color offset | `get_color_offset(catalog_band, filt)` | Not needed — SED integration gives true LSST AB mag | ✓ simpler |
| Extinction | `slice_point['A_{filt}']` or `ax1[filt] * ebv` | Same — applied after `synthesize_mag_at_z` | ✓ identical |
| Phase convention | `time_rel` = days from peak; GP templates have negative phases | `time_rel` = days from peak; physical phase\_grid = 1–400 post-peak → pre-peak obs return nan | ⚠ see below |

### Phase convention warning ⚠

`synthesize_mag_at_z` returns `np.nan` for `phase_rest < 1.0` (the physical
template's minimum phase). Any OpSim observation occurring before the SLSN peak
will yield no magnitude, contributing no SNR. This means:
- The **rising light curve criterion** in `detect_slsn()` (criterion 2, lines 55–78)
  requires Δmag < −0.1 (brightening) over 0.5–30 days. If pre-peak mags are NaN,
  no brightening can be detected.
- **Detection efficiency will be lower** for the physical path than the GP path,
  independent of the physical SED quality, because the rising criterion cannot be
  satisfied without pre-peak coverage.
- Characterize, ELAsTiCC, and Villar metrics are less affected (they work with
  post-peak coverage), but `detect_slsn()` is a prerequisite for
  `SLSN_CharacterizeMetric` and `SLSN_SpecTriggerMetric`.

**Resolution options (not implemented here, flagged for decision):**
1. Accept reduced detection efficiency as physically correct (physical model
   only constrains post-peak).
2. Extend PHASE\_GRID to include negative phases (pre-peak) by running the
   physical model at negative phases relative to peak, which requires knowing
   each event's `|texplosion|` to set the pre-explosion T=0 boundary correctly.
3. Relax the rising criterion when using physical templates (a separate detect
   function variant for physical).

---

## Section 7 — Diagnostic and Figure Compatibility

| Function (diagnostics.py) | Line | Reads from | GP-specific attribute? | Works unchanged with physical? | Minimum change if not |
|---|---|---|---|---|---|
| `plot_mc_rate_uncertainty_panel()` | 2465 | `pop_data[model][metric_key][cadence]` (`.npy` array) and `pop_data[model]['peak_time']` | None — pure array arithmetic | ✓ Yes, if `.npy` and `peak_time` arrays have same shape and dtype | None |
| `plot_mc_rate_uncertainty()` | 2307 | `detect_vals` (`.npy` array), `peak_times` (from population pickle) | None | ✓ Yes | None |
| `plot_healpix_efficiency()` | 1019 | `bundle_metric_values`, `ra_rad`, `dec_rad` (all from population/MAF) | None | ✓ Yes | None |
| `plot_population_rate_vs_redshift()` | 1870 | `population_slicer.slice_points['z']`, `distance`, `peak_time` | None | ✓ Yes, same slice_points keys | None |
| `compare_simulated_vs_observed_rates()` | 1812 | `population_slicer.slice_points` — `z`, `distance` | None | ✓ Yes | None |
| `plot_population_diagnostics()` | 922 | `ra_rad`, `dec_rad`, `peak_times`, `distances_mpc`, `z_vals`, `ebv`, `gall`, `galb` | None — all from slice_points | ✓ Yes | None |
| `plot_detection_diagnostics()` | 1981 | `df_obs` DataFrame (from `_build_detection_dataframe`) and population | Accesses `templates.data[idx][f]` via `_cadence_peak_for_sid` in `run_slsn_detect` (runners.py line 283) | ⚠ **No** — `_cadence_peak_for_sid` calls `templates.interp(tt, f, idx)` which fails on empty `.data` | Add physical branch in `_cadence_peak_for_sid` or skip peak columns for physical runs |
| `plot_sky_detection()` | 2164 | `detect_vals` (`.npy`), sky coordinates | None | ✓ Yes | None |
| `diagnose_abs_from_templates()` | 115 | `templates_file` pickle, per-event CSV | Reads `lightcurves[idx][band]` directly from pickle | ⚠ **No** — physical lightcurves are empty dicts, no bands to compare residuals against | Not applicable to physical path — GP-only diagnostic |
| `list_template_bands()` | 61 | `templates_file` pickle | Reads `lightcurves[template_idx]` keys | ⚠ **No** — empty dict, returns empty list | Physical path: query `sed_grid` wavelength keys instead (different diagnostic) |
| `characterize_template_coverage()` | Called by `validate_template_quality` | `templates_file` pickle | Reads per-band `ph`, `mag` arrays | ⚠ **No** — empty dicts have no per-band data | Not applicable to physical path |

### Summary

**Seven of ten diagnostic functions work unchanged** with physical results because
they consume `.npy` arrays and population pickle `slice_points`, which are
format-agnostic.

**Three functions are GP-specific:**
1. `plot_detection_diagnostics()` (indirectly, via `_cadence_peak_for_sid` in `run_slsn_detect`)
   — needs a one-liner guard to skip or use `synthesize_mag_at_z` for peak column computation.
2. `diagnose_abs_from_templates()` — GP-only residual diagnostic. Not needed for physical runs.
3. `list_template_bands()` / `characterize_template_coverage()` — GP-only template inspection.
   Not needed for production metric runs.

The production analysis notebook (`analysis_multimodel_comparison.ipynb`) calls
only `plot_mc_rate_uncertainty_panel`, `plot_healpix_efficiency`,
`plot_population_diagnostics`, `compare_simulated_vs_observed_rates`, and
`plot_population_rate_vs_redshift` — **all five are format-agnostic and work
unchanged.** ✓

---

*End of audit. All source reading done on branch `unified-template-interface`, 2026-05-08.*
