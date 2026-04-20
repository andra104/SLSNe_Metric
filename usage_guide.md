# SLSN Metrics — Usage Guide

## Overview

This pipeline predicts SLSN detection rates for the Rubin LSST survey using
MAF (Metrics Analysis Framework) and OpSim cadence databases. It evaluates
three physically distinct rate models across multiple survey cadences.

## Module Structure

```
slsn_metrics/
├── constants.py                   ← Physical constants, cosmology
├── export_slsne_photometry.py     ← Data cleaning & cenwave
├── gp_build.py                    ← GP fitting & t0 selection
├── model.py                       ← LC class & templates
├── population.py                  ← Population generation, rate models
├── metrics.py                     ← MAF metrics
├── runners.py                     ← Execution wrappers
├── diagnostics.py                 ← QA & plotting
└── paths.py                       ← Machine-agnostic paths
```

---

## Example 1: Build Templates from Catalog

Run once. Templates do not change unless you add new events to the catalog.

```python
from pathlib import Path
import pandas as pd
from slsn_metrics import (
    process_all_events, per_filter_cenwave,
    attach_cenwave_to_perevent_csv, LC, CatalogInputs
)

# Step 1: Export and clean photometry
supernovae_dir = Path("data/SLSNe_raw/")
out_per = Path("output/per_event_files/")
out_all = Path("output/all_events/")

index_df, _, _ = process_all_events(
    supernovae_dir, out_per, out_all,
    write_parquet=True, write_csv=True
)

# NOTE: The _cenwave.csv files in output/per_event_files/ were generated
# once using this loop and committed to git as data artifacts (branch: main).
# If running on a fresh clone or adding new events, re-run Steps 1-2 before
# building templates. The cenwave lookup uses two reference tables at
# output/all_events/filter_reference.csv and generic_reference.csv, originally
# built in notebooks/all_run.ipynb (main branch only, cell 25) using
# build_user_filter_map(). Both reference files must exist before this loop
# will produce valid output.

# Step 2: Attach cenwave (per-event)
for name in index_df[index_df['status'] == 'ok']['event']:
    cen_map = per_filter_cenwave(supernovae_dir, name, verbose=False)
    attach_cenwave_to_perevent_csv(out_per / f"{name}.csv", cen_map)

# Step 3: Build GP templates
params = pd.read_csv("output/all_events/allparameter.csv")
inputs = CatalogInputs(
    photometry_dir=out_per,
    params_table=params,
    name_col="name",
    z_col="redshift_med",
    peak_mjd_col="Peak_MJD_med"
)

templates = LC.from_catalog(
    inputs,
    filename_pattern="{name}_cenwave.csv",
    save_to=Path("output/SLSNe/shared/templates.pkl")
)
```

---

## Example 2: Pre-compute Magnitude Grid

Run once after building templates. Provides 100-500x speedup over on-the-fly synthesis.

```python
from slsn_metrics import LC

templates = LC(load_from="output/SLSNe/shared/templates.pkl")
templates.build_magnitude_grid(
    save_to="output/SLSNe/shared/mag_grid.pkl"
)
```

---

## Example 3: Generate Population (Tabulated Rate Model)

One population per rate model. Reused across all cadences for that model.

```python
from slsn_metrics import LC, generate_SLSN_PopSlicer
from slsn_metrics.paths import get_rate_csv_path, get_shared_output_dir

templates = LC(load_from="output/SLSNe/shared/templates.pkl")
shared_dir = get_shared_output_dir()

# Generate Fe-dependent population (primary science result)
population = generate_SLSN_PopSlicer(
    lc_model=templates,
    rate_model='tabulated',
    tabulated_csv=get_rate_csv_path(),   # output/SLSNe/shared/fiducial_models.csv
    model_name='fe_dependent',
    z_min=0.1,
    z_max=2.0,
    gal_lat_cut=15.0,
    seed=42,
    save_to=shared_dir / 'population_fe_dependent.pkl'
)

# Or load existing population
population = generate_SLSN_PopSlicer(
    lc_model=templates,
    load_from=str(shared_dir / 'population_fe_dependent.pkl')
)
```

### Rate model options

| `model_name` | Physics | CSV column |
|---|---|---|
| `'fe_dependent'` | Iron abundance threshold — **primary result** | f_Fe_mixed (col 2) |
| `'naive'` | SFR only, no metallicity | computed from Ψ(z) |
| `'o_dependent'` | Oxygen abundance threshold | f_OH (col 1) |

---

## Example 4: Run All Three Metrics (Production)

All three metrics run in one MAF pass. Detection hierarchy is enforced
internally — characterization and spec trigger re-run the detection check
before applying their own criteria.

```python
from slsn_metrics.runners import run_slsn_multi_metrics

summary = run_slsn_multi_metrics(
    templates=templates,
    population=population,
    cadences=['baseline_v5.1.1_10yrs'],
    model_name='fe_dependent',
    z_min=0.1,
    z_max=2.0,
    store_obs_mode='none',    # production: summary + .npy files only
    save_summary=True,
    verbose=True
)

print(summary)
# cadence               metric  n_events  n_success  efficiency
# baseline_v5.1.1_10yrs  SLSN_Detect      2243377    145000    0.065
# baseline_v5.1.1_10yrs  SLSN_Characterize ...
# baseline_v5.1.1_10yrs  SLSN_SpecTrigger  ...
```

---

## Example 5: Diagnostic Run (50k events, visit detail)

For notebook plots. Uses `store_obs_mode='full'` to store visit arrays.
Only practical at reduced population size.

```python
import numpy as np
from rubin_sim.maf.slicers import UserPointsSlicer

# Subsample 50k events from full population
sp = population.slice_points
n_full = len(sp['distance'])
rng = np.random.default_rng(42)
keep = np.sort(rng.choice(n_full, size=50000, replace=False))

ra_sub = np.degrees(sp['ra'][keep])
dec_sub = np.degrees(sp['dec'][keep])
sub_pop = UserPointsSlicer(ra=ra_sub, dec=dec_sub, badval=0)
for key in sp.keys():
    arr = np.asarray(sp[key])
    if arr.shape and arr.shape[0] == n_full:
        sub_pop.slice_points[key] = arr[keep]
    else:
        sub_pop.slice_points[key] = sp[key]

# Run with full observation storage
from slsn_metrics.metrics import (
    SLSN_Detect_Metric, SLSN_CharacterizeMetric, SLSN_SpecTriggerMetric
)

metrics_list = [
    SLSN_Detect_Metric(lc_model=templates, mjd0=60980.5, store_obs_mode='full'),
    SLSN_CharacterizeMetric(lc_model=templates, mjd0=60980.5, store_obs_mode='full'),
    SLSN_SpecTriggerMetric(lc_model=templates, mjd0=60980.5, store_obs_mode='full'),
]

summary = run_slsn_multi_metrics(
    templates=templates,
    population=sub_pop,
    cadences=['baseline_v5.1.1_10yrs'],
    metrics_list=metrics_list,
    store_obs_mode='full'
)

# Extract obs_records for diagnostic plots
import pandas as pd
df_detect = pd.DataFrame.from_dict(
    metrics_list[0].obs_records, orient='index'
).reset_index(drop=True)

df_char = pd.DataFrame.from_dict(
    metrics_list[1].obs_records, orient='index'
).reset_index(drop=True)

df_spec = pd.DataFrame.from_dict(
    metrics_list[2].obs_records, orient='index'
).reset_index(drop=True)
```

---

## Example 6: Load Results and Compute Adam's Fixed-z Bin

```python
import numpy as np
import pickle

# Load population
with open('output/SLSNe/shared/population_fe_dependent.pkl', 'rb') as f:
    pop_data = pickle.load(f)
z_vals = pop_data['slice_points']['z']

# Load metric values
detect = np.load(
    'output/SLSNe/fe_dependent/'
    'metric_values_Detect_fe_dependent_baseline_v5.1.1_10yrs_z0.1-2.0_260402.npy'
)

# Adam's fixed-z bin comparison (z=1.2-1.8 and z=1.3-1.7)
for z_lo, z_hi in [(1.2, 1.8), (1.3, 1.7)]:
    mask = (z_vals >= z_lo) & (z_vals <= z_hi)
    n_det = detect[mask].sum()
    eff = detect[mask].mean()
    print(f"z={z_lo}-{z_hi}: {n_det:.0f} detected, {100*eff:.1f}% efficiency")
```

---

## Example 7: Run at Command Line (MSI)

### Single job
```bash
conda activate rubin_sim_2.6.1

# Dry run first — verify all paths exist
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --dry-run

# Real run
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --store-obs-mode none \
    --n-cores 4
```

### SLURM batch (all models × all cadences)
```bash
# Submit all 9 jobs — fe/o first, naive last
bash submit_slsn_batch.slurm

# Monitor
squeue -u andra104
tail -f output/SLSNe/logs/fe_dependent_baseline_v5.1.1_10yrs_JOBID.out
```

### What the log shows (with timestamps)
```
[16:42:49] Python started — beginning imports
[16:43:12] All imports complete — pipeline starting.
[16:43:12] CELL 2 — Load Templates
[16:43:45]   Loaded 165 templates  (33.1s)
[16:43:45] CELL 3 — Population [fe_dependent]
[16:44:01]   Population ready: 2,243,377 events  (16.2s)
[16:44:01] CELL 4 — MAF Metrics [baseline_v5.1.1_10yrs]
[16:44:01]   Calling MAF run_all() — silent until complete, this is the long step
[18:30:00] CELL 5 — Results  |  metrics runtime: 106.0 min
[18:30:00] DONE: fe_dependent x baseline_v5.1.1_10yrs
```

---

## Shared Files Location

All shared inputs and outputs live in `output/SLSNe/shared/`:

| File | Description | Regenerate when |
|---|---|---|
| `templates.pkl` | 165 GP-fitted SLSN templates | New catalog events |
| `mag_grid.pkl` | Pre-computed apparent magnitude grid | Templates change |
| `population_fe_dependent.pkl` | 2.24M simulated SLSNe | Model or z range changes |
| `population_naive.pkl` | ~6M simulated SLSNe | Model or z range changes |
| `population_o_dependent.pkl` | ~3M simulated SLSNe | Model or z range changes |
| `fiducial_models.csv` | Ben's tabulated R(z) | New CSV from collaborator |

---

## Performance Notes

- **Import time on SLURM cold node**: ~1-2 minutes (rubin_sim is large)
- **Population generation**: ~10-30 minutes (depends on model size)
- **MAF metrics (2.24M events, 1 cadence)**: ~1.5-3 hours (single MAF pass)
- **MAF metrics (old 3-pass method)**: ~5-9 hours — fixed by bundling all metrics

## Troubleshooting

| Problem | Solution |
|---|---|
| `KeyError: 'Cenwave'` | Run `attach_cenwave_to_perevent_csv()` first |
| Import errors | `conda activate rubin_sim_2.6.1` |
| Cadence DB missing | Download from LSST S3 to `cadences/` |
| Job silent for 25+ min | Normal — MAF has no internal progress output |
| Population too large | `--max-events 50000` for diagnostic runs |
| Results missing after run | Check `output/SLSNe/{model}/` for stamped files |
| Wrong population loaded | Check `model_name` in population `slice_points` |
