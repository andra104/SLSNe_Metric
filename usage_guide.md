# SLSN Metrics — Usage Guide

## Module Structure

```
slsn_metrics/
├── __init__.py                    # Clean imports
├── constants.py                   # Physical constants, cosmology
├── export_slsne_photometry.py     # Data cleaning & cenwave
├── gp_build.py                    # GP fitting & t0 selection
├── model.py                       # LC class & templates
├── population.py                  # Population generation
├── metrics.py                     # MAF metrics
└── diagnostics.py                 # QA & plotting
```

## Workflow Overview

```
Raw catalog data
    ↓
[export_slsne_photometry] → Clean CSV/Parquet + Cenwave
    ↓
[gp_build + model] → SLSN templates (absolute mags)
    ↓
[population] → Simulated SLSN population
    ↓
[metrics] → MAF evaluation (detect/characterize/trigger)
    ↓
[diagnostics] → QA plots & residuals
```

## Example 1: Build Templates from Catalog

```python
from pathlib import Path
import pandas as pd
from slsn_metrics import (
    process_all_events,
    per_filter_cenwave,
    attach_cenwave_to_perevent_csv,
    LC,
    CatalogInputs
)

# Step 1: Export and clean photometry
supernovae_dir = Path("data/SLSNe_raw/")
out_per = Path("data/per_event/")
out_all = Path("data/all_events/")

index_df, _, _ = process_all_events(
    supernovae_dir, out_per, out_all,
    write_parquet=True, write_csv=True
)

# Step 2: Attach cenwave (per-event)
event_root = supernovae_dir
for name in index_df[index_df['status'] == 'ok']['event']:
    cen_map = per_filter_cenwave(event_root, name, verbose=False)
    csv_path = out_per / f"{name}.csv"
    attach_cenwave_to_perevent_csv(csv_path, cen_map)

# Step 3: Build templates with GP
params = pd.read_csv("data/all_parameters.txt", sep=r"\s+")
inputs = CatalogInputs(
    photometry_dir=Path("data/per_event/"),
    params_table=params,
    name_col="name",
    z_col="redshift_med",
    peak_mjd_col="Peak_MJD_med"
)

templates = LC.from_catalog(
    inputs,
    filename_pattern="{name}_cenwave.csv",
    tpad_pre_days=5.0,
    tpad_post_days=160.0,
    n_time=220,
    save_to=Path("slsn_templates.pkl")
)

print(f"Built {len(templates.names)} templates")
```

## Example 2: Build Magnitude Grid (Fast Path)

```python
from slsn_metrics import LC
import numpy as np

# Load templates
templates = LC(load_from="slsn_templates.pkl")

# Build grid (takes ~5-10 min once, saves 100x+ in simulations)
templates.build_magnitude_grid(
    z_grid=np.linspace(0.02, 2.0, 50),
    phase_grid=np.geomspace(0.1, 160, 200),
    filters='ugrizy',
    save_to="mag_grid.pkl"
)
```

## Example 3: Generate Population

```python
from slsn_metrics import generate_SLSN_PopSlicer, LC
from pathlib import Path

templates = LC(load_from="slsn_templates.pkl")

# Option: load pre-built mag grid for speed
# templates.load_magnitude_grid("mag_grid.pkl")

pop = generate_SLSN_PopSlicer(
    lc_model=templates,
    t_start=1,
    t_end=3652,
    z_min=0.1,
    z_max=2.0,
    rate_density=1e-7,  # Mpc^-3 yr^-1
    seed=42,
    healpix_cache_file=Path("healpix_cache.pkl"),  # Reuse sky draws
    save_to=Path("slsn_population.pkl")
)

print(f"Generated {len(pop.slice_points['sid'])} SLSNe")
```

## Example 4: Run MAF Metrics

```python
import rubin_sim.maf as maf
from slsn_metrics import SLSN_Detect_Metric, LC

# Load templates
templates = LC(load_from="slsn_templates.pkl")
templates.load_magnitude_grid("mag_grid.pkl")  # Use fast path

# Load population
from slsn_metrics import generate_SLSN_PopSlicer
pop = generate_SLSN_PopSlicer(
    templates, 
    load_from="slsn_population.pkl"
)

# Define metric
metric = SLSN_Detect_Metric(
    lc_model=templates,
    mjd0=60980.5,
    store_obs_mode="meta"  # Options: "none", "meta", "diag", "full"
)

# Run with MAF
bundle = maf.MetricBundle(
    metric=metric,
    slicer=pop,
    constraint="",
    run_name="baseline_v3_4"
)

# Load OpSim and evaluate
db = maf.OpsimDatabase("baseline_v3_4_10yrs.db")
bundle_dict = {'detect': bundle}
bg = maf.MetricBundleGroup(bundle_dict, db, out_dir="output")
bg.run_all()

# Results
detection_rate = bundle.metric_values.mean()
print(f"Detection efficiency: {detection_rate:.2%}")
```

## Example 5: Diagnostic Checks

```python
from slsn_metrics.diagnostics import (
    diagnose_abs_from_templates,
    plot_residuals,
    plot_event_obs_vs_model,
    characterize_template_coverage
)
from pathlib import Path

# Get z and t0 for an event
from slsn_metrics import list_t0_meta
meta = list_t0_meta("slsn_templates.pkl")
event = meta.iloc[0]
z, t0 = event['z'], event['t0_used']

# Check residuals: obs vs. (template + DM)
resid = diagnose_abs_from_templates(
    event_name=event['name'],
    per_event_dir=Path("data/per_event/"),
    templates_file=Path("slsn_templates.pkl"),
    template_idx=0,
    z=z,
    t0=t0,
    return_with_phase=True
)

# Plot
plot_residuals(resid, vs_phase=True, title=event['name'])

# Overlay obs + model
plot_event_obs_vs_model(
    event_name=event['name'],
    per_event_dir=Path("data/per_event/"),
    templates_file=Path("slsn_templates.pkl"),
    template_idx=0,
    z=z,
    t0=t0
)

# Template coverage summary
cov = characterize_template_coverage("slsn_templates.pkl")
print(cov[['name', 'z', 'n_bands', 'phase_span']].head())
```

## Example 6: All Three Metrics

```python
from slsn_metrics import (
    SLSN_Detect_Metric,
    SLSN_CharacterizeMetric,
    SLSN_SpecTriggerMetric,
    LC
)

templates = LC(load_from="slsn_templates.pkl")
templates.load_magnitude_grid("mag_grid.pkl")

# Define all metrics
detect = SLSN_Detect_Metric(lc_model=templates, store_obs_mode="meta")
char = SLSN_CharacterizeMetric(lc_model=templates, store_obs_mode="meta")
spec = SLSN_SpecTriggerMetric(lc_model=templates, store_obs_mode="meta")

# Use with MAF bundle
bundles = {
    'detect': maf.MetricBundle(detect, slicer=pop, constraint=""),
    'characterize': maf.MetricBundle(char, slicer=pop, constraint=""),
    'spec_trigger': maf.MetricBundle(spec, slicer=pop, constraint="")
}

bg = maf.MetricBundleGroup(bundles, db, out_dir="output")
bg.run_all()

# Compare efficiencies
print(f"Detection: {bundles['detect'].metric_values.mean():.2%}")
print(f"Characterization: {bundles['characterize'].metric_values.mean():.2%}")
print(f"Spec trigger: {bundles['spec_trigger'].metric_values.mean():.2%}")
```

## Testing & Profiling

### Quick functionality test
```python
# Test constants
from slsn_metrics import dm_from_z
assert abs(dm_from_z(1.0) - 44.12) < 0.1

# Test GP fitting
from slsn_metrics import fit_2d_gp
import numpy as np
t = np.linspace(0, 100, 50)
nu = np.full_like(t, 5e14)
flux = np.exp(-0.5 * ((t - 50)/10)**2)
fluxerr = np.full_like(flux, 0.1)
gp_predict = fit_2d_gp(t, nu, flux, fluxerr)
print(" GP fitting works")

# Test template building (small sample)
# ... (use 2-3 events)
```

### Profile evaluator
```python
import cProfile
import pstats

def profile_evaluate():
    # Setup metric, pop, OpSim slice
    # ...
    for i in range(100):
        metric.run(dataSlice, slice_point)

cProfile.run('profile_evaluate()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Performance Tips

1. **Use magnitude grids**: `templates.build_magnitude_grid()` — 100x+ speedup
2. **Cache HEALPix draws**: Pass `healpix_cache_file` to avoid regenerating sky
3. **Limit storage**: Use `store_obs_mode="meta"` or `"none"` for large runs
4. **Phase windowing**: Set `metric.phase_window_rest = (5.0, 160.0)` to trim early
5. **Diagnostic sampling**: Enable only for QA, not production runs

## Troubleshooting

### Import errors
```python
# If you get import errors, check sys.path
import sys
sys.path.insert(0, '/path/to/slsn_metrics/')
```

### Missing Cenwave
```python
# If templates fail due to missing Cenwave:
from slsn_metrics import per_filter_cenwave, attach_cenwave_to_perevent_csv
# ... reprocess events with cenwave attachment
```

### Slow evaluation
```python
# Check if mag_grid is loaded
assert hasattr(templates, 'mag_grid'), "Build/load mag_grid first"
assert hasattr(templates, '_interps'), "Interpolators not built"
```

### Memory issues
```python
# For large populations, use store_obs_mode="none"
metric = SLSN_Detect_Metric(lc_model=templates, store_obs_mode="none")
```

## Best Practices

1. **Always attach cenwave** before template building
2. **Build mag grids once**, reuse across runs
3. **Cache populations** when testing cadence variations
4. **Use diagnostics** after template building to check residuals
5. **Profile before optimizing** — measure first
6. **Version control**: Include template/grid hashes in metadata

## Debugging

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check template metadata
from slsn_metrics import list_t0_meta
meta = list_t0_meta("slsn_templates.pkl")
print(meta[['name', 't0_cat', 't0_data', 'source']])

# Inspect a single evaluation
metric._mag_cache.clear()  # Clear cache
snr, filt, mjd, obs = metric.run(dataSlice, slice_point)
print(f"Obs: {len(mjd)}, Detected: {(snr >= 5).sum()}")
```

## Further Reading

- **Firth+2015**: SLSN detection criteria
- **Inserra+2024**: Characterization requirements
- **Tyler's GP paper**: 2D GP methodology for transients
- **Rubin MAF docs**: `rubin_sim.maf` usage

---

**Questions?** Check function docstrings with `help(function_name)` or inspect source code directly.
