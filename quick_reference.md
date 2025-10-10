# SLSN Metrics — Quick Reference Card

## Complete File Structure

```
slsn_metrics/
│
├── __init__.py                     # Package initialization & clean imports
│
├── constants.py                    # Constants & utilities
│   ├── Cosmology lookups (dm_from_z, z_from_comoving_fast)
│   ├── Physical constants (C_MS, F0_JY, etc.)
│   ├── Unit conversions (mag↔flux)
│   ├── Bandpass cache (get_lsst_bands)
│   └── Phase binning (phase_bucket_vec, PHASE_BIN_STEP)
│
├── export_slsne_photometry.py     # Data cleaning & export
│   ├── canonical_filter()         # Normalize filter names
│   ├── process_all_events()       # Batch export
│   ├── per_filter_cenwave()       # Calculate central wavelengths
│   ├── attach_cenwave_to_perevent_csv()  # Add Cenwave column
│   ├── audit_cenwave_mode()       # QA: check consistency
│   └── audit_cenwave_variability() # QA: check time variation
│
├── gp_build.py                     # GP fitting
│   ├── default_gp_kernel()        # Matérn-3/2 kernel
│   ├── fit_2d_gp()                # 2D (time, freq) GP fit
│   ├── gp_predict_surface()       # Predict at multiple bands
│   └── pick_t0_hybrid()           # Data-driven t0 selection
│
├── model.py                        # 🏗️ Template building
│   ├── LC class                   # Main template container
│   │   ├── from_catalog()        # Build from photometry
│   │   ├── build_magnitude_grid() # Pre-compute mag grid
│   │   ├── load_magnitude_grid() # Load pre-computed grid
│   │   └── interp()              # Interpolate abs mag
│   ├── CatalogInputs              # Configuration dataclass
│   ├── synthesize_mag_at_z()     # SED synthesis (single call)
│   ├── template_index_for_event() # Name → index
│   ├── get_median_cenwave_map()  # Cached cenwave per event
│   └── list_t0_meta()            # Extract t0 metadata
│
├── population.py                   # Population generation
│   └── generate_SLSN_PopSlicer()  # Volumetric sampling
│       ├── Cached HEALPix sky    # Reusable coordinate draws
│       ├── z from comoving volume
│       ├── Random template assignment
│       ├── EBV + per-filter A_f   # Persisted extinction
│       └── Peak mag summaries     # For cuts/analysis
│
├── metrics.py                      # MAF metrics
│   ├── evaluate_slsn()            # Core evaluator (fast)
│   ├── SLSN_Base_Metric           # Base class
│   ├── SLSN_Detect_Metric         # Detection (Firth+2015)
│   ├── SLSN_CharacterizeMetric    # Characterization (Inserra+2024)
│   ├── SLSN_SpecTriggerMetric     # Spec trigger (brightness+color)
│   └── detect_slsn()              # Detection logic
│
└── diagnostics.py                  # QA & plotting
    ├── diagnose_abs_from_templates()  # Obs vs. (M + DM)
    ├── diagnose_internal_consistency() # GP refit check
    ├── plot_residuals()               # Residual plots
    ├── plot_event_obs()               # Obs-only plot
    ├── plot_event_model()             # Model-only plot
    ├── plot_event_obs_vs_model()      # Overlay plot
    ├── characterize_template_coverage() # Coverage summary
    ├── compute_slsn_properties()      # Physical properties
    ├── assess_literature_coverage()   # Compare to lit
    └── find_missing_archetypes()      # Gap analysis
```

## ⚡ One-Liners for Common Tasks

### Import everything you need
```python
from slsn_metrics import (
    LC, CatalogInputs,                    # Model building
    generate_SLSN_PopSlicer,              # Population
    SLSN_Detect_Metric,                   # Metrics
    diagnose_abs_from_templates,          # Diagnostics
    per_filter_cenwave,                   # Cenwave
)
```

### Build templates
```python
templates = LC.from_catalog(
    CatalogInputs("data/per_event/", params_df),
    save_to="templates.pkl"
)
```

### Pre-compute magnitude grid (once, 5-10 min)
```python
templates.build_magnitude_grid(save_to="mag_grid.pkl")
# Then always: templates.load_magnitude_grid("mag_grid.pkl")
```

### Generate population
```python
pop = generate_SLSN_PopSlicer(templates, save_to="pop.pkl")
# Or reload: pop = generate_SLSN_PopSlicer(templates, load_from="pop.pkl")
```

### Run detection metric
```python
metric = SLSN_Detect_Metric(lc_model=templates)
bundle = maf.MetricBundle(metric, pop, constraint="")
# ... run MAF
```

### Quick diagnostic
```python
resid = diagnose_abs_from_templates(
    "SN2015bn", per_event_dir, templates_file, 
    template_idx=0, z=0.5, t0=57500
)
```

## Key Design Principles

### Module Independence
- **constants.py**: No internal dependencies
- **export_slsne_photometry.py**: Only uses constants
- **gp_build.py**: Only uses constants
- **model.py**: Uses constants + gp_build + export (safe)
- **population.py**: Uses constants + model (no circular deps)
- **metrics.py**: Uses constants + model (evaluator only)
- **diagnostics.py**: Uses all (but lazy imports matplotlib)

### Import Order (Never Circular)
```python
constants → export_slsne_photometry
         → gp_build
         → model (uses gp_build, export)
         → population (uses model)
         → metrics (uses model, constants)
         → diagnostics (uses all, imports plt lazily)
```

### Performance Hierarchy
```
Fastest:  mag_grid interpolation (templates.build_magnitude_grid)
  ↓ 100x
Fast:     Cached SED synthesis (phase bucketing)
  ↓ 10x
Slow:     On-demand SED synthesis (synthesize_mag_at_z)
  ↓ 100x
Slowest:  GP refitting (diagnose_internal_consistency with refit_gp=True)
```

## Color Coding by Purpose

-  **constants.py**: Shared infrastructure
-  **export_slsne_photometry.py**: Data preparation
-  **gp_build.py**: Statistical modeling
-  **model.py**: Core physics engine
-  **population.py**: Survey simulation
-  **metrics.py**: Science metrics
-  **diagnostics.py**: Quality assurance

## Workflow States

```
State 1: Raw catalog files (Sebastian's git)
    ↓ [process_all_events]
State 2: Clean per-event CSV/Parquet
    ↓ [per_filter_cenwave + attach_cenwave]
State 3: Per-event CSV with Cenwave column
    ↓ [LC.from_catalog]
State 4: Template pickle (lightcurves + sed_grid)
    ↓ [build_magnitude_grid] ← OPTIONAL BUT RECOMMENDED
State 5: Mag grid pickle (fast evaluator)
    ↓ [generate_SLSN_PopSlicer]
State 6: Population pickle (slice_points)
    ↓ [MAF metrics]
State 7: Detection results (metric_values)
    ↓ [diagnostics]
State 8: QA plots & analysis
```

## Common Gotchas & Solutions

| Problem | Solution |
|---------|----------|
| `KeyError: 'Cenwave'` | Run `attach_cenwave_to_perevent_csv()` first |
| Slow evaluator | Build and load mag_grid |
| Import errors | Check `sys.path`, use relative imports |
| Memory issues | Set `store_obs_mode="none"` in metrics |
| GP fit fails | Check for <5 points or bad data |
| Template names missing | Rebuild with latest `LC.from_catalog()` |
| Cache not reused | Verify `healpix_cache_file` path exists |
| Circular import | Never import metrics/diagnostics in constants/model |

## Minimal Test Suite

```python
# Test 1: Constants
from slsn_metrics import dm_from_z
assert 44.0 < dm_from_z(1.0) < 44.5, "DM(z=1) sanity check"

# Test 2: GP fitting
from slsn_metrics import fit_2d_gp
import numpy as np
t = np.linspace(0, 100, 20)
nu = np.full_like(t, 5e14)
flux = np.random.rand(20) + 1.0
fluxerr = np.full_like(flux, 0.1)
gp_pred = fit_2d_gp(t, nu, flux, fluxerr)
assert callable(gp_pred), "GP predict should be callable"

# Test 3: Template building (use 1-2 events)
# ... subset your catalog
templates = LC.from_catalog(small_inputs)
assert len(templates.data) >= 1, "Should build at least 1 template"

# Test 4: Population generation
pop = generate_SLSN_PopSlicer(templates, t_end=365, rate_density=1e-6)
assert len(pop.slice_points['sid']) > 0, "Should generate events"

# Test 5: Metric evaluation (mock OpSim slice)
metric = SLSN_Detect_Metric(lc_model=templates)
# ... create mock dataSlice and slice_point
result = metric.run(mock_slice, mock_point)
assert result in [0.0, 1.0], "Should return 0 or 1"

print(" All tests passed")
```

## Learning Path

1. **Week 1**: Understand data flow
   - Read `export_slsne_photometry.py` 
   - Process 5 events manually
   - Inspect output CSVs

2. **Week 2**: Build small template set
   - Use `LC.from_catalog()` with 10 events
   - Run diagnostics on 2-3 templates
   - Understand GP fitting

3. **Week 3**: Scale up
   - Build full template set (50-100 events)
   - Pre-compute magnitude grid
   - Profile evaluator

4. **Week 4**: Run full sim
   - Generate population (1000s of events)
   - Run all three metrics
   - Analyze results

## Pro Tips

1. **Always version your data**: Include template hash in filenames
2. **Use joblib for templates**: Faster load than pickle
3. **Cache everything reusable**: HEALPix draws, mag grids, populations
4. **Profile before optimizing**: Use `cProfile` to find bottlenecks
5. **Keep diagnostics separate**: Don't run GP refits in production
6. **Document your workflow**: Track which params/versions you used
7. **Test incrementally**: Build 1 template → 10 → 100, checking at each step

## Help Commands

```python
# Get function help
help(LC.from_catalog)

# List available attributes
dir(templates)

# Check what's in a pickle
import pickle
with open("templates.pkl", "rb") as f:
    obj = pickle.load(f)
    print(obj.keys())

# Verify imports
import slsn_metrics
print(slsn_metrics.__version__)
print(slsn_metrics.__file__)  # Location
```

---

**Remember**: Start small, test often, cache aggressively, profile before optimizing!
