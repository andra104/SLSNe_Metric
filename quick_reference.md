# SLSN Metrics — Quick Reference Card

## File Structure

```
SLSNe_Metric/
├── py_files/slsn_metrics/          ← Python package
│   ├── constants.py                ← Physical constants, cosmology lookups
│   ├── export_slsne_photometry.py  ← Data cleaning, cenwave attachment
│   ├── gp_build.py                 ← GP fitting, t0 selection
│   ├── gomez_models.py             ← Vendored Gomez+2024 magnetar physics (used by mosfit_interface)
│   ├── mosfit_interface.py         ← Physical SED templates from MOSFiT posteriors (drop-in for LC.from_catalog)
│   ├── model.py                    ← LC class, templates, magnitude grid
│   ├── population.py               ← Population generation, rate models
│   ├── metrics.py                  ← MAF metrics (detect/characterize/spectrigger/villar/elasticc)
│   ├── runners.py                  ← Execution wrappers
│   ├── diagnostics.py              ← QA plots
│   └── paths.py                    ← Machine-agnostic path resolution
│
├── run_slsn_pipeline.py            ← CLI: one (model × cadence) run
├── launch_all.py                   ← CLI: all (model × cadence) in parallel
├── submit_slsn_batch.slurm         ← SLURM: all jobs batch submission
├── submit_slsn_pipeline.slurm      ← SLURM: single job submission
│
├── cadences/                       ← OpSim .db files
└── output/SLSNe/
    ├── shared/                     ← templates.pkl, population_*.pkl, fiducial_models.csv
    ├── fe_dependent/               ← results: summary + metric_values .npy
    ├── naive/
    ├── o_dependent/
    └── logs/                       ← SLURM job logs
```

---

## Workflow States

```
State 1: Raw catalog (Gómez+2024, 265 events)
    ↓ [process_all_events + attach_cenwave]
State 2: Per-event CSV with Cenwave column
    ↓ [LC.from_catalog]                          ← build once, never again
State 3: templates.pkl  (165 valid GP templates)
    ↓ [build_magnitude_grid]                     ← build once, never again
State 4: mag_grid.pkl  (100-500x speedup)
    ↓ [run_slsn_pipeline.py --model <name>]      ← one population per model
State 5: population_{model}.pkl  (2-7M events)
    ↓ [run_slsn_pipeline.py --cadence <name>]    ← reuse population across cadences
State 6: metric_values_{metric}_{model}_{cadence}_{YYMMDD_HHMM}.npy
         summary_{model}_{cadence}_{YYMMDD_HHMM}.csv
    ↓ [analysis notebook]
State 7: Detection efficiency, N(SLSNe), redshift-binned counts
```

---

## Reload Flags

| Component | Regenerate when... | How | CLI flag? |
|---|---|---|---|
| Templates | Adding events to catalog | `LC.from_catalog()` in notebook | No — manual notebook only |
| Mag grid | Templates change | `templates.build_magnitude_grid()` in notebook | No — manual notebook only |
| Population | Changing model, z range, or CSV | `--regen-population` | Yes |
| Kernel | Never | — | — |

**Never regenerate templates or mag grid between cadence runs.**
**Always regenerate population when switching rate model.**
**`--regen-templates` and `--regen-mag-grid` are not yet implemented.**
**Template and mag grid rebuilds must be triggered manually in a notebook.**
**See mosfit_interface.py docstring for the full rebuild checklist.**

---

## Rate Models

| Model | `--model` flag | Physics | Priority |
|---|---|---|---|
| Fe-dependent | `fe_dependent` | Rate suppressed by iron abundance | **Primary** |
| Naive | `naive` | Rate tracks SFR only, no metallicity | Reference |
| O-dependent | `o_dependent` | Rate suppressed by oxygen abundance | Completeness |

All anchored to **Frohmaier+2021**: 35 Gpc⁻³ yr⁻¹ at z=0.17.
Rate CSV: `output/SLSNe/shared/fiducial_models.csv` (columns: z, f_OH, f_Fe_mixed).

---

## CLI Quick Reference

### Dry run (verify paths before computing)
```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --dry-run
```

### Production run (full population, no visit detail)
```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --store-obs-mode none \
    --n-cores 4
```

### Diagnostic run (50k events, full visit detail for plots)
```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --store-obs-mode full \
    --max-events 50000
```

### All models × all cadences (parallel)
```bash
python3 launch_all.py \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --cadences baseline_v5.1.1_10yrs four_roll_v5.0.0_10yrs noroll_v5.0.0_10yrs \
    --models naive fe_dependent o_dependent \
    --dry-run   # remove --dry-run to actually launch
```

### SLURM batch submission (MSI)
```bash
bash submit_slsn_batch.slurm
```

---

## SLURM Monitoring

```bash
# Is the job running?
squeue -u andra104

# What's in the log?
tail -f output/SLSNe/logs/{model}_{cadence}_{jobid}.out

# Any errors?
cat output/SLSNe/logs/{model}_{cadence}_{jobid}.err

# Recently completed jobs + memory usage
sacct -u andra104 --starttime=now-24hours \
      --format=JobID,JobName,State,Elapsed,MaxRSS

# Is it computing or stuck? (on compute node)
ssh acn162 'ps aux | grep andra104 | grep -v grep'
ssh acn162 'ls -la /proc/{PID}/fd'   # what files are open
```

---

## Output Files

Each production run writes to `output/SLSNe/{model}/`:

```
summary_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.csv
    cadence, metric, n_events, n_success, efficiency

metric_values_detect_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.npy
metric_values_characterize_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.npy
metric_values_spectrigger_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.npy
metric_values_villar_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.npy
metric_values_elasticc_{model}_{cadence}_z0.1-2.0_{YYMMDD_HHMM}.npy
    Per-event 0/1 arrays (length = population size)
    Join with population pickle for per-event analysis
```

### Loading results for analysis
```python
import numpy as np
import pickle

# Load population
with open('output/SLSNe/shared/population_fe_dependent.pkl', 'rb') as f:
    pop_data = pickle.load(f)
z_vals = pop_data['slice_points']['z']

# Load metric results
detect = np.load('output/SLSNe/fe_dependent/metric_values_detect_fe_dependent_baseline_v5.1.1_10yrs_z0.1-2.0_260402_1015.npy')

# Detection efficiency in Adam's redshift bin
mask = (z_vals >= 1.2) & (z_vals <= 1.8)
n_detected_highz = detect[mask].sum()
efficiency_highz = detect[mask].mean()
print(f"z=1.2-1.8: {n_detected_highz:.0f} detected, {100*efficiency_highz:.1f}% efficiency")
```

---

## Two-Run Strategy

| Run type | Population | `--store-obs-mode` | Purpose |
|---|---|---|---|
| Production | Full (2-7M) | `none` | Paper numbers |
| Diagnostic | 50k | `full` | Notebook plots |

Production gives efficiency numbers. Diagnostic gives visit-level detail
for `plot_metrics_mosaic_grid` and other diagnostic plots.

---

## Metrics

All three run in **one MAF pass** per cadence. Detection hierarchy enforced internally.

| Metric | Key criteria | Reference |
|---|---|---|
| `SLSN_Detect` | ≥2 filters SNR≥5, rising LC, ≥15 day baseline | Firth+2015 |
| `SLSN_Characterize` | Detect + ≥5 epochs, ≥3 filters, near/post-peak | Inserra+2024 |
| `SLSN_SpecTrigger` | Detect + near-peak epoch, mag ≤ 23.0 | — |

---

## Common Gotchas

| Problem | Solution |
|---|---|
| `KeyError: 'Cenwave'` | Run `attach_cenwave_to_perevent_csv()` first |
| Slow evaluator | Build and load mag_grid |
| File collision across models | `model_name` must be in `build_filenames` — already fixed |
| SLURM job silent for 25+ min | Normal — MAF `run_all()` has no internal progress output |
| Population too large for RAM | Use `--max-events 50000` for diagnostic runs |
| Wrong conda env | `conda activate rubin_sim_2.6.1` before any run |
| Cadence DB missing | Download from LSST sim-data S3 bucket to `cadences/` |
| Results overwrite on rerun | Date stamp in filename prevents this — e.g. `_260402.npy` |

---

## Module Import Order (Never Circular)

```
constants → export_slsne_photometry
         → gp_build
         → model  (uses gp_build, export)
         → population  (uses model, paths)
         → metrics  (uses model, constants)
         → runners  (uses metrics, paths)
         → diagnostics  (uses all, lazy matplotlib)
```

---

## Hardcoded Parameters
These values are baked into the code and affect science results. Not exposed as CLI flags.
Check the cited file and line before changing any of these.

### population.py
| Parameter | Value | Location | Justification |
|---|---|---|---|
| `RATE_REF_GPC3` | 35.0 Gpc⁻³ yr⁻¹ | `population.py` module level | Frohmaier+2021 local rate at z=0.17 |
| `Z_REF` | 0.17 | `population.py` module level | Reference redshift for Frohmaier+2021 |
| `OH_max` | 8.3 | `generate_SLSN_PopSlicer()` default | Metallicity threshold for SLSN host galaxies |
| `seed` | 42 | `generate_SLSN_PopSlicer()` default | Random seed — changing gives different population |
| `gal_lat_cut` | None | `generate_SLSN_PopSlicer()` default | None = Galactic plane included; pass 15.0 to exclude |
| CSFRD formula | Madau & Dickinson 2014 | `cosmic_sfr_density_MD14()` | Specific paper choice affects R(z) shape |

### metrics.py
| Parameter | Value | Location | Justification |
|---|---|---|---|
| `mag_limit` | 23.0 | `SLSN_SpecTriggerMetric.__init__` | Keck 1-hr ToO limit (confirmed by Adam Miller) |
| `peak_window` | 20.0 days | `SLSN_SpecTriggerMetric.__init__` | Window around peak for SpecTrigger detections |
| `n_near_peak` | 2 | `SLSN_SpecTriggerMetric.__init__` | Minimum detections required near peak |
| `decline_limit` | 1.0 mag | `SLSN_SpecTriggerMetric.__init__` | Max brightness change over 30 days (slow-evolution cut) |
| `SNR threshold` | 5 | `detect_slsn()` | Detection significance cut for all five metrics |
| `mjd0` | 60980.5 | All metric `__init__` defaults | Survey start MJD (~2023-10-01); verify against actual Rubin start |

---

## Key References

- Frohmaier+2021 — SLSN rate anchor (35 Gpc⁻³ yr⁻¹ at z=0.17)
- Madau & Dickinson 2014 — cosmic SFR density Ψ(z)
- Gómez+2024 — SLSN template catalog (265 events, 165 valid)
- Firth+2015 — detection criteria
- Inserra+2024 — characterization criteria
- Tremonti+2004 — mass-metallicity relation
- Andrews & Martini 2013 — MZR redshift evolution
