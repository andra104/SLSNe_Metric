# SLSNe_Metric

Detection rate predictions for Superluminous Supernovae (SLSNe) with the Vera C. Rubin Observatory LSST survey, using the Rubin MAF (Metrics Analysis Framework) and OpSim cadence databases.

## Science Overview

This pipeline predicts how many SLSNe Rubin will detect per year under different survey cadences, and how that number depends on the assumed SLSN rate evolution model. Three physically distinct rate models are evaluated:

| Model | Description | Priority |
|---|---|---|
| `fe_dependent` | Rate suppressed by iron abundance — primary science result | **Highest** |
| `naive` | Rate tracks cosmic SFR only, no metallicity dependence | Reference |
| `o_dependent` | Rate suppressed by oxygen abundance — completeness check | Lower |

All models are anchored to the Frohmaier et al. (2021) measured local rate of 35 Gpc⁻³ yr⁻¹ at z=0.17. Rate evolution R(z) is loaded from `fiducial_models.csv` (provided by B. Margalit).

## Repository Structure

```
SLSNe_Metric/
├── py_files/slsn_metrics/      ← Python package (7 modules)
│   ├── constants.py            ← Physical constants, cosmology
│   ├── model.py                ← LC class, GP templates, magnitude grid
│   ├── population.py           ← Population generation, rate models
│   ├── metrics.py              ← MAF metrics (detect, characterize, spec trigger)
│   ├── runners.py              ← Execution wrappers
│   ├── diagnostics.py          ← QA plots
│   └── paths.py                ← Machine-agnostic path resolution
├── run_slsn_pipeline.py        ← CLI entry point (one model × one cadence)
├── launch_all.py               ← Parallel launcher (all models × all cadences)
├── submit_slsn_batch.slurm     ← SLURM batch submission (MSI)
├── submit_slsn_pipeline.slurm  ← SLURM single job submission (MSI)
├── cadences/                   ← OpSim .db files
└── output/
    └── SLSNe/
        ├── shared/             ← templates.pkl, population_*.pkl, fiducial_models.csv
        ├── fe_dependent/       ← results for Fe-dependent model
        ├── naive/              ← results for naive model
        ├── o_dependent/        ← results for O-dependent model
        └── logs/               ← SLURM job logs
```

## Quick Start

### Prerequisites

```bash
conda activate rubin_sim_2.6.1
```

### Dry Run (verify paths before computing)

```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --dry-run
```

### Single Job (one model × one cadence)

```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --store-obs-mode none \
    --n-cores 4
```

### Full Batch on MSI (all models × all cadences)

```bash
bash submit_slsn_batch.slurm
```

Submits 9 jobs simultaneously. Priority order: fe_dependent → o_dependent → naive (naive runs after the others complete).

### Diagnostic Run (50k events, full observation detail)

```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadence baseline_v5.1.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --store-obs-mode full \
    --max-events 50000
```

## Output Files

Each run produces stamped output files in `output/SLSNe/{model}/`:

```
summary_{model}_{cadence}_z0.1-2.0_{YYMMDD}.csv
metric_values_Detect_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
metric_values_Characterize_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
metric_values_SpecTrigger_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
```

The `.npy` files are per-event 0/1 detection arrays (one entry per simulated SLSN). Joined with the population pickle they give full per-event analysis.

## Reload Flags

When to regenerate each component:

| Component | Regenerate when... | Flag |
|---|---|---|
| Templates | Adding new events to Gómez+2024 catalog | `LC.from_catalog()` |
| Mag grid | Templates change | `templates.build_magnitude_grid()` |
| Population | Changing rate model, z range, or CSV values | `--regen-population` |
| Kernel | Never — GP kernel is fixed | — |

## Rate Models

Rate evolution R(z) follows:

```
R(z) = Rate_ref × [Ψ(z) / Ψ(z_ref)] × [f(z) / f(z_ref)]
```

Where:
- `Rate_ref = 35 Gpc⁻³ yr⁻¹` (Frohmaier+2021 at z_ref = 0.17)
- `Ψ(z)` = Madau & Dickinson 2014 cosmic SFR density
- `f(z)` = fraction of SF in low-metallicity galaxies (from `fiducial_models.csv`)

The naive model sets f(z) = constant (no metallicity dependence).

## Metrics

Three MAF metrics are evaluated in a single pass per cadence:

| Metric | Criteria | Reference |
|---|---|---|
| `SLSN_Detect` | ≥2 filters SNR≥5, rising LC, ≥15 day baseline | Firth+2015 |
| `SLSN_Characterize` | Detection + ≥5 epochs, ≥3 filters, near/post-peak coverage | Inserra+2024 |
| `SLSN_SpecTrigger` | Detection + near-peak epoch, mag ≤ 21.0, blue color | — |

Characterization and spec trigger enforce detection internally — they re-run the detection check before applying their additional criteria.

## Cadences

| Cadence | Description |
|---|---|
| `baseline_v5.1.1_10yrs` | Standard WFD baseline |
| `four_roll_v5.0.0_10yrs` | Four-roll WFD variant |
| `noroll_v5.0.0_10yrs` | No-roll WFD variant |
| `poor_weather_v5.0.1_10yrs` | Poor weather baseline (lower priority) |

## Key References

- Frohmaier et al. (2021) — SLSN local rate anchor
- Madau & Dickinson (2014) — cosmic SFR density
- Gómez et al. (2024) — SLSN template catalog (265 events, 165 valid)
- Firth et al. (2015) — detection criteria
- Inserra et al. (2024) — characterization criteria

## Collaborators

- B. Margalit — theoretical rate models, `fiducial_models.csv`
- V. Shah — detection metrics, Monte Carlo uncertainty methods
- A. Miller — science interpretation, fixed-z bin comparison (z=1.2–1.8)
- C. Andrade — pipeline implementation
