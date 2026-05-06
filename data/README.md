
# data/

External scientific inputs committed as versioned data artifacts.

These files are **not pipeline outputs** — they are observational data,

MOSFiT posteriors, and rate models that the pipeline reads as inputs.

---

## Directory Structure

data/ 

├── rate_models/ ← Tabulated R(z) rate models from B. Margalit 

├── slsne_catalog/ ← Gómez+2024 MOSFiT posterior medians 

├── per_event/ ← Reformatted per-event photometry CSVs 

└── all_events/ ← Compiled all-event catalog files


---

## rate_models/

| File | Description | Source |
|------|-------------|--------|
| `fiducial_models.csv` | Tabulated R(z) for fe_dependent, o_dependent, naive models | B. Margalit (private communication, 2026-03-30) |
| `fiducial_models_uncertainty.csv` | R(z) Z_max variants for uncertainty analysis | B. Margalit (private communication, 2026-04-07) |

Column structure of `fiducial_models.csv`:
- col 0: redshift z (0 to 6, step 0.06)
- col 1: f_OH (O-dependent metallicity fraction)
- col 2: f_Fe_mixed (Fe-dependent metallicity fraction)

---

## slsne_catalog/

| File | Description | Source |
|------|-------------|--------|
| `all_parameters.txt` | MOSFiT posterior medians for 265 Gómez+2024 SLSNe-I | Gómez+2024 (arXiv:2407.07946); Gómez git commit `3023697`; snapshot 2026-04-09 |

This file is read directly by `mosfit_interface.py` to build physical SED templates.
The committed snapshot corresponds to the 260409 canonical production runs.
If Gómez updates his repository, copy the new version here and document the date.

---

## per_event/

Reformatted per-event photometry CSVs produced by `export_slsne_photometry.py`
from the raw Gómez+2024 photometry files in `SLSNe/slsne/`.

| File pattern | Description |
|-------------|-------------|
| `{event}.csv` | Raw photometry for one event (filter, mjd, mag, magerr) |
| `{event}_cenwave.csv` | Same, with central wavelength column attached |

265 events × 2 file types = 530 files (plus 1 extra raw CSV = 531 total).
Generated: 2026-02-20. Source: Gómez+2024 catalog via `SLSNe/` vendored clone.

To regenerate:
```python
from slsn_metrics.export_slsne_photometry import process_all_events
process_all_events()
```

---

## all_events/

Compiled catalog files aggregating all 265+ events.

| File | Description |
|------|-------------|
| `all_objects.csv` | Full photometry for all events combined |
| `allevent_redshift_med.csv` | Median redshift per event |
| `allparameter.csv` | MOSFiT parameters for all events (reformatted) |
| `filter_reference.csv` | Filter name → central wavelength mapping |
| `generic_reference.csv` | Generic bandpass reference table |

Generated: 2026-02-20. Source: Gómez+2024 catalog via `export_slsne_photometry.py`.

---

## What is NOT here

- `cadences/*.db` — OpSim cadence databases (100s MB–1 GB each, not committed).
  Download from the LSST sim-data S3 bucket. See `docs/` for instructions.
- `output/SLSNe/shared/*.pkl` — generated templates, populations, mag grids.
  These are pipeline outputs, not inputs. Regenerate via the pipeline.
