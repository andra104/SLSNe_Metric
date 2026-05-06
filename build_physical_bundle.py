#!/usr/bin/env python3
"""
build_physical_bundle.py — Build physical magnitude grid and populations.

Runs three steps in sequence:
  1. Load physical templates from physical_templates.pkl
  2. Build or resume physical magnitude grid (z=0.02-5.0, phase=1-400d)
  3. Build populations for all three rate models (z_max=5.0)

Physical templates must be pre-built. Run prototype_physical_templates.ipynb
with REBUILD_PHYSICAL_TEMPLATES=True to generate physical_templates.pkl first.

Usage
-----
python3 build_physical_bundle.py [options]

Flags
-----
  --skip-mag-grid      Skip magnitude grid build even if file is missing
  --skip-populations   Skip all population generation
  --regen-population   Force regenerate populations even if pkl files exist
  --dry-run            Print resolved paths and exit without doing any work

Grid parameters (fixed)
-----------------------
  z_grid    : linspace(0.02, 5.0, 100)
  phase_grid: geomspace(1, 100, 50) + linspace(100, 400, 30)  [80 pts]
  filters   : ugrizy

Rebuild controls
----------------
  Templates  : NOT handled here — rebuild in prototype_physical_templates.ipynb
  Mag grid   : built here if physical_mag_grid.pkl does not exist;
               resumes from checkpoint if interrupted
  Population : --regen-population flag regenerates even if pickle exists
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np


def _log(msg):
    """Timestamped print — flushes immediately to SLURM log."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# First output before heavy imports
_log("Python started — beginning imports")

# ---------------------------------------------------------------------------
# Paths — hardcoded relative to this script
# ---------------------------------------------------------------------------
REPO_ROOT               = Path(__file__).resolve().parent
SHARED_DIR              = REPO_ROOT / 'output' / 'SLSNe' / 'shared'
PHYSICAL_TEMPLATES_FILE = SHARED_DIR / 'physical_templates.pkl'
PHYSICAL_MAG_GRID_FILE  = SHARED_DIR / 'physical_mag_grid.pkl'
PHYSICAL_SED_CACHE_FILE = SHARED_DIR / 'physical_sed_cache.pkl'
RATE_CSV                = SHARED_DIR / 'fiducial_models.csv'

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
Z_GRID = np.linspace(0.02, 5.0, 100)

PHASE_GRID = np.concatenate([
    np.geomspace(1.0,  100.0, 50),
    np.linspace(100.0, 400.0, 30),
])  # 80 points

FILTERS = list('ugrizy')

MODELS = ['fe_dependent', 'o_dependent', 'naive']


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build physical magnitude grid and populations for all three rate models."
    )
    p.add_argument('--skip-mag-grid', action='store_true',
                   help="Skip magnitude grid build even if physical_mag_grid.pkl is missing.")
    p.add_argument('--skip-populations', action='store_true',
                   help="Skip all population generation.")
    p.add_argument('--regen-population', action='store_true',
                   help="Force regenerate populations even if pkl files exist.")
    p.add_argument('--regen-templates', action='store_true',
                   help="Force rebuild physical templates even if pkl exists.")
    p.add_argument('--dry-run', action='store_true',
                   help="Print resolved paths and exit without doing any work.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- resolve repo root and add package to path ---
    repo_root = REPO_ROOT
    sys.path.insert(0, str(repo_root / 'py_files'))
    _log("  importing slsn_metrics (rubin_sim loads here — may take several minutes)...")

    from slsn_metrics.model import LC
    from slsn_metrics.population import generate_SLSN_PopSlicer
    from slsn_metrics.paths import get_log_dir
    get_log_dir('build')   # auto-creates output/logs/build/
    _log("All imports complete — pipeline starting.")

    # --- dry run: print config and exit ---
    if args.dry_run:
        print("\n=== DRY RUN — resolved configuration ===")
        print(f"  repo root              : {repo_root}")
        ALL_PARAMS_FILE = repo_root / 'SLSNe' / 'slsne' / 'ref_data' / 'all_parameters.txt'
        print(f"  all_parameters.txt     : {ALL_PARAMS_FILE}  "
              f"{'OK' if ALL_PARAMS_FILE.exists() else 'MISSING'}")
        print(f"  physical_templates.pkl : {PHYSICAL_TEMPLATES_FILE}  "
              f"{'OK' if PHYSICAL_TEMPLATES_FILE.exists() else 'will build'}")
        print(f"  physical_sed_cache.pkl : {PHYSICAL_SED_CACHE_FILE}  "
              f"{'OK' if PHYSICAL_SED_CACHE_FILE.exists() else 'will build'}")
        print(f"  physical_mag_grid.pkl  : {PHYSICAL_MAG_GRID_FILE}  "
              f"{'exists' if PHYSICAL_MAG_GRID_FILE.exists() else 'will build'}")
        print(f"  rate CSV               : {RATE_CSV}  "
              f"{'OK' if RATE_CSV.exists() else 'MISSING'}")
        print(f"  z_grid                 : linspace(0.02, 5.0, 100)  "
              f"[{Z_GRID.min():.2f}, {Z_GRID.max():.2f}]")
        print(f"  phase_grid             : {len(PHASE_GRID)} pts  "
              f"[{PHASE_GRID.min():.1f}, {PHASE_GRID.max():.1f}] days")
        print(f"  filters                : {FILTERS}")
        print(f"  models                 : {MODELS}")
        print(f"  skip_mag_grid          : {args.skip_mag_grid}")
        print(f"  skip_populations       : {args.skip_populations}")
        print(f"  regen_population       : {args.regen_population}")
        print()
        for model in MODELS:
            pop_pkl = SHARED_DIR / f'population_{model}_physical.pkl'
            print(f"  population_{model}_physical : "
                  f"{'exists' if pop_pkl.exists() else 'will build'}")
        print("=========================================\n")
        return

    # --- validate required inputs ---
    if not args.skip_populations and not RATE_CSV.exists():
        sys.exit(
            f"FATAL ERROR: rate CSV not found:\n"
            f"  {RATE_CSV}\n"
            f"Copy fiducial_models.csv to that path, or use --skip-populations."
        )

    # Track status for summary table
    status = {}

    # -----------------------------------------------------------------------
    # Step 1 — Build or load physical templates
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    _log("STEP 1 — Physical Templates")
    t0_step = datetime.now()
    print(f"{'='*60}")

    from slsn_metrics.mosfit_interface import build_physical_templates

    ALL_PARAMS_FILE = repo_root / 'SLSNe' / 'slsne' / 'ref_data' / 'all_parameters.txt'
    if not ALL_PARAMS_FILE.exists():
        sys.exit(
            f"FATAL ERROR: all_parameters.txt not found:\n"
            f"  {ALL_PARAMS_FILE}\n"
            f"This file contains Gomez+2024 MOSFiT posterior medians for 265 events."
        )

    if PHYSICAL_TEMPLATES_FILE.exists() and not args.regen_templates:
        _log(f"  Loading from {PHYSICAL_TEMPLATES_FILE}")
        try:
            import joblib
            payload = joblib.load(PHYSICAL_TEMPLATES_FILE)
        except Exception as exc:
            sys.exit(f"FATAL ERROR: could not load physical_templates.pkl: {exc}")

        templates = LC(
            lightcurves = payload['lightcurves'],
            t_grid      = payload.get('t_grid'),
            names       = payload.get('names'),
        )
        templates.sed_grid = payload['sed_grid']
        elapsed = (datetime.now() - t0_step).total_seconds()
        _log(f"  Loaded {len(templates.names)} events  ({elapsed:.1f}s)")
        status['physical_templates.pkl'] = ('LOADED', PHYSICAL_TEMPLATES_FILE)

    else:
        if args.regen_templates:
            _log("  --regen-templates set: rebuilding.")
        else:
            _log("  physical_templates.pkl not found — building from scratch.")
        _log(f"  Input : {ALL_PARAMS_FILE}")
        _log(f"  Cache : {PHYSICAL_SED_CACHE_FILE}")
        _log(f"  Output: {PHYSICAL_TEMPLATES_FILE}")
        _log("  NOTE: slsnni() runs for all 265 events (~15-20 min first run).")
        _log("        Subsequent runs load from cache in seconds.")

        templates = build_physical_templates(
            params_file = ALL_PARAMS_FILE,
            save_to     = PHYSICAL_TEMPLATES_FILE,
            cache_file  = PHYSICAL_SED_CACHE_FILE,
        )
        elapsed = (datetime.now() - t0_step).total_seconds()
        _log(f"  Built {len(templates.names)} events  ({elapsed/60:.1f} min)")
        _log(f"  Wavelength pts: {templates.sed_grid[0]['lam_rest_A'].shape[0]}")
        status['physical_templates.pkl'] = ('BUILT', PHYSICAL_TEMPLATES_FILE)
        
    n_templates = len(templates.names)
    _log(f"  First event : {templates.names[0]}")
    _log(f"  Last event  : {templates.names[-1]}")

    if not templates.sed_grid:
        sys.exit(
            "FATAL ERROR: sed_grid is empty after loading/building templates.\n"
            "Delete physical_templates.pkl and physical_sed_cache.pkl and rerun."
        )

    # -----------------------------------------------------------------------
    # Step 2 — Build or load physical magnitude grid
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    _log("STEP 2 — Physical Magnitude Grid")
    t0_step = datetime.now()
    print(f"{'='*60}")

    if args.skip_mag_grid:
        _log("  --skip-mag-grid set: skipping magnitude grid step.")
        status['physical_mag_grid.pkl'] = ('SKIPPED', PHYSICAL_MAG_GRID_FILE)

    elif PHYSICAL_MAG_GRID_FILE.exists():
        _log(f"  {PHYSICAL_MAG_GRID_FILE.name} already exists — loading.")
        templates.load_magnitude_grid(PHYSICAL_MAG_GRID_FILE)
        grid_shape = templates.mag_grid['g'].shape
        elapsed = (datetime.now() - t0_step).total_seconds()
        _log(f"  Loaded.  Shape (g-band): {grid_shape}  ({elapsed:.1f}s)")
        _log(f"  z range    : [{templates.mag_grid_axes['z'].min():.2f}, "
             f"{templates.mag_grid_axes['z'].max():.2f}]")
        _log(f"  phase range: [{templates.mag_grid_axes['phase'].min():.1f}, "
             f"{templates.mag_grid_axes['phase'].max():.1f}] days")
        status['physical_mag_grid.pkl'] = ('LOADED', PHYSICAL_MAG_GRID_FILE)

    else:
        # Check for existing checkpoint from a prior interrupted build
        checkpoint_file = PHYSICAL_MAG_GRID_FILE.with_suffix('.checkpoint.pkl')
        if checkpoint_file.exists():
            try:
                import pickle
                with open(checkpoint_file, 'rb') as f:
                    ckpt = pickle.load(f)
                last_completed = ckpt.get('last_completed', '?')
                n_total_ckpt   = ckpt.get('n_templates', n_templates)
                _log(f"  Resuming from checkpoint at template "
                     f"{last_completed}/{n_total_ckpt}")
            except Exception:
                _log("  Checkpoint found but unreadable — starting fresh.")
        else:
            _log("  No existing grid or checkpoint found. Building from scratch.")

        _log(f"  Templates  : {n_templates} events")
        _log(f"  z grid     : [{Z_GRID.min():.2f}, {Z_GRID.max():.2f}]  "
             f"n={len(Z_GRID)}")
        _log(f"  phase grid : [{PHASE_GRID.min():.1f}, {PHASE_GRID.max():.1f}] days  "
             f"n={len(PHASE_GRID)}")
        _log(f"  filters    : {FILTERS}")
        _log(f"  Output     : {PHYSICAL_MAG_GRID_FILE}")
        _log("  NOTE: This is the slow step (~20-60 min on MSI). "
             "Checkpointing every 50 templates.")

        templates.build_magnitude_grid(
            z_grid           = Z_GRID,
            phase_grid       = PHASE_GRID,
            filters          = FILTERS,
            save_to          = PHYSICAL_MAG_GRID_FILE,
            checkpoint_every = 50,
        )

        grid_shape = templates.mag_grid['g'].shape
        elapsed    = (datetime.now() - t0_step).total_seconds()
        _log(f"  Built.  Shape (g-band): {grid_shape}  ({elapsed/60:.1f} min)")
        _log(f"  z range    : [{templates.mag_grid_axes['z'].min():.2f}, "
             f"{templates.mag_grid_axes['z'].max():.2f}]")
        _log(f"  phase range: [{templates.mag_grid_axes['phase'].min():.1f}, "
             f"{templates.mag_grid_axes['phase'].max():.1f}] days")
        status['physical_mag_grid.pkl'] = ('BUILT', PHYSICAL_MAG_GRID_FILE)

    # -----------------------------------------------------------------------
    # Step 3 — Build populations for all three rate models
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    _log("STEP 3 — Physical Template Populations  [z_max=5.0, tabulated R(z)]")
    print(f"{'='*60}")

    if args.skip_populations:
        _log("  --skip-populations set: skipping all population generation.")
        for model in MODELS:
            pop_pkl = SHARED_DIR / f'population_{model}_physical.pkl'
            status[f'population_{model}_physical'] = ('SKIPPED', pop_pkl)
    else:
        for model in MODELS:
            pop_pkl = SHARED_DIR / f'population_{model}_physical.pkl'

            print(f"\n{'='*60}")
            _log(f"  Model: {model}")
            t0_pop = datetime.now()
            print(f"{'='*60}")

            if pop_pkl.exists() and not args.regen_population:
                _log(f"  Loading existing population: {pop_pkl.name}")
                population = generate_SLSN_PopSlicer(
                    lc_model         = templates,
                    load_from        = str(pop_pkl),
                    make_debug_plots = False,
                )
                n_events = len(population.slice_points['distance'])
                elapsed  = (datetime.now() - t0_pop).total_seconds()
                _log(f"  Loaded {n_events:,} events  ({elapsed:.1f}s)")
                status[f'population_{model}_physical'] = ('LOADED', pop_pkl)

            else:
                if args.regen_population and pop_pkl.exists():
                    _log("  --regen-population set: regenerating.")
                else:
                    _log("  No existing population found. Generating.")

                population = generate_SLSN_PopSlicer(
                    lc_model         = templates,
                    t_start          = 1.0,
                    t_end            = 3652.0,
                    z_min            = 0.1,
                    z_max            = 5.0,
                    rate_model       = 'tabulated',
                    tabulated_csv    = str(RATE_CSV),
                    model_name       = model,
                    gal_lat_cut      = 15.0,
                    seed             = 42,
                    save_to          = str(pop_pkl),
                    make_debug_plots = False,
                )
                n_events = len(population.slice_points['distance'])
                elapsed  = (datetime.now() - t0_pop).total_seconds()
                _log(f"  Built {n_events:,} events  ({elapsed/60:.1f} min)")
                _log(f"  Saved → {pop_pkl}")
                status[f'population_{model}_physical'] = ('BUILT', pop_pkl)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    _log("SUMMARY")
    print(f"{'='*60}")

    col_w = 38
    print(f"  {'Component':<{col_w}} {'Status':<10}  Path")
    print(f"  {'-'*col_w} {'-'*8}  {'-'*44}")
    for component, (state, path) in status.items():
        rel = str(path).replace(str(repo_root) + os.sep, '')
        print(f"  {component:<{col_w}} {state:<10}  {rel}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    import traceback
    try:
        main()
    except SystemExit:
        raise   # let sys.exit() pass through normally
    except Exception as exc:
        _log(f"FATAL ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
