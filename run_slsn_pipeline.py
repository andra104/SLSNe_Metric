#!/usr/bin/env python3
"""
run_slsn_pipeline.py — Command-line entry point for SLSN metric runs.

Runs one (model, cadence) combination end-to-end:
  1. Load templates (read-only, shared)
  2. Load or generate population for the requested rate model
  3. Run all three MAF metrics against the requested cadence
  4. Save results to output/SLSNe/{model_name}/

Usage
-----
python run_slsn_pipeline.py \\
    --model fe_dependent \\
    --cadence baseline_v3.4 \\
    --templates-pkl output/SLSNe/shared/templates.pkl \\
    [--rate-csv output/SLSNe/shared/fiducial_models.csv] \\
    [--regen-population] \\
    [--n-cores 4] \\
    [--dry-run]

Rate models
-----------
  naive         SLSN rate tracks cosmic SFR only (no metallicity dependence)
  fe_dependent  Iron-abundance threshold — PRIMARY science result
  o_dependent   Oxygen-abundance threshold — completeness check

Rebuild controls
----------------
  Templates  : NOT handled here — rebuild manually in a notebook via LC.from_catalog()
               --regen-templates flag does not exist yet
  Mag grid   : NOT handled here — rebuild manually in a notebook via templates.build_magnitude_grid()
               --regen-mag-grid flag does not exist yet
  Population : --regen-population flag regenerates even if pickle exists
  Kernel     : never changes — fixed Matern-3/2 in gp_build.py

  See mosfit_interface.py docstring for the full rebuild checklist when
  switching from GP to physical templates.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime


def _log(msg):
    """Timestamped print — flushes immediately to SLURM log."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# First output before heavy imports
_log("Python started — beginning imports")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run SLSN detection pipeline for one (model, cadence) pair."
    )

    # Required
    p.add_argument('--model', required=True,
                   choices=['naive', 'fe_dependent', 'o_dependent'],
                   help="Rate model to use.")
    p.add_argument('--cadences', required=True, nargs='+',
                   help="One or more OpSim cadence names (without .db). "
                        "Multiple cadences run in parallel when --n-workers > 1. "
                        "e.g. --cadences baseline_v5.1.1_10yrs four_roll_v5.0.0_10yrs")
    p.add_argument('--templates-pkl', required=True,
                   help="Path to templates pickle file.")

    # Optional with defaults
    p.add_argument('--rate-csv', default=None,
                   help="Path to fiducial_models.csv. "
                        "Defaults to output/SLSNe/shared/fiducial_models.csv")
    p.add_argument('--regen-population', action='store_true',
                   help="Force regeneration of population even if pickle exists.")
    p.add_argument('--z-min', type=float, default=0.1,
                   help="Minimum redshift. Default 0.1")
    p.add_argument('--z-max', type=float, default=2.0,
                   help="Maximum redshift. Default 2.0")
    p.add_argument('--t-start', type=float, default=1.0,
                   help="Survey start (days). Default 1")
    p.add_argument('--t-end', type=float, default=3652.0,
                   help="Survey end (days). Default 3652 (10 yr)")
    p.add_argument('--gal-lat-cut', type=float, default=15.0,
                   help="Galactic latitude cut (degrees). Default 15")
    p.add_argument('--mjd0', type=float, default=60980.5,
                   help="Survey start MJD. Default 60980.5")
    p.add_argument('--seed', type=int, default=42,
                   help="Random seed. Default 42")
    p.add_argument('--n-cores', type=int, default=1,
                   help="Number of cores for MAF (passed to rubin_sim). Default 1")
    p.add_argument('--n-workers', type=int, default=1,
                   help="Number of parallel workers for metrics evaluation. "
                        "Splits population into N chunks, runs simultaneously. "
                        "Default 1 (sequential). Use 4 for ~4x speedup on MSI.")
    p.add_argument('--store-obs-mode', default='none',
                   choices=['none', 'meta', 'full'],
                   help="Observation storage mode. "
                        "'none': summary only (production). "
                        "'meta': per-event metadata, no visit arrays. "
                        "'full': full visit arrays for diagnostic plots (use with --max-events). "
                        "Default: none")
    p.add_argument('--max-events', type=int, default=None,
                   help="Cap population at this many events after generation. "
                        "Use with --store-obs-mode full for diagnostic runs (e.g. 50000). "
                        "Default: None (use full population).")
    p.add_argument('--only-metrics', nargs='+', default=None,
                   metavar='METRIC',
                   choices=['detect', 'characterize', 'villar', 'elasticc', 'spectrigger'],
                   help="Run only the listed metrics instead of all five. "
                        "e.g. --only-metrics spectrigger  "
                        "Valid: detect characterize villar elasticc spectrigger")
    p.add_argument('--dry-run', action='store_true',
                   help="Print resolved paths and parameters, then exit.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- resolve repo root and add package to path ---
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / 'py_files'))
    _log("  importing slsn_metrics (rubin_sim loads here — may take several minutes)...")

    from slsn_metrics.paths import (
        get_rate_csv_path, get_shared_output_dir, get_output_dir,
        get_cadence_path, print_paths
    )
    from slsn_metrics.model import LC
    from slsn_metrics.population import generate_SLSN_PopSlicer
    from slsn_metrics.runners import run_slsn_multi_metrics
    _log("All imports complete — pipeline starting.")

    # --- resolve paths ---
    templates_pkl = Path(args.templates_pkl)
    rate_csv      = Path(args.rate_csv) if args.rate_csv else get_rate_csv_path()
    shared_dir    = get_shared_output_dir('SLSNe')
    pop_pkl       = shared_dir / f"population_{args.model}.pkl"
    output_dir    = get_output_dir('SLSNe', subdir=args.model)

    # --- dry run: print config and exit ---
    if args.dry_run:
        print("\n=== DRY RUN — resolved configuration ===")
        print(f"  model          : {args.model}")
        print(f"  cadences       : {args.cadences}")
        for c in args.cadences:
            cdb = get_cadence_path(c)
            print(f"  cadence db     : {cdb}  {'OK' if cdb.exists() else 'MISSING'}")
        print(f"  templates pkl  : {templates_pkl}  {'OK' if templates_pkl.exists() else 'MISSING'}")
        print(f"  rate CSV       : {rate_csv}  {'OK' if rate_csv.exists() else 'MISSING'}")
        print(f"  population pkl : {pop_pkl}  {'exists' if pop_pkl.exists() else 'will generate'}")
        print(f"  output dir     : {output_dir}")
        print(f"  z range        : {args.z_min} – {args.z_max}")
        print(f"  survey days    : {args.t_start} – {args.t_end}")
        print(f"  gal lat cut    : {args.gal_lat_cut} deg")
        print(f"  seed           : {args.seed}")
        print(f"  regen pop      : {args.regen_population}")
        print(f"  only-metrics   : {args.only_metrics if args.only_metrics else 'all (detect, characterize, villar, elasticc, spectrigger)'}")
        print("=========================================\n")
        return

    # --- validate inputs ---
    if not templates_pkl.exists():
        sys.exit(f"ERROR: templates pickle not found: {templates_pkl}")
    for c in args.cadences:
        cdb = get_cadence_path(c)
        if not cdb.exists():
            sys.exit(f"ERROR: cadence database not found: {cdb}")
    if not rate_csv.exists():
        sys.exit(f"ERROR: rate CSV not found: {rate_csv}\n"
                 f"       Copy fiducial_models.csv to {rate_csv} or pass --rate-csv")

    # --- Step 1: Load templates ---
    print(f"\n{'='*60}")
    _log("CELL 2 — Load Templates")
    t0_step = datetime.now()
    print(f"{'='*60}")
    templates = LC(load_from=str(templates_pkl))
    elapsed = (datetime.now() - t0_step).total_seconds()
    _log(f"  Loaded {len(templates.data)} templates from {templates_pkl}  ({elapsed:.1f}s)")

    # --- Step 2: Load or generate population ---
    print(f"\n{'='*60}")
    _log(f"CELL 3 — Population [{args.model}] for {len(args.cadences)} cadence(s)")
    t0_step = datetime.now()
    print(f"{'='*60}")

    if pop_pkl.exists() and not args.regen_population:
        print(f"  Loading existing population: {pop_pkl}")
        population = generate_SLSN_PopSlicer(
            lc_model=templates,
            load_from=str(pop_pkl),
            make_debug_plots=False
        )
    else:
        if args.regen_population:
            print(f"  --regen-population set: regenerating.")
        else:
            print(f"  No existing population found. Generating.")
        population = generate_SLSN_PopSlicer(
            lc_model=templates,
            t_start=args.t_start,
            t_end=args.t_end,
            z_min=args.z_min,
            z_max=args.z_max,
            rate_model='tabulated',
            tabulated_csv=str(rate_csv),
            model_name=args.model,
            gal_lat_cut=args.gal_lat_cut,
            seed=args.seed,
            save_to=str(pop_pkl),
            make_debug_plots=False
        )

    n_events = len(population.slice_points['distance'])
    elapsed = (datetime.now() - t0_step).total_seconds()
    _log(f"  Population ready: {n_events:,} events  ({elapsed:.1f}s)")

    # --- Apply max_events cap (for diagnostic runs) ---
    if args.max_events is not None and n_events > args.max_events:
        print(f"  Capping population: {n_events:,} -> {args.max_events:,} events")
        import numpy as np
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_events, size=args.max_events, replace=False)
        keep = np.sort(keep)
        from rubin_sim.maf.slicers import UserPointsSlicer
        sp = population.slice_points
        ra_sub  = np.degrees(sp['ra'][keep])
        dec_sub = np.degrees(sp['dec'][keep])
        sub_pop = UserPointsSlicer(ra=ra_sub, dec=dec_sub, badval=0)
        for key in sp.keys():
            try:
                arr = np.asarray(sp[key])
                if arr.shape and arr.shape[0] == n_events:
                    sub_pop.slice_points[key] = arr[keep]
                else:
                    sub_pop.slice_points[key] = sp[key]
            except Exception:
                sub_pop.slice_points[key] = sp[key]
        sub_pop.slice_points['sid'] = np.arange(args.max_events)
        population = sub_pop
        print(f"  Subsample ready: {args.max_events:,} events")

    # --- Step 3: Run metrics ---
    print(f"\n{'='*60}")
    _log(f"CELL 4 — MAF Metrics {args.cadences}")
    _log(f"  {n_events:,} events x {len(args.cadences)} cadence(s)")
    _log("  Calling MAF run_all() — silent until complete, this is the long step")
    t0_step = datetime.now()
    print(f"{'='*60}")

    if args.only_metrics:
        _all_valid = ['detect', 'characterize', 'villar', 'elasticc', 'spectrigger']
        _skipped = [m for m in _all_valid if m not in args.only_metrics]
        print(f"""
============================================================
  WARNING — PARTIAL METRIC RUN (--only-metrics active)
  Running : {sorted(args.only_metrics)}
  Skipped : {_skipped}
  Only these .npy files will be written/updated.
  All other existing .npy files are untouched.
============================================================
""", flush=True)

    from slsn_metrics.runners import run_slsn_multi_metrics_parallel
    if args.n_workers > 1 and len(args.cadences) > 1:
        _log(f"  Using {args.n_workers} parallel workers for "
             f"{len(args.cadences)} cadences")
        summary = run_slsn_multi_metrics_parallel(
            templates=templates,
            population=population,
            cadences=args.cadences,
            n_workers=args.n_workers,
            output_dir=str(output_dir),
            mjd0=args.mjd0,
            save_summary=True,
            verbose=True,
            store_obs_mode=args.store_obs_mode,
            model_name=args.model,
            z_min=args.z_min,
            z_max=args.z_max,
            only_metrics=args.only_metrics,
        )
    else:
        summary = run_slsn_multi_metrics(
            templates=templates,
            population=population,
            cadences=args.cadences,
            output_dir=str(output_dir),
            mjd0=args.mjd0,
            save_summary=True,
            make_plots=False,
            verbose=True,
            store_obs_mode=args.store_obs_mode,
            model_name=args.model,
            z_min=args.z_min,
            z_max=args.z_max,
            only_metrics=args.only_metrics,
        )

    print(f"\n{'='*60}")
    elapsed = (datetime.now() - t0_step).total_seconds()
    _log(f"CELL 5 — Results  |  metrics runtime: {elapsed/60:.1f} min")
    _log(f"DONE: {args.model} x {args.cadences}")
    _log(f"Results: {output_dir}")
    print(f"{'='*60}\n")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    import traceback
    try:
        main()
    except SystemExit:
        raise   # let argparse / sys.exit() pass through normally
    except Exception as exc:
        # Flush a timestamped error so it's never buried in SLURM output.
        # Re-raise so the full traceback also appears, then exit non-zero.
        _log(f"FATAL ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
