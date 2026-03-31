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

Reload flags (when to set --regen-population)
---------------------------------------------
  Templates  : never reloaded here (read-only)
  Mag grid   : never reloaded here (read-only)
  Population : set --regen-population when changing model, z range, or CSV
  Kernel     : never touched here
"""

import argparse
import sys
import os
from pathlib import Path

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
    p.add_argument('--cadence', required=True,
                   help="OpSim cadence name (without .db), e.g. baseline_v3.4")
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

    from slsn_metrics.paths import (
        get_rate_csv_path, get_shared_output_dir, get_output_dir,
        get_cadence_path, print_paths
    )
    from slsn_metrics.model import LC
    from slsn_metrics.population import generate_SLSN_PopSlicer
    from slsn_metrics.runners import run_slsn_multi_metrics

    # --- resolve paths ---
    templates_pkl = Path(args.templates_pkl)
    rate_csv      = Path(args.rate_csv) if args.rate_csv else get_rate_csv_path()
    shared_dir    = get_shared_output_dir('SLSNe')
    pop_pkl       = shared_dir / f"population_{args.model}.pkl"
    output_dir    = get_output_dir('SLSNe', subdir=args.model)
    cadence_db    = get_cadence_path(args.cadence)

    # --- dry run: print config and exit ---
    if args.dry_run:
        print("\n=== DRY RUN — resolved configuration ===")
        print(f"  model          : {args.model}")
        print(f"  cadence        : {args.cadence}")
        print(f"  cadence db     : {cadence_db}  {'OK' if cadence_db.exists() else 'MISSING'}")
        print(f"  templates pkl  : {templates_pkl}  {'OK' if templates_pkl.exists() else 'MISSING'}")
        print(f"  rate CSV       : {rate_csv}  {'OK' if rate_csv.exists() else 'MISSING'}")
        print(f"  population pkl : {pop_pkl}  {'exists' if pop_pkl.exists() else 'will generate'}")
        print(f"  output dir     : {output_dir}")
        print(f"  z range        : {args.z_min} – {args.z_max}")
        print(f"  survey days    : {args.t_start} – {args.t_end}")
        print(f"  gal lat cut    : {args.gal_lat_cut} deg")
        print(f"  seed           : {args.seed}")
        print(f"  regen pop      : {args.regen_population}")
        print("=========================================\n")
        return

    # --- validate inputs ---
    if not templates_pkl.exists():
        sys.exit(f"ERROR: templates pickle not found: {templates_pkl}")
    if not cadence_db.exists():
        sys.exit(f"ERROR: cadence database not found: {cadence_db}")
    if not rate_csv.exists():
        sys.exit(f"ERROR: rate CSV not found: {rate_csv}\n"
                 f"       Copy fiducial_models.csv to {rate_csv} or pass --rate-csv")

    # --- Step 1: Load templates ---
    print(f"\n{'='*60}")
    print(f"STEP 1: Loading templates")
    print(f"{'='*60}")
    templates = LC(load_from=str(templates_pkl))
    print(f"  Loaded {len(templates.data)} templates from {templates_pkl}")

    # --- Step 2: Load or generate population ---
    print(f"\n{'='*60}")
    print(f"STEP 2: Population  [{args.model}]")
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
    print(f"  Population size: {n_events:,} events")

    # --- Step 3: Run metrics ---
    print(f"\n{'='*60}")
    print(f"STEP 3: Metrics  [{args.cadence}]")
    print(f"{'='*60}")

    summary = run_slsn_multi_metrics(
        templates=templates,
        population=population,
        cadences=[args.cadence],
        output_dir=str(output_dir),
        mjd0=args.mjd0,
        save_summary=True,
        make_plots=False,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"DONE: {args.model} x {args.cadence}")
    print(f"Results: {output_dir}")
    print(f"{'='*60}\n")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
