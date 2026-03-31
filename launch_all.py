#!/usr/bin/env python3
"""
launch_all.py — Launch all (model, cadence) combinations in parallel.

Runs run_slsn_pipeline.py once per (model, cadence) pair as a background
process. All jobs run simultaneously. Logs go to output/SLSNe/logs/.

Usage
-----
# Run all three models against all cadences:
python launch_all.py \\
    --templates-pkl output/SLSNe/shared/templates.pkl \\
    --cadences baseline_v3.4 rolling_v3.4 \\
    --models naive fe_dependent o_dependent

# Run only the two priority models:
python launch_all.py \\
    --templates-pkl output/SLSNe/shared/templates.pkl \\
    --cadences baseline_v3.4 \\
    --models naive fe_dependent

# Dry run — print all commands without executing:
python launch_all.py \\
    --templates-pkl output/SLSNe/shared/templates.pkl \\
    --cadences baseline_v3.4 rolling_v3.4 \\
    --models naive fe_dependent o_dependent \\
    --dry-run

Notes
-----
- Templates and mag grid are shared (read-only) across all jobs.
- Population is generated once per model, then reused across cadences.
  Set --regen-population to force fresh generation.
- Each job writes its own log: output/SLSNe/logs/{model}_{cadence}.log
- To convert to SLURM: ask Claude to write a .slurm version of this script.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch all (model, cadence) pipeline combinations."
    )
    p.add_argument('--templates-pkl', required=True,
                   help="Path to templates pickle.")
    p.add_argument('--cadences', nargs='+', required=True,
                   help="One or more cadence names (without .db).")
    p.add_argument('--models', nargs='+',
                   default=['naive', 'fe_dependent'],
                   choices=['naive', 'fe_dependent', 'o_dependent'],
                   help="Models to run. Default: naive fe_dependent")
    p.add_argument('--rate-csv', default=None,
                   help="Path to fiducial_models.csv. Uses default if omitted.")
    p.add_argument('--regen-population', action='store_true',
                   help="Pass --regen-population to each pipeline run.")
    p.add_argument('--n-cores', type=int, default=1,
                   help="Cores per job. Default 1.")
    p.add_argument('--dry-run', action='store_true',
                   help="Print commands without running them.")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    pipeline  = repo_root / 'run_slsn_pipeline.py'

    # Log directory
    log_dir = repo_root / 'output' / 'SLSNe' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build all (model, cadence) combinations
    jobs = [(m, c) for m in args.models for c in args.cadences]

    print(f"\n{'='*60}")
    print(f"SLSN PIPELINE LAUNCHER")
    print(f"{'='*60}")
    print(f"  Models   : {args.models}")
    print(f"  Cadences : {args.cadences}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Log dir  : {log_dir}")
    print(f"{'='*60}\n")

    processes = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model, cadence in jobs:
        log_file = log_dir / f"{model}_{cadence}_{timestamp}.log"

        # Build command
        cmd = [
            sys.executable, str(pipeline),
            '--model',         model,
            '--cadence',       cadence,
            '--templates-pkl', args.templates_pkl,
            '--n-cores',       str(args.n_cores),
        ]
        if args.rate_csv:
            cmd += ['--rate-csv', args.rate_csv]
        if args.regen_population:
            cmd += ['--regen-population']

        print(f"  Job: {model} x {cadence}")
        print(f"       log -> {log_file}")
        print(f"       cmd -> {' '.join(cmd)}\n")

        if not args.dry_run:
            log_fh = open(log_file, 'w')
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(repo_root)
            )
            processes.append((model, cadence, proc, log_file, log_fh))

    if args.dry_run:
        print("Dry run complete. No jobs launched.")
        return

    # Wait for all jobs and report
    print(f"\nAll {len(processes)} jobs launched. Waiting for completion...\n")
    results = []
    for model, cadence, proc, log_file, log_fh in processes:
        proc.wait()
        log_fh.close()
        status = 'OK' if proc.returncode == 0 else f'FAILED (code {proc.returncode})'
        results.append((model, cadence, status, log_file))
        print(f"  [{status}] {model} x {cadence}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    n_ok     = sum(1 for _, _, s, _ in results if s == 'OK')
    n_failed = len(results) - n_ok
    print(f"  Completed: {n_ok}/{len(results)}")
    if n_failed:
        print(f"  FAILED ({n_failed}):")
        for model, cadence, status, log_file in results:
            if status != 'OK':
                print(f"    {model} x {cadence} — see {log_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
