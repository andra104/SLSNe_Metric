#!/usr/bin/env python3
"""
adaptive_efficiency.py — Adaptive z-bin stopping wrapper.

Submits run_slsn_pipeline.py jobs one z-bin at a time via SLURM
dependency chains. After each bin completes, reads the summary CSV
and checks the stopping criterion. Stops when ELAsTiCC n_success == 0
across ALL cadences for TWO consecutive bins.

Uses full population per bin (no --max-events cap). Each bin job
runs all 4 cadences in parallel via --n-workers 4.

Usage:
    python3 adaptive_efficiency.py [--models MODEL [MODEL ...]]
                                   [--z-start FLOAT]
                                   [--z-step FLOAT]
                                   [--account STR]
                                   [--partition STR]
                                   [--dry-run]

Outputs:
    output/SLSNe/adaptive/{model}_adaptive_results.csv
        One row per z-bin per cadence per metric.
        Appended after each bin completes.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
TEMPLATES_PKL = REPO / 'output/SLSNe/shared/physical_templates.pkl'
CADENCES = [
    'baseline_v5.1.1_10yrs',
    'baseline_v5.3.0_10yrs',
    'noroll_v5.0.0_10yrs',
    'poor_weather_v5.0.1_10yrs',
]
CONDA_INIT = (
    'source /common/software/install/migrated/anaconda/'
    'miniconda3_4.8.3-jupyter/etc/profile.d/conda.sh && '
    'conda activate rubin_sim_2.6.1'
)
STOPPING_METRIC = 'SLSN_ELAsTiCC_Metric'
STOP_CONSEC     = 2      # consecutive zero bins before stopping
N_WORKERS       = 4      # one per cadence
MEM             = '48G'
TIME_PER_BIN    = '02:00:00'  # generous per-bin walltime

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_summary_csvs(model, z_lo, z_hi, output_dir):
    """
    Find all per-cadence summary CSVs for this z-bin.
    Pattern: summary_{model}_{cadence}_z{z_lo}-{z_hi}_{date}.csv
    Returns list of Path objects — one per cadence if all present.
    """
    z_tag = f'z{z_lo:.1f}-{z_hi:.1f}'
    return sorted(output_dir.glob(f'summary_{model}_*_{z_tag}_*.csv'))


def read_bin_results(csvs, z_lo, z_hi):
    """
    Read all per-cadence summary CSVs for one z-bin.
    Returns DataFrame with columns:
        z_lo, z_hi, cadence, metric, n_events, n_success, efficiency, sigma
    """
    rows = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            df['z_lo'] = z_lo
            df['z_hi'] = z_hi
            rows.append(df)
        except Exception as e:
            log(f"  WARNING: could not read {csv.name}: {e}")
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    # Add Poisson sigma
    eff = combined['efficiency'].values
    n   = combined['n_events'].values
    combined['sigma'] = np.where(
        n > 0,
        np.sqrt(eff * (1.0 - eff) / np.maximum(n, 1)),
        np.nan
    )
    return combined


def check_stop(bin_results_df, consecutive_zeros):
    """
    Check stopping criterion for the current bin.
    Returns True (stop) if ELAsTiCC n_success == 0 for ALL cadences.
    consecutive_zeros is the running count of zero bins so far.
    """
    if bin_results_df.empty:
        log("  WARNING: empty results — treating as non-zero, continuing")
        return False, 0

    elasticc = bin_results_df[
        bin_results_df['metric'] == STOPPING_METRIC
    ]
    if elasticc.empty:
        log(f"  WARNING: {STOPPING_METRIC} not found in results")
        return False, consecutive_zeros

    all_zero = (elasticc['n_success'] == 0).all()
    if all_zero:
        consecutive_zeros += 1
        log(f"  ELAsTiCC n_success == 0 across all cadences "
            f"({consecutive_zeros}/{STOP_CONSEC} consecutive zero bins)")
    else:
        consecutive_zeros = 0
        total = int(elasticc['n_success'].sum())
        log(f"  ELAsTiCC detections: {total} total across all cadences — continuing")

    return consecutive_zeros >= STOP_CONSEC, consecutive_zeros


def submit_bin_job(model, z_lo, z_hi, account, partition, output_dir,
                   dependency_job_id=None, dry_run=False,
                   store_obs_mode="none", cadences=None):
    """
    Submit one z-bin SLURM job via sbatch --wrap.
    Returns job ID string, or 'DRY_RUN' if dry_run=True.
    """
    z_lo_s = f'{z_lo:.1f}'
    z_hi_s = f'{z_hi:.1f}'
    job_name = f'adap_{model[:4]}_{z_lo_s}'
    log_file = (REPO / 'output/SLSNe/logs' /
                f'adaptive_{model}_z{z_lo_s}-{z_hi_s}_%j.out')
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd_inner = (
        f'cd {REPO} && '
        f'{CONDA_INIT} && '
        f'python3 run_slsn_pipeline.py '
        f'--model {model} '
        f'--cadences {" ".join(cadences or CADENCES)} '
        f'--templates-pkl {TEMPLATES_PKL} '
        f'--z-min {z_lo_s} '
        f'--z-max {z_hi_s} '
        f'--store-obs-mode {store_obs_mode} '
        f'--n-workers {N_WORKERS}'
    )

    sbatch_cmd = [
        'sbatch',
        f'--job-name={job_name}',
        f'--account={account}',
        f'--partition={partition}',
        '--nodes=1',
        '--ntasks=1',
        f'--cpus-per-task={N_WORKERS}',
        f'--mem={MEM}',
        f'--time={TIME_PER_BIN}',
        f'--output={log_file}',
    ]
    if dependency_job_id:
        sbatch_cmd.append(f'--dependency=afterok:{dependency_job_id}')
    sbatch_cmd += ['--wrap', cmd_inner]

    if dry_run:
        log(f"  DRY RUN: {' '.join(sbatch_cmd[:6])} ... z={z_lo_s}-{z_hi_s}")
        return 'DRY_RUN'

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ERROR submitting job: {result.stderr.strip()}")
        sys.exit(1)
    job_id = result.stdout.strip().split()[-1]
    return job_id


def wait_for_job(job_id, poll_seconds=30):
    """
    Poll squeue until job_id is no longer running/pending.
    Returns True if job completed successfully (COMPLETED),
    False if FAILED/CANCELLED/TIMEOUT.
    """
    log(f"  Waiting for job {job_id}...")
    while True:
        result = subprocess.run(
            ['sacct', '-j', job_id,
             '--format=JobID,State', '--noheader', '--parsable2'],
            capture_output=True, text=True
        )
        lines = [l for l in result.stdout.strip().splitlines()
                 if '.' not in l.split('|')[0]]  # skip .batch .extern
        if lines:
            state = lines[0].split('|')[1].strip()
            if state == 'COMPLETED':
                return True
            if state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL'):
                log(f"  Job {job_id} ended with state: {state}")
                return False
            if state in ('RUNNING', 'PENDING', 'COMPLETING'):
                time.sleep(poll_seconds)
                continue
        time.sleep(poll_seconds)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_adaptive(model, z_start, z_step, account, partition, dry_run,
                 store_obs_mode="none", cadences=None):
    """Run adaptive z-bin loop for one model."""
    output_dir  = REPO / 'output/SLSNe' / model
    adaptive_dir = REPO / 'output/SLSNe/adaptive'
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    results_csv = adaptive_dir / f'{model}_adaptive_results.csv'

    log(f"\n{'='*60}")
    log(f"Model: {model}")
    log(f"Z start: {z_start:.1f}, step: {z_step:.1f}")
    log(f"Stop criterion: ELAsTiCC n_success==0 for all cadences, "
        f"{STOP_CONSEC} consecutive bins")
    log(f"{'='*60}")

    consecutive_zeros = 0
    z_lo = z_start
    all_rows = []
    bin_num  = 0

    while True:
        z_hi   = round(z_lo + z_step, 2)
        bin_num += 1
        log(f"\nBin {bin_num}: z=[{z_lo:.1f}, {z_hi:.1f})")
        # Dry run guard — stop at z=2.0 to avoid infinite loop
        # Real runs stop via check_stop() after reading actual CSV results
        if dry_run and z_lo >= 2.0:
            log("  DRY RUN: reached z=2.0 — stopping (real runs use adaptive criterion)")
            break

        # Submit job — no dependency for first bin
        dep = None
        job_id = submit_bin_job(
            model, z_lo, z_hi, account, partition,
            output_dir, dependency_job_id=dep, dry_run=dry_run,
            store_obs_mode=store_obs_mode, cadences=cadences
        )
        log(f"  Submitted job {job_id}")

        if not dry_run:
            success = wait_for_job(job_id)
            if not success:
                log(f"  Job {job_id} failed — stopping adaptive run")
                break

            # Find and read summary CSVs
            csvs = find_summary_csvs(model, z_lo, z_hi, output_dir)
            _cad_list = cadences or CADENCES
            if len(csvs) < len(_cad_list):
                log(f"  WARNING: expected {len(_cad_list)} summary CSVs, "
                    f"found {len(csvs)}")

            bin_df = read_bin_results(csvs, z_lo, z_hi)

            # Append to running results CSV
            if not bin_df.empty:
                all_rows.append(bin_df)
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_csv(results_csv, index=False)
                log(f"  Results appended to {results_csv.name}")

                # Print per-metric summary for this bin
                for metric in bin_df['metric'].unique():
                    mdf = bin_df[bin_df['metric'] == metric]
                    avg_eff = mdf['efficiency'].mean()
                    log(f"    {metric}: avg_eff={avg_eff:.4f} "
                        f"(n_events={mdf['n_events'].iloc[0]:,})")

            # Check stopping criterion
            stop, consecutive_zeros = check_stop(bin_df, consecutive_zeros)
            if stop:
                log(f"\n{'='*60}")
                log(f"STOPPING: {STOP_CONSEC} consecutive zero-detection bins")
                log(f"Detection horizon for {model}: z ~ {z_lo:.1f}")
                log(f"Full results: {results_csv}")
                log(f"{'='*60}")
                break
        else:
            # Dry run: just show what would happen
            log(f"  DRY RUN: would wait for job, read CSVs, check criterion")

        z_lo = z_hi


def main():
    p = argparse.ArgumentParser(
        description='Adaptive z-bin efficiency stopping wrapper'
    )
    p.add_argument('--models', nargs='+',
                   default=['naive_physical',
                            'o_dependent_physical',
                            'fe_dependent_physical'],
                   help='Models to run')
    p.add_argument('--z-start', type=float, default=0.1,
                   help='Starting redshift (default: 0.1)')
    p.add_argument('--z-step', type=float, default=0.1,
                   help='Z-bin width (default: 0.1)')
    p.add_argument('--account', default='cough052',
                   help='SLURM account (default: cough052)')
    p.add_argument('--partition', default='agsmall',
                   help='SLURM partition (default: agsmall)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print jobs without submitting')
    p.add_argument('--store-obs-mode', default='none',
                   choices=['none', 'meta', 'diag', 'full'],
                   help='Observation storage mode (default: none)')
    p.add_argument('--cadences', nargs='+',
                   default=None,
                   help='Cadences to run (default: all 4)')
    args = p.parse_args()

    # Validate
    if not TEMPLATES_PKL.exists():
        sys.exit(f"ERROR: templates not found: {TEMPLATES_PKL}")

    # Run models sequentially (each model's bins are sequential)
    # Models themselves could run in parallel but we keep it simple —
    # sequential avoids any MSI queue pressure and is easy to monitor
    for model in args.models:
        pop_pkl = REPO / 'output/SLSNe/shared' / f'population_{model}.pkl'
        if not pop_pkl.exists():
            log(f"WARNING: population not found for {model} — skipping")
            continue
        run_adaptive(
            model=model,
            z_start=args.z_start,
            z_step=args.z_step,
            account=args.account,
            partition=args.partition,
            dry_run=args.dry_run,
            store_obs_mode=args.store_obs_mode,
            cadences=args.cadences,
        )

    log("\nAll models complete.")


if __name__ == '__main__':
    main()
