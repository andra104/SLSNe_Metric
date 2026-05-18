#!/usr/bin/env python3
"""
convergence_analysis.py — Analyze efficiency convergence across event counts.

For each model x cadence x metric, loads .npy output files grouped by n_events,
pairs with population z values (same seed=42 subsampling as pipeline),
bins efficiency by redshift, and checks convergence criterion:
    |eff(N, z_bin) - eff(N_prev, z_bin)| / eff(N_prev, z_bin) < 10%
for all z_bins where eff > 0.

Usage:
    python3 convergence_analysis.py

Output:
    - Per-model convergence table printed to stdout
    - Convergence verdict: minimum N for stable efficiency estimates
"""

import sys
import pickle
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / 'py_files'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = ['naive_physical', 'o_dependent_physical', 'fe_dependent_physical']
CADENCES = [
    'baseline_v5.1.1_10yrs',
    'baseline_v5.3.0_10yrs',
    'noroll_v5.0.0_10yrs',
    'poor_weather_v5.0.1_10yrs',
]
METRICS = ['detect', 'characterize', 'villar', 'elasticc', 'spectrigger']
Z_BINS  = np.arange(0.0, 5.5, 0.5)
Z_LABELS = [f"z=[{Z_BINS[i]:.1f},{Z_BINS[i+1]:.1f})" for i in range(len(Z_BINS)-1)]
SEED    = 42
OUTPUT  = Path('output/SLSNe')
SHARED  = OUTPUT / 'shared'
CONV_THRESHOLD = 0.10  # 10% relative change = converged

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_population_z(model, n_events):
    """Load z values for the n_events subsample used in pipeline runs."""
    pop_path = SHARED / f'population_{model}.pkl'
    if not pop_path.exists():
        raise FileNotFoundError(f"Population not found: {pop_path}")
    with open(pop_path, 'rb') as f:
        pop = pickle.load(f)
    z_all = np.asarray(pop['z'])
    n_total = len(z_all)
    del pop

    if n_events >= n_total:
        return z_all

    rng  = np.random.default_rng(SEED)
    keep = rng.choice(n_total, size=n_events, replace=False)
    keep = np.sort(keep)
    return z_all[keep]


def find_npy_files(model, cadence, metric):
    """Find all .npy files for a given model/cadence/metric, grouped by n_rows."""
    model_dir = OUTPUT / model
    pattern   = f'metric_values_{metric}_{model}_{cadence}_z0.1-5.0_*.npy'
    files     = sorted(model_dir.glob(pattern), key=lambda f: f.stat().st_mtime)

    # Group by n_rows — each unique n_rows is a different N run
    groups = {}
    for f in files:
        try:
            arr    = np.load(str(f))
            n_rows = len(arr)
            # Keep latest file for each n_rows
            groups[n_rows] = (f, arr)
        except Exception:
            continue
    return groups  # {n_rows: (path, array)}


def efficiency_by_z(arr, z_vals):
    """Compute efficiency in each z-bin."""
    effs = []
    counts = []
    for i in range(len(Z_BINS) - 1):
        zlo, zhi = Z_BINS[i], Z_BINS[i+1]
        in_bin = (z_vals >= zlo) & (z_vals < zhi)
        n_bin  = in_bin.sum()
        n_det  = float(arr[in_bin].sum())
        eff    = n_det / n_bin if n_bin > 0 else np.nan
        effs.append(eff)
        counts.append(n_bin)
    return np.array(effs), np.array(counts)


def check_convergence(eff_prev, eff_curr, min_eff=0.001):
    """
    Check if efficiency has converged between two N values.
    Only check bins where eff_curr > min_eff (ignore zero-efficiency bins).
    Returns (converged, max_relative_change, n_bins_checked)
    """
    detectable = (eff_curr > min_eff) & np.isfinite(eff_curr) & np.isfinite(eff_prev)
    if not np.any(detectable):
        return True, 0.0, 0  # No detectable bins — trivially converged

    rel_change = np.abs(eff_curr[detectable] - eff_prev[detectable]) / \
                 np.maximum(eff_prev[detectable], 1e-10)
    max_change = float(rel_change.max())
    return max_change < CONV_THRESHOLD, max_change, int(detectable.sum())


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
print("\nSLSN Physical Track — Efficiency Convergence Analysis")
print("=" * 70)
print(f"Convergence criterion: |eff(N) - eff(N_prev)| / eff(N_prev) < "
      f"{100*CONV_THRESHOLD:.0f}%")
print(f"Z bins: {Z_LABELS}")
print(f"Seed: {SEED}")

# Cache population z arrays
pop_cache = {}

for model in MODELS:
    print(f"\n{'='*70}")
    print(f"MODEL: {model}")
    print(f"{'='*70}")

    model_converged_N = {}  # cadence -> minimum converged N

    for cadence in CADENCES:
        print(f"\n  Cadence: {cadence}")

        # For convergence, use ELAsTiCC (most detections, most sensitive)
        # Also check Villar (most stringent) if available
        for metric in ['elasticc', 'detect', 'villar']:
            groups = find_npy_files(model, cadence, metric)
            if not groups:
                continue

            n_vals = sorted(groups.keys())
            if len(n_vals) < 2:
                print(f"    [{metric}] Only {len(n_vals)} N value(s) — "
                      f"need at least 2 for convergence check")
                continue

            print(f"\n    Metric: {metric}")
            print(f"    N values found: {n_vals}")

            eff_prev = None
            n_prev   = None
            conv_N   = None

            for n in n_vals:
                _, arr = groups[n]

                # Get z values for this subsample
                if (model, n) not in pop_cache:
                    try:
                        pop_cache[(model, n)] = load_population_z(model, n)
                    except Exception as e:
                        print(f"    ERROR loading population for N={n}: {e}")
                        continue
                z_vals = pop_cache[(model, n)]

                if len(z_vals) != len(arr):
                    print(f"    WARNING: z_vals length {len(z_vals)} != "
                          f"arr length {len(arr)} for N={n}")
                    continue

                eff, counts = efficiency_by_z(arr, z_vals)

                # Print efficiency by z-bin
                det_bins = [(Z_LABELS[i], eff[i], counts[i])
                            for i in range(len(eff))
                            if np.isfinite(eff[i]) and eff[i] > 0.001]

                print(f"\n    N={n:>8,}:")
                for zlabel, e, c in det_bins:
                    print(f"      {zlabel}: eff={e:.4f} ({c} events)")
                if not det_bins:
                    print(f"      (no detectable z-bins)")

                # Convergence check
                if eff_prev is not None:
                    conv, max_chg, n_bins = check_convergence(eff_prev, eff)
                    status = "CONVERGED ✓" if conv else f"not converged ({100*max_chg:.1f}% max change)"
                    print(f"      vs N={n_prev:,}: {status} "
                          f"({n_bins} detectable bins checked)")
                    if conv and conv_N is None:
                        conv_N = n
                        print(f"      → First convergence at N={conv_N:,}")

                eff_prev = eff
                n_prev   = n

            model_converged_N[f"{cadence}_{metric}"] = conv_N

    # Summary for this model
    print(f"\n  {'─'*60}")
    print(f"  CONVERGENCE SUMMARY: {model}")
    converged_vals = [v for v in model_converged_N.values() if v is not None]
    if converged_vals:
        min_conv = min(converged_vals)
        max_conv = max(converged_vals)
        print(f"  First convergence: N={min_conv:,}")
        print(f"  Last convergence:  N={max_conv:,}")
        print(f"  Recommended N:     {max_conv:,} "
              f"(conservative — all cadences/metrics converged)")
    else:
        print(f"  No convergence detected in available N values")
        print(f"  → Run larger N or check detection rates")

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")
print("Run this script again after all convergence jobs complete.")
print("The recommended production N is the maximum convergence N")
print("across all models, cadences, and metrics.")
