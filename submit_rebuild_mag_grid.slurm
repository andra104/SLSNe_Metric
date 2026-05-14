#!/bin/bash
# =============================================================================
# submit_rebuild_mag_grid.slurm — Rebuild physical mag grid only
#
# Deletes corrupt physical_mag_grid.pkl (built before calc_mag fix — all NaN)
# and rebuilds using build_physical_bundle.py --skip-populations
#
#   Step 1 : Load physical_templates.pkl
#   Step 2 : Rebuild physical magnitude grid (z=0.02-5.0, 100pts; phase=1-400d, 79pts)
#   Step 3 : SKIPPED (populations already exist and are correct)
#
# Physical templates must exist before submitting this job.
# Build them first:
#   sbatch submit_physical_templates.slurm
#   — or —
#   Run prototype_physical_templates.ipynb with REBUILD_PHYSICAL_TEMPLATES=True
#
# Usage:
#   sbatch submit_physical_bundle.slurm
#
# Output (all written to output/SLSNe/shared/):
#   physical_mag_grid.pkl
#   population_fe_dependent_physical.pkl
#   population_o_dependent_physical.pkl
#   population_naive_physical.pkl
#
# Checkpointing:
#   Mag grid checkpoints every 50 templates to physical_mag_grid.checkpoint.pkl
#   Re-submit this job to resume if interrupted — build_physical_bundle.py
#   detects the checkpoint automatically.
# =============================================================================

#SBATCH --job-name=rebuild_mag_grid
#SBATCH --account=cough052
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=output/logs/build/rebuild_mag_grid_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andra104@umn.edu

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter

mkdir -p $REPO/output/logs/build

# ---------------------------------------------------------------------------
# Pre-flight check: verify physical_templates.pkl exists before any computation.
# Exits immediately if missing rather than burning 12 hours of SLURM time.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "PRE-FLIGHT CHECK — physical_templates.pkl"
echo "============================================================"

python3 - << 'PYCHECK'
import sys
from pathlib import Path

REPO     = Path('/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric')
PKL_PATH = REPO / 'output' / 'SLSNe' / 'shared' / 'physical_templates.pkl'

print(f"  Checking: {PKL_PATH}")

if not PKL_PATH.exists():
    print(f"  STATUS: FAILED")
    print(f"  physical_templates.pkl not found.")
    print(f"  Build it first:")
    print(f"    sbatch submit_physical_templates.slurm")
    print(f"    — or —")
    print(f"    Run prototype_physical_templates.ipynb with REBUILD_PHYSICAL_TEMPLATES=True")
    sys.exit(1)

try:
    import joblib
    payload  = joblib.load(PKL_PATH)
    n_events = len(payload.get('names', []))
    has_sed  = bool(payload.get('sed_grid'))
    print(f"  Events   : {n_events}")
    print(f"  sed_grid : {'present' if has_sed else 'MISSING — rebuild templates'}")
    if not has_sed:
        print(f"  STATUS: FAILED — sed_grid is empty")
        sys.exit(1)
    print(f"  STATUS: OK")
except Exception as e:
    print(f"  STATUS: FAILED — could not load pkl: {e}")
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "Pre-flight check failed. Aborting job."
    exit 1
fi
echo "============================================================"


cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1

echo 'Node    : '$(hostname)
echo 'Start   : '$(date)

# Delete corrupt mag grid (built before calc_mag fix, all NaN)
# build_physical_bundle.py will rebuild automatically when file is missing
echo "Deleting corrupt physical_mag_grid.pkl (all-NaN, built before calc_mag fix)"
rm -f $REPO/output/SLSNe/shared/physical_mag_grid.pkl
rm -f $REPO/output/SLSNe/shared/physical_mag_grid.checkpoint.pkl
echo "Rebuilding mag grid only (skipping populations)"
python3 build_physical_bundle.py --skip-populations

echo 'End     : '$(date)
