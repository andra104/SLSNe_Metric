#!/bin/bash
#SBATCH --job-name=slsn_phys_validate
#SBATCH --account=cough052
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=output/SLSNe/logs/slsn_%j_%x.out
#SBATCH --error=output/SLSNe/logs/slsn_%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andra104@umn.edu

# ============================================================
# SLSN Pipeline — SLURM submission script
# Runs one (model, cadence) combination end-to-end.
#
# Usage:
#   sbatch submit_slsn_pipeline.slurm
#
# To change model or cadence, edit MODEL and CADENCE below.
# To run all combinations, use launch_all_slurm.sh instead.
# ============================================================

# --- Configuration ---
MODEL="fe_dependent_physical"           # naive | fe_dependent | o_dependent | *_physical variants
CADENCE="baseline_v5.1.0_10yrs"
TEMPLATES="output/SLSNe/shared/physical_templates.pkl"

# --- Environment ---
cd /users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric

# Initialize conda properly for non-interactive shell
source /home/compsci/conda/etc/profile.d/conda.sh 2>/dev/null || \
source /panfs/jay/groups/0/cough052/shared/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(/home/compsci/conda/bin/conda shell.bash hook 2>/dev/null)"
conda activate rubin_sim_2.6.1

echo "============================================================"
echo "SLSN Pipeline SLURM Job"
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Model        : $MODEL"
echo "Cadence      : $CADENCE"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Memory       : 16G"
echo "Start time   : $(date)"
echo "============================================================"

# --- Create log directory ---
mkdir -p output/SLSNe/logs

# --- Run pipeline ---
python3 run_slsn_pipeline.py \
    --model        $MODEL \
    --cadences      $CADENCE \
    --templates-pkl $TEMPLATES \
    --z-max        5.0 \
    --max-events   10000 \
    --only-metrics detect \
    --n-cores      $SLURM_CPUS_PER_TASK

EXIT_CODE=$?

echo "============================================================"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
