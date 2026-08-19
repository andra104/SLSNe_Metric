#!/bin/bash
# =============================================================================
# submit_adaptive_spectrigger.slurm
# Runs adaptive_efficiency.py for all 3 models, v5.3.0 only, 36-hour walltime.
# One job per model so they run in parallel — avoids sequential timeout.
# All 5 metrics run per bin (including fixed SpecTrigger brightness criterion).
# 36 hours provides buffer for queue latency on child jobs (each bin ~35 min
# including queue wait; naive needs ~48 bins = ~28 hrs worst case).
# =============================================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter
PARTITION=agsmall
ACCOUNT=cough052
CADENCE="baseline_v5.3.0_10yrs"

mkdir -p $REPO/output/SLSNe/logs

submit_model() {
    local MODEL=$1
    local JOB_ID=$(sbatch \
        --account=$ACCOUNT \
        --partition=$PARTITION \
        --ntasks=1 \
        --mem=16G \
        --time=36:00:00 \
        --job-name=adap_${MODEL:0:6} \
        --output=$REPO/output/SLSNe/logs/adaptive_${MODEL}_%j.out \
        --mail-type=END,FAIL \
        --mail-user=andra104@umn.edu \
        --wrap="
cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1
echo 'Model: $MODEL'
echo 'Cadence: $CADENCE'
echo 'Node: '\$(hostname)
echo 'Start: '\$(date)
python adaptive_efficiency.py \
    --models $MODEL \
    --cadences $CADENCE \
    --store-obs-mode none
echo 'End: '\$(date)
" | awk '{print $NF}')
    echo "  Submitted $MODEL: job $JOB_ID"
    echo $JOB_ID
}

echo "=== Submitting adaptive jobs (36hr walltime, v5.3.0 only) ==="
submit_model "naive_physical"
submit_model "o_dependent_physical"
submit_model "fe_dependent_physical"
echo "=== Done. Monitor: squeue -u andra104 ==="
