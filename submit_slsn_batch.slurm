#!/bin/bash
# =============================================================================
# submit_slsn_batch.slurm — Submit all models x all cadences
#
# Structure: ONE job per model, all 3 cadences run in parallel within each job
#   Job 1: fe_dependent x [baseline, four_roll, noroll]  — primary science
#   Job 2: o_dependent  x [baseline, four_roll, noroll]  — completeness
#   Job 3: naive        x [baseline, four_roll, noroll]  — reference (runs last)
#
# Speedup: 3 cadences run simultaneously per job (~13 hr vs ~40 hr sequential)
# Memory:  3 workers x ~600MB population = ~1.8GB + MAF overhead → 128GB safe
#
# Usage:
#   bash submit_slsn_batch.slurm
#
# Output per cadence in output/SLSNe/{model}/:
#   summary_{model}_{cadence}_z0.1-2.0_{YYMMDD}.csv
#   metric_values_detect_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
#   metric_values_characterize_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
#   metric_values_spectrigger_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
# =============================================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
TEMPLATES=$REPO/output/SLSNe/shared/templates.pkl
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter

PARTITION=agsmall
ACCOUNT=cough052
MEM=128G
TIME=24:00:00
CPUS=4
N_WORKERS=3

CADENCES="four_roll_v5.0.0_10yrs noroll_v5.0.0_10yrs poor_weather_v5.0.1_10yrs"

mkdir -p $REPO/output/SLSNe/logs

echo "============================================================"
echo "SLSN BATCH LAUNCHER"
echo "============================================================"
echo "Repo      : $REPO"
echo "Partition : $PARTITION / $ACCOUNT"
echo "Resources : $CPUS CPUs, $MEM RAM, $TIME"
echo "Cadences  : $CADENCES"
echo "Workers   : $N_WORKERS (one per cadence)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: submit one job for one model x all cadences
# ---------------------------------------------------------------------------
submit_job() {
    local MODEL=$1
    local DEPENDENCY=$2

    local LOG=$REPO/output/SLSNe/logs/${MODEL}_%j.out
    local ERR=$REPO/output/SLSNe/logs/${MODEL}_%j.err

    local DEP_FLAG=""
    if [ -n "$DEPENDENCY" ]; then
        DEP_FLAG="--dependency=afterok:$DEPENDENCY"
    fi

    local JOB_ID=$(sbatch \
        --job-name=slsn_${MODEL:0:4} \
        --account=$ACCOUNT \
        --partition=$PARTITION \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --time=$TIME \
        --output=$LOG \
        --error=$ERR \
        --mail-type=END,FAIL \
        --mail-user=andra104@umn.edu \
        $DEP_FLAG \
        --wrap="
cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1
echo '==== SLSN Pipeline ===='
echo 'Model   : $MODEL'
echo 'Cadences: $CADENCES'
echo 'Workers : $N_WORKERS'
echo 'Node    : '\$(hostname)
echo 'Start   : '\$(date)
echo '======================'
python3 run_slsn_pipeline.py \
    --model $MODEL \
    --cadences $CADENCES \
    --templates-pkl $TEMPLATES \
    --store-obs-mode none \
    --n-cores $CPUS \
    --n-workers $N_WORKERS
echo 'End: '\$(date)
" | awk '{print $NF}')

    echo "  [job $JOB_ID] $MODEL x all cadences${DEPENDENCY:+ (after $DEPENDENCY)}"
    echo $JOB_ID
}

# ---------------------------------------------------------------------------
# Priority 1: fe_dependent — primary science, no dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 1: fe_dependent ---"
FE_JOB=$(submit_job "fe_dependent" "")

# ---------------------------------------------------------------------------
# Priority 2: o_dependent — no dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 2: o_dependent ---"
OD_JOB=$(submit_job "o_dependent" "")

# ---------------------------------------------------------------------------
# Priority 3: naive — runs after fe + o complete
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 3: naive (after fe_dependent + o_dependent) ---"
NAIVE_JOB=$(submit_job "naive" "${FE_JOB}:${OD_JOB}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SUBMITTED"
echo "  fe_dependent : $FE_JOB"
echo "  o_dependent  : $OD_JOB"
echo "  naive        : $NAIVE_JOB (depends on $FE_JOB + $OD_JOB)"
echo ""
echo "Monitor:"
echo "  squeue -u andra104"
echo "  tail -f output/SLSNe/logs/fe_dependent_JOBID.out"
echo "============================================================"
