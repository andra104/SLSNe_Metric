#!/bin/bash
# =============================================================================
# submit_slsn_batch.slurm — Submit all (model x cadence) SLSN pipeline jobs
#
# Priority order:
#   1. fe_dependent x all cadences  (primary science result)
#   2. o_dependent  x all cadences  (completeness check)
#   3. naive        x all cadences  (reference model, largest — runs last)
#
# Usage:
#   bash submit_slsn_batch.slurm
#
# Output files per job:
#   output/SLSNe/{model}/
#       summary_{model}_{cadence}_z0.1-2.0_{YYMMDD}.csv
#       metric_values_Detect_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
#       metric_values_Characterize_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
#       metric_values_SpecTrigger_{model}_{cadence}_z0.1-2.0_{YYMMDD}.npy
#   output/SLSNe/logs/
#       {model}_{cadence}_{jobid}.out
#       {model}_{cadence}_{jobid}.err
# =============================================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
TEMPLATES=$REPO/output/SLSNe/shared/templates.pkl
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter

PARTITION=agsmall
ACCOUNT=cough052
MEM=64G
TIME=24:00:00
CPUS=4

# Priority cadences
CADENCES=(
    "baseline_v5.1.1_10yrs"
    "four_roll_v5.0.0_10yrs"
    "noroll_v5.0.0_10yrs"
)

mkdir -p $REPO/output/SLSNe/logs

echo "============================================================"
echo "SLSN BATCH LAUNCHER"
echo "============================================================"
echo "Repo      : $REPO"
echo "Partition : $PARTITION / $ACCOUNT"
echo "Resources : $CPUS CPUs, $MEM RAM, $TIME"
echo "Cadences  : ${CADENCES[@]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: submit one job, print and return job ID
# ---------------------------------------------------------------------------
submit_job() {
    local MODEL=$1
    local CADENCE=$2
    local DEPENDENCY=$3

    local LOG=$REPO/output/SLSNe/logs/${MODEL}_${CADENCE}_%j.out
    local ERR=$REPO/output/SLSNe/logs/${MODEL}_${CADENCE}_%j.err

    local DEP_FLAG=""
    if [ -n "$DEPENDENCY" ]; then
        DEP_FLAG="--dependency=afterok:$DEPENDENCY"
    fi

    local JOB_ID=$(sbatch \
        --job-name=slsn_${MODEL:0:2}_${CADENCE:0:8} \
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
echo 'Cadence : $CADENCE'
echo 'Node    : '\$(hostname)
echo 'Start   : '\$(date)
echo '======================'
python3 run_slsn_pipeline.py \
    --model $MODEL \
    --cadence $CADENCE \
    --templates-pkl $TEMPLATES \
    --store-obs-mode none \
    --n-cores $CPUS \
    --n-workers $CPUS
echo 'End: '\$(date)
" | awk '{print $NF}')

    echo "  [job $JOB_ID] $MODEL x $CADENCE${DEPENDENCY:+ (after $DEPENDENCY)}"
    echo $JOB_ID
}

# ---------------------------------------------------------------------------
# Priority 1: fe_dependent — primary science result, no dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 1: fe_dependent ---"
FE_JOBS=()
for CADENCE in "${CADENCES[@]}"; do
    JOB_ID=$(submit_job "fe_dependent" "$CADENCE" "")
    FE_JOBS+=($JOB_ID)
done

# ---------------------------------------------------------------------------
# Priority 2: o_dependent — completeness check, no dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 2: o_dependent ---"
OD_JOBS=()
for CADENCE in "${CADENCES[@]}"; do
    JOB_ID=$(submit_job "o_dependent" "$CADENCE" "")
    OD_JOBS+=($JOB_ID)
done

# ---------------------------------------------------------------------------
# Priority 3: naive — largest population, runs after fe + o finish
# ---------------------------------------------------------------------------
ALL_PREV=$(printf "%s:" "${FE_JOBS[@]}" "${OD_JOBS[@]}")
ALL_PREV=${ALL_PREV%:}   # strip trailing colon

echo ""
echo "--- Priority 3: naive (after fe_dependent + o_dependent) ---"
NAIVE_JOBS=()
for CADENCE in "${CADENCES[@]}"; do
    JOB_ID=$(submit_job "naive" "$CADENCE" "$ALL_PREV")
    NAIVE_JOBS+=($JOB_ID)
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SUBMITTED"
echo "  fe_dependent : ${FE_JOBS[@]}"
echo "  o_dependent  : ${OD_JOBS[@]}"
echo "  naive        : ${NAIVE_JOBS[@]}"
echo ""
echo "Monitor:"
echo "  squeue -u andra104"
echo "  tail -f output/SLSNe/logs/<model>_<cadence>_<jobid>.out"
echo "============================================================"
