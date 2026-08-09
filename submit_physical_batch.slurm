#!/bin/bash
# =============================================================================
# submit_physical_batch.slurm — Submit all physical models x all cadences
#
# Structure: ONE job per model, all 4 cadences run in parallel within each job
#   Job 1: fe_dependent_physical x [baseline_v5.1.1, baseline_v5.3.0, noroll, poor_weather]
#   Job 2: o_dependent_physical  x [baseline_v5.1.1, baseline_v5.3.0, noroll, poor_weather]
#   Job 3: naive_physical        x [baseline_v5.1.1, baseline_v5.3.0, noroll, poor_weather]
#
# Physical track specifics:
#   - Uses physical_templates.pkl (MOSFiT posteriors, Gomez+2024)
#   - z=0.1-5.0 (full physical range, vs z=0.1-2.0 for GP)
#   - Physical metric variants auto-selected via _is_physical_track in runners.py
#   - MJD0 auto-read from each cadence db at runtime
#   - baseline_v5.3.0 MJD0=61208.2 handled correctly by auto-read
#   - Memory: 4 workers x ~4GB population load = ~16GB + MAF overhead → 64GB safe
#
# Usage:
#   bash submit_physical_batch.slurm
#
# Output per cadence in output/SLSNe/{model}/:
#   summary_{model}_{cadence}_z0.1-5.0_{YYMMDD}.csv
#   metric_values_detect_{model}_{cadence}_z0.1-5.0_{YYMMDD}.npy
#   metric_values_characterize_{model}_{cadence}_z0.1-5.0_{YYMMDD}.npy
#   metric_values_villar_{model}_{cadence}_z0.1-5.0_{YYMMDD}.npy
#   metric_values_spectrigger_{model}_{cadence}_z0.1-5.0_{YYMMDD}.npy
#   metric_values_elasticc_{model}_{cadence}_z0.1-5.0_{YYMMDD}.npy
# =============================================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
TEMPLATES=$REPO/output/SLSNe/shared/physical_templates.pkl
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter

PARTITION=agsmall
ACCOUNT=cough052
MEM=64G
TIME=24:00:00
CPUS=4
N_WORKERS=4

CADENCES="baseline_v5.3.0_10yrs"

mkdir -p $REPO/output/logs/pipeline

# ---------------------------------------------------------------------------
# Pre-flight check: verify metrics in runners.py _name_map match instantiated
# Catches mismatches before burning compute hours on misnamed/missing .npy files
# ---------------------------------------------------------------------------
echo "============================================================"
echo "PRE-FLIGHT METRIC ALIGNMENT CHECK"
echo "============================================================"

python3 - << 'PYCHECK'
import re, sys
txt = open('py_files/slsn_metrics/runners.py').read()
name_map_hits  = re.findall(r"'(SLSN_[^']+)'\s*:\s*'([^']+)'", txt)
name_map       = dict(name_map_hits)
mapped_classes = set(name_map.keys())
mapped_shorts  = set(name_map.values())
instantiated   = set(re.findall(r'(SLSN_\w+Metric)\(lc_model=', txt))
print(f"  Metrics instantiated in runners.py : {sorted(instantiated)}")
print(f"  Metrics in _name_map               : {sorted(mapped_classes)}")
print(f"  Output .npy short names            : {sorted(mapped_shorts)}")
missing_from_map  = instantiated - mapped_classes
missing_from_list = mapped_classes - instantiated
ok = True
if missing_from_map:
    print(f"  WARNING: instantiated but not in _name_map: {missing_from_map}")
    ok = False
if missing_from_list:
    print(f"  WARNING: in _name_map but not instantiated: {missing_from_list}")
    ok = False
if ok:
    print(f"  STATUS: OK - all {len(instantiated)} metrics aligned")
else:
    print(f"  STATUS: FAILED - fix runners.py before submitting")
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "Pre-flight check failed. Aborting submission."
    exit 1
fi
echo "============================================================"


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

    local LOG=$REPO/output/logs/pipeline/${MODEL}_%j.out
    local ERR=$REPO/output/logs/pipeline/${MODEL}_%j.err

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
    --z-min 0.1 \
    --z-max 5.0 \
    --max-events 1000000 \
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
FE_JOB=$(submit_job "fe_dependent_physical" "")

# ---------------------------------------------------------------------------
# Priority 2: o_dependent — no dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 2: o_dependent ---"
OD_JOB=$(submit_job "o_dependent_physical" "")

# ---------------------------------------------------------------------------
# Priority 3: naive — no dependencies, runs simultaneously
# ---------------------------------------------------------------------------
echo ""
echo "--- Priority 3: naive ---"
NAIVE_JOB=$(submit_job "naive_physical" "")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SUBMITTED"
echo "  fe_dependent : $FE_JOB"
echo "  o_dependent  : $OD_JOB"
echo "  naive        : $NAIVE_JOB"
echo ""
echo "Monitor:"
echo "  squeue -u andra104"
echo "  tail -f output/logs/pipeline/fe_dependent_JOBID.out"
echo "============================================================"
