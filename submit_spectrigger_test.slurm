#!/bin/bash
#SBATCH --job-name=slsn_spec_test
#SBATCH --account=cough052
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/SLSNe/logs/slsn_spec_test_%j.out
#SBATCH --error=output/SLSNe/logs/slsn_spec_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andra104@umn.edu

# ============================================================
# SpecTrigger test run — validate brightness fix before
# committing to full production run.
# One cadence (v5.3.0), all 3 models, all 5 metrics,
# 1000 events each, z=0.1-1.0 (where SpecTrigger is nonzero)
# ============================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
TEMPLATES=$REPO/output/SLSNe/shared/physical_templates.pkl
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter

cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1

echo "==== SpecTrigger Validation ===="
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "================================"

mkdir -p output/SLSNe/logs

for MODEL in naive_physical o_dependent_physical fe_dependent_physical; do
    echo ""
    echo "--- Model: $MODEL ---"
    python3 run_slsn_pipeline.py \
        --model $MODEL \
        --cadences baseline_v5.3.0_10yrs \
        --templates-pkl $TEMPLATES \
        --z-min 0.1 \
        --z-max 1.0 \
        --max-events 1000 \
        --store-obs-mode none \
        --n-workers 4 \
        --n-cores $SLURM_CPUS_PER_TASK

    EXIT=$?
    echo "  Exit code: $EXIT"
    if [ $EXIT -ne 0 ]; then
        echo "  FAILED: $MODEL — aborting"
        exit $EXIT
    fi
done

echo ""
echo "==== Results summary ===="
for MODEL in naive_physical o_dependent_physical fe_dependent_physical; do
    echo "--- $MODEL ---"
    ls -lt $REPO/output/SLSNe/${MODEL}/summary_${MODEL}_baseline_v5.3.0_10yrs_z0.1-1.0_*.csv 2>/dev/null | head -1
    LATEST=$(ls -t $REPO/output/SLSNe/${MODEL}/summary_${MODEL}_baseline_v5.3.0_10yrs_z0.1-1.0_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        grep "spectrigger\|SpecTrigger" $LATEST || echo "  No SpecTrigger row found"
    fi
done

echo ""
echo "End : $(date)"
