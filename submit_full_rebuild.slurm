#!/bin/bash
# =============================================================================
# submit_full_rebuild.slurm — Full physical track rebuild chain
#
# Sequence:
#   Job 1: build_physical_bundle.py --regen-templates --regen-population
#           Rebuilds templates, mag grid, and all 3 populations
#   Jobs 2-4: submit_physical_batch.slurm (3 model jobs, after Job 1)
#           Runs MAF metrics for all models x all cadences
#   Job 5: adaptive_efficiency.py wrapper (after Jobs 2-4)
#           Runs adaptive z-bin efficiency stopping loop
#
# Usage:
#   bash submit_full_rebuild.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f output/logs/build/phys_bundle_JOBID.log
# =============================================================================

REPO=/users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric
CONDA_BASE=/common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter
ACCOUNT=cough052
PARTITION=agsmall

mkdir -p $REPO/output/logs/build
mkdir -p $REPO/output/logs/pipeline
mkdir -p $REPO/output/SLSNe/logs

# Delete old SED cache — built with wrong _flam_to_fnu_jy_at_10pc units
# Must be removed so build_physical_bundle.py rebuilds from scratch
echo "Deleting old physical_sed_cache.pkl (built with wrong unit conversion)"
rm -f $REPO/output/SLSNe/shared/physical_sed_cache.pkl
echo "Deleted."

echo "============================================================"
echo "FULL PHYSICAL TRACK REBUILD"
echo "============================================================"
echo "Repo: $REPO"
echo "Start: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Job 1: Build templates + mag grid + populations
# ---------------------------------------------------------------------------
BUNDLE_JOB=$(sbatch \
    --job-name=phys_bundle \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=16:00:00 \
    --output=$REPO/output/logs/build/phys_bundle_%j.log \
    --mail-type=END,FAIL \
    --mail-user=andra104@umn.edu \
    --wrap="
cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1
echo 'Node  : '\$(hostname)
echo 'Start : '\$(date)
python3 build_physical_bundle.py --regen-templates --regen-population
echo 'End   : '\$(date)
" | awk '{print $NF}')

echo "Job 1 (bundle): $BUNDLE_JOB"

# ---------------------------------------------------------------------------
# Jobs 2-4: MAF metrics — one job per model, all cadences in parallel
# depends on Job 1 completing successfully
# ---------------------------------------------------------------------------
MODELS="fe_dependent_physical o_dependent_physical naive_physical"
CADENCES="baseline_v5.1.1_10yrs baseline_v5.3.0_10yrs noroll_v5.0.0_10yrs poor_weather_v5.0.1_10yrs"
TEMPLATES=$REPO/output/SLSNe/shared/physical_templates.pkl
N_WORKERS=4
CPUS=4
MEM=64G

METRIC_JOBS=""
for MODEL in $MODELS; do
    JOB=$(sbatch \
        --job-name=slsn_${MODEL:0:4} \
        --account=$ACCOUNT \
        --partition=$PARTITION \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --time=24:00:00 \
        --output=$REPO/output/logs/pipeline/${MODEL}_%j.out \
        --error=$REPO/output/logs/pipeline/${MODEL}_%j.err \
        --mail-type=END,FAIL \
        --mail-user=andra104@umn.edu \
        --dependency=afterok:$BUNDLE_JOB \
        --wrap="
cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1
echo 'Node  : '\$(hostname)
echo 'Start : '\$(date)
python3 run_slsn_pipeline.py \
    --model $MODEL \
    --cadences $CADENCES \
    --templates-pkl $TEMPLATES \
    --z-min 0.1 \
    --z-max 5.0 \
    --store-obs-mode none \
    --n-workers $N_WORKERS
echo 'End   : '\$(date)
" | awk '{print $NF}')
    echo "Job (metrics $MODEL): $JOB"
    if [ -z "$METRIC_JOBS" ]; then
        METRIC_JOBS="$JOB"
    else
        METRIC_JOBS="$METRIC_JOBS:$JOB"
    fi
done

# ---------------------------------------------------------------------------
# Job 5: Adaptive efficiency — after ALL metric jobs complete
# ---------------------------------------------------------------------------
ADAPTIVE_JOB=$(sbatch \
    --job-name=adaptive \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=8G \
    --time=72:00:00 \
    --output=$REPO/output/SLSNe/logs/adaptive_wrapper_%j.out \
    --mail-type=END,FAIL \
    --mail-user=andra104@umn.edu \
    --dependency=afterok:$METRIC_JOBS \
    --wrap="
cd $REPO
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1
echo 'Node  : '\$(hostname)
echo 'Start : '\$(date)
python3 adaptive_efficiency.py \
    --models naive_physical o_dependent_physical fe_dependent_physical \
    --z-start 0.1 \
    --z-step 0.1
echo 'End   : '\$(date)
" | awk '{print $NF}')

echo "Job 5 (adaptive): $ADAPTIVE_JOB"

echo "============================================================"
echo "ALL JOBS SUBMITTED"
echo "  Job 1 bundle:   $BUNDLE_JOB"
echo "  Jobs 2-4 metrics: $METRIC_JOBS"
echo "  Job 5 adaptive: $ADAPTIVE_JOB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Bundle log: tail -f output/logs/build/phys_bundle_${BUNDLE_JOB}.log"
echo "============================================================"
