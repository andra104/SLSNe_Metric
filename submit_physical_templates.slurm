#!/bin/bash
#SBATCH --job-name=phys_mag_grid
#SBATCH --account=cough052
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=output/logs/build/phys_mag_grid_%j.log

source ~/.bashrc
conda activate rubin_sim_2.6.1
cd /users/1/andra104/Documents/Research/Transient_Metrics/SLSNe_Metric

jupyter nbconvert --to notebook --execute \
    notebooks/prototype_physical_templates.ipynb \
    --output notebooks/prototype_physical_templates_executed.ipynb \
    --ExecutePreprocessor.timeout=10800
