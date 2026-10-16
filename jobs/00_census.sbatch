#!/bin/bash
# Step 0: STAC item census of MPC sentinel-2-l2a over the ROI (metadata only, needs internet). From crop_phenology_zarr/:
#   sbatch jobs/00_census.sbatch --years 2019 2022 2024
# Province: CPZ_PROVINCE=SK|MB|BC|AB (default SK); inventory/fill cover CPZ_ACTIVE (default SK,MB). sbatch passes the env.
#SBATCH --job-name=cpz-census
#SBATCH --account=def-kanj241_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --output=logs/00_census-%A_%a.out
module load StdEnv/2023 python/3.11
source "$HOME/envs/cropclassify/bin/activate"
cd "$SLURM_SUBMIT_DIR"
python -u 00_census.py "$@"
