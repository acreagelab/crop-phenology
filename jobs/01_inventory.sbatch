#!/bin/bash
# Step 1: S2 store inventory and date-level audit against the census. From crop_phenology_zarr/:
#   sbatch jobs/01_inventory.sbatch [--years 2019 2024] [--masks]
# Province: CPZ_PROVINCE=SK|MB|BC|AB (default SK); inventory/fill cover CPZ_ACTIVE (default SK,MB). sbatch passes the env.
#SBATCH --job-name=cpz-inv
#SBATCH --account=def-kanj241_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --output=logs/01_inventory-%A_%a.out
module load StdEnv/2023 python/3.11
source "$HOME/envs/cropclassify/bin/activate"
export NUMBA_CACHE_DIR="$SCRATCH/crop_phenology_zarr/numba_cache"
cd "$SLURM_SUBMIT_DIR"
python -u 01_inventory.py "$@"
