#!/bin/bash
# Step 2: per-pixel phenology (int16 tiles, dates sidecars); the year's tiles are split over array tasks.
# From crop_phenology_zarr/ (run 01_inventory first; tiles with the current params_hash and inputs are skipped):
#   CPZ_PROVINCE=SK sbatch --array=0-4 jobs/02_phenology.sbatch --year 2019   (~126 tiles per task, 32 workers)
#   sbatch --cpus-per-task=9 jobs/02_phenology.sbatch --area tile_shift_2019   (or orbit_stripe_2024, zone_edge_2022)
# Province: CPZ_PROVINCE=SK|MB|BC|AB (default SK); inventory/fill cover CPZ_ACTIVE (default SK,MB). sbatch passes the env.
#SBATCH --job-name=cpz-pheno
#SBATCH --account=def-kanj241_cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7000M
#SBATCH --output=logs/02_phenology-%A_%a.out
module load StdEnv/2023 python/3.11
source "$HOME/envs/cropclassify/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$SCRATCH/crop_phenology_zarr/numba_cache"
cd "$SLURM_SUBMIT_DIR"
python -u 02_phenology.py --workers "$SLURM_CPUS_PER_TASK" "$@"
