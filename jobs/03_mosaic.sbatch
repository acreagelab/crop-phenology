#!/bin/bash
# Step 3: mosaic one year or a test area. From crop_phenology_zarr/:
#   CPZ_PROVINCE=MB sbatch jobs/03_mosaic.sbatch --year 2019   |   sbatch jobs/03_mosaic.sbatch --area tile_shift_2019
#   add --allow-missing to mosaic with absent step-2 tiles (listed in mosaic_qc_<year>.json)
# Province: CPZ_PROVINCE=SK|MB|BC|AB (default SK); inventory/fill cover CPZ_ACTIVE (default SK,MB). sbatch passes the env.
#SBATCH --job-name=cpz-mosaic
#SBATCH --account=def-kanj241_cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --output=logs/03_mosaic-%j.out
module load StdEnv/2023 python/3.11
source "$HOME/envs/cropclassify/bin/activate"
export NUMBA_CACHE_DIR="$SCRATCH/crop_phenology_zarr/numba_cache"
cd "$SLURM_SUBMIT_DIR"
python -u 03_mosaic.py --threads "$SLURM_CPUS_PER_TASK" "$@"
