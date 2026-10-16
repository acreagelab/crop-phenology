#!/bin/bash
# Step 1b: re-download the fill_<year>.csv dates into $SCRATCH/crop_phenology_zarr/fill (gaig-stac venv).
# Needs internet (MPC STAC + blob reads): compute nodes of fir reach MPC (earlier gaig-stac runs did).
# The year's fill tiles are split over the array tasks. From crop_phenology_zarr/:
#   sbatch jobs/01b_fill.sbatch --year 2019 --area tile_shift_2019
#   sbatch --array=0-9 jobs/01b_fill.sbatch --year 2019
# Province: CPZ_PROVINCE=SK|MB|BC|AB (default SK); inventory/fill cover CPZ_ACTIVE (default SK,MB). sbatch passes the env.
#SBATCH --job-name=cpz-fill
#SBATCH --account=def-kanj241_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --output=logs/01b_fill-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
GAIG_PY=${CPZ_GAIG_PY:-/project/def-kanj241/hvt632/gaig_stack/.venv/bin/python}
curl -sS -m 20 -o /dev/null -w 'mpc stac http %{http_code}\n' https://planetarycomputer.microsoft.com/api/stac/v1
"$GAIG_PY" -u 01b_fill.py --tile-workers 4 "$@"
