import os
from pathlib import Path

YEARS = list(range(2018, 2026))

# Provinces. Tiles are province-free: one job, one tile file and one fill store per tile-year, built from all
# stores of the ACTIVE provinces, so a border tile is computed once and both provinces mosaic the same pixels.
# PROVINCE selects the ROI, mosaic, report and ancillary layers of a run (env CPZ_PROVINCE or --province).
# Sentinel-2 L2A Zarr stores (gaig-stac): s2_<year>.zarr with bands (tile, t, 3660, 3660, 10) uint16,
# masks (tile, t, y, x) 1 = SCL-valid, doys (tile, t) 0 = empty slot; attrs['mgrs_tiles'] = tile order
_MAIN = Path("/project/def-kanj241/main_data_repo")
_ACILAB = Path("/project/def-acilab/hvt632")
SCRATCH = Path(os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'hvt632')}"))
PROVINCES = {
    "SK": dict(name="Saskatchewan", abbr="sk", stores={
        "sk_main": _MAIN / "homaeye/data/s2_saskatchewan",
        "sk_triangle": _ACILAB / "s2_saskatchewan_triangle",
        "sk_priority": _MAIN / "homaeye/data/s2_saskatchewan/s2_saskatchewan_priority"},
        cropmask=_MAIN / "crop_mask_raw/crop_mask_sk.tif"),
    "MB": dict(name="Manitoba", abbr="mb", stores={"mb_main": _ACILAB / "s2_manitoba"},
               cropmask=_MAIN / "crop_mask_raw/crop_mask_mb.tif"),
    "BC": dict(name="British Columbia", abbr="bc", stores={"bc_main": SCRATCH / "s2_stores/bc"}, cropmask=None),
    "AB": dict(name="Alberta", abbr="ab", stores={"ab_main": SCRATCH / "s2_stores/ab"},
               cropmask=_MAIN / "crop_mask_raw/crop_mask_ab.tif"),
}
ACTIVE = os.environ.get("CPZ_ACTIVE", "SK,MB").split(",")
PROVINCE = os.environ.get("CPZ_PROVINCE", "SK")
EDGE_BUFFER_KM = 2.0        # province mosaics extend this far past the province border (edge-match overlap)
S2_STORES = {k: v for c in ACTIVE for k, v in PROVINCES[c]["stores"].items()}
STORE_BANDS = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
SUB = 2500  # populated top-left part of the 3660 x 3660 store arrays (25 km at 10 m)
# Ids holding the wrong land in the SK stores (s2_saskatchewan/NOTICE.txt, 2026-10-12); never used
MISLABELLED = {"12UYA7500", "12UYA7525", "12UYA7550", "12UYA7575", "12UYB7500", "12UYV7525", "12UYV7550", "12UYV7575",
               "13UGQ7525", "13UGQ7550", "13UGQ7575", "13UGR7500", "13UGR7525", "13UGR7550", "13UGR7575", "13UGS7500"}

# Region of interest: cropland ROI per province (feature selected by ROI_FIELD); inventory = ACTIVE provinces
REGIONS = [PROVINCES[c]["name"] for c in ACTIVE]
ROI = Path("/project/def-kanj241/main_data_repo/Prairies_intersected/Prairies_Provinces_intersected.shp")
ROI_FIELD = "PRENAME"

# Mosaic grid: EPSG:4326 lattice of the DEM origin and pixel size (extended past the DEM, all provinces), cut to
# the ROI bounds
DEM = Path("/project/def-kanj241/nazifa/inputs/DEM/final/deriv_var_slope.tif")

# Data acquisition and input QC (00_census, 01 audit, 01b_fill, cpz/store.py)
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
MAX_CLOUD = 80.0            # item filter of the stores; items below it are expected in the store
FOOTPRINT_MIN_FRAC = 0.01   # item expected for a tile when footprint covers >= this share of the tile
PB_CUTOFF = "2022-01-25"    # store removed the -1000 DN offset by date (strictly after this date)
PB_OFFSET = 1000
PB_OFFSET_MIN = "04.00"     # processing baselines >= this carry the offset
FILL_DOY = (45, 345)        # DOY range re-downloaded by 01b_fill
FILL_TCHUNK = 16
SPRING = (90, 170)
SPRING_MIN_DATES = 8
SEASON = (170, 300)
EMPTY_SUSPECT_CC = 20.0     # stored date with no valid pixel although item cloud cover < this
MAX_UNFILLABLE = 2

# Phenology (per pixel, no composites, no fixed calendar window for the metrics)
VI = "evi2"                 # 'evi2' or 'ndvi'
VI_RANGE = (-0.2, 1.0)      # observations outside are invalid; the fitted curve is clipped to it
BASE_PCT = 10               # floor = this percentile of the pixel's snow-free valid obs
LAMBDA_REF = 300.0          # Whittaker lambda at LAMBDA_NREF obs in SEARCH (2nd differences, daily grid)
LAMBDA_NREF = 54
LAMBDA_CLIP = (0.5, 2.0)    # lam = LAMBDA_REF * clip(n_obs / LAMBDA_NREF, *LAMBDA_CLIP)
ENVELOPE_ITER = 2           # upper-envelope passes (obs below the fit get w0 * ENVELOPE_W)
ENVELOPE_W = 0.3
SEARCH = (60, 330)          # DOY range of peak and crossings (crossing outside -> NaN + bit, never clamped)
SOS_FRAC = 0.2              # SoS / EoS at base_L / base_R + frac * (peak - base_L / base_R)
EOS_FRAC = 0.2
MID_FRAC = 0.5              # SoS50 / EoS50
MIN_AMP = 0.08              # min(peak - base_L, peak - base_R) below this -> no season
MIN_OBS = 8                 # fewer valid obs in SEARCH -> few_obs bit
MAX_GAP = 30                # valid-obs gap (d) around SoS / EoS above this -> gap bits
DATE_GAP = 20               # acquisition-date gap (d) around SoS / EoS above this -> date_gap bits (input)
SNOW_NDSI = 0.4             # snow = NDSI > SNOW_NDSI and B03 > SNOW_B03 (reflectance)
SNOW_B03 = 0.10
HAZE_B02 = 0.12             # B02 above this (not snow) -> initial weight HAZE_W
HAZE_W = 0.3
SPIKE = 0.15                # obs above both neighbours (each within SPIKE_DAYS) by more than this -> dropped
SPIKE_DAYS = 20
DIP = 0.05                  # obs more than this below the floor (residual cloud / shadow) -> dropped
DORMANT_STEP = 10           # dormant pseudo-obs at the floor every DORMANT_STEP d, >= DORMANT_GAP d outside
DORMANT_GAP = 20            # the first / last snow-free obs, weight DORMANT_W (plus day 1 and day ndays)
DORMANT_W = 0.5
SNOW_SOS_WIN = 15           # snow obs before PoS within this many days of SoS -> snow_at_sos bit
GREEN_START_FRAC = 0.5      # first in-season obs already above base_L + frac * (peak - base_L) -> green_at_start
OVERSHOOT_TOL = 0.05        # peak above max obs + tol -> overshoot bit

# Step-2 output: one int16 GeoTIFF per tile-year; physical = stored * scale (scale 1 if not listed)
OUT_BANDS = ["SoS", "SoS50", "PoS", "EoS50", "EoS", "LoS", "AUC", "amplitude", "base", "n_obs", "max_gap",
             "gap_sos", "gap_eos", "qa"]
OUT_SCALE = {"AUC": 0.01, "amplitude": 1e-4, "base": 1e-4}
OUT_DTYPE = "int16"
NODATA = -32768
QA_BITS = {"low_amplitude": 1, "few_obs": 2, "gap_at_sos": 4, "gap_at_eos": 8, "sos_outside_search": 16,
           "eos_outside_search": 32, "date_gap_sos": 64, "date_gap_eos": 128, "snow_at_sos": 256,
           "green_at_start": 512, "overshoot": 1024}
QA_CLEAN_SOS = 1 | 2 | 4 | 16 | 64 | 256 | 512 | 1024  # 1879: SoS clean when (qa & QA_CLEAN_SOS) == 0
QA_CLEAN_EOS = 1 | 2 | 8 | 32 | 128 | 1024              # 1195
QA_INPUT = 64 | 128                                     # bits caused by missing acquisitions

# Mosaic and report
OVERLAP_RULE = "native"
OVR_NEAREST = {"n_obs", "max_gap", "gap_sos", "gap_eos", "qa", "source"}
CROP_WC_CLASS = 40
# QC thresholds: name -> (warn, fail, higher_is_bad)
#   higher_is_bad: value <= warn PASS, <= fail WARN, else FAIL; otherwise value >= warn PASS, >= fail WARN, else FAIL
QC = {
    "completeness": (0.98, 0.90, False), "spring_gap": (20, 30, True), "season_gap": (25, 35, True),
    "sibling_deficit": (4, 9, True), "sos_valid_share": (0.90, 0.75, False), "gap_sos_share": (0.10, 0.30, True),
    "eos_out_share": (0.15, 0.35, True), "overshoot_share": (0.001, 0.01, True),
    "crop_amp_share": (0.70, 0.50, False), "snow_at_sos_share": (0.10, 0.20, True), "base_p05": (0.05, 0.0, False),
    "median_n_obs": (20, 10, False), "tile_shift_days": (7, 12, True), "stripe_days": (5, 8, True),
    "spike_ratio": (3, 5, True), "seam_ratio": (1.3, 1.5, True), "seam_offset_days": (5, 10, True),
    "overlap_bias_days": (2, 5, True), "overlap_mad_days": (4, 6, True), "overlap_share10": (0.15, 0.25, True),
    "georef_shift_px": (0.5, 1.0, True), "roi_uncovered_share": (1e-4, 1e-3, True),
    "native_violation_share": (1e-4, 1e-3, True), "runtime_s": (180, 400, True),
    "mosaic_s_per_tile": (30, 60, True), "date_resid": (0.15, 0.25, True), "b02_p01": (0.06, 0.09, True),
    "summer_snow_frac": (0.01, 0.05, True), "seam_excess_share": (0.02, 0.05, True),
    "edge_match_share": (1.0, 0.9999, False),
}
SEAM_MIN_PAIRS = 500         # boundaries with fewer clean pixel pairs can WARN but not FAIL
SEAM_PLACEBO_Q = (0.95, 0.99)  # seam warn / fail = max(QC value, this quantile of the placebo boundaries)
SEAM_PLACEBO_MIN = 30        # fewer placebo boundaries per band: plain QC thresholds
PLAUSIBLE = {"SoS": (110, 170), "SoS50": (125, 185), "PoS": (180, 225), "EoS50": (215, 275), "EoS": (225, 290)}
MAP_RANGES = {"SoS": (100, 180), "SoS50": (110, 190), "PoS": (170, 240), "EoS50": (200, 290), "EoS": (220, 320),
              "LoS": (40, 200), "AUC": (0, 150), "amplitude": (0, 0.8), "base": (0, 0.3), "n_obs": (0, 80),
              "max_gap": (0, 60), "gap_sos": (0, 60), "gap_eos": (0, 60)}

# Areas for the issue tests (3x3 tiles each)
TEST_AREAS = {
    "tile_shift_2019": dict(year=2019, centre=(-109.69, 53.35),
                            tiles=["12UWE5025", "12UWE7525", "12UXE0025", "12UWE5000", "12UWE7500", "12UXE0000",
                                   "12UWD5075", "12UWD7575", "12UXD0075"]),
    "orbit_stripe_2024": dict(year=2024, centre=(-105.7, 53.3),
                              tiles=["13UDV2525", "13UDV5025", "13UDV7525", "13UDV2500", "13UDV5000", "13UDV7500",
                                     "13UDU2575", "13UDU5075", "13UDU7575"]),
    "zone_edge_2022": dict(year=2022, centre=(-108.0, 51.9),
                           tiles=["12UXC7525", "12UXC7550", "12UXC7575", "12UYC0025", "12UYC0050", "12UYC0075",
                                  "13UBT7525", "13UBT7550", "13UBT7575", "13UCT0025", "13UCT0050", "13UCT0075"]),
}

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WORK = Path(os.environ.get("CPZ_WORK", SCRATCH / "crop_phenology_zarr"))  # tiles and mosaics: large, not backed up
TILES = WORK / "tiles"           # shared by all provinces
FILL = WORK / "fill"             # per-tile fill stores s2_<year>/<tile>.zarr (01b_fill), shared
MOSAIC = WORK / "mosaic" / PROVINCE
INVENTORY = OUT / "inventory"    # one inventory over the ACTIVE provinces
REPORT = OUT / "report" / PROVINCE
CENSUS_DIR = INVENTORY / ("census_" + "-".join(sorted(ACTIVE)))  # census bbox = ACTIVE ROI: one cache per set
TIMING = INVENTORY / "timing"
REF_DIR = WORK / "reference"
GAIG_PY = Path(os.environ.get("CPZ_GAIG_PY", "/project/def-kanj241/hvt632/gaig_stack/.venv/bin/python"))
# MGRS 25 km tile grids (copies of main_data_repo/homaeye/data/*_mgrs_tiles_25km_corrected.geojson)
TILE_LISTS = [ROOT / "tile_grids" / "saskatchewan_mgrs_tiles_25km_corrected.geojson",
              ROOT / "tile_grids" / "canadian_prairie_mgrs_tiles_25km_corrected.geojson"]
WORLDCOVER_DIR = _MAIN / "landcover_data"  # <year>/lc_<tile>.tif 2019-2024, tile grid, zones 10-15
CROPMASK = PROVINCES[PROVINCE]["cropmask"]  # EPSG:32612 10 m, >0 crop; None (BC): WorldCover cropland only
ACI_DIR = _MAIN / "aafc_crop_type"  # <y>/aci_<y>_<abbr>.tif, embedded Albers: use file CRS
