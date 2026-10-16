"""Step 2: per-pixel phenology for each tile-year (cpz.lsp) from the merged base + fill series (cpz.store).

Reads jobs.csv (step 1). All dates of the year are used (winter dates set the dormant base). Per 512 x 512
block aligned with the store chunks: read, screen (snow / haze / VI range), fit (cpz.lsp.block), encode to int16.
Outputs per tile-year, in the tile's UTM grid (10 m, 2500 x 2500):
  <TILES>/<tile>/<year>/pheno_<tile>_<year>.tif   int16, bands config.OUT_BANDS, nodata config.NODATA,
                                                  band scales config.OUT_SCALE, tag params_hash
  <TILES>/<tile>/<year>/dates_<tile>_<year>.csv   per date: doy, source (0 base, 2 fill), n_px, valid_frac
                                                  (valid / n_px), snow_frac and haze_frac (shares of the valid obs),
                                                  evi2_p50, b02_p01, b02_p50, b04_p50 (reflectance, valid
                                                  non-snow obs; NaN when there are none)
  outputs/inventory/timing/<name>_<year>_<jobid>_<task>.csv (name = test area or province; finished = UTC time)
Tiles are shared by the provinces: the run covers the tiles of the year that intersect the ROI of config.PROVINCE
plus EDGE_BUFFER_KM (env CPZ_PROVINCE). A tile is skipped when its tif exists with the current params_hash and the
same input_hash (store, index and the merged dates with their source and offset fix; unless --overwrite), so a
border tile whose store or fill changed is recomputed and one mosaic never mixes parameters or inputs. Tiles
written before input_hash existed match on store, n_dates and n_fill. A DOY integrity failure writes nothing.

    python -u 02_phenology.py --year 2019 [--tiles T ...] [--area NAME] [--workers N] [--overwrite]
                              [--jobs CSV] [--tiles-root DIR]
    python -u 02_phenology.py --year 2019 --debug-block 12UWE7525 1024 1024 --out DIR   (one block, login node)
    SLURM array: the year's tiles are split over the array tasks
"""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

import config
from cpz import lsp
from cpz.grid import footprint, province_roi, tile_grid
from cpz.store import DoyIntegrityError, series

BLOCK = 512
NB = len(config.OUT_BANDS)
BI = {b: i for i, b in enumerate(config.OUT_BANDS)}
DATE_COLS = ["doy", "source", "n_px", "valid_frac", "snow_frac", "haze_frac", "evi2_p50", "b02_p01", "b02_p50",
             "b04_p50"]


def out_path(root, tile, year):
    return Path(root) / tile / str(year) / f"pheno_{tile}_{year}.tif"


def code_version():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "uncommitted"


def tif_tags(path):
    try:
        with rasterio.open(path) as f:
            return f.tags()
    except rasterio.errors.RasterioIOError:
        return {}


def input_hash(job, s):
    h = hashlib.sha1(f"{job.get('store_path', '')}|{job.get('index', -1)}".encode())
    for a in (s.doys, s.source, s.offset_fix):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def same_input(tags, job, s, ih):
    if "input_hash" in tags:
        return tags["input_hash"] == ih
    return (tags.get("store", "") == str(job.get("store_path", "")) and tags.get("n_dates") == str(len(s.doys))
            and tags.get("n_fill") == str(int((s.source == 2).sum())))


class Stats:
    """Per-date accumulators (lsp.date_stats) over all blocks of a tile-year."""

    def __init__(self, T):
        self.counts = np.zeros((T, 4), np.int64)
        self.h_vi = np.zeros((T, lsp.EVI_BINS), np.int64)
        self.h_b02 = np.zeros((T, lsp.REF_BINS), np.int64)
        self.h_b04 = np.zeros((T, lsp.REF_BINS), np.int64)

    def add(self, raw, valid, bidx, params):
        lsp.date_stats(raw, valid, bidx, params, self.counts, self.h_vi, self.h_b02, self.h_b04)

    def frame(self, doys, source):
        n, v, s, h = (self.counts[:, i].astype(float) for i in range(4))
        with np.errstate(invalid="ignore", divide="ignore"):
            d = dict(doy=doys.astype(int), source=source.astype(int), n_px=n.astype(np.int64),
                     valid_frac=np.where(n > 0, v / n, np.nan), snow_frac=np.where(v > 0, s / v, np.nan),
                     haze_frac=np.where(v > 0, h / v, np.nan),
                     evi2_p50=lsp.hist_quantile(self.h_vi, 0.5, lsp.EVI_LO, lsp.EVI_STEP),
                     b02_p01=lsp.hist_quantile(self.h_b02, 0.01, 0.0, lsp.REF_STEP),
                     b02_p50=lsp.hist_quantile(self.h_b02, 0.5, 0.0, lsp.REF_STEP),
                     b04_p50=lsp.hist_quantile(self.h_b04, 0.5, 0.0, lsp.REF_STEP))
        return pd.DataFrame(d)[DATE_COLS].round(5)


def run_block(s, r0, r1, c0, c1, nd, params, bidx, stats):
    """int16 (NB, h, w) for one block; stats accumulated in place."""
    raw, valid = s.read(r0, r1, c0, c1)
    stats.add(raw, valid, bidx, params)
    vi, w, snow = lsp.screen_block(raw, valid, bidx, params)
    del raw, valid
    doys = s.doys.astype(np.int64)
    res = lsp.block(vi, w, snow, doys, doys, nd, params)
    return lsp.encode(res.reshape(NB, r1 - r0, c1 - c0), config)


def write_tif(path, out, epsg, tr, tags):
    prof = dict(driver="GTiff", width=out.shape[2], height=out.shape[1], count=NB, dtype=config.OUT_DTYPE,
                nodata=config.NODATA, crs=f"EPSG:{epsg}", transform=tr, tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", predictor=2)
    if out.shape[1] < 512 or out.shape[2] < 512:
        prof.update(tiled=False)
        prof.pop("blockxsize")
        prof.pop("blockysize")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **prof) as f:
        f.write(out)
        for i, b in enumerate(config.OUT_BANDS, 1):
            f.set_band_description(i, b)
        f.scales = tuple(float(config.OUT_SCALE.get(b, 1.0)) for b in config.OUT_BANDS)
        f.offsets = (0.0,) * NB
        f.update_tags(**tags)
    os.replace(tmp, path)


def tags_for(job, s, phash, version, sos_valid):
    return dict(method="crop_phenology_zarr cpz.lsp v2 (per-pixel screened weighted Whittaker, left/right bases)",
                vi=config.VI, lambda_ref=config.LAMBDA_REF, lambda_nref=config.LAMBDA_NREF,
                lambda_clip=str(tuple(config.LAMBDA_CLIP)), search=str(tuple(config.SEARCH)),
                sos_frac=config.SOS_FRAC, eos_frac=config.EOS_FRAC, mid_frac=config.MID_FRAC, min_amp=config.MIN_AMP,
                snow_ndsi=config.SNOW_NDSI, snow_b03=config.SNOW_B03, qa_bits=json.dumps(config.QA_BITS),
                out_scale=json.dumps(config.OUT_SCALE), params_hash=phash, code_version=version, year=s.year,
                store=job.get("store_path", ""), store_index=job.get("index", -1), fill_path=job.get("fill_path", ""),
                n_dates=len(s.doys), n_fill=int((s.source == 2).sum()), n_offset_fix=int(s.offset_fix.sum()),
                input_status=job.get("input_status", ""), sos_valid=round(sos_valid, 5), input_hash=input_hash(job, s))


def clean(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v


def process(job):
    t0 = time.time()
    tile, year = job["tile"], int(job["year"])
    phash = job["params_hash"]
    path = out_path(job["tiles_root"], tile, year)
    rec = dict(tile=tile, year=year, seconds=0.0, status="", n_dates=0, n_fill=0, sos_valid=np.nan,
               clean_sos_share=np.nan, max_rss_mb=np.nan, params_hash=phash)
    try:
        s = series(job)
    except DoyIntegrityError as e:
        return dict(rec, status="doy_integrity_fail", note=str(e))
    tags = tif_tags(path) if path.exists() and not job["overwrite"] else {}
    if tags.get("params_hash") == phash and same_input(tags, job, s, input_hash(job, s)):
        return dict(rec, status="exists")
    nd, params, bidx = lsp.ndays(year), lsp.params_array(config), lsp.band_indices(config)
    out = np.full((NB, config.SUB, config.SUB), config.NODATA, np.int16)
    stats = Stats(len(s.doys))
    if len(s.doys):
        for r0 in range(0, config.SUB, BLOCK):
            for c0 in range(0, config.SUB, BLOCK):
                r1, c1 = min(r0 + BLOCK, config.SUB), min(c0 + BLOCK, config.SUB)
                out[:, r0:r1, c0:c1] = run_block(s, r0, r1, c0, c1, nd, params, bidx, stats)
    sos, qa = out[BI["SoS"]], out[BI["qa"]]
    sos_valid = float((sos != config.NODATA).mean())
    clean_sos = float(((sos != config.NODATA) & (qa != config.NODATA) & ((qa & config.QA_CLEAN_SOS) == 0)).mean())
    epsg, tr = tile_grid(tile)
    write_tif(path, out, epsg, tr, tags_for(job, s, phash, job["code_version"], sos_valid))
    stats.frame(s.doys, s.source).to_csv(path.parent / f"dates_{tile}_{year}.csv", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return dict(rec, seconds=round(time.time() - t0, 1), status="done" if len(s.doys) else "no_dates",
                n_dates=len(s.doys), n_fill=int((s.source == 2).sum()), sos_valid=round(sos_valid, 5),
                clean_sos_share=round(clean_sos, 5), max_rss_mb=round(rss, 1))


def process_safe(job):
    try:
        return process(job)
    except Exception as e:  # noqa: BLE001
        return dict(tile=job["tile"], year=int(job["year"]), seconds=np.nan, status=f"error: {type(e).__name__}: {e}"[:300],
                    n_dates=0, n_fill=0, sos_valid=np.nan, clean_sos_share=np.nan, max_rss_mb=np.nan,
                    params_hash=job["params_hash"])


def debug_block(job, r0, c0, out_dir):
    t0 = time.time()
    tile, year = job["tile"], int(job["year"])
    s = series(job)
    r1, c1 = min(r0 + BLOCK, config.SUB), min(c0 + BLOCK, config.SUB)
    nd, params, bidx = lsp.ndays(year), lsp.params_array(config), lsp.band_indices(config)
    stats = Stats(len(s.doys))
    t1 = time.time()
    out = run_block(s, r0, r1, c0, c1, nd, params, bidx, stats)
    t2 = time.time()
    epsg, tr = tile_grid(tile)
    out_dir = Path(out_dir)
    p = out_dir / f"pheno_{tile}_{year}_r{r0}_c{c0}.tif"
    write_tif(p, out, epsg, tr * Affine.translation(c0, r0),
              tags_for(job, s, job["params_hash"], job["code_version"], float((out[BI["SoS"]] != config.NODATA).mean())))
    df = stats.frame(s.doys, s.source)
    df.to_csv(out_dir / f"dates_{tile}_{year}_r{r0}_c{c0}.csv", index=False)
    dec = lsp.decode(out, config)
    qa = out[BI["qa"]]
    print(f"{tile} {year} block r{r0} c{c0} ({r1 - r0}x{c1 - c0}): {len(s.doys)} dates "
          f"({int((s.source == 2).sum())} fill), read+fit {t2 - t1:.1f}s, total {time.time() - t0:.1f}s, "
          f"{(t2 - t1) / ((r1 - r0) * (c1 - c0)) * 1e6:.1f} us/px")
    for b in ("n_obs", "SoS", "SoS50", "PoS", "EoS50", "EoS", "LoS", "amplitude", "base"):
        v = dec[BI[b]]
        print(f"  {b:9s} median {np.nanmedian(v):8.3f}  valid {np.isfinite(v).mean():.3f}")
    print(f"  clean SoS share {(((qa & config.QA_CLEAN_SOS) == 0) & (out[BI['SoS']] != config.NODATA)).mean():.3f}")
    for name, bit in config.QA_BITS.items():
        print(f"  qa {bit:5d} {name:20s} {((qa & bit) > 0).mean():.3f}")
    df["month"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df.doy - 1, unit="D")
    m = df.groupby(df.month.dt.month).agg(dates=("doy", "size"), valid_frac=("valid_frac", "mean"),
                                         snow_frac=("snow_frac", "mean"), haze_frac=("haze_frac", "mean"),
                                         evi2_p50=("evi2_p50", "median"), b02_p01=("b02_p01", "median"))
    print(m.round(3).to_string())
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--tiles", nargs="+")
    ap.add_argument("--area", choices=list(config.TEST_AREAS), help="tiles and year of a test area")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=Path, default=config.INVENTORY / "jobs.csv")
    ap.add_argument("--tiles-root", type=Path, default=config.TILES)
    ap.add_argument("--debug-block", nargs=3, metavar=("TILE", "R0", "C0"))
    ap.add_argument("--out", type=Path, help="output directory of --debug-block")
    a = ap.parse_args()
    if a.area:
        a.year, a.tiles = config.TEST_AREAS[a.area]["year"], config.TEST_AREAS[a.area]["tiles"]
    if a.year is None:
        ap.error("--year or --area is required")
    jobs = pd.read_csv(a.jobs, dtype={"offset_fix_doys": str, "fill_path": str, "input_status": str})
    jobs = jobs[jobs.year == a.year]
    common = dict(params_hash=lsp.params_hash(config), code_version=code_version(), overwrite=a.overwrite,
                  tiles_root=str(a.tiles_root))
    if a.debug_block:
        if a.out is None:
            ap.error("--debug-block needs --out")
        tile, r0, c0 = a.debug_block[0], int(a.debug_block[1]), int(a.debug_block[2])
        sel = jobs[jobs.tile == tile]
        if sel.empty:
            raise SystemExit(f"{tile} not in {a.jobs} for {a.year}")
        job = {k: clean(v) for k, v in sel.iloc[0].to_dict().items()}
        debug_block(dict(job, **common), r0, c0, a.out)
        return
    if a.tiles:
        missing = set(a.tiles) - set(jobs.tile)
        if missing:
            raise SystemExit(f"not in {a.jobs} for {a.year}: {sorted(missing)}")
        jobs = jobs[jobs.tile.isin(a.tiles)]
    else:
        region = province_roi()
        jobs = jobs[[footprint(t).intersects(region) for t in jobs.tile]]
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    recs = [dict({k: clean(v) for k, v in r.items()}, **common) for r in jobs.iloc[task::ntasks].to_dict("records")]
    name = a.area or config.PROVINCE
    print(f"{name} {a.year}: {len(recs)} tile-years (task {task + 1}/{ntasks}), {a.workers} workers, "
          f"params {common['params_hash'][:8]}", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(process_safe, recs):
            rows.append(r)
            print(f"  {r['tile']} {r['year']}: {r['status']} {r['seconds']}s sos_valid {r['sos_valid']} "
                  f"clean {r['clean_sos_share']}", flush=True)
    cols = ["tile", "year", "seconds", "status", "n_dates", "n_fill", "sos_valid", "clean_sos_share", "max_rss_mb",
            "params_hash", "finished"]
    config.TIMING.mkdir(parents=True, exist_ok=True)
    jobid = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    df = pd.DataFrame(rows).assign(finished=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    df.reindex(columns=cols).to_csv(config.TIMING / f"{name}_{a.year}_{jobid}_{task}.csv", index=False)
    print(f"done {len(rows)} tile-years in {time.time() - t0:.0f}s", flush=True)
    bad = [r["tile"] for r in rows if r["status"] not in ("done", "exists", "no_dates")]
    if bad:
        raise SystemExit(f"{len(bad)} tile-years failed: {bad}")


if __name__ == "__main__":
    main()
