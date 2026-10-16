"""Step 1b: re-download the dates of fill_<year>.csv (01_inventory audit) into per-tile fill stores.

Runs with the gaig-stac venv (config.GAIG_PY) and reuses its reader unchanged: per date,
gaig_stac.s2_download._daily_mosaic (first SCL-valid item per pixel, bilinear 20 m bands, SCL_INVALID)
inside gaig_stac.retry.read_with_retry. Items are fetched by id. One processing-baseline family is
used per date (date <= PB_CUTOFF: PB < PB_OFFSET_MIN items if any; later: PB >= PB_OFFSET_MIN items if
any) and the +PB_OFFSET DN offset is removed (floor 0) only when the chosen family carries it, so the
stores keep the pre-2022 DN convention whatever the date. No max_t.

Output /scratch/hvt632/crop_phenology_zarr/fill/s2_<year>/<tile>.zarr (zarr v2):
  bands (T, 2500, 2500, 10) u2  chunks (FILL_TCHUNK, 512, 512, 10), blosc zstd 1 shuffle
  masks (T, 2500, 2500)     u1  1 = SCL-valid
  doys  (T,)                i2  written last = commit sentinel; 0 = failed date
A tile whose doys array is non-zero is skipped unless --redo. Per date: outputs/inventory/
fill_log_<year>.csv (appended) and fill_failed_<year>.csv (latest status failed).
--dry-run: one STAC id query per tile, prints the plan (dates, PB family, offset removal, RAM and disk
estimate) and writes only --out-root/plan_<year>.csv.

    $GAIG_PY -u 01b_fill.py --year 2019 [--tiles T ...|--area NAME] [--dry-run] [--tile-workers 4] [--redo]
             [--fill-csv PATH] [--out-root PATH] [--log-dir PATH]
    SLURM array: the tiles are split over the array tasks (jobs/01b_fill.sbatch)
"""
import argparse
import fcntl
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gaig_stac._runtime  # noqa: F401
import config
import dask
import mgrs
import zarr
from gaig_stac.backends import get_s2_backend, open_catalog
from gaig_stac.retry import READ_RETRIES, read_with_retry, search_with_retry
from gaig_stac.s2_download import _daily_mosaic
from gaig_stac.tile_grid import tile_grid
from gaig_stac.zarr_store import set_blosc_threads, zstd_compressor

HW = config.SUB
NB = len(config.STORE_BANDS)
LOG_COLUMNS = ["time", "tile", "date", "item_ids", "pb", "offset_removed", "status", "attempts", "valid_pct", "error"]
_LOG_LOCK = threading.Lock()


def gaig_commit():
    try:
        return subprocess.run(["git", "-C", str(config.GAIG_PY.parents[2]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def check_grid(tile):
    grid = tile_grid(tile)
    zone, hemi, e, n = mgrs.MGRS().MGRSToUTM(tile)
    tr = grid.transform
    exp = (10.0, 0.0, float(e), 0.0, -10.0, float(n) + 25000.0)
    assert np.allclose((tr.a, tr.b, tr.c, tr.d, tr.e, tr.f), exp) and grid.width == HW and grid.height == HW, \
        f"{tile}: gaig tile_grid {tuple(tr)[:6]} {grid.width}x{grid.height} != MGRS origin {exp}"
    return grid


def pb_carries_offset(pb, d):
    try:
        return float(pb) >= float(config.PB_OFFSET_MIN)
    except (TypeError, ValueError):
        return d > config.PB_CUTOFF


def choose_family(d, items):
    """Items of one PB family for date d (iso) and whether that family carries the offset."""
    flag = [pb_carries_offset(it.properties.get("s2:processing_baseline"), d) for it in items]
    new = [it for it, f in zip(items, flag) if f]
    old = [it for it, f in zip(items, flag) if not f]
    if d <= config.PB_CUTOFF:
        use = old or new
    else:
        use = new or old
    return use, bool(use) and use is new


def search_ids(catalog, ids, label):
    out = []
    for k in range(0, len(ids), 100):
        chunk = ids[k:k + 100]
        out += search_with_retry(lambda: list(catalog.search(collections=[config.S2_COLLECTION], ids=chunk,
                                                               limit=100).items()), label=label)
    return out


def plan_tile(catalog, tile, rows):
    """Per fill date: chosen items, PB, offset_removed. rows: the tile's fill_<year>.csv rows."""
    ids = sorted({i for s in rows.item_ids.fillna("") for i in str(s).split(";") if i})
    found = {it.id: it for it in search_ids(catalog, ids, f"[fill {tile}]")} if ids else {}
    plan = []
    for d, g in rows.groupby("date", sort=True):
        want = sorted({i for s in g.item_ids.fillna("") for i in str(s).split(";") if i})
        items = [found[i] for i in want if i in found]
        use, off = choose_family(d, items)
        plan.append(dict(tile=tile, date=d, doy=date.fromisoformat(d).timetuple().tm_yday,
                         reason=";".join(sorted(set(g.reason))), items=use, item_ids=[it.id for it in use],
                         n_missing_ids=len(want) - len(items),
                         pb=";".join(sorted({str(it.properties.get("s2:processing_baseline")) for it in use})),
                         offset_removed=off,
                         cc_min=min((float(it.properties.get("eo:cloud_cover", np.nan)) for it in use),
                                    default=np.nan)))
    return plan


def committed(path):
    try:
        return bool(np.any(zarr.open_group(str(path), mode="r")["doys"][:] != 0))
    except Exception:  # noqa: BLE001
        return False


def append_log(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            df.to_csv(f, header=f.tell() == 0, index=False)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_failed(log_dir, year):
    p = log_dir / f"fill_log_{year}.csv"
    if not p.exists():
        return
    log = pd.read_csv(p).drop_duplicates(["tile", "date"], keep="last")
    tmp = log_dir / f".fill_failed_{year}.{os.getpid()}.csv"
    log[log.status == "failed"].to_csv(tmp, index=False)
    tmp.replace(log_dir / f"fill_failed_{year}.csv")


def read_date(p, grid, backend, chunksize=2048):
    calls = [0]

    def build(use):
        calls[0] += 1
        return _daily_mosaic(use, grid, chunksize, backend)

    bands, mask, valid_pct = read_with_retry(build, p["items"], sign=backend.sign, is_stale=backend.is_stale,
                                             label=f"[fill {p['tile']}] {p['date']}")
    bands = np.ascontiguousarray(bands[:HW, :HW])
    if p["offset_removed"]:
        bands -= np.minimum(bands, np.uint16(config.PB_OFFSET))
    return bands, np.ascontiguousarray(mask[:HW, :HW]), float(valid_pct), calls[0]


def fill_tile(catalog, backend, tile, rows, year, out_root, log_dir, redo=False):
    """Download one tile's fill dates; returns (tile, n_ok, n_failed) or n_ok = -1 when skipped."""
    path = out_root / f"s2_{year}" / f"{tile}.zarr"
    if committed(path) and not redo:
        return tile, -1, 0
    grid = check_grid(tile)
    plan = plan_tile(catalog, tile, rows)
    T = len(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w")
    comp, tc = zstd_compressor(1), config.FILL_TCHUNK
    zb = g.create_dataset("bands", shape=(T, HW, HW, NB), chunks=(tc, 512, 512, NB), dtype="u2", compressor=comp,
                          fill_value=0)
    zm = g.create_dataset("masks", shape=(T, HW, HW), chunks=(tc, 512, 512), dtype="u1", compressor=comp,
                          fill_value=0)
    zd = g.create_dataset("doys", shape=(T,), chunks=(max(T, 1),), dtype="i2", compressor=comp, fill_value=0)
    doys = np.zeros(T, np.int16)
    valid = [0.0] * T
    log = []
    for c0 in range(0, T, tc):
        c1 = min(c0 + tc, T)
        bb = np.zeros((c1 - c0, HW, HW, NB), np.uint16)
        mb = np.zeros((c1 - c0, HW, HW), np.uint8)
        for k in range(c0, c1):
            p = plan[k]
            rec = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), tile=tile, date=p["date"],
                       item_ids=";".join(p["item_ids"]), pb=p["pb"], offset_removed=p["offset_removed"],
                       attempts=0, valid_pct=np.nan, error="")
            if not p["items"]:
                log.append(dict(rec, status="failed", error="no item of the fill list in the catalogue"))
                continue
            try:
                b, m, vp, n = read_date(p, grid, backend)
            except Exception as e:  # noqa: BLE001
                log.append(dict(rec, status="failed", attempts=READ_RETRIES, error=f"{type(e).__name__}: {e}"[:300]))
                logging.warning(f"[fill {year} {tile}] {p['date']} failed: {e}")
                continue
            bb[k - c0], mb[k - c0] = b, m
            doys[k], valid[k] = p["doy"], round(vp, 3)
            log.append(dict(rec, status="ok" if vp > 0 else "empty", attempts=n, valid_pct=round(vp, 3)))
        zb[c0:c1] = bb
        zm[c0:c1] = mb
        del bb, mb
    g.attrs.update(tile=tile, year=year, dates=[p["date"] for p in plan], item_ids=[p["item_ids"] for p in plan],
                   processing_baseline=[p["pb"] for p in plan], offset_removed=[bool(p["offset_removed"]) for p in plan],
                   valid_pct=valid, cloud_cover_min=[None if not np.isfinite(p["cc_min"]) else p["cc_min"] for p in plan],
                   reason=[p["reason"] for p in plan], source=f"mpc {config.S2_COLLECTION}",
                   gaig_stac_commit=gaig_commit(), created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    zd[:] = doys
    append_log(log_dir / f"fill_log_{year}.csv", log)
    n_fail = int((doys == 0).sum())
    if n_fail > config.MAX_UNFILLABLE:
        logging.warning(f"[fill {year} {tile}] {n_fail} failed dates > MAX_UNFILLABLE {config.MAX_UNFILLABLE}")
    return tile, int((doys > 0).sum()), n_fail


def dry_run(catalog, fl, year, out_root):
    rows = []
    for tile, g in fl.groupby("tile"):
        check_grid(tile)
        plan = plan_tile(catalog, tile, g)
        n = len(plan)
        fam = pd.Series([p["pb"] for p in plan]).value_counts().to_dict()
        off = sum(p["offset_removed"] for p in plan)
        miss = sum(p["n_missing_ids"] for p in plan)
        empty = sum(not p["items"] for p in plan)
        ram = config.FILL_TCHUNK * 0.125 + 2.5
        print(f"{tile}: {n} dates (DOY {plan[0]['doy']}-{plan[-1]['doy']}), PB {fam}, offset_removed {off}/{n}, "
              f"ids not in catalogue {miss}, dates without items {empty}, RAM ~{ram:.1f} GB, "
              f"disk ~{n * 42 / 1024:.2f} GB", flush=True)
        rows += [dict(year=year, tile=tile, date=p["date"], doy=p["doy"], reason=p["reason"],
                      item_ids=";".join(p["item_ids"]), pb=p["pb"], offset_removed=p["offset_removed"],
                      n_missing_ids=p["n_missing_ids"], cc_min=p["cc_min"]) for p in plan]
    plan = pd.DataFrame(rows)
    if len(plan):
        out_root.mkdir(parents=True, exist_ok=True)
        plan.to_csv(out_root / f"plan_{year}.csv", index=False)
        print(f"{len(plan)} tile-dates on {plan.tile.nunique()} tiles, ~{len(plan) * 42 / 1024:.1f} GB; "
              f"plan -> {out_root / f'plan_{year}.csv'}", flush=True)
    return plan


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--tiles", nargs="+")
    ap.add_argument("--area", choices=list(config.TEST_AREAS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tile-workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--fill-csv", type=Path)
    ap.add_argument("--out-root", type=Path, default=config.FILL)
    ap.add_argument("--log-dir", type=Path, default=config.INVENTORY)
    a = ap.parse_args()
    if a.area:
        a.year, a.tiles = config.TEST_AREAS[a.area]["year"], a.tiles or config.TEST_AREAS[a.area]["tiles"]
    if a.year is None:
        ap.error("--year or --area is required")
    fl = pd.read_csv(a.fill_csv or config.INVENTORY / f"fill_{a.year}.csv", dtype={"date": str, "item_ids": str})
    fl = fl[fl.year == a.year]
    if a.tiles:
        fl = fl[fl.tile.isin(a.tiles)]
    tiles = sorted(fl.tile.unique())
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    tiles = tiles[task::ntasks]
    fl = fl[fl.tile.isin(tiles)]
    print(f"fill {a.year}: {len(fl)} tile-dates on {len(tiles)} tiles (task {task + 1}/{ntasks})"
          f"{' DRY RUN' if a.dry_run else ''}", flush=True)
    backend = get_s2_backend("mpc")
    catalog = open_catalog(backend, pool_maxsize=max(10, a.tile_workers * 2))
    if a.dry_run:
        dry_run(catalog, fl, a.year, a.out_root)
        return
    dask.config.set(scheduler="threads", num_workers=4)
    set_blosc_threads(a.tile_workers)
    t0 = time.time()
    with ThreadPoolExecutor(max(1, a.tile_workers)) as ex:
        futs = {ex.submit(fill_tile, catalog, backend, t, g, a.year, a.out_root, a.log_dir, a.redo): t
                for t, g in fl.groupby("tile")}
        for f in as_completed(futs):
            try:
                t, n, nf = f.result()
                print(f"  {t}: {'committed before, skipped' if n < 0 else f'{n} dates, {nf} failed'}", flush=True)
            except Exception as e:  # noqa: BLE001
                logging.error(f"[fill {a.year} {futs[f]}] crashed: {type(e).__name__}: {e}")
    write_failed(a.log_dir, a.year)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
