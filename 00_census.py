"""Step 0: per-item STAC census of MPC sentinel-2-l2a over the ROI (no cloud filter, metadata only).

For each year and month, a paged search over the ROI bounds (or the footprints of --tiles) writes one
row per item: id, date (UTC), datetime, mgrs_tile, relative_orbit, cloud_cover, processing_baseline,
platform, footprint_wkt. Months are resumable part files items_<year>_<mm>.csv.gz; when the year's
months are complete they are concatenated (deduplicated by id) into items_<year>.csv.gz.
items_<year>.meta.json records the months and the tile restriction (01_inventory audits only those).

    python -u 00_census.py --years 2019 [--months 5 6] [--tiles 12UWE7500] [--out DIR] [--overwrite]
"""
import argparse
import calendar
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pystac_client import Client
from shapely.geometry import shape
from shapely.ops import unary_union

import config
from cpz.grid import footprint, roi

COLUMNS = ["id", "date", "datetime", "mgrs_tile", "relative_orbit", "cloud_cover", "processing_baseline",
           "platform", "footprint_wkt"]


def retry(fn, label, attempts=3, base=10.0):
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == attempts - 1:
                raise
            print(f"{label}: {type(e).__name__}: {e}; retry in {base * 2 ** k:.0f} s", flush=True)
            time.sleep(base * 2 ** k)


def pb_str(v):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):05.2f}"
    except ValueError:
        return str(v)


def item_row(it):
    p = it.properties
    dt = p.get("datetime") or ""
    return dict(id=it.id, date=dt[:10], datetime=dt, mgrs_tile=p.get("s2:mgrs_tile", ""),
                relative_orbit=p.get("sat:relative_orbit"), cloud_cover=p.get("eo:cloud_cover"),
                processing_baseline=pb_str(p.get("s2:processing_baseline")), platform=p.get("platform", ""),
                footprint_wkt=shape(it.geometry).wkt if it.geometry else "")


def search_month(client, bbox, year, month):
    last = calendar.monthrange(year, month)[1]
    dt = f"{year}-{month:02d}-01T00:00:00Z/{year}-{month:02d}-{last}T23:59:59Z"

    def run():
        s = client.search(collections=[config.S2_COLLECTION], bbox=list(bbox), datetime=dt, limit=1000)
        return [item_row(it) for it in s.items()]

    return retry(run, f"search {year}-{month:02d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=config.YEARS)
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=config.CENSUS_DIR)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    geom = unary_union([footprint(t) for t in a.tiles]) if a.tiles else roi()
    bbox = tuple(round(v, 6) for v in geom.bounds)
    client = retry(lambda: Client.open(config.STAC_URL), "open")
    print(f"census {config.STAC_URL} {config.S2_COLLECTION} bbox {bbox} -> {a.out}", flush=True)
    for y in a.years:
        for m in a.months:
            part = a.out / f"items_{y}_{m:02d}.csv.gz"
            if part.exists() and not a.overwrite:
                continue
            t0 = time.time()
            df = pd.DataFrame(search_month(client, bbox, y, m), columns=COLUMNS)
            tmp = part.with_name(part.name + ".tmp")
            df.to_csv(tmp, index=False, compression="gzip")
            tmp.replace(part)
            print(f"{y}-{m:02d}: {len(df)} items, {df.date.nunique()} dates, {time.time() - t0:.0f} s", flush=True)
        parts = sorted(a.out.glob(f"items_{y}_[0-9][0-9].csv.gz"))
        months = sorted(int(p.name[-9:-7]) for p in parts)
        if not parts:
            continue
        df = pd.concat([pd.read_csv(p, dtype={"processing_baseline": str}) for p in parts], ignore_index=True)
        df = df.drop_duplicates("id").sort_values(["date", "id"])
        df.to_csv(a.out / f"items_{y}.csv.gz", index=False, compression="gzip")
        meta = dict(year=y, months=months, tiles=a.tiles, bbox=bbox, stac_url=config.STAC_URL,
                    collection=config.S2_COLLECTION, created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (a.out / f"items_{y}.meta.json").write_text(json.dumps(meta, indent=1))
        pbs = df.processing_baseline.fillna("").value_counts().sort_index().to_dict()
        print(f"{y}: {len(df)} items, {df.date.nunique()} dates, months {months}, PB {pbs}", flush=True)


if __name__ == "__main__":
    main()
