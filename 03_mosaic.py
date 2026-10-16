"""Step 3: mosaic the int16 step-2 tiles of one year onto the DEM-aligned EPSG:4326 grid, clipped to the ROI of
config.PROVINCE (env CPZ_PROVINCE) extended EDGE_BUFFER_KM into the neighbouring provinces' ROI. The lattice is
the same for every province and the tiles are shared, so adjacent province mosaics are identical in the overlap.

Tile list from jobs.csv (year, in_roi, not mislabelled; fill-only tiles included); a missing tile file stops the
run unless --allow-missing. Every tile is validated first (14 OUT_BANDS, int16, NODATA, tile_grid CRS and
transform, 2500 x 2500, one params_hash, identical scales). Source ids are the stable idx of tile_index.csv.

Exact nearest neighbour: each target pixel centre is projected with PROJ into the tile's UTM CRS and takes the
tile pixel containing it; all bands of a pixel come from one tile. Priority per pixel: 4 native UTM zone with
data (qa != NODATA and n_obs > 0), 3 other zone with data, 2 native without data, 1 other zone without data.
Where two tiles with data overlap, their SoS50 / EoS50 / n_obs agreement is accumulated (overlap_<y>.csv);
seam statistics at every source boundary are accumulated on the final SoS50 and EoS50 (seams_<y>.csv). The same
statistics on placebo boundaries (every tile split into quadrants at its box centre, rows placebo=True) are the
no-seam baseline that 04 calibrates the seam thresholds against. --seams-only recomputes seams_<y>.csv from an
existing mosaic (source_<y>.csv / .tif, qa, SoS50, EoS50) without the step-2 tiles; mosaic_qc_<y>.json is kept.

Writes <out-root>/[<name>/]<year>/: pheno_<band>_<year>.tif (int16, tile scales), source_<year>.tif (uint16, 0 =
none) and .csv, overlap_<year>.csv, seams_<year>.csv, mosaic_qc_<year>.json, pheno_<year>.vrt (14 bands +
source) and utm<zz>_<year>.vrt (native 10 m tiles of one zone). Overviews: nearest for counts, gaps, qa and
source, average otherwise. The grid is 9.4 x 15.2 m at 52N (N-S undersampled); the UTM VRTs are lossless.

    python -u 03_mosaic.py --year 2019 | --area tile_shift_2019 | --year 2022 --bbox lon0 lat0 lon1 lat1 --name N
        [--jobs F] [--tile-index F] [--tiles-root D] [--out-root D] [--allow-missing] [--seams-only]
"""
import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import box

import config
from cpz.grid import footprint, province_roi, target_grid, tile_grid
from cpz.qc import BoundaryAccumulator, OverlapAccumulator

CHUNK = 2048
BLOCK = 512
NB = len(config.OUT_BANDS)
BI = {b: i for i, b in enumerate(config.OUT_BANDS)}
GDAL_TYPES = {"int16": "Int16", "uint16": "UInt16"}
SEAM_BANDS = ("SoS50", "EoS50")
PLACEBO_STEP = 16384  # placebo id = idx + PLACEBO_STEP * quadrant (0-3); needs idx < PLACEBO_STEP


def code_version():
    try:
        r = subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "uncommitted"
    except OSError:
        return "uncommitted"


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def select_tiles(a, region=None):
    jobs = pd.read_csv(a.jobs, dtype={"tile": str}, keep_default_na=False)
    j = jobs[(jobs.year.astype(int) == a.year) & as_bool(jobs.in_roi) & ~as_bool(jobs.mislabelled)].copy()
    if a.area:
        j = j[j.tile.isin(config.TEST_AREAS[a.area]["tiles"])]
    j = j.drop_duplicates("tile").reset_index(drop=True)
    fps = [footprint(t) for t in j.tile]
    if region is not None:
        keep = [fp.intersects(region) for fp in fps]
        j, fps = j[keep].reset_index(drop=True), [f for f, k in zip(fps, keep) if k]
    j["path"] = [str(Path(a.tiles_root) / t / str(a.year) / f"pheno_{t}_{a.year}.tif") for t in j.tile]
    j["epsg"] = [tile_grid(t)[0] for t in j.tile]
    b = np.array([fp.bounds for fp in fps]).reshape(-1, 4)
    j["lon_min"], j["lat_min"], j["lon_max"], j["lat_max"] = b.T if len(j) else ([],) * 4
    ti = pd.read_csv(a.tile_index, dtype={"tile": str})
    idx = dict(zip(ti.tile, ti.idx.astype(int)))
    absent = sorted(set(j.tile) - set(idx))
    if absent:
        raise SystemExit(f"tiles missing from {a.tile_index}: {absent}")
    j["idx"] = j.tile.map(idx).astype(int)
    return j


def validate(tiles):
    hashes, scales, errors = set(), None, []
    for t in tiles.itertuples():
        epsg, tr = tile_grid(t.tile)
        with rasterio.open(t.path) as s:
            e = []
            if s.count != NB or list(s.descriptions) != config.OUT_BANDS:
                e.append(f"bands {s.count} {s.descriptions}")
            if set(s.dtypes) != {config.OUT_DTYPE} or s.nodata != config.NODATA:
                e.append(f"dtype {set(s.dtypes)} nodata {s.nodata}")
            if s.crs is None or s.crs.to_epsg() != epsg or not s.transform.almost_equals(tr, 1e-6):
                e.append(f"grid {s.crs} {tuple(s.transform)[:6]} != EPSG:{epsg} {tuple(tr)[:6]}")
            if (s.height, s.width) != (config.SUB, config.SUB):
                e.append(f"shape {s.height}x{s.width}")
            hashes.add(s.tags().get("params_hash", ""))
            if scales is None:
                scales = tuple(s.scales)
            elif tuple(s.scales) != scales:
                e.append(f"scales {s.scales} != {scales}")
        errors += [f"{t.tile}: {x}" for x in e]
    if len(hashes) > 1 or "" in hashes:
        errors.append(f"params_hash not unique: {sorted(hashes)}")
    if errors:
        raise SystemExit("tile validation failed:\n  " + "\n  ".join(errors))
    return hashes.pop() if hashes else "", scales


def sample(path, lon, lat):
    """int16 (NB, ...) values of the tile pixels containing the points, NODATA outside the tile."""
    with rasterio.open(path) as s:
        x, y = Transformer.from_crs(4326, s.crs.to_epsg(), always_xy=True).transform(lon, lat)
        col, row = ~s.transform * (x, y)
        col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        inside = (row >= 0) & (row < s.height) & (col >= 0) & (col < s.width)
        out = np.full((s.count,) + lon.shape, config.NODATA, "int16")
        if inside.any():
            r0, r1, c0, c1 = row[inside].min(), row[inside].max() + 1, col[inside].min(), col[inside].max() + 1
            a = s.read(window=Window(c0, r0, c1 - c0, r1 - r0))
            out[:, inside] = a[:, row[inside] - r0, col[inside] - c0]
    return out


def decode(v, band, scales):
    a = v.astype("float32") * np.float32(scales[BI[band]])
    a[v == config.NODATA] = np.nan
    return a


def clean_masks(qa):
    has = qa != config.NODATA
    return {"SoS50": has & ((qa & config.QA_CLEAN_SOS) == 0), "EoS50": has & ((qa & config.QA_CLEAN_EOS) == 0)}


def tile_centres(tiles, tr):
    if tiles.idx.max() >= PLACEBO_STEP:
        raise SystemExit(f"tile idx {tiles.idx.max()} >= PLACEBO_STEP {PLACEBO_STEP}")
    mid_r, mid_c = np.zeros(PLACEBO_STEP, np.int64), np.zeros(PLACEBO_STEP, np.int64)
    mid_r[tiles.idx] = np.rint(((tiles.lat_min + tiles.lat_max) / 2 - tr.f) / tr.e)
    mid_c[tiles.idx] = np.rint(((tiles.lon_min + tiles.lon_max) / 2 - tr.c) / tr.a)
    return mid_r, mid_c


def placebo_ids(idx, r0, mid_r, mid_c):
    rows = np.arange(r0, r0 + idx.shape[0])[:, None]
    cols = np.arange(idx.shape[1])[None, :]
    q = (rows >= mid_r[idx]).astype("uint16") + 2 * (cols >= mid_c[idx]).astype("uint16")
    return np.where(idx > 0, idx + PLACEBO_STEP * q, 0).astype("uint16")


def seam_frame(seams, placebo, names):
    s = []
    for is_placebo, accs in ((False, seams), (True, placebo)):
        for b, acc in accs.items():
            df = acc.to_frame()
            if is_placebo:
                df = df[df.idx_a % PLACEBO_STEP == df.idx_b % PLACEBO_STEP]
            df.insert(0, "band", b)
            df["placebo"] = is_placebo
            s.append(df)
    s = pd.concat(s, ignore_index=True)
    s.insert(3, "tile_a", (s.idx_a % PLACEBO_STEP).map(names))
    s.insert(4, "tile_b", (s.idx_b % PLACEBO_STEP).map(names))
    return s


def seams_from_mosaic(out, year, tiles, tr, H, W):
    mid_r, mid_c = tile_centres(tiles, tr)
    seams = {b: BoundaryAccumulator() for b in SEAM_BANDS}
    placebo = {b: BoundaryAccumulator() for b in SEAM_BANDS}
    paths = dict(source=out / f"source_{year}.tif", qa=out / f"pheno_qa_{year}.tif",
                 **{b: out / f"pheno_{b}_{year}.tif" for b in SEAM_BANDS})
    fs = {k: rasterio.open(p) for k, p in paths.items()}
    try:
        for k, f in fs.items():
            if f.shape != (H, W) or not f.transform.almost_equals(tr):
                raise SystemExit(f"{paths[k]}: grid {f.shape} {f.transform} differs from {(H, W)} {tr}")
        for r0 in range(0, H, CHUNK):
            w = Window(0, r0, W, min(CHUNK, H - r0))
            idx = fs["source"].read(1, window=w)
            pidx = placebo_ids(idx, r0, mid_r, mid_c)
            cm = clean_masks(fs["qa"].read(1, window=w))
            for b in SEAM_BANDS:
                v = fs[b].read(1, window=w)
                val = v.astype("float32") * np.float32(fs[b].scales[0])
                val[v == config.NODATA] = np.nan
                seams[b].update(idx, val, cm[b])
                placebo[b].update(pidx, val, cm[b])
    finally:
        for f in fs.values():
            f.close()
    s = seam_frame(seams, placebo, dict(zip(tiles.idx, tiles.tile)))
    s.to_csv(out / f"seams_{year}.csv", index=False)
    return s


def effective_px(tr):
    g = Geod(ellps="WGS84")
    out = {}
    for lat in (49, 52, 55):
        dx = g.inv(-105, lat, -105 + tr.a, lat)[2]
        dy = g.inv(-105, lat, -105, lat - tr.e)[2]
        out[f"{lat}N"] = [round(dx, 2), round(dy, 2)]
    return out


def vrt_band(i, dtype, desc, nodata, scale, sources):
    s = [f'  <VRTRasterBand dataType="{GDAL_TYPES[dtype]}" band="{i}">',
         f"    <Description>{escape(desc)}</Description>", f"    <NoDataValue>{nodata}</NoDataValue>",
         f"    <Offset>0</Offset>", f"    <Scale>{scale!r}</Scale>"]
    for fn, rel, band, (sw, sh), (dx, dy) in sources:
        s += ["    <SimpleSource>",
              f'      <SourceFilename relativeToVRT="{int(rel)}">{escape(str(fn))}</SourceFilename>',
              f"      <SourceBand>{band}</SourceBand>",
              f'      <SourceProperties RasterXSize="{sw}" RasterYSize="{sh}" DataType="{GDAL_TYPES[dtype]}" '
              f'BlockXSize="{BLOCK}" BlockYSize="{BLOCK}"/>',
              f'      <SrcRect xOff="0" yOff="0" xSize="{sw}" ySize="{sh}"/>',
              f'      <DstRect xOff="{dx}" yOff="{dy}" xSize="{sw}" ySize="{sh}"/>', "    </SimpleSource>"]
    return s + ["  </VRTRasterBand>"]


def write_vrt(path, W, H, crs_wkt, tr, bands):
    gt = ", ".join(repr(float(v)) for v in (tr.c, tr.a, tr.b, tr.f, tr.d, tr.e))
    lines = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">', f"  <SRS>{escape(crs_wkt)}</SRS>",
             f"  <GeoTransform>{gt}</GeoTransform>"]
    for i, b in enumerate(bands, 1):
        lines += vrt_band(i, *b)
    Path(path).write_text("\n".join(lines + ["</VRTDataset>", ""]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--area", choices=list(config.TEST_AREAS))
    ap.add_argument("--bbox", type=float, nargs=4)
    ap.add_argument("--name")
    ap.add_argument("--jobs", type=Path, default=config.INVENTORY / "jobs.csv")
    ap.add_argument("--tile-index", type=Path, default=config.INVENTORY / "tile_index.csv")
    ap.add_argument("--tiles-root", type=Path, default=config.TILES)
    ap.add_argument("--out-root", type=Path, default=config.MOSAIC)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--seams-only", action="store_true", help="recompute seams_<y>.csv from the existing mosaic")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    t0 = time.time()
    name = a.name
    if a.area:
        name, a.year = a.name or a.area, config.TEST_AREAS[a.area]["year"]
    if a.year is None:
        ap.error("--year or --area required")
    if a.bbox and not name:
        name = "bbox"
    if a.seams_only:
        out = a.out_root / name / str(a.year) if name else a.out_root / str(a.year)
        src = pd.read_csv(out / f"source_{a.year}.csv", dtype={"tile": str})
        b = np.array([footprint(t).bounds for t in src.tile]).reshape(-1, 4)
        src["lon_min"], src["lat_min"], src["lon_max"], src["lat_max"] = b.T
        with rasterio.open(out / f"source_{a.year}.tif") as f:
            tr, (H, W) = f.transform, f.shape
        s = seams_from_mosaic(out, a.year, src, tr, H, W)
        r, p = s[~s.placebo], s[s.placebo]
        print(f"{config.PROVINCE} seams {len(r)} rows, placebo {len(p)} rows, ratio median {r.ratio.median():.3f} / "
              f"{p.ratio.median():.3f} (placebo) -> {out / f'seams_{a.year}.csv'} in {time.time() - t0:.0f}s", flush=True)
        return
    region = province_roi()
    if a.bbox:
        region = region.intersection(box(*a.bbox))
    tiles = select_tiles(a, region)
    if a.area:
        region = region.intersection(box(tiles.lon_min.min(), tiles.lat_min.min(), tiles.lon_max.max(),
                                         tiles.lat_max.max()))
    exists = np.array([Path(p).exists() for p in tiles.path], bool)
    missing = sorted(tiles.tile[~exists])
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} step-2 tiles missing (use --allow-missing): {missing}")
    tiles = tiles[exists].reset_index(drop=True)
    if tiles.empty:
        raise SystemExit(f"no step-2 tiles for {a.year}")
    params_hash, scales = validate(tiles)
    tr, H, W = target_grid(region.bounds)
    with rasterio.open(config.DEM) as d:
        dem_off = [(tr.c - d.transform.c) / d.transform.a, (tr.f - d.transform.f) / d.transform.e]
    out = a.out_root / name / str(a.year) if name else a.out_root / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{config.PROVINCE} {a.year}: {len(tiles)} tiles -> {W} x {H} px, res {tr.a:.15g}, origin ({tr.c:.10f}, {tr.f:.10f}), "
          f"params {params_hash[:8]}", flush=True)

    eff = effective_px(tr)
    ver = code_version()
    tags = dict(year=a.year, vi=config.VI, params_hash=params_hash, qa_bits=json.dumps(config.QA_BITS),
                out_scale=json.dumps(config.OUT_SCALE), overlap_rule=config.OVERLAP_RULE,
                resampling="exact nearest (PROJ)", effective_px_m=json.dumps(eff), n_tiles=len(tiles),
                code_version=ver, province=config.PROVINCE, edge_buffer_km=config.EDGE_BUFFER_KM)
    prof = dict(driver="GTiff", width=W, height=H, count=1, crs="EPSG:4326", transform=tr, tiled=True, blockxsize=BLOCK,
                blockysize=BLOCK, compress="deflate", predictor=2, BIGTIFF="YES", num_threads=a.threads)
    files = {}
    for b in config.OUT_BANDS:
        f = rasterio.open(out / f"pheno_{b}_{a.year}.tif.part", "w", dtype="int16", nodata=config.NODATA, **prof)
        f.set_band_description(1, b)
        f.scales = (scales[BI[b]],)
        f.update_tags(band=b, overview_resampling="nearest" if b in config.OVR_NEAREST else "average", **tags)
        files[b] = f
    f = rasterio.open(out / f"source_{a.year}.tif.part", "w", dtype="uint16", nodata=0, **prof)
    f.set_band_description(1, "source")
    f.update_tags(band="source", overview_resampling="nearest", **tags)
    files["source"] = f

    boxes = tiles[["lon_min", "lat_min", "lon_max", "lat_max"]].to_numpy()
    lon_all = tr.c + (np.arange(W) + 0.5) * tr.a
    native = np.floor((lon_all + 180) / 6).astype(int) + 1
    ov = OverlapAccumulator()
    seams = {b: BoundaryAccumulator() for b in SEAM_BANDS}
    placebo = {b: BoundaryAccumulator() for b in SEAM_BANDS}
    mid_r, mid_c = tile_centres(tiles, tr)
    t_tile = dict.fromkeys(tiles.tile, 0.0)
    n_roi = n_unc = n_viol = 0
    iq, ino = BI["qa"], BI["n_obs"]
    for r0 in range(0, H, CHUNK):
        h = min(CHUNK, H - r0)
        lat_all = tr.f + (r0 + np.arange(h) + 0.5) * tr.e
        ctr = Affine(tr.a, 0, tr.c, 0, tr.e, tr.f + r0 * tr.e)
        inside = geometry_mask([region], out_shape=(h, W), transform=ctr, invert=True)
        lat_top, lat_bot = lat_all[0] - tr.e / 2, lat_all[-1] + tr.e / 2
        hit = tiles[(boxes[:, 1] < lat_top) & (boxes[:, 3] > lat_bot)]
        val = np.full((NB, h, W), config.NODATA, "int16")
        idx = np.zeros((h, W), "uint16")
        prio = np.zeros((h, W), "uint8")
        nat_has = np.zeros((h, W), bool)
        for t in hit.itertuples() if inside.any() else []:
            c0 = max(int(np.floor((t.lon_min - tr.c) / tr.a)) - 1, 0)
            c1 = min(int(np.ceil((t.lon_max - tr.c) / tr.a)) + 1, W)
            rr0 = max(int(np.floor((lat_all[0] - t.lat_max) / -tr.e)) - 1, 0)
            rr1 = min(int(np.ceil((lat_all[0] - t.lat_min) / -tr.e)) + 1, h)
            if c0 >= c1 or rr0 >= rr1:
                continue
            ts = time.time()
            lon, lat = np.meshgrid(lon_all[c0:c1], lat_all[rr0:rr1])
            v = sample(t.path, lon, lat)
            sub = (slice(rr0, rr1), slice(c0, c1))
            q = v[iq]
            anyv = (q != config.NODATA) & inside[sub]
            has = anyv & (v[ino] > 0)
            nat = (native[c0:c1] == t.epsg % 100)[None, :]
            p = np.where(has, np.where(nat, 4, 3), np.where(anyv, np.where(nat, 2, 1), 0)).astype("uint8")
            nat_has[sub] |= nat & has
            ps, vs, ids = prio[sub], val[:, rr0:rr1, c0:c1], idx[sub]
            both = (ps >= 3) & has
            if both.any():
                old, new = vs[:, both], v[:, both]
                ov.update(ids[both], t.idx, {k: decode(old[BI[k]], k, scales) for k in ("SoS50", "EoS50", "n_obs")},
                          {k: decode(new[BI[k]], k, scales) for k in ("SoS50", "EoS50", "n_obs")},
                          clean_masks(old[iq]), clean_masks(new[iq]))
            take = p > ps
            vs[:, take] = v[:, take]
            ids[take] = t.idx
            prio[sub] = np.maximum(ps, p)
            t_tile[t.tile] += time.time() - ts
        n_roi += int(inside.sum())
        n_unc += int((inside & (idx == 0)).sum())
        n_viol += int(((prio == 3) & nat_has).sum())
        cm = clean_masks(val[iq])
        pidx = placebo_ids(idx, r0, mid_r, mid_c)
        for b in SEAM_BANDS:
            d = decode(val[BI[b]], b, scales)
            seams[b].update(idx, d, cm[b])
            placebo[b].update(pidx, d, cm[b])
        win = Window(0, r0, W, h)
        for i, b in enumerate(config.OUT_BANDS):
            files[b].write(val[i], 1, window=win)
        files["source"].write(idx, 1, window=win)
        print(f"  rows {r0}-{r0 + h} / {H}: {len(hit)} tiles, {time.time() - t0:.0f}s", flush=True)
    t_write = time.time()
    factors = [f for f in (2, 4, 8, 16, 32, 64, 128) if min(H, W) // f >= 256]
    for b, f in files.items():
        if factors:
            f.build_overviews(factors, Resampling.nearest if b in config.OVR_NEAREST else Resampling.average)
        p = f.name
        f.close()
        os.replace(p, p[:-len(".part")])

    src = tiles[["idx", "tile", "epsg"]].copy()
    for c, jc in (("store", "store"), ("store_path", "store_path"), ("store_index", "index"), ("fill_path", "fill_path"),
                  ("n_dates", "n_dates"), ("input_status", "input_status")):
        src[c] = tiles[jc] if jc in tiles else ""
    src["path"], src["params_hash"] = tiles.path, params_hash
    src.to_csv(out / f"source_{a.year}.csv", index=False)
    names = dict(zip(tiles.idx, tiles.tile))
    o = ov.to_frame()
    o.insert(2, "tile_a", o.idx_a.map(names))
    o.insert(3, "tile_b", o.idx_b.map(names))
    o.to_csv(out / f"overlap_{a.year}.csv", index=False)
    s = seam_frame(seams, placebo, names)
    s.to_csv(out / f"seams_{a.year}.csv", index=False)

    crs_wkt = CRS.from_epsg(4326).to_wkt()
    bands = [(f"pheno_{b}_{a.year}.tif", True, 1, (W, H), (0, 0)) for b in config.OUT_BANDS]
    write_vrt(out / f"pheno_{a.year}.vrt", W, H, crs_wkt, tr,
              [("int16", b, config.NODATA, scales[BI[b]], [bands[i]]) for i, b in enumerate(config.OUT_BANDS)]
              + [("uint16", "source", 0, 1.0, [(f"source_{a.year}.tif", True, 1, (W, H), (0, 0))])])
    for epsg, g in tiles.groupby("epsg"):
        trs = [tile_grid(t)[1] for t in g.tile]
        x0, y1 = min(t.c for t in trs), max(t.f for t in trs)
        x1, y0 = max(t.c for t in trs) + 25000, min(t.f for t in trs) - 25000
        zw, zh = int(round((x1 - x0) / 10)), int(round((y1 - y0) / 10))
        ztr = Affine(10.0, 0, x0, 0, -10.0, y1)
        write_vrt(out / f"utm{epsg % 100}_{a.year}.vrt", zw, zh, CRS.from_epsg(int(epsg)).to_wkt(), ztr,
                  [("int16", b, config.NODATA, scales[i],
                    [(Path(p).resolve(), False, i + 1, (config.SUB, config.SUB),
                      (int(round((t.c - x0) / 10)), int(round((y1 - t.f) / 10)))) for p, t in zip(g.path, trs)])
                   for i, b in enumerate(config.OUT_BANDS)])

    pooled = ov.pooled()
    qc = dict(year=a.year, name=name, province=config.PROVINCE, edge_buffer_km=config.EDGE_BUFFER_KM,
              params_hash=params_hash, code_version=ver, n_tiles=len(tiles),
              grid=dict(res=[tr.a, -tr.e], origin=[tr.c, tr.f], H=H, W=W, dem_offset_px=dem_off,
                        effective_px_m=eff, bounds=list(region.bounds)),
              roi_pixels=n_roi, covered_share=1 - n_unc / n_roi if n_roi else None,
              roi_uncovered_share=n_unc / n_roi if n_roi else None,
              native_violation_share=n_viol / max(n_roi - n_unc, 1), native_violation_pixels=n_viol,
              overlap_pooled=pooled, missing_tiles=missing, scales=dict(zip(config.OUT_BANDS, scales)),
              seconds_per_tile=t_tile, mosaic_s_per_tile=float(np.mean(list(t_tile.values()))),
              overview_seconds=time.time() - t_write, total_seconds=time.time() - t0,
              peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (out / f"mosaic_qc_{a.year}.json").write_text(json.dumps(qc, indent=1, default=float))
    worst = s[~s.placebo].sort_values("ratio", ascending=False).head(1)
    print(f"overlap pooled SoS50 {pooled['SoS50']}; seams {len(s)} rows, worst ratio "
          f"{worst.ratio.iloc[0] if len(worst) else float('nan'):.2f}; uncovered {qc['roi_uncovered_share']}; "
          f"native violations {n_viol}", flush=True)
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    with rasterio.Env(COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW="2", GDAL_TIFF_OVR_BLOCKSIZE=str(BLOCK)):
        main()
