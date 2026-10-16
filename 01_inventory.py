"""Step 1: inventory of the Sentinel-2 Zarr stores and date-level input audit.

scan: one row per tile-year and store of the ACTIVE provinces (dates in the year and in config.SEARCH, longest
gap between dates in SEARCH, ROI, mislabelled id of the SK stores). The job list keeps, per tile-year in the ROI of
the ACTIVE provinces, one store (most dates in SEARCH), so a border tile held by two provinces' stores gets one job
and the census fill completes it to the same date set whichever province is mosaicked; ROI parts no job covers are listed as gaps and matched to tile ids of
config.TILE_LISTS that no store holds (hole_tiles.csv; fill-only candidates).

audit (when CENSUS_DIR/items_<year>.csv.gz from 00_census.py exists): every tile-date of a job gets a
status against the STAC catalogue, using the share of the 25 km tile square covered by each item
footprint (tile UTM):
  ok              stored, an item covers >= FOOTPRINT_MIN_FRAC of the tile
  orphan          stored, no item covers >= FOOTPRINT_MIN_FRAC (any cloud cover)
  missing         expected (item with cloud_cover < MAX_CLOUD and frac >= FOOTPRINT_MIN_FRAC), not stored
  truncated       missing, the store row is full (slots_full) and the date is after the last stored date
  cloud_filtered  not stored, covering items exist but all have cloud_cover >= MAX_CLOUD
  edge_only       not stored, only items with 0 < frac < FOOTPRINT_MIN_FRAC
  empty_suspect   (--masks) stored, no valid pixel although the best covering item has cc < EMPTY_SUSPECT_CC
and stored dates a baseline status (PB_OFFSET_MIN family of the covering expected items):
  baseline_fix    date <= PB_CUTOFF, all items carry the offset: the store kept +1000 DN (fixed at read time)
  baseline_mixed  both PB families present (pixels in two conventions; re-download)
  baseline_nofix  date > PB_CUTOFF, no item carries the offset: the store removed 1000 DN wrongly (re-download)
Only the census months (and --tiles of the census, items_<year>.meta.json) are audited.

Fill stores of 01b_fill.py (FILL/s2_<year>/<tile>.zarr with non-zero doys) are merged into the date
lists (fill wins). Per tile-year: completeness (expected dates in FILL_DOY stored / expected), the
same with fill, spring / season / search gaps, sibling deficit (DOYs in SEARCH present in >= 2 and in
at least half of the other sub-tiles of the same 100 km granule but not here), input_flags and input_status (worst of the QC
checks; sibling_deficit alone gives at most WARN).

Writes to outputs/inventory/: store_tiles.csv, jobs.csv (rows of other years are kept), gaps_<y>.csv,
hole_tiles.csv, tile_index.csv (append-only idx >= 1), inventory_<y>.png and, with a census,
date_audit_<y>.csv.gz, fill_<y>.csv, upstream_issues_<y>.csv.

    python -u 01_inventory.py [--years 2019 2024] [--masks] [--census-dir D] [--fill-root D] [--out D]
"""
import argparse
import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import zarr
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import unary_union

import config
from cpz.grid import fill_geom, footprint, plot_geom, roi, set_extent, tile_grid

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}
FILL_REASONS = ("missing", "truncated", "empty_suspect", "baseline_mixed", "baseline_nofix")
INT_COLUMNS = ["index", "max_t", "n_dates", "n_dates_search", "max_date_gap", "n_fill", "n_expected", "n_dates_merged",
               "spring_dates", "spring_gap", "season_gap", "max_date_gap_merged", "sibling_deficit"]
NEW_COLUMNS = ["fill_path", "n_fill", "offset_fix_doys", "n_expected", "completeness", "completeness_filled",
               "n_dates_merged", "spring_dates", "spring_gap", "season_gap", "max_date_gap_merged",
               "sibling_deficit", "input_flags", "input_status"]


def date_gap(d, lo, hi):
    d = np.asarray(d)
    s = np.sort(d[(d >= lo) & (d <= hi)])
    return int(np.diff(np.r_[lo, s, hi]).max())


def doy_date(year, doy):
    return (date(year, 1, 1) + timedelta(days=int(doy) - 1)).isoformat()


def date_doy(d):
    return date.fromisoformat(d).timetuple().tm_yday


def qc_status(name, value):
    warn, fail, higher_bad = config.QC[name]
    if value is None or not np.isfinite(value):
        return "SKIP"
    if higher_bad:
        return "PASS" if value <= warn else "WARN" if value <= fail else "FAIL"
    return "PASS" if value >= warn else "WARN" if value >= fail else "FAIL"


def worst(statuses):
    s = [x for x in statuses if x in RANK]
    return max(s, key=RANK.get) if s else "PASS"


def doys_ok(d, year):
    d = np.asarray(d)
    nd = 366 if calendar.isleap(year) else 365
    return bool(d.size == 0 or (d.min() >= 1 and d.max() <= nd and np.all(np.diff(d) > 0)))


def scan(years):
    rows, doys = [], {}
    lo, hi = config.SEARCH
    for name, root in config.S2_STORES.items():
        for y in years:
            p = root / f"s2_{y}.zarr"
            if not p.exists():
                continue
            g = zarr.open_group(str(p), mode="r")
            for i, (t, d) in enumerate(zip(g.attrs["mgrs_tiles"], g["doys"][:])):
                d = d[d > 0].astype(int)
                doys[(name, y, t)] = d
                rows.append(dict(year=y, tile=t, store=name, store_path=str(p), index=i, max_t=g["doys"].shape[1],
                                 n_dates=len(d), n_dates_search=int(((d >= lo) & (d <= hi)).sum()),
                                 max_date_gap=date_gap(d, lo, hi), slots_full=len(d) == g["doys"].shape[1],
                                 mislabelled=name.startswith("sk_") and t in config.MISLABELLED))
    return pd.DataFrame(rows), doys


def load_census(census_dir, year):
    p = census_dir / f"items_{year}.csv.gz"
    if not p.exists():
        return None, None
    df = pd.read_csv(p, dtype={"processing_baseline": str, "id": str, "date": str})
    df["processing_baseline"] = df.processing_baseline.fillna("")
    mp = census_dir / f"items_{year}.meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return df, meta


def tile_item_fractions(items, tiles):
    """Rows (tile, item, frac) for every census item whose footprint covers frac > 0 of the tile square."""
    wkt = items.footprint_wkt.fillna("").astype(str).to_numpy()
    geoms = shapely.from_wkt(np.where(wkt == "", None, wkt))
    ok = ~shapely.is_missing(geoms)
    out = []
    grids = {t: tile_grid(t) for t in tiles}
    for epsg in sorted({e for e, _ in grids.values()}):
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        g = shapely.transform(geoms[ok], lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
        g = shapely.make_valid(g)
        tree = shapely.STRtree(g)
        ts = [t for t, (e, _) in grids.items() if e == epsg]
        sq = np.array([box(grids[t][1].c, grids[t][1].f - 25000, grids[t][1].c + 25000, grids[t][1].f) for t in ts])
        si, gi = tree.query(sq, predicate="intersects")
        frac = shapely.area(shapely.intersection(sq[si], g[gi])) / 25000.0 ** 2
        out.append(pd.DataFrame(dict(tile=np.array(ts)[si], item=np.flatnonzero(ok)[gi], frac=frac)))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["tile", "item", "frac"])
    return df[df.frac > 0]


def pb_offset(pb):
    try:
        return float(pb) >= float(config.PB_OFFSET_MIN)
    except (TypeError, ValueError):
        return None


def baseline_status(d, pbs):
    fam = {f for f in (pb_offset(p) for p in pbs) if f is not None}
    if fam == {True, False}:
        return "baseline_mixed"
    if d <= config.PB_CUTOFF and fam == {True}:
        return "baseline_fix"
    if d > config.PB_CUTOFF and fam == {False}:
        return "baseline_nofix"
    return ""


def classify_dates(items, stored_doys, year, slots_full, months=None, valid_frac=None):
    """Status per tile-date.

    items: the tile's census items with frac > 0 (columns id, date, cloud_cover, processing_baseline, frac);
    stored_doys: base-store DOYs; valid_frac: optional {doy: tile valid fraction} (--masks).
    """
    stored = {doy_date(year, d): int(d) for d in stored_doys}
    last = max(stored) if stored else None
    dates = sorted(set(items.date) | set(stored))
    if months:
        dates = [d for d in dates if int(d[5:7]) in months]
    by_date = dict(tuple(items.groupby("date"))) if len(items) else {}
    rows = []
    for d in dates:
        it = by_date.get(d, items.iloc[:0])
        cover = it[it.frac >= config.FOOTPRINT_MIN_FRAC]
        exp = cover[cover.cloud_cover < config.MAX_CLOUD]
        use = exp if len(exp) else cover
        is_stored = d in stored
        if is_stored:
            status = "ok" if len(cover) else "orphan"
            if (status == "ok" and valid_frac is not None and valid_frac.get(stored[d], 1.0) == 0
                    and cover.cloud_cover.min() < config.EMPTY_SUSPECT_CC):
                status = "empty_suspect"
        elif len(exp):
            status = "truncated" if slots_full and last is not None and d > last else "missing"
        elif len(cover):
            status = "cloud_filtered"
        else:
            status = "edge_only"
        rows.append(dict(year=year, date=d, doy=date_doy(d), status=status, expected=bool(len(exp)), stored=is_stored,
                         baseline=baseline_status(d, use.processing_baseline) if is_stored else "",
                         n_items=len(it), cc_min=float(use.cloud_cover.min()) if len(use) else np.nan,
                         frac_max=round(float(it.frac.max()), 4) if len(it) else 0.0,
                         item_ids=";".join(use.id), pbs=";".join(sorted(set(use.processing_baseline)))))
    cols = ["year", "date", "doy", "status", "expected", "stored", "baseline", "n_items", "cc_min", "frac_max",
            "item_ids", "pbs"]
    return pd.DataFrame(rows, columns=cols)


def fill_rows(audit, hole=False):
    lo, hi = config.FILL_DOY
    a = audit[(audit.doy >= lo) & (audit.doy <= hi)]
    if hole:
        sel = a[a.expected].assign(reason="hole")
    else:
        st = a[a.status.isin(FILL_REASONS)].assign(reason=lambda x: x.status)
        bl = a[a.baseline.isin(FILL_REASONS) & ~a.status.isin(FILL_REASONS)].assign(reason=lambda x: x.baseline)
        sel = pd.concat([st, bl])
    return sel[["year", "date", "doy", "reason", "item_ids", "pbs", "cc_min"]].sort_values("doy")


def fill_doys(fill_root, year, tile):
    p = fill_root / f"s2_{year}" / f"{tile}.zarr"
    if not p.exists():
        return "", np.zeros(0, int)
    try:
        d = np.asarray(zarr.open_group(str(p), mode="r")["doys"][:]).astype(int)
    except (KeyError, FileNotFoundError, ValueError):
        return "", np.zeros(0, int)
    d = np.unique(d[d > 0])
    return (str(p), d) if len(d) else ("", d)


def sibling_deficit(dates_by_tile):
    """DOYs in SEARCH present in >= 2 and >= half of the other sub-tiles of the same 100 km granule, absent here.

    The majority rule ignores dates that only part of a granule gets legitimately (orbit-swath edges, and
    dates whose only item touches the tile's lon/lat search bbox)."""
    lo, hi = config.SEARCH
    sets = {t: {int(x) for x in d if lo <= x <= hi} for t, d in dates_by_tile.items()}
    out = {}
    for t, s in sets.items():
        sib = [sets[u] for u in sets if u != t and u[:5] == t[:5]]
        cnt = {}
        for u in sib:
            for x in u:
                cnt[x] = cnt.get(x, 0) + 1
        need = max(2, len(sib) / 2)
        out[t] = sum(1 for x, c in cnt.items() if c >= need and x not in s)
    return out


def input_summary(base_doys, fill_d, audit, year, sib, integrity_ok):
    """New jobs.csv columns of one tile-year (audit None without a census)."""
    merged = np.union1d(np.asarray(base_doys, int), fill_d)
    fset = set(int(x) for x in fill_d)
    r = dict(n_fill=len(fill_d), n_dates_merged=len(merged), sibling_deficit=int(sib))
    r["spring_dates"] = int(((merged >= config.SPRING[0]) & (merged <= config.SPRING[1])).sum())
    r["spring_gap"] = date_gap(merged, *config.SPRING)
    r["season_gap"] = date_gap(merged, *config.SEASON)
    r["max_date_gap_merged"] = date_gap(merged, *config.SEARCH)
    flags, checks = [], {}
    if audit is not None:
        lo, hi = config.FILL_DOY
        a = audit[(audit.doy >= lo) & (audit.doy <= hi)]
        exp = a[a.expected]
        r["n_expected"] = len(exp)
        r["completeness"] = float(exp.stored.mean()) if len(exp) else np.nan
        r["completeness_filled"] = float((exp.stored | exp.doy.isin(fset)).mean()) if len(exp) else np.nan
        fix = audit[(audit.baseline == "baseline_fix") & ~audit.doy.isin(fset)].doy
        r["offset_fix_doys"] = ";".join(str(int(x)) for x in sorted(fix))
        trunc_left = int(((audit.status == "truncated") & ~audit.doy.isin(fset)).sum())
        checks["completeness"] = qc_status("completeness", r["completeness_filled"])
        checks["truncation"] = "FAIL" if trunc_left else "PASS"
        if checks["completeness"] in ("WARN", "FAIL"):
            flags.append("short")
        if trunc_left:
            flags.append("truncated")
        if len(fix):
            flags.append("baseline_fix")
        for b in ("baseline_mixed", "baseline_nofix"):
            if ((audit.baseline == b) & ~audit.doy.isin(fset)).any():
                flags.append(b)
        for s in ("orphan", "empty_suspect"):
            if ((audit.status == s) & ~audit.doy.isin(fset)).any():
                flags.append(s)
    else:
        r.update(n_expected=-1, completeness=np.nan, completeness_filled=np.nan, offset_fix_doys="")
    checks["spring_gap"] = ("FAIL" if r["spring_dates"] < config.SPRING_MIN_DATES
                            else qc_status("spring_gap", r["spring_gap"]))
    checks["season_gap"] = qc_status("season_gap", r["season_gap"])
    checks["doy_integrity"] = "PASS" if integrity_ok else "FAIL"
    sd = qc_status("sibling_deficit", sib)
    checks["sibling_deficit"] = "WARN" if sd == "FAIL" else sd
    for k in ("spring_gap", "season_gap", "sibling_deficit"):
        if checks[k] != "PASS":
            flags.append(k)
    if not integrity_ok:
        flags.append("doy_integrity")
    r["input_flags"] = ";".join(flags)
    r["input_status"] = worst(checks.values())
    return r


def update_tile_index(path, tiles):
    """Append-only idx (>= 1) per tile; existing ids never change."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["idx", "tile"])
    new = sorted(set(tiles) - set(old.tile))
    start = int(old.idx.max()) + 1 if len(old) else 1
    df = pd.concat([old, pd.DataFrame(dict(idx=range(start, start + len(new)), tile=new))], ignore_index=True)
    df["idx"] = df.idx.astype(int)
    df.to_csv(path, index=False)
    return df


def tile_list_features():
    feats = {}
    for p in config.TILE_LISTS:
        for f in json.loads(p.read_text())["features"]:
            t = f["properties"].get("MGRS")
            if t and t not in feats:
                feats[t] = shape(f["geometry"])
    return feats


def hole_tiles(parts, year, stored_tiles, feats, min_area=1e-5):
    rows = []
    for p in parts:
        if p.area <= min_area:
            continue
        pt = p.representative_point()
        for t, g in feats.items():
            if t not in config.MISLABELLED and t not in stored_tiles and g.covers(pt):
                rows.append(dict(year=year, tile=t, lon=round(pt.x, 5), lat=round(pt.y, 5), area_deg2=p.area))
    return rows


def masks_valid_frac(store_path, index, doys):
    g = zarr.open_group(store_path, mode="r")
    t_hi = int(np.flatnonzero(g["doys"][index] > 0).max()) + 1
    row = np.asarray(g["doys"][index][:t_hi])
    n = np.zeros(t_hi)
    for r0 in range(0, config.SUB, 512):
        for c0 in range(0, config.SUB, 512):
            m = g["masks"][index, :t_hi, r0:min(r0 + 512, config.SUB), c0:min(c0 + 512, config.SUB)]
            n += (m > 0).sum(axis=(1, 2))
    return {int(d): float(v) / config.SUB ** 2 for d, v in zip(row, n) if d > 0}


def merge_years(path, df, years):
    if path.exists():
        old = pd.read_csv(path, dtype={"offset_fix_doys": str, "input_flags": str, "fill_path": str})
        keep = old[~old.year.isin(years)]
        if len(keep):
            df = pd.concat([keep, df], ignore_index=True)
    for c in INT_COLUMNS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    df = df.sort_values(["year", "tile"] + (["store"] if "store" in df else []))
    df.to_csv(path, index=False)
    return df


def plot_year(path, y, jy, fps, gap, region):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    panels = [("n_dates_search", "viridis", 0, None, "acquisition dates in DOY %d-%d" % config.SEARCH),
              ("max_date_gap", "magma_r", 0, None, "longest gap between dates (days)"),
              ("completeness", "RdYlGn", 0.8, 1.0, "completeness (expected dates stored, DOY %d-%d)" % config.FILL_DOY),
              ("completeness_filled", "RdYlGn", 0.8, 1.0, "completeness with fill")]
    fail = jy[jy.input_status == "FAIL"]
    for ax, (col, cmap, vmin, vmax, label) in zip(axes.flat, panels):
        vals = pd.to_numeric(jy[col], errors="coerce")
        if vmax is None:
            vmax = max(vals.quantile(0.98), 1)
        cm, norm = plt.get_cmap(cmap), plt.Normalize(vmin, vmax)
        for t, v in zip(jy.tile, vals):
            fill_geom(ax, fps[t], color=cm(norm(v)) if np.isfinite(v) else "#BBBBBB", ec="white", lw=0.2)
        for t in fail.tile:
            fill_geom(ax, fps[t], fill=False, hatch="///", ec="black", lw=0.3)
        fill_geom(ax, gap, color="#D7263D", alpha=0.6)
        plot_geom(ax, region, color="black", lw=1)
        set_extent(ax, region)
        ax.set_title(label)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=norm), ax=ax, shrink=0.75, pad=0.02, label=label)
    axes[0, 1].legend(handles=[Patch(color="#D7263D", alpha=0.6, label="ROI without tile"),
                               Patch(fill=False, hatch="///", ec="black", label="input_status FAIL"),
                               Patch(color="#BBBBBB", label="no census"),
                               plt.Line2D([], [], color="black", label="ROI")],
                      loc="upper left", bbox_to_anchor=(1.25, 1), frameon=False)
    fig.suptitle(f"S2 store inventory {y}: {len(jy)} tiles in the ROI, input_status "
                 f"{jy.input_status.value_counts().to_dict()}", fontsize=14)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=config.YEARS)
    ap.add_argument("--masks", action="store_true", help="read store masks for empty_suspect (SLURM only)")
    ap.add_argument("--census-dir", type=Path, default=config.CENSUS_DIR)
    ap.add_argument("--fill-root", type=Path, default=config.FILL)
    ap.add_argument("--out", type=Path, default=config.INVENTORY)
    a = ap.parse_args()
    inv = a.out
    inv.mkdir(parents=True, exist_ok=True)
    region = roi()
    feats = tile_list_features()

    st, store_doys = scan(a.years)
    fps = {t: footprint(t) for t in st.tile.unique()}
    st["in_roi"] = st.tile.map(lambda t: fps[t].intersects(region))
    merge_years(inv / "store_tiles.csv", st, a.years)
    jobs = (st[st.in_roi & ~st.mislabelled].sort_values("n_dates_search", ascending=False)
            .drop_duplicates(["year", "tile"]).sort_values(["year", "tile"])).reset_index(drop=True)
    print(f"store tile-years {len(st)} {st.groupby('store').size().to_dict()}; in ROI {int(st.in_roi.sum())}; "
          f"mislabelled {int(st.mislabelled.sum())}; date slots full (possible truncation) {int(st.slots_full.sum())}")

    holes, out_jobs = [], []
    for y in a.years:
        jy = jobs[jobs.year == y].copy()
        gap = region.difference(unary_union([fps[t] for t in jy.tile]))
        parts = [p for p in getattr(gap, "geoms", [gap]) if not p.is_empty]
        pd.DataFrame([dict(lon=p.centroid.x, lat=p.centroid.y, area_deg2=p.area) for p in parts]).to_csv(
            inv / f"gaps_{y}.csv", index=False)
        hy = pd.DataFrame(hole_tiles(parts, y, set(st[st.year == y].tile), feats),
                          columns=["year", "tile", "lon", "lat", "area_deg2"]).drop_duplicates("tile")
        holes.append(hy)

        census, meta = load_census(a.census_dir, y)
        audit_tiles = set()
        if census is not None:
            audit_tiles = set(meta.get("tiles") or (set(jy.tile) | set(hy.tile))) & (set(jy.tile) | set(hy.tile))
            months = meta.get("months")
            fr = tile_item_fractions(census, sorted(audit_tiles))
            fr = fr.join(census.drop(columns="footprint_wkt"), on="item")
            fr_by_tile = dict(tuple(fr.groupby("tile")))
        base = {r.tile: store_doys[(r.store, y, r.tile)] for r in jy.itertuples()}
        fills = {t: fill_doys(a.fill_root, y, t) for t in set(jy.tile) | set(hy.tile)}
        for h in hy.tile:
            if fills[h][0]:
                fps.setdefault(h, footprint(h))
                jy = pd.concat([jy, pd.DataFrame([dict(year=y, tile=h, store="fill", store_path="", index=-1, max_t=0,
                                                       n_dates=0, n_dates_search=0,
                                                       max_date_gap=date_gap([], *config.SEARCH), slots_full=False,
                                                       mislabelled=False, in_roi=True)])], ignore_index=True)
                base[h] = np.zeros(0, int)
        sib = sibling_deficit({t: np.union1d(base[t], fills[t][1]) for t in jy.tile})

        audits, fill_list, upstream, extra = [], [], [], []
        for r in jy.itertuples():
            t = r.tile
            audit = None
            if t in audit_tiles:
                vf = masks_valid_frac(r.store_path, r.index, base[t]) if a.masks and r.store_path else None
                items = fr_by_tile.get(t, fr.iloc[:0])
                audit = classify_dates(items, base[t], y, bool(r.slots_full), months, vf).assign(tile=t)
                audits.append(audit)
                fill_list.append(fill_rows(audit).assign(tile=t))
                c = audit.status.value_counts().to_dict()
                b = audit.baseline.value_counts().to_dict()
                upstream.append(dict(year=y, tile=t, store=r.store, n_stored=int(audit.stored.sum()),
                                     n_expected=int(audit.expected.sum()),
                                     **{k: int(c.get(k, 0)) for k in ("missing", "truncated", "orphan",
                                                                      "empty_suspect", "cloud_filtered")},
                                     missing_clear=int(((audit.status == "missing") & (audit.cc_min < 30)
                                                        & audit.doy.between(*config.SEARCH)).sum()),
                                     **{k: int(b.get(k, 0)) for k in ("baseline_fix", "baseline_mixed",
                                                                      "baseline_nofix")}))
            extra.append(dict(fill_path=fills[t][0],
                              **input_summary(base[t], fills[t][1], audit, y, sib[t], doys_ok(base[t], y))))
        for h in hy.tile:
            if h in audit_tiles and not fills[h][0]:
                audit = classify_dates(fr_by_tile.get(h, fr.iloc[:0]), [], y, False, months).assign(tile=h)
                audits.append(audit)
                fill_list.append(fill_rows(audit, hole=True).assign(tile=h))
        jy = pd.concat([jy.reset_index(drop=True), pd.DataFrame(extra)[NEW_COLUMNS]], axis=1)
        out_jobs.append(jy)

        if census is not None:
            cols = ["year", "tile", "date", "doy", "status", "baseline", "n_items", "cc_min", "frac_max",
                    "expected", "stored", "item_ids", "pbs"]
            pd.concat(audits)[cols].to_csv(inv / f"date_audit_{y}.csv.gz", index=False, compression="gzip")
            fl = pd.concat(fill_list) if fill_list else pd.DataFrame()
            fcols = ["year", "tile", "date", "doy", "reason", "item_ids", "pbs", "cc_min"]
            fl = fl[fcols] if len(fl) else pd.DataFrame(columns=fcols)
            fl.to_csv(inv / f"fill_{y}.csv", index=False)
            up = pd.DataFrame(upstream)
            issue = up[["missing", "truncated", "orphan", "empty_suspect", "baseline_fix", "baseline_mixed",
                        "baseline_nofix"]].sum(axis=1) > 0 if len(up) else []
            (up[issue] if len(up) else up).to_csv(inv / f"upstream_issues_{y}.csv", index=False)
            print(f"{y} audit ({len(audit_tiles)} tiles, months {months}): "
                  f"{pd.concat(audits).status.value_counts().to_dict()}; baseline "
                  f"{pd.concat(audits).baseline.replace('', 'none').value_counts().to_dict()}; "
                  f"fill list {len(fl)} tile-dates on {fl.tile.nunique()} tiles", flush=True)
        else:
            print(f"{y}: no census in {a.census_dir} (sibling and gap screens only, n_expected = -1)", flush=True)
        plot_year(inv / f"inventory_{y}.png", y, jy, fps, gap, region)

    jobs = pd.concat(out_jobs, ignore_index=True)
    jobs = merge_years(inv / "jobs.csv", jobs, a.years)
    hz = pd.concat(holes, ignore_index=True)
    merge_years(inv / "hole_tiles.csv", hz, a.years)
    update_tile_index(inv / "tile_index.csv", jobs.tile.unique())
    jy = jobs[jobs.year.isin(a.years)]
    print(jy.groupby("year").agg(tiles=("tile", "size"), dates_median=("n_dates_search", "median"),
                                 dates_min=("n_dates_search", "min"), max_gap_median=("max_date_gap_merged", "median"),
                                 gap_over_30d=("max_date_gap_merged", lambda s: int((s > 30).sum())),
                                 fill=("n_fill", lambda s: int((s > 0).sum())),
                                 sibling_warn=("sibling_deficit", lambda s: int((s > config.QC["sibling_deficit"][0]).sum())),
                                 PASS=("input_status", lambda s: int((s == "PASS").sum())),
                                 WARN=("input_status", lambda s: int((s == "WARN").sum())),
                                 FAIL=("input_status", lambda s: int((s == "FAIL").sum()))).to_string())
    if len(hz):
        print("ROI hole tiles (fill-only candidates):", hz.groupby("tile").year.apply(list).to_dict())
    print(f"wrote {inv}", flush=True)


if __name__ == "__main__":
    main()
