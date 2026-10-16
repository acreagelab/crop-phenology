"""Shared QC library for steps 03 (mosaic) and 04 (report).

Check rows: every test returns Check(stage, area, year, test, artefact, unit, id, metric, value, n, note) whose
status follows config.QC[name] = (warn, fail, higher_is_bad): higher_is_bad -> value <= warn PASS, <= fail WARN,
else FAIL; otherwise value >= warn PASS, >= fail WARN, else FAIL. A non-finite value gives SKIP. FLAGGED_INPUT
and SKIP can be set explicitly.

Readers: read_tile (int16 step-2 tile -> decoded float32 at 10*step m), cropland (WorldCover cropland AND the
province crop masks on the tile grid), aci_on_tile (AAFC ACI with the file's own CRS).
Artefact metrics: spike_ratio / edge_share_ratio (clamping), boundary_matched_density (orbit stripes),
tile_residuals / neighbours (tile shifts), BoundaryAccumulator (source-boundary seams, streamed by row chunks),
OverlapAccumulator (cross-zone overlap agreement), phase_shift (georeferencing).
Maps: roi_map, categorical_qa.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

import config
from cpz.grid import footprint, tile_grid

COLUMNS = ["stage", "area", "year", "test", "artefact", "unit", "id", "metric", "value", "warn", "fail", "status", "n",
           "note"]
ARTEFACTS = {"tile_shift", "orbit_stripe", "clamping", "unbounded", "geometry", "input", "plausibility", "external"}
STATUSES = ("PASS", "WARN", "FAIL", "FLAGGED_INPUT", "SKIP")
SEVERITY = {"SKIP": 0, "PASS": 1, "WARN": 2, "FLAGGED_INPUT": 3, "FAIL": 4}
NBIN = 367
ACI_CODES = {"barley": 133, "oats": 136, "winter wheat": 145, "durum": 147, "spring wheat": 146, "canola": 153,
             "flax": 154, "peas": 162, "lentils": 174}
QA_CLASSES = ["clean", "gap at SoS", "outside search", "few obs", "low amplitude", "overshoot", "snow / green start"]
_QA_RULES = [(4 | 64, 1), (16 | 32, 2), (2, 3), (1, 4), (1024, 5), (256 | 512, 6)]


def status(name, value, thr=None):
    warn, fail, hib = thr if thr is not None else config.QC[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "SKIP"
    if not np.isfinite(v):
        return "SKIP"
    if hib:
        return "PASS" if v <= warn else "WARN" if v <= fail else "FAIL"
    return "PASS" if v >= warn else "WARN" if v >= fail else "FAIL"


@dataclass
class Check:
    stage: str
    area: str
    year: int
    test: str
    artefact: str
    unit: str
    id: str
    metric: str
    value: float
    n: int = 0
    note: str = ""
    status: str = None
    qc: str = None
    thr: tuple = None
    warn: float = np.nan
    fail: float = np.nan

    def __post_init__(self):
        if self.artefact not in ARTEFACTS:
            raise ValueError(f"unknown artefact {self.artefact}")
        name = self.qc or self.metric
        thr = self.thr if self.thr is not None else config.QC.get(name)
        if thr is not None:
            self.warn, self.fail = thr[0], thr[1]
        if self.status is None:
            self.status = status(name, self.value, thr) if thr is not None else "SKIP"
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status}")

    def row(self):
        d = asdict(self)
        return {k: d[k] for k in COLUMNS}


def checks_frame(rows):
    return pd.DataFrame([r.row() if isinstance(r, Check) else r for r in rows], columns=COLUMNS)


def write_checks(rows, path):
    df = checks_frame(rows)
    df.to_csv(path, index=False)
    return df


def worst(statuses):
    s = [x for x in statuses if x in SEVERITY]
    return max(s, key=SEVERITY.get) if s else "SKIP"


def read_tile(path, bands, step=2):
    """{band: float32 (2500/step)^2} physical values (stored * dataset scale), dataset nodata -> NaN."""
    out = {}
    with rasterio.open(path) as r:
        desc = list(r.descriptions)
        names = desc if all(desc) else list(config.OUT_BANDS)
        for b in bands:
            i = names.index(b)
            a = r.read(i + 1)[::step, ::step]
            v = a.astype("float32") * np.float32(r.scales[i])
            if r.nodata is not None:
                v[a == r.nodata] = np.nan
            out[b] = v
    return out


def _onto_tile(path, tile, step, dtype="uint8"):
    """Raster at path -> (n, n) array on the tile grid at 10*step m, nearest; None if no overlap."""
    epsg, tr = tile_grid(tile)
    n = config.SUB // step
    dtr = tr * Affine.scale(step)
    with rasterio.open(path) as s:
        if (s.crs.to_epsg() == epsg and s.transform.almost_equals(tr, precision=1e-6) and s.width == config.SUB
                and s.height == config.SUB):
            return s.read(1)[::step, ::step].astype(dtype)
        l, t = tr.c, tr.f
        b = transform_bounds(f"EPSG:{epsg}", s.crs, l, t - 25000, l + 25000, t, densify_pts=21)
        pad = 4 * abs(s.transform.a)
        w = from_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, s.transform)
        c0, r0 = max(int(np.floor(w.col_off)), 0), max(int(np.floor(w.row_off)), 0)
        c1 = min(int(np.ceil(w.col_off + w.width)), s.width)
        r1 = min(int(np.ceil(w.row_off + w.height)), s.height)
        if c1 <= c0 or r1 <= r0:
            return None
        w = Window(c0, r0, c1 - c0, r1 - r0)
        src = s.read(1, window=w)
        dst = np.zeros((n, n), src.dtype)
        reproject(src, dst, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=dtr,
                  dst_crs=f"EPSG:{epsg}", resampling=Resampling.nearest, src_nodata=s.nodata, dst_nodata=0)
    return dst.astype(dtype)


def worldcover_path(tile, year):
    y = int(np.clip(year, 2019, 2024))
    p = config.WORLDCOVER_DIR / str(y) / f"lc_{tile}.tif"
    if p.exists():
        return p
    for d in sorted(config.WORLDCOVER_DIR.glob("*")):
        q = d / f"lc_{tile}.tif"
        if q.exists():
            return q
    return None


def cropland(tile, year, step=2, with_note=False):
    """WorldCover == CROP_WC_CLASS AND crop_mask_sk > 0 on the tile grid at 10*step m (bool)."""
    n = config.SUB // step
    notes = []
    wc = worldcover_path(tile, year)
    if wc is not None:
        m = _onto_tile(wc, tile, step)
        m = np.zeros((n, n), bool) if m is None else m == config.CROP_WC_CLASS
        if wc.parent.name != str(int(np.clip(year, 2019, 2024))):
            notes.append(f"worldcover {wc.parent.name}")
    else:
        m = np.ones((n, n), bool)
        notes.append("no worldcover: crop mask only")
    cm = None
    for p in cropmasks():
        x = _onto_tile(p, tile, step)
        if x is not None:
            cm = x if cm is None else np.maximum(cm, x)
    if cm is not None:
        m &= cm > 0
    else:
        notes.append("no crop mask: WorldCover cropland only")
    return (m, "; ".join(notes)) if with_note else m


def cropmasks():
    """Existing province crop masks, config.PROVINCE first (border tiles take the union of all that overlap)."""
    order = [config.PROVINCE] + [c for c in config.PROVINCES if c != config.PROVINCE]
    ps = [config.PROVINCES[c]["cropmask"] for c in order]
    return [p for p in ps if p is not None and p.exists()]


def aci_on_tile(tile, year, step=2):
    """AAFC ACI on the tile grid: config.PROVINCE file first, the other provinces' files where it has no data."""
    out = None
    order = [config.PROVINCE] + [c for c in config.PROVINCES if c != config.PROVINCE]
    for c in order:
        p = config.ACI_DIR / str(year) / f"aci_{year}_{config.PROVINCES[c]['abbr']}.tif"
        x = _onto_tile(p, tile, step) if p.exists() else None
        if x is not None:
            out = x if out is None else np.where(out > 0, out, x)
            if (out > 0).all():
                break
    return out


def _hist(values):
    v = np.asarray(values, "float64").ravel()
    v = v[np.isfinite(v)]
    return np.bincount(np.clip(np.rint(v), 0, NBIN - 1).astype(np.int64), minlength=NBIN)


def spike_ratio(values_int=None, counts=None, win=7, min_share=0.01):
    """Per-DOY count / median(count in d-win..d+win excluding d); returns (frame, max ratio over DOYs with
    share >= min_share). counts: precomputed DOY histogram (length NBIN) instead of values."""
    c = _hist(values_int) if counts is None else np.asarray(counts, "int64")
    tot = c.sum()
    rows = []
    for d in np.flatnonzero(c):
        nb = np.r_[c[max(d - win, 0):d], c[d + 1:d + win + 1]]
        exp = float(np.median(nb)) if nb.size else 0.0
        rows.append((d, c[d], c[d] / tot, exp, c[d] / max(exp, 1.0)))
    df = pd.DataFrame(rows, columns=["doy", "count", "share", "expected", "ratio"])
    big = df[df.share >= min_share]
    return df, float(big.ratio.max()) if len(big) else np.nan


def edge_share_ratio(values=None, doys=(60, 61, 329, 330, 141, 240), counts=None, win=7, search=None):
    """For each DOY: share / local expected share (median share of d+-win inside SEARCH, excluding the listed
    DOYs). Ratio NaN when the local expectation is 0 and the DOY is empty."""
    c = _hist(values) if counts is None else np.asarray(counts, "int64")
    tot = max(int(c.sum()), 1)
    s0, s1 = search or config.SEARCH
    rows = []
    for d in doys:
        nb = [k for k in range(d - win, d + win + 1) if k != d and k not in doys and s0 <= k <= s1]
        exp = float(np.median(c[nb])) / tot if nb else 0.0
        share = c[d] / tot
        ratio = share / max(exp, 1.0 / tot) if (share > 0 or exp > 0) else np.nan
        rows.append((d, share, exp, ratio))
    return pd.DataFrame(rows, columns=["doy", "share", "expected", "ratio"])


def otsu(values, bins=None):
    v = np.asarray(values, "float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    lo, hi = np.floor(v.min()), np.ceil(v.max()) + 1
    h, e = np.histogram(v, bins=bins or max(int(hi - lo), 2), range=(lo, hi))
    p = h / h.sum()
    w0 = np.cumsum(p)
    m = np.cumsum(p * e[:-1])
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = -1
    return float(e[np.argmax(sb) + 1])


def boundary_matched_density(sos, n_obs, crop, classes=None, px_m=20, buffer_m=1000, min_px=500, min_sep=10):
    """Median SoS of low-n_obs minus high-n_obs crop pixels within buffer_m of the n_obs class boundary
    (Otsu split of the crop n_obs histogram), class-matched (classes with >= min_px per side), area-weighted.
    Returns (diff, n, naive_diff); diff NaN (SKIP) if the class means differ by < min_sep or too few pixels."""
    from scipy.ndimage import distance_transform_edt
    fin = np.isfinite(n_obs)
    ok = crop & fin & np.isfinite(sos)
    if ok.sum() < 2 * min_px:
        return np.nan, 0, np.nan
    thr = otsu(n_obs[ok])
    lo, hi = fin & (n_obs < thr), fin & (n_obs >= thr)
    naive = np.nan
    if (ok & lo).any() and (ok & hi).any():
        naive = float(np.median(sos[ok & lo]) - np.median(sos[ok & hi]))
        if abs(n_obs[ok & hi].mean() - n_obs[ok & lo].mean()) < min_sep:
            return np.nan, 0, naive
    else:
        return np.nan, 0, naive
    lim = buffer_m / px_m
    near_lo = ok & lo & (distance_transform_edt(~hi) <= lim)
    near_hi = ok & hi & (distance_transform_edt(~lo) <= lim)
    groups = [None] if classes is None else [c for c in np.unique(classes[near_lo | near_hi]) if c != 0]
    num = den = 0.0
    for c in groups:
        a, b = (near_lo, near_hi) if c is None else (near_lo & (classes == c), near_hi & (classes == c))
        na, nb = int(a.sum()), int(b.sum())
        if na < min_px or nb < min_px:
            continue
        num += (np.median(sos[a]) - np.median(sos[b])) * (na + nb)
        den += na + nb
    return (num / den if den else np.nan), int(den), naive


def neighbours(tiles, buffer=0.01):
    """{tile: [tiles whose EPSG:4326 footprints intersect after a buffer of `buffer` deg]}, across zones."""
    fp = {t: footprint(t).buffer(buffer) for t in tiles}
    return {t: [u for u in tiles if u != t and fp[t].intersects(fp[u])] for t in tiles}


def tile_residuals(stats, nbrs):
    """stats: DataFrame tile, value[, class][, n]. Residual = value - median of the neighbours' values,
    class-matched when a class column exists, then area-weighted (n) over the tile's classes."""
    s = stats.dropna(subset=["value"]).copy()
    if "class" not in s:
        s["class"] = 0
    if "n" not in s:
        s["n"] = 1
    by = {(t, c): v for t, c, v in zip(s.tile, s["class"], s.value)}
    rows = []
    for t, g in s.groupby("tile"):
        num = den = 0.0
        n_nb = 0
        for c, v, n in zip(g["class"], g.value, g.n):
            vals = [by[(u, c)] for u in nbrs.get(t, []) if (u, c) in by]
            if not vals:
                continue
            num += (v - np.median(vals)) * n
            den += n
            n_nb = max(n_nb, len(vals))
        rows.append(dict(tile=t, residual=num / den if den else np.nan, n_nb=n_nb, n=int(g.n.sum())))
    return pd.DataFrame(rows, columns=["tile", "residual", "n_nb", "n"])


def _hist_median(h, lo=0):
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    return float(lo + np.searchsorted(c, 0.5 * c[-1]))


class BoundaryAccumulator:
    """Seam statistics at source-index boundaries, streamed over row chunks (top to bottom).

    update(idx, val, clean) per chunk: idx (h, W) uint16 (0 = none), val (h, W) float32 decoded DOY (NaN),
    clean (h, W) bool. Per unordered pair a < b: n and sum |delta| over horizontally and vertically adjacent
    pixels with different idx (both clean and finite); per idx the within-tile adjacent |delta|; and DOY
    histograms of the clean values 1..K pixels into each side along the row (left/right neighbours) or column
    (upper/lower neighbours) while the idx is unchanged. The accumulator keeps the last K rows of the previous
    chunk itself (the carry), so the result does not depend on the chunking; to_frame() flushes the boundaries
    whose lower strips are still open."""

    def __init__(self, K=32):
        self.K = K
        self.keys = {}
        self.cross = np.zeros((0, 2))
        self.hist = np.zeros((0, 2, NBIN), np.int64)
        self.win_s = np.zeros(65536)
        self.win_n = np.zeros(65536, np.int64)
        self.buf = None
        self.done = 0
        self.row0 = 0

    def _rows(self, keys):
        new = [k for k in np.unique(keys) if k not in self.keys]
        if new:
            for k in new:
                self.keys[int(k)] = len(self.keys)
            self.cross = np.vstack([self.cross, np.zeros((len(new), 2))])
            self.hist = np.concatenate([self.hist, np.zeros((len(new), 2, NBIN), np.int64)])
        return np.array([self.keys[int(k)] for k in keys], np.int64) if len(keys) else np.zeros(0, np.int64)

    def _pairs(self, ia, ib, va, vb, ca, cb):
        diff = (ia != ib) & (ia > 0) & (ib > 0)
        good = ca & cb & np.isfinite(va) & np.isfinite(vb)
        same = (ia == ib) & (ia > 0) & good
        if same.any():
            d = np.abs(va[same] - vb[same])
            self.win_s += np.bincount(ia[same], weights=d, minlength=65536)
            self.win_n += np.bincount(ia[same], minlength=65536)
        x = diff & good
        if x.any():
            a, b = np.minimum(ia[x], ib[x]).astype(np.int64), np.maximum(ia[x], ib[x]).astype(np.int64)
            k = a * 65536 + b
            u, inv = np.unique(k, return_inverse=True)
            r = self._rows(u)
            np.add.at(self.cross, (r[inv], 0), 1)
            np.add.at(self.cross, (r[inv], 1), np.abs(va[x] - vb[x]))

    def _strips(self, I, V, C, rr, cc, dr, dc, own, side_of):
        """Walk 1..K pixels from (rr, cc) in direction (dr, dc) while I == own; add clean finite values."""
        alive = np.ones(rr.size, bool)
        H, W = I.shape
        for k in range(self.K):
            r, c = rr + dr * k, cc + dc * k
            inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            alive &= inb
            rs, cs = np.where(alive, r, 0), np.where(alive, c, 0)
            alive &= I[rs, cs] == own
            if not alive.any():
                break
            m = alive & C[rs, cs] & np.isfinite(V[rs, cs])
            if m.any():
                b = np.clip(np.rint(V[rs[m], cs[m]]), 0, NBIN - 1).astype(np.int64)
                np.add.at(self.hist, (side_of[0][m], side_of[1][m], b), 1)

    def _boundary(self, I, V, C, r1, c1, r2, c2, d):
        """Boundary between (r1, c1) and (r2, c2) = (r1, c1) + d: strips go -d from side 1 and +d from side 2."""
        a1, a2 = I[r1, c1], I[r2, c2]
        m = (a1 != a2) & (a1 > 0) & (a2 > 0)
        if not m.any():
            return
        r1, c1, r2, c2, a1, a2 = r1[m], c1[m], r2[m], c2[m], a1[m].astype(np.int64), a2[m].astype(np.int64)
        lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
        rows = self._rows(lo * 65536 + hi)
        s1 = np.where(a1 == lo, 0, 1)
        self._strips(I, V, C, r1, c1, -d[0], -d[1], a1, (rows, s1))
        self._strips(I, V, C, r2, c2, d[0], d[1], a2, (rows, 1 - s1))

    def update(self, idx, val, clean, carry=None):
        idx = np.asarray(idx)
        if idx.dtype.kind not in "u":
            idx = idx.astype(np.int64)
        val = np.asarray(val, "float32")
        clean = np.asarray(clean, bool)
        self._pairs(idx[:, :-1], idx[:, 1:], val[:, :-1], val[:, 1:], clean[:, :-1], clean[:, 1:])
        rr, cc = np.nonzero(idx[:, :-1] != idx[:, 1:])
        self._boundary(idx, val, clean, rr, cc, rr, cc + 1, (0, 1))
        if self.buf is None:
            self.buf = (idx, val, clean)
        else:
            self.buf = tuple(np.concatenate([b, x]) for b, x in zip(self.buf, (idx, val, clean)))
        self._vertical(final=False)

    def _vertical(self, final):
        I, V, C = self.buf
        top = self.row0 + I.shape[0]
        last = top - 1 if final else top - 1 - self.K
        if last <= self.done:
            return
        a = np.arange(self.done, last) - self.row0
        self._pairs(I[a], I[a + 1], V[a], V[a + 1], C[a], C[a + 1])
        rr, cc = np.nonzero(I[a] != I[a + 1])
        rr = a[rr]
        self._boundary(I, V, C, rr, cc, rr + 1, cc, (1, 0))
        self.done = last
        keep = max(self.done - self.K + 1 - self.row0, 0)
        if keep:
            self.buf = tuple(b[keep:] for b in self.buf)
            self.row0 += keep

    def to_frame(self):
        if self.buf is not None:
            self._vertical(final=True)
        rows = []
        wm = np.where(self.win_n > 0, self.win_s / np.maximum(self.win_n, 1), np.nan)
        for k, i in self.keys.items():
            a, b = divmod(k, 65536)
            n = int(self.cross[i, 0])
            cross = self.cross[i, 1] / n if n else np.nan
            within = float(np.nanmean([wm[a], wm[b]])) if np.isfinite([wm[a], wm[b]]).any() else np.nan
            ma, mb = _hist_median(self.hist[i, 0]), _hist_median(self.hist[i, 1])
            rows.append(dict(idx_a=a, idx_b=b, n_pairs=n, mean_abs_cross=cross, mean_abs_within=within,
                             ratio=cross / within if within and np.isfinite(within) and within > 0 else np.nan,
                             median_a=ma, median_b=mb, offset=mb - ma, n_a=int(self.hist[i, 0].sum()),
                             n_b=int(self.hist[i, 1].sum())))
        cols = ["idx_a", "idx_b", "n_pairs", "mean_abs_cross", "mean_abs_within", "ratio", "median_a", "median_b",
                "offset", "n_a", "n_b"]
        return pd.DataFrame(rows, columns=cols).sort_values(["idx_a", "idx_b"]).reset_index(drop=True)


class OverlapAccumulator:
    """Agreement of two tiles that both have data at the same mosaic pixels (cross-zone overlaps).

    update(idx_old, idx_new, a_vals, b_vals, clean_a, clean_b): idx_* int arrays or scalars; *_vals dicts with
    decoded 'SoS50', 'EoS50', 'n_obs'; clean_* dicts {band: bool} or one bool array for all bands. delta is
    oriented per unordered pair as value(idx_b) - value(idx_a) with idx_a < idx_b; histograms -100..100 d."""

    BANDS = ("SoS50", "EoS50")
    R = 100

    def __init__(self):
        self.h = {}
        self.dn = {}

    def update(self, idx_old, idx_new, a_vals, b_vals, clean_a=None, clean_b=None):
        n = np.asarray(a_vals[self.BANDS[0]]).size
        io = np.broadcast_to(np.asarray(idx_old, np.int64), (n,)).ravel()
        inew = np.broadcast_to(np.asarray(idx_new, np.int64), (n,)).ravel()
        flip = io > inew
        lo, hi = np.minimum(io, inew), np.maximum(io, inew)
        key = lo * 65536 + hi
        for band in self.BANDS:
            ca = np.ones(n, bool) if clean_a is None else np.ravel(clean_a[band] if isinstance(clean_a, dict) else clean_a)
            cb = np.ones(n, bool) if clean_b is None else np.ravel(clean_b[band] if isinstance(clean_b, dict) else clean_b)
            d = np.ravel(b_vals[band]) - np.ravel(a_vals[band])
            d = np.where(flip, -d, d)
            m = ca & cb & np.isfinite(d)
            for k in np.unique(key[m]):
                s = m & (key == k)
                h = np.bincount(np.clip(np.rint(d[s]), -self.R, self.R).astype(np.int64) + self.R,
                                minlength=2 * self.R + 1)
                self.h[(int(k), band)] = self.h.get((int(k), band), 0) + h
        if "n_obs" in a_vals:
            d = np.ravel(b_vals["n_obs"]) - np.ravel(a_vals["n_obs"])
            d = np.where(flip, -d, d)
            m = np.isfinite(d)
            for k in np.unique(key[m]):
                s = m & (key == k)
                c, t = self.dn.get(int(k), (0, 0.0))
                self.dn[int(k)] = (c + int(s.sum()), t + float(d[s].sum()))

    @staticmethod
    def metrics(h):
        n = int(h.sum())
        if n == 0:
            return dict(n=0, bias=np.nan, mad=np.nan, share10=np.nan)
        x = np.arange(-OverlapAccumulator.R, OverlapAccumulator.R + 1)
        ha = np.bincount(np.abs(x), weights=h).astype(np.int64)
        return dict(n=n, bias=_hist_median(h, -OverlapAccumulator.R), mad=_hist_median(ha),
                    share10=float(h[np.abs(x) > 10].sum() / n))

    def to_frame(self):
        rows = []
        for (k, band), h in sorted(self.h.items()):
            a, b = divmod(k, 65536)
            c, t = self.dn.get(k, (0, 0.0))
            rows.append(dict(idx_a=a, idx_b=b, band=band, **self.metrics(h), mean_dn_obs=t / c if c else np.nan))
        return pd.DataFrame(rows, columns=["idx_a", "idx_b", "band", "n", "bias", "mad", "share10", "mean_dn_obs"])

    def pooled(self):
        out = {}
        for band in self.BANDS:
            hs = [h for (k, b), h in self.h.items() if b == band]
            out[band] = self.metrics(np.sum(hs, axis=0) if hs else np.zeros(2 * self.R + 1, np.int64))
        return out


def phase_shift(a, b, whiten=0.5):
    """Sub-pixel (dy, dx) with b(y, x) ~ a(y - dy, x - dx): FFT phase correlation on Hann-windowed, mean-removed
    images (NaN -> mean) with a parabolic peak fit. The cross-power is divided by |R|**whiten (1 = pure phase
    correlation, 0 = plain cross-correlation); 0.5 stays robust for smooth fields with resampling noise."""
    a = np.asarray(a, "float64")
    b = np.asarray(b, "float64")
    a = np.where(np.isfinite(a), a, np.nanmean(a)) - np.nanmean(a)
    b = np.where(np.isfinite(b), b, np.nanmean(b)) - np.nanmean(b)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    R = np.fft.fft2(b * win) * np.conj(np.fft.fft2(a * win))
    R /= np.maximum(np.abs(R), 1e-12) ** whiten
    c = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(c), c.shape)
    out = []
    for p, n, ax in ((py, c.shape[0], 0), (px, c.shape[1], 1)):
        get = (lambda q: c[q % n, px]) if ax == 0 else (lambda q: c[py, q % n])
        y0, y1, y2 = get(p - 1), get(p), get(p + 1)
        den = y0 - 2 * y1 + y2
        sub = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        s = p + sub
        out.append(s - n if s > n / 2 else s)
    return tuple(out)


def categorical_qa(qa):
    """qa (int or float with NaN) -> uint8 class (QA_CLASSES order, first match), 255 = nodata."""
    q = np.asarray(qa)
    nod = ~np.isfinite(q) if q.dtype.kind == "f" else (q == config.NODATA)
    qi = np.where(nod, 0, q).astype(np.int64)
    out = np.zeros(qi.shape, np.uint8)
    done = np.zeros(qi.shape, bool)
    for bits, cls in _QA_RULES:
        m = ~done & ((qi & bits) != 0)
        out[m] = cls
        done |= m
    out[nod] = 255
    return out


def roi_mask(region, shape, extent):
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds as tfb
    return geometry_mask([region], out_shape=shape, transform=tfb(extent[0], extent[2], extent[1], extent[3],
                                                                    shape[1], shape[0]), invert=True)


def roi_map(ax, arr, extent, region, cmap, vmin, vmax, label, cax_side="right", norm=None, colorbar=True):
    """arr in (left, right, bottom, top) extent; outside the ROI white, thin dark-grey ROI outline, colorbar in an
    inset axes just outside the plot (follows the aspect-constrained axes box). Returns the image."""
    import matplotlib.pyplot as plt
    from cpz.grid import plot_geom
    a = np.ma.masked_invalid(np.asarray(arr, "float32"))
    if region is not None:
        a = np.ma.masked_where(~roi_mask(region, a.shape, extent), a)
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cm.set_bad("white")
    im = ax.imshow(a, extent=extent, cmap=cm, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm,
                   interpolation="nearest")
    if region is not None:
        plot_geom(ax, region, color="#52514e", lw=0.6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians((extent[2] + extent[3]) / 2)))
    ax.tick_params(labelsize=8, colors="#52514e")
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    ax.set_anchor("NW")
    if colorbar:
        rect = {"right": [1.02, 0, 0.03, 1], "left": [-0.12, 0, 0.03, 1], "bottom": [0, -0.1, 1, 0.03]}[cax_side]
        cax = ax.inset_axes(rect)
        cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal" if cax_side == "bottom" else "vertical")
        cb.set_label(label, color="#52514e", fontsize=9)
        cb.ax.tick_params(labelsize=8, colors="#52514e")
        cb.outline.set_edgecolor("#c3c2b7")
    return im
