"""Merged reader of one tile-year: base gaig-stac store row plus the optional per-tile fill store.

series(job) takes one jobs.csv row (01_inventory) and returns the sorted union of the base-store
dates (slots with doys > 0) and the committed fill dates (doys > 0). On a duplicate DOY the fill
date wins (fill holds only missing dates or dates to be replaced). Base dates listed in the job's
offset_fix_doys carry the PB >= 04.00 +1000 DN offset that the store did not remove; read() removes
it (floor 0). Bands come back in config.STORE_BANDS order in the pre-2022 DN convention (0 = nodata)
for the tile's top-left SUB x SUB grid, which is the full fill grid.
"""
import calendar
from dataclasses import dataclass, field

import numpy as np
import zarr

import config


class DoyIntegrityError(ValueError):
    pass


def _empty(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in ("", "nan", "None")


def _int_list(v):
    if _empty(v):
        return []
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    if isinstance(v, float):
        return [int(v)]
    return [int(float(s)) for s in str(v).split(";") if s.strip()]


def _check(doys, ndays, what):
    d = np.asarray(doys)
    if d.size and (d.min() < 1 or d.max() > ndays):
        raise DoyIntegrityError(f"{what}: DOY outside 1..{ndays}: {d[(d < 1) | (d > ndays)].tolist()}")
    if d.size > 1 and np.any(np.diff(d) <= 0):
        raise DoyIntegrityError(f"{what}: DOYs not strictly increasing / unique: {d.tolist()}")


@dataclass
class Series:
    tile: str
    year: int
    doys: np.ndarray
    source: np.ndarray
    offset_fix: np.ndarray
    _base: object = field(default=None, repr=False)
    _index: int = field(default=-1, repr=False)
    _t_hi: int = field(default=0, repr=False)
    _base_t: np.ndarray = field(default=None, repr=False)
    _base_pos: np.ndarray = field(default=None, repr=False)
    _fill: object = field(default=None, repr=False)
    _fill_t: np.ndarray = field(default=None, repr=False)
    _fill_pos: np.ndarray = field(default=None, repr=False)

    def read(self, r0, r1, c0, c1):
        h, w = r1 - r0, c1 - c0
        T = len(self.doys)
        nb = len(config.STORE_BANDS)
        bands = np.zeros((T, h, w, nb), np.uint16)
        valid = np.zeros((T, h, w), bool)
        if self._base is not None and len(self._base_t):
            b = self._base["bands"][self._index, :self._t_hi, r0:r1, c0:c1, :]
            m = self._base["masks"][self._index, :self._t_hi, r0:r1, c0:c1]
            bands[self._base_pos] = b[self._base_t]
            valid[self._base_pos] = m[self._base_t] > 0
        if self._fill is not None and len(self._fill_t):
            t0, t1 = int(self._fill_t.min()), int(self._fill_t.max()) + 1
            b = self._fill["bands"][t0:t1, r0:r1, c0:c1, :]
            m = self._fill["masks"][t0:t1, r0:r1, c0:c1]
            bands[self._fill_pos] = b[self._fill_t - t0]
            valid[self._fill_pos] = m[self._fill_t - t0] > 0
        fix = np.flatnonzero(self.offset_fix)
        if len(fix):
            sub = bands[fix]
            sub -= np.minimum(sub, np.uint16(config.PB_OFFSET))
            bands[fix] = sub
        return bands, valid


def series(job):
    """Series of one jobs.csv row (dict); NaN / '' fields count as empty."""
    tile, year = str(job["tile"]), int(job["year"])
    ndays = 366 if calendar.isleap(year) else 365
    base = fill = None
    bd = np.zeros(0, np.int64)
    bt = np.zeros(0, np.int64)
    t_hi = 0
    index = -1
    if not _empty(job.get("store_path")):
        index = int(float(job["index"]))
        base = zarr.open_group(str(job["store_path"]), mode="r")
        row = np.asarray(base["doys"][index]).astype(np.int64)
        bt = np.flatnonzero(row > 0)
        bd = row[bt]
        t_hi = int(bt.max()) + 1 if len(bt) else 0
        _check(bd, ndays, f"{tile} {year} base")
    fd = np.zeros(0, np.int64)
    ft = np.zeros(0, np.int64)
    if not _empty(job.get("fill_path")):
        fill = zarr.open_group(str(job["fill_path"]), mode="r")
        row = np.asarray(fill["doys"][:]).astype(np.int64)
        ft = np.flatnonzero(row > 0)
        fd = row[ft]
        order = np.argsort(fd, kind="stable")
        ft, fd = ft[order], fd[order]
        _check(fd, ndays, f"{tile} {year} fill")
    keep = ~np.isin(bd, fd)
    bt, bd = bt[keep], bd[keep]
    doys = np.concatenate([bd, fd])
    src = np.concatenate([np.zeros(len(bd), np.uint8), np.full(len(fd), 2, np.uint8)])
    order = np.argsort(doys, kind="stable")
    doys, src = doys[order], src[order]
    _check(doys, ndays, f"{tile} {year} merged")
    pos = np.empty(len(order), np.int64)
    pos[order] = np.arange(len(order))
    fix_doys = set(_int_list(job.get("offset_fix_doys")))
    offset_fix = (src == 0) & np.isin(doys, list(fix_doys))
    return Series(tile=tile, year=year, doys=doys.astype(np.int16), source=src, offset_fix=offset_fix,
                  _base=base, _index=index, _t_hi=t_hi, _base_t=bt, _base_pos=pos[:len(bd)],
                  _fill=fill, _fill_t=ft, _fill_pos=pos[len(bd):])
