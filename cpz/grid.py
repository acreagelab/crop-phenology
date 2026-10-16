"""Tile grids (MGRS 25 km sub-tiles, 10 m UTM), ROI and the DEM-aligned EPSG:4326 mosaic grid."""
from pathlib import Path

import mgrs
import numpy as np
import rasterio
import shapefile
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform, unary_union

import config

_MGRS = mgrs.MGRS()


def tile_grid(tile, res=10.0, size=25000.0):
    """EPSG code and north-up transform of an MGRS sub-tile id (SW corner from MGRSToUTM)."""
    zone, hemi, e, n = _MGRS.MGRSToUTM(tile)
    return (32600 if hemi == "N" else 32700) + zone, Affine(res, 0, float(e), 0, -res, float(n) + size)


def footprint(tile, dst=4326, n=25):
    epsg, tr = tile_grid(tile)
    l, t = tr.c, tr.f
    r, b = l + 25000, t - 25000
    s = np.linspace(0, 1, n, endpoint=False)
    xs = np.r_[l + s * (r - l), np.full(n, r), r - s * (r - l), np.full(n, l)]
    ys = np.r_[np.full(n, b), b + s * (t - b), np.full(n, t), t - s * (t - b)]
    return Polygon(zip(*Transformer.from_crs(epsg, dst, always_xy=True).transform(xs, ys)))


def reproject_geom(geom, src, dst):
    return transform(Transformer.from_crs(src, dst, always_xy=True).transform, geom)


def roi(regions=None):
    """Union of the regions (default config.REGIONS = ACTIVE provinces) features of config.ROI, EPSG:4326."""
    regions = config.REGIONS if regions is None else regions
    path = Path(config.ROI)
    r = shapefile.Reader(str(path))
    names = [f[0] for f in r.fields[1:]]
    geoms = [shape(s.__geo_interface__) for s, rec in zip(r.shapes(), r.records())
             if dict(zip(names, rec)).get(config.ROI_FIELD) in regions]
    if not geoms:
        raise ValueError(f"no {config.ROI_FIELD} in {regions} in {path}")
    g = unary_union(geoms)
    prj = path.with_suffix(".prj")
    src = CRS.from_wkt(prj.read_text()) if prj.exists() else CRS.from_epsg(4326)
    return g if src.equals(CRS.from_epsg(4326), ignore_axis_order=True) else reproject_geom(g, src, 4326)


def province_roi(code=None, buffer_km=None):
    """Cropland ROI of one province, extended buffer_km (default EDGE_BUFFER_KM) into the cropland ROI of the
    neighbouring provinces, so adjacent province mosaics share an overlap strip (edge match, 04)."""
    code = code or config.PROVINCE
    buffer_km = config.EDGE_BUFFER_KM if buffer_km is None else buffer_km
    own = roi([config.PROVINCES[code]["name"]])
    if buffer_km <= 0:
        return own
    lat = (own.bounds[1] + own.bounds[3]) / 2
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={(own.bounds[0] + own.bounds[2]) / 2} +units=m")
    grown = reproject_geom(reproject_geom(own, 4326, aeqd).buffer(buffer_km * 1000), aeqd, 4326)
    allp = roi([p["name"] for p in config.PROVINCES.values()])
    return own.union(grown.intersection(allp))


def target_grid(bounds):
    """DEM-aligned transform and (H, W) covering lon0, lat0, lon1, lat1."""
    with rasterio.open(config.DEM) as r:
        x0, y0, rx, ry = r.transform.c, r.transform.f, r.transform.a, -r.transform.e
    c0, c1 = np.floor((bounds[0] - x0) / rx), np.ceil((bounds[2] - x0) / rx)
    r0, r1 = np.floor((y0 - bounds[3]) / ry), np.ceil((y0 - bounds[1]) / ry)
    return Affine(rx, 0, x0 + c0 * rx, 0, -ry, y0 - r0 * ry), int(r1 - r0), int(c1 - c0)


def plot_geom(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if not p.is_empty:
            ax.plot(*p.exterior.xy, **kw)


def fill_geom(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if not p.is_empty and p.geom_type == "Polygon":
            ax.fill(*p.exterior.xy, **kw)


def set_extent(ax, geom, pad=0.1):
    x0, y0, x1, y1 = geom.bounds
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
