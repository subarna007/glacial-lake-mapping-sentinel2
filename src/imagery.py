"""Search and load Sentinel-2 L2A imagery from the Planetary Computer STAC API."""
import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import xarray as xr
from odc.geo.geobox import GeoBox
from pyproj import Transformer

from . import config


def open_catalog():
    # Items are signed right before loading (see load); signed URLs expire after about an
    # hour, and building all training tiles takes longer than that.
    return pystac_client.Client.open(config.STAC_URL)


def search(bbox, start, end, max_cloud=config.MAX_SCENE_CLOUD, catalog=None):
    """Sentinel-2 L2A items intersecting a lon/lat bbox between two ISO dates."""
    catalog = catalog or open_catalog()
    found = catalog.search(
        collections=[config.COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(found.items())
    items.sort(key=lambda it: it.datetime)
    return items


def season_dates(year, season=config.SEASON):
    return f"{year}-{season[0]}", f"{year}-{season[1]}"


def items_touching(items, lonlat_bbox):
    """Items whose footprint bbox overlaps a lon/lat bbox (cheap filter before loading)."""
    x0, y0, x1, y1 = lonlat_bbox
    keep = []
    for it in items:
        a0, b0, a1, b1 = it.bbox
        if a0 < x1 and a1 > x0 and b0 < y1 and b1 > y0:
            keep.append(it)
    return keep


_to_utm = Transformer.from_crs("EPSG:4326", config.CRS, always_xy=True)
_to_lonlat = Transformer.from_crs(config.CRS, "EPSG:4326", always_xy=True)


def utm_bounds(lonlat_bbox):
    x0, y0, x1, y1 = _to_utm.transform_bounds(*lonlat_bbox)
    snap = config.RES
    return (np.floor(x0 / snap) * snap, np.floor(y0 / snap) * snap,
            np.ceil(x1 / snap) * snap, np.ceil(y1 / snap) * snap)


def lonlat_bounds(utm_bbox):
    return _to_lonlat.transform_bounds(*utm_bbox)


def geobox(utm_bbox):
    return GeoBox.from_bbox(utm_bbox, crs=config.CRS, resolution=config.RES)


def load(items, gbox, chunks=None):
    """Stack of all scenes on a grid, one time step per solar day."""
    items = [planetary_computer.sign(it) for it in items]
    return odc.stac.load(
        items,
        bands=config.BANDS + ["SCL"],
        geobox=gbox,
        groupby="solar_day",
        chunks=chunks,
        resampling={"*": "bilinear", "SCL": "nearest"},
        fail_on_error=False,
    )


def reflectance(ds):
    """Surface reflectance (band, time, y, x) as float32, NaN where cloudy or missing."""
    refl = ds[config.BANDS].to_array("band").astype("float32")
    offset = xr.where(ds.time >= np.datetime64(config.OFFSET_DATE), 1000.0, 0.0).astype("float32")
    refl = ((refl - offset) / 10000.0).clip(min=0)
    valid = (~ds.SCL.isin(config.SCL_MASK)) & (ds[config.BANDS[0]] > 0)
    # surface reflectance above 1.5 is physically impossible; such values come from
    # broken reads and would otherwise pass straight into the median
    valid = valid & (refl < 1.5).all("band")
    return refl.where(valid)


def band_medians(image):
    """Median reflectance of each band over the observed pixels (a quick quality check)."""
    return {b: float(np.nanmedian(image[i])) for i, b in enumerate(config.BANDS)}


def looks_broken(image, reference=None, factor=3.0):
    """True if any band median is more than `factor` times off the reference medians
    (or, without a reference, if the visible bands are not of similar size)."""
    med = band_medians(image)
    if reference:
        return any(not (reference[b] / factor <= med[b] <= reference[b] * factor) for b in med)
    vis = [med["B02"], med["B03"], med["B04"]]
    return max(vis) > factor * max(min(vis), 1e-4)


def composite(items, gbox, chunks=None):
    """Median of all clear observations. Returns (bands, H, W) float32 and the clear count (H, W)."""
    if not items:
        raise ValueError("no Sentinel-2 scenes for this area and period")
    ds = load(items, gbox, chunks=chunks)
    refl = reflectance(ds)
    if chunks:
        refl = refl.chunk({"time": -1})
    med = refl.median("time", skipna=True)
    count = refl.isel(band=0).notnull().sum("time")
    # .compute() works for both numpy- and dask-backed arrays and on every xarray version
    med, count = med.compute(), count.compute()
    return med.values.astype("float32"), count.values.astype("int16")


def scenes(items, gbox, min_clear=0.7, chunks=None):
    """Single scenes (no compositing), keeping those with at least `min_clear` clear pixels.

    Yields (date, (bands, H, W) array, clear fraction), one scene at a time to save memory.
    """
    ds = load(items, gbox, chunks=chunks)
    refl = reflectance(ds)
    for t in range(refl.sizes["time"]):
        arr = refl.isel(time=t).values.astype("float32")
        # share of the area with a usable observation; this also catches scenes that only
        # cover part of the area (the edge of the satellite's swath)
        clear = float(np.isfinite(arr[0]).mean())
        if clear < min_clear:
            continue
        yield str(refl.time.values[t])[:10], arr, clear
