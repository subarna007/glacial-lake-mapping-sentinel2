"""Cut 256 x 256 training tiles from post-monsoon Sentinel-2 composites and burn in the
Hi-MAG lake outlines.

    python -m src.chips --labels data/labels_2017.gpkg --out data/chips
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box

from . import config, imagery
from .split import assign_split, in_bbox

TILE_M = config.TILE * config.RES


def choose_tiles(lakes, max_pos, neg_ratio, seed=config.SEED):
    """Tiles containing lakes, plus lake-free tiles from the same mountain areas."""
    rng = np.random.default_rng(seed)
    cx, cy = lakes.geometry.centroid.x.values, lakes.geometry.centroid.y.values
    pos = set(zip((cx // TILE_M).astype(int), (cy // TILE_M).astype(int)))

    # candidate negatives: tiles 2-6 tiles away from a lake (high mountain terrain)
    neg = set()
    for i, j in zip((cx // TILE_M).astype(int), (cy // TILE_M).astype(int)):
        di, dj = rng.integers(-6, 7, size=2)
        if max(abs(di), abs(dj)) >= 2:
            neg.add((i + di, j + dj))
    neg -= pos
    neg = {c for c in neg if len(lakes.sindex.query(box(*cell_bounds(c)))) == 0}

    rows = []
    for kind, cells in (("lake", pos), ("no_lake", neg)):
        for c in sorted(cells):
            lon, lat = cell_center_lonlat(c)
            split = assign_split(lon)
            if split is None or in_bbox(lon, lat, config.MONITOR_BBOX) or not in_bbox(lon, lat, config.STUDY_BBOX):
                continue
            rows.append({"i": c[0], "j": c[1], "kind": kind, "split": split, "lon": lon, "lat": lat})
    df = pd.DataFrame(rows)
    take_pos = df[df.kind == "lake"].sample(frac=1, random_state=seed).head(max_pos)
    n_neg = int(len(take_pos) * neg_ratio)
    take_neg = df[df.kind == "no_lake"].sample(frac=1, random_state=seed).head(n_neg)
    return pd.concat([take_pos, take_neg]).reset_index(drop=True)


def cell_bounds(c):
    i, j = c
    return (i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M)


def cell_center_lonlat(c):
    x0, y0, x1, y1 = cell_bounds(c)
    lon0, lat0, lon1, lat1 = imagery.lonlat_bounds((x0, y0, x1, y1))
    return (lon0 + lon1) / 2, (lat0 + lat1) / 2


def burn(lakes, gbox):
    bounds = box(*gbox.boundingbox)
    hit = lakes.iloc[lakes.sindex.query(bounds)]
    if len(hit) == 0:
        return np.zeros(tuple(gbox.shape), dtype="uint8")
    transform = getattr(gbox, "affine", None) or gbox.transform
    return rasterize(((g, 1) for g in hit.geometry), out_shape=tuple(gbox.shape), transform=transform,
                     fill=0, dtype="uint8")


def make_tile(row, items, lakes, out_dir):
    path = out_dir / f"{row.split}_{row.i}_{row.j}.npz"
    if path.exists():
        return path.name, None
    gbox = imagery.geobox(cell_bounds((row.i, row.j)))
    tile_items = imagery.items_touching(items, imagery.lonlat_bounds(cell_bounds((row.i, row.j))))
    x, clear = imagery.composite(tile_items, gbox)
    y = burn(lakes, gbox)
    np.savez_compressed(path, x=x.astype("float16"), y=y, clear=clear)
    return path.name, float(np.isfinite(x[0]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"data/labels_{config.LABEL_YEAR}.gpkg")
    ap.add_argument("--year", type=int, default=config.LABEL_YEAR)
    ap.add_argument("--out", default="data/chips")
    ap.add_argument("--max-lake-tiles", type=int, default=900)
    ap.add_argument("--neg-ratio", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lakes = gpd.read_file(args.labels).to_crs(config.CRS)
    tiles = choose_tiles(lakes, args.max_lake_tiles, args.neg_ratio)
    print(tiles.groupby(["split", "kind"]).size().unstack(fill_value=0))

    start, end = imagery.season_dates(args.year)
    items = imagery.search(config.STUDY_BBOX, start, end)
    print(f"{len(items)} Sentinel-2 scenes between {start} and {end}")

    done = {}
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(make_tile, r, items, lakes, out): r.Index for r in tiles.itertuples()}
        for n, job in enumerate(as_completed(jobs), 1):
            try:
                name, cover = job.result()
                done[jobs[job]] = (name, cover)
            except Exception as e:  # one failed tile should not stop the rest
                print(f"tile {jobs[job]} failed: {e}")
            if n % 50 == 0:
                print(f"{n}/{len(tiles)} tiles")

    tiles["file"] = [done.get(k, (None, None))[0] for k in tiles.index]
    tiles.to_csv(out / "tiles.csv", index=False)
    print(f"saved {tiles.file.notna().sum()} tiles to {out}")


if __name__ == "__main__":
    main()
