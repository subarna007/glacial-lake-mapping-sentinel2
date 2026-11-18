"""Download the Hi-MAG inventory and save the lakes of one year inside the study area.

    python -m src.labels --out data/labels_2017.gpkg
"""
import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from . import config


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def find_layers(folder):
    """All vector files in the unzipped archive (shapefiles, GeoPackages, file geodatabases)."""
    folder = Path(folder)
    layers = [p for p in folder.rglob("*") if p.suffix.lower() in (".shp", ".gpkg")]
    layers += [p for p in folder.rglob("*.gdb") if p.is_dir()]
    return sorted(layers)


def read_year(folder, year, layer=None):
    """Return the lake polygons of one year.

    Hi-MAG is published as one layer per year; the year appears in the file name. If no
    file name contains the year, look for a column holding years instead.
    """
    layers = [Path(layer)] if layer else find_layers(folder)
    if not layers:
        raise FileNotFoundError(f"no vector files found in {folder}")
    print("vector files in archive:")
    for p in layers:
        print("   ", p.relative_to(folder) if not layer else p)

    named = [p for p in layers if str(year) in p.name]
    if named:
        print(f"using {named[0].name}")
        return gpd.read_file(named[0])

    for p in layers:
        gdf = gpd.read_file(p)
        for col in gdf.columns:
            if col == "geometry":
                continue
            values = gdf[col].astype(str).str[:4]
            if (values == str(year)).any() and values.str.isdigit().mean() > 0.9:
                print(f"using {p.name}, column {col} == {year}")
                return gdf[values == str(year)].copy()
    raise ValueError(
        f"could not find the {year} lakes; pass the right file with --layer (see the list above)"
    )


def clip_to_study_area(gdf, bbox=config.STUDY_BBOX):
    if gdf.crs is None:
        print("layer has no CRS, assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.intersects(box(*bbox))].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)  # fixes self-intersecting outlines
    gdf = gdf.to_crs(config.CRS)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    return gdf.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=config.LABEL_YEAR)
    ap.add_argument("--data", default="data")
    ap.add_argument("--layer", default=None, help="path to the layer to use, if auto-detection fails")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out or data / f"labels_{args.year}.gpkg")
    archive = download(config.HIMAG_URL, data / "himag.zip")
    folder = data / "himag"
    if not folder.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        # some archives contain further zips, one per year
        for inner in folder.rglob("*.zip"):
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.with_suffix(""))

    lakes = clip_to_study_area(read_year(folder, args.year, args.layer))
    os.makedirs(out.parent, exist_ok=True)
    lakes.to_file(out, driver="GPKG")
    print(f"{len(lakes)} lakes, {lakes.area_km2.sum():.1f} km2 in total -> {out}")
    print(lakes.area_km2.describe().round(4).to_string())


if __name__ == "__main__":
    main()
