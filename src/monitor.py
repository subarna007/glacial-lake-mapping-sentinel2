"""Apply the trained U-Net to the Lende and upper Bhotekoshi catchments above Rasuwagadhi,
the source area of the floods of 8 July 2025 and 26 August 2026. This area was never used
for training.

1. One post-monsoon composite per year (2017-2025): how many lakes, how large, which grew.
2. Single clear scenes around the two floods: did a lake appear and disappear?

    python -m src.monitor --model results/unet --out results/monitor
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config, imagery
from .evaluate import load_model, rgb, unet_predictor
from .lakes import remove_small, sliding_predict, track

PX_KM2 = (config.RES / 1000) ** 2
CHUNKS = {"x": 2048, "y": 2048}
EVENT_WINDOWS = {
    "2025 flood (8 July)": ("2025-05-01", "2025-08-15"),
    "2026 flood (26 August)": ("2026-07-01", "2026-10-31"),
}


def water_mask(predict, image, threshold):
    prob = sliding_predict(predict, image, tile=config.TILE, overlap=64)
    return remove_small(np.nan_to_num(prob) > threshold, config.MIN_LAKE_PX), np.isfinite(prob)


def to_lonlat(gbox, rows, cols):
    a = getattr(gbox, "affine", None) or gbox.transform
    xs = a.c + (np.asarray(cols) + 0.5) * a.a
    ys = a.f + (np.asarray(rows) + 0.5) * a.e
    lon, lat = imagery._to_lonlat.transform(xs, ys)
    return np.round(lon, 4), np.round(lat, 4)


def annual(predict, threshold, gbox, out, catalog):
    masks, coverage = {}, {}
    for year in config.MONITOR_YEARS:
        start, end = imagery.season_dates(year)
        items = imagery.search(config.MONITOR_BBOX, start, end, catalog=catalog)
        if not items:
            print(f"{year}: no scenes")
            continue
        image, clear = imagery.composite(items, gbox, chunks=CHUNKS)
        if imagery.looks_broken(image):
            print(f"{year}: composite failed the band check {imagery.band_medians(image)}; skipped")
            plt.imsave(out / f"rgb_{year}_rejected.png", rgb(image[:, ::4, ::4]))
            continue
        mask, valid = water_mask(predict, image, threshold)
        masks[year] = mask
        coverage[year] = float(valid.mean())
        print(f"{year}: {len(items)} scenes, {coverage[year]:.1%} of the area observed, "
              f"{mask.sum() * PX_KM2:.2f} km2 of lake water")
        if year in (config.MONITOR_YEARS[0], config.MONITOR_YEARS[-1]):
            plt.imsave(out / f"rgb_{year}.png", rgb(image[:, ::4, ::4]))
            np.save(out / f"mask_{year}.npy", mask)
        del image

    table, lake_ids = track(masks, PX_KM2, config.MIN_LAKE_PX)
    years = list(masks)
    table["lon"], table["lat"] = to_lonlat(gbox, table.row, table.col)
    first, last = years[0], years[-1]
    table["change_km2"] = table[last] - table[first]
    table["change_pct"] = np.where(table[first] > 0, 100 * table.change_km2 / table[first], np.nan)
    table = table.sort_values(last, ascending=False)
    table.drop(columns=["row", "col"]).to_csv(out / "lakes_by_year.csv", index=False, float_format="%.4f")

    totals = pd.DataFrame({"year": years,
                           "lakes": [int((table[y] > 0).sum()) for y in years],
                           "area_km2": [float(table[y].sum()) for y in years],
                           "observed_share": [coverage[y] for y in years]})
    totals.to_csv(out / "totals_by_year.csv", index=False, float_format="%.4f")
    plot_annual(table, totals, years, out)
    return table, totals


def plot_annual(table, totals, years, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(totals.year, totals.area_km2, "o-", color="#1f6f8b")
    ax.set_ylabel("Total lake area (km²)")
    ax.set_title("All mapped lakes in the Lende / upper Bhotekoshi area", fontsize=9)
    ax2 = ax.twinx()
    ax2.bar(totals.year, totals.lakes, alpha=0.2, color="#555")
    ax2.set_ylabel("Number of lakes")
    ax = axes[1]
    for _, r in table.head(8).iterrows():
        ax.plot(years, [r[y] for y in years], "o-", ms=3, label=f"{r.lat:.3f}N {r.lon:.3f}E")
    ax.set_ylabel("Lake area (km²)")
    ax.set_title("The eight largest lakes", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    for a in axes:
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_annual.png", dpi=200)
    plt.close(fig)


def events(predict, threshold, gbox, out, catalog):
    rows = []
    for name, (start, end) in EVENT_WINDOWS.items():
        items = imagery.search(config.MONITOR_BBOX, start, end, max_cloud=70, catalog=catalog)
        print(f"{name}: {len(items)} scenes to check")
        for date, image, clear in imagery.scenes(items, gbox, min_clear=0.6, chunks=CHUNKS):
            mask, valid = water_mask(predict, image, threshold)
            rows.append({"event": name, "date": date, "clear_share": round(clear, 3),
                         "lake_area_km2": float(mask.sum() * PX_KM2),
                         "lakes": int(_count(mask))})
            (out / "scenes").mkdir(exist_ok=True)
            plt.imsave(out / "scenes" / f"{date}.png", rgb(image[:, ::4, ::4]))
            print(f"  {date}: {clear:.0%} clear, {rows[-1]['lake_area_km2']:.2f} km2 water")
    df = pd.DataFrame(rows)
    df.to_csv(out / "event_scenes.csv", index=False)
    return df


def _count(mask):
    from scipy import ndimage
    return ndimage.label(mask)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/unet")
    ap.add_argument("--out", default="results/monitor")
    ap.add_argument("--skip-events", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.model, device)
    predict = unet_predictor(model, meta, device)
    gbox = imagery.geobox(imagery.utm_bounds(config.MONITOR_BBOX))
    print(f"monitoring grid {gbox.shape} at {config.RES} m")
    catalog = imagery.open_catalog()

    table, totals = annual(predict, meta["threshold"], gbox, out, catalog)
    print(totals.to_string(index=False))
    if not args.skip_events:
        events(predict, meta["threshold"], gbox, out, catalog)
    (out / "settings.json").write_text(json.dumps({"bbox": config.MONITOR_BBOX, "years": config.MONITOR_YEARS,
                                                   "threshold": meta["threshold"]}, indent=2))


if __name__ == "__main__":
    main()
