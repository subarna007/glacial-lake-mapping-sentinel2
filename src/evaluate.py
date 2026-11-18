"""Compare the U-Net with NDWI and MNDWI thresholds on the held-out test strips.

    python -m src.evaluate --chips data/chips --model results/unet --out results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config, indices, metrics
from .data import load_tile, normalise, tile_table
from .lakes import remove_small
from .train import build_model

PX_KM2 = (config.RES / 1000) ** 2
NAMES = {"ndwi": "NDWI threshold", "mndwi": "MNDWI threshold", "unet": "U-Net (ResNet-34)"}


def rgb(x):
    img = np.nan_to_num(x[[2, 1, 0]].transpose(1, 2, 0))
    return np.clip(img / np.percentile(img, 99) if img.max() > 0 else img, 0, 1)


def load_model(model_dir, device):
    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = build_model()
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=device))
    return model.to(device).eval(), meta


def unet_predictor(model, meta, device):
    mean, std = np.array(meta["mean"], "float32"), np.array(meta["std"], "float32")

    @torch.no_grad()
    def predict(batch):
        z = np.stack([normalise(b, mean, std) for b in batch])
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(torch.from_numpy(z).to(device))).float()
        return p[:, 0].cpu().numpy()
    return predict


def evaluate(preds, labels, valids, n_boot=1000):
    tile_counts = [metrics.counts(p, l, v) for p, l, v in zip(preds, labels, valids)]
    res = metrics.scores(np.sum(tile_counts, 0))
    res.update(metrics.bootstrap(tile_counts, n=n_boot, seed=config.SEED))
    rows, false = [], 0
    for p, l in zip(preds, labels):
        p = remove_small(p, config.MIN_LAKE_PX)
        rows += metrics.lake_detection(p, l, min_px=config.MIN_LAKE_PX)
        false += metrics.false_lakes(p, l, min_px=config.MIN_LAKE_PX)
    area_km2 = len(preds) * (config.TILE ** 2) * PX_KM2
    res["lake_recall"] = float(np.mean([r[1] for r in rows])) if rows else float("nan")
    res["lakes"] = len(rows)
    res["false_lakes_per_100km2"] = 100 * false / area_km2
    res["recall_by_size"] = metrics.recall_by_size(rows, PX_KM2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="data/chips")
    ap.add_argument("--model", default="results/unet")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiles = tile_table(args.chips)
    fit = tiles[tiles.split != "test"]
    test = tiles[tiles.split == "test"].reset_index(drop=True)
    print(f"test tiles: {len(test)} ({(test.kind == 'lake').sum()} with lakes)")

    # index thresholds are tuned on the training and validation tiles, never on test
    thresholds = {}
    fit_tiles = [load_tile(Path(args.chips) / f) for f in fit.file]
    for name, fn in indices.INDICES.items():
        t, iou = indices.best_threshold([fn(x) for x, _ in fit_tiles], [y for _, y in fit_tiles])
        thresholds[name] = t
        print(f"{name}: threshold {t} (IoU {iou:.3f} on train+val)")
    del fit_tiles

    model, meta = load_model(args.model, device)
    predict = unet_predictor(model, meta, device)

    xs, labels, valids = [], [], []
    for f in test.file:
        x, y = load_tile(Path(args.chips) / f)
        xs.append(x)
        labels.append(y.astype(bool))
        valids.append(np.isfinite(x).all(0))
    preds = {name: [indices.predict(x, name, thresholds[name]) for x in xs] for name in indices.INDICES}
    probs = np.concatenate([predict(np.stack(xs[i:i + 16])) for i in range(0, len(xs), 16)])
    preds["unet"] = [p > meta["threshold"] for p in probs]

    results = {}
    for name, p in preds.items():
        results[name] = evaluate(p, labels, valids)
        results[name]["threshold"] = thresholds.get(name, meta["threshold"])
    summary = {"test_tiles": len(test), "test_area_km2": len(test) * config.TILE ** 2 * PX_KM2,
               "label_year": config.LABEL_YEAR, "methods": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_table(results, out / "results.md", len(test))
    plot_size_recall(results, out / "fig_recall_by_size.png")
    plot_examples(xs, labels, preds, test, out / "fig_examples.png")
    print((out / "results.md").read_text())


def write_table(results, path, n_tiles):
    lines = ["| Method | IoU (95% CI) | F1 | Precision | Recall | Lakes found | False lakes per 100 km2 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lo, hi = r["iou_ci"]
        lines.append(f"| {NAMES[name]} | {r['iou']:.3f} ({lo:.3f}-{hi:.3f}) | {r['f1']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['lake_recall']:.1%} of {r['lakes']} | "
                     f"{r['false_lakes_per_100km2']:.1f} |")
    lines += ["", f"Test set: {n_tiles} tiles of 2.56 km x 2.56 km in two held-out longitude strips. "
              "Pixel scores are summed over all test pixels; intervals come from 1,000 bootstrap "
              "resamples of tiles. A lake counts as found when at least half of it is predicted as water; "
              f"lakes smaller than {config.MIN_LAKE_PX} pixels (0.0081 km2) are not counted.", "",
              "| Lake size (km2) | " + " | ".join(NAMES[n] for n in results) + " |",
              "|---|" + "---|" * len(results)]
    for i, b in enumerate(next(iter(results.values()))["recall_by_size"]):
        cells = [f"{r['recall_by_size'][i]['recall']:.1%}" for r in results.values()]
        lines.append(f"| {b['size_km2']} ({b['lakes']} lakes) | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def plot_size_recall(results, path):
    bins = [b["size_km2"] for b in next(iter(results.values()))["recall_by_size"]]
    w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = {"ndwi": "#9aa5ab", "mndwi": "#5f7c8a", "unet": "#1f6f8b"}
    for k, (name, r) in enumerate(results.items()):
        vals = [b["recall"] for b in r["recall_by_size"]]
        ax.bar(np.arange(len(bins)) + (k - (len(results) - 1) / 2) * w, vals, w, label=NAMES[name], color=colors[name])
    ax.set_xticks(range(len(bins)), [f"{b} km²" for b in bins])
    ax.set_ylabel("Share of lakes found")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Lakes found on the test strips, by lake size", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_examples(xs, labels, preds, test, path, n=6):
    lake_idx = [i for i in range(len(xs)) if labels[i].sum() > config.MIN_LAKE_PX]
    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(lake_idx, size=min(n, len(lake_idx)), replace=False)
    cols = ["Sentinel-2 (RGB)", "Hi-MAG label", "NDWI threshold", "U-Net"]
    fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.3 * len(pick)))
    for r, i in enumerate(pick):
        for c, img in enumerate([rgb(xs[i]), labels[i], preds["ndwi"][i], preds["unet"][i]]):
            ax = axes[r, c]
            ax.imshow(img, cmap=None if c == 0 else "Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        axes[r, 0].set_ylabel(f"{test.lon[i]:.2f}E {test.lat[i]:.2f}N", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
