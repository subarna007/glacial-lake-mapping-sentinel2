"""Pixel and lake-level scores with bootstrap confidence intervals."""
import numpy as np
from scipy import ndimage


def counts(pred, label, valid=None):
    """TP, FP, FN for one tile."""
    pred, label = pred.astype(bool), label.astype(bool)
    if valid is not None:
        pred, label = pred[valid], label[valid]
    return np.array([(pred & label).sum(), (pred & ~label).sum(), (~pred & label).sum()], dtype=np.int64)


def scores(c):
    """IoU, F1, precision and recall from summed TP, FP, FN."""
    tp, fp, fn = (float(v) for v in c)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def bootstrap(tile_counts, n=1000, seed=0):
    """95% interval of IoU and F1, resampling tiles with replacement."""
    tile_counts = np.asarray(tile_counts)
    rng = np.random.default_rng(seed)
    ious, f1s = [], []
    for _ in range(n):
        s = tile_counts[rng.integers(0, len(tile_counts), len(tile_counts))].sum(0)
        sc = scores(s)
        ious.append(sc["iou"])
        f1s.append(sc["f1"])
    return {"iou_ci": tuple(np.percentile(ious, [2.5, 97.5])), "f1_ci": tuple(np.percentile(f1s, [2.5, 97.5]))}


def lake_detection(pred, label, min_px=0, covered=0.5, valid=None):
    """One row per labelled lake: its size in pixels and whether the prediction covers
    at least `covered` of it.

    With `valid`, predictions in pixels without data are ignored and lakes that are less
    than half observed are skipped, so no method gets credit for guessing inside gaps."""
    if valid is not None:
        pred = pred & valid
        obs = ndimage.label(label.astype(bool))
        seen = ndimage.mean(valid, obs[0], index=np.arange(1, obs[1] + 1)) if obs[1] else []
        keep = np.zeros(obs[1] + 1, bool)
        keep[1:] = np.asarray(seen) >= 0.5
        label = keep[obs[0]]
    lab, n = ndimage.label(label.astype(bool))
    rows = []
    if n == 0:
        return rows
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    hits = ndimage.sum(pred.astype(bool), lab, index=np.arange(1, n + 1))
    for size, hit in zip(sizes, hits):
        if size >= min_px:
            rows.append((int(size), bool(hit / size >= covered)))
    return rows


def false_lakes(pred, label, min_px=0, valid=None):
    """Predicted water bodies (at least `min_px`) that touch no labelled lake at all."""
    if valid is not None:
        pred = pred & valid
    lab, n = ndimage.label(pred.astype(bool))
    if n == 0:
        return 0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    overlap = ndimage.sum(label.astype(bool), lab, index=np.arange(1, n + 1))
    return int(((sizes >= min_px) & (overlap == 0)).sum())


SIZE_BINS_KM2 = [(0.0081, 0.02), (0.02, 0.1), (0.1, np.inf)]


def recall_by_size(rows, px_area_km2=1e-4, bins=SIZE_BINS_KM2):
    """Share of labelled lakes detected, per lake-size class."""
    out = []
    sizes = np.array([r[0] for r in rows], dtype=float) * px_area_km2
    hits = np.array([r[1] for r in rows], dtype=bool)
    for lo, hi in bins:
        m = (sizes >= lo) & (sizes < hi)
        out.append({"size_km2": f"{lo}-{hi}" if np.isfinite(hi) else f">= {lo}",
                    "lakes": int(m.sum()),
                    "detected": int(hits[m].sum()),
                    "recall": float(hits[m].mean()) if m.any() else float("nan")})
    return out
