"""Turning water masks into lakes: sliding-window prediction, small-object removal and
following each lake through the years."""
import numpy as np
import pandas as pd
from scipy import ndimage


def remove_small(mask, min_px):
    lab, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def _weights(tile):
    """Weights that fall off towards the tile edge, so overlapping windows blend smoothly."""
    r = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype("float32")
    w = np.minimum.outer(r, r)
    return w / w.max()


def sliding_predict(predict_batch, image, tile=256, overlap=64, batch=16):
    """Water probability for a whole (bands, H, W) image.

    `predict_batch` takes an array (N, bands, tile, tile) and returns (N, tile, tile)
    probabilities. The image is padded so every pixel is covered by at least one window.
    """
    c, h, w = image.shape
    step = tile - overlap
    ph = max(tile, int(np.ceil((h - tile) / step)) * step + tile)
    pw = max(tile, int(np.ceil((w - tile) / step)) * step + tile)
    padded = np.full((c, ph, pw), np.nan, dtype="float32")
    padded[:, :h, :w] = image
    prob = np.zeros((ph, pw), dtype="float32")
    weight = np.zeros((ph, pw), dtype="float32")
    wt = _weights(tile)

    corners = [(y, x) for y in range(0, ph - tile + 1, step) for x in range(0, pw - tile + 1, step)]
    for i in range(0, len(corners), batch):
        part = corners[i:i + batch]
        windows = np.stack([padded[:, y:y + tile, x:x + tile] for y, x in part])
        probs = predict_batch(windows)
        for (y, x), p in zip(part, probs):
            prob[y:y + tile, x:x + tile] += p * wt
            weight[y:y + tile, x:x + tile] += wt
    prob = prob / np.maximum(weight, 1e-6)
    prob = prob[:h, :w]
    prob[~np.isfinite(image[0])] = np.nan
    return prob


def track(masks, px_area_km2=1e-4, min_px=81):
    """Follow lakes through a series of masks on the same grid.

    A lake is a connected component of the union of all masks, so a lake that grows, shrinks
    or splits keeps one id. Returns a table with one row per lake and one column per key of
    `masks` (area in km2), plus centroid row/column, and the lake-id raster.
    """
    keys = list(masks)
    cleaned = {k: remove_small(np.nan_to_num(m).astype(bool), min_px) for k, m in masks.items()}
    union = np.zeros_like(next(iter(cleaned.values())), dtype=bool)
    for m in cleaned.values():
        union |= m
    lab, n = ndimage.label(union)
    if n == 0:
        return pd.DataFrame(columns=["lake_id", "row", "col"] + keys), lab
    ids = np.arange(1, n + 1)
    rows = {"lake_id": ids}
    cents = ndimage.center_of_mass(union, lab, ids)
    rows["row"] = [int(round(c[0])) for c in cents]
    rows["col"] = [int(round(c[1])) for c in cents]
    for k in keys:
        rows[k] = ndimage.sum(cleaned[k], lab, ids) * px_area_km2
    return pd.DataFrame(rows), lab
