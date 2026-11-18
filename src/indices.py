"""Water indices used as the baseline: NDWI (McFeeters, 1996) and MNDWI (Xu, 2006)."""
import numpy as np

from .config import BAND_INDEX


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def ndwi(x):
    """(green - NIR) / (green + NIR). `x` is (bands, H, W) reflectance."""
    return _ratio(x[BAND_INDEX["B03"]], x[BAND_INDEX["B08"]])


def mndwi(x):
    """(green - SWIR1) / (green + SWIR1)."""
    return _ratio(x[BAND_INDEX["B03"]], x[BAND_INDEX["B11"]])


INDICES = {"ndwi": ndwi, "mndwi": mndwi}


def threshold_counts(index_maps, labels, thresholds):
    """TP, FP, FN for every threshold, summed over all tiles (NaN pixels ignored)."""
    thresholds = np.asarray(thresholds)
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for idx, lab in zip(index_maps, labels):
        ok = np.isfinite(idx)
        v, l = idx[ok], lab[ok].astype(bool)
        pred = v[None, :] > thresholds[:, None]
        counts[:, 0] += (pred & l).sum(1)
        counts[:, 1] += (pred & ~l).sum(1)
        counts[:, 2] += (~pred & l).sum(1)
    return counts


def best_threshold(index_maps, labels, thresholds=np.round(np.arange(-0.4, 0.71, 0.02), 2)):
    """Threshold with the highest IoU on the given tiles."""
    c = threshold_counts(index_maps, labels, thresholds)
    iou = c[:, 0] / np.maximum(c.sum(1), 1)
    i = int(np.argmax(iou))
    return float(thresholds[i]), float(iou[i])


def predict(x, name, threshold):
    idx = INDICES[name](x)
    return np.nan_to_num(idx, nan=-1.0) > threshold
