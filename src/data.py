"""Loading the saved tiles."""
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CLEAR = 0.8  # tiles with more than 20% missing pixels (cloud in every scene) are skipped


def load_tile(path):
    with np.load(path) as z:
        return z["x"].astype("float32"), z["y"].astype("uint8")


def tile_table(chip_dir, min_clear=MIN_CLEAR):
    """tiles.csv plus the share of pixels with a clear observation, keeping usable tiles only."""
    chip_dir = Path(chip_dir)
    df = pd.read_csv(chip_dir / "tiles.csv")
    df = df[df.file.notna()].copy()
    clear = []
    for f in df.file:
        x, _ = load_tile(chip_dir / f)
        clear.append(float(np.isfinite(x[0]).mean()))
    df["clear"] = clear
    dropped = (df.clear < min_clear).sum()
    if dropped:
        print(f"skipping {dropped} tiles with less than {min_clear:.0%} clear pixels")
    return df[df.clear >= min_clear].reset_index(drop=True)


def band_stats(chip_dir, files):
    """Per-band mean and standard deviation over the training tiles."""
    s = s2 = n = 0
    for f in files:
        x, _ = load_tile(Path(chip_dir) / f)
        x = x.reshape(x.shape[0], -1)
        ok = np.isfinite(x)
        s = s + np.where(ok, x, 0).sum(1)
        s2 = s2 + np.where(ok, x ** 2, 0).sum(1)
        n = n + ok.sum(1)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype("float32"), std.astype("float32")


def normalise(x, mean, std):
    """Standardise each band; missing pixels become 0 (the band mean)."""
    z = (x - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(z, nan=0.0).astype("float32")
