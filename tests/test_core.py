"""Checks for the parts that do not need imagery: indices, scores, lake tracking and the split.

    python -m pytest tests
"""
import numpy as np
import pytest

from src import config, indices, metrics
from src.lakes import remove_small, sliding_predict, track
from src.split import assign_split, in_bbox


def make_image(h=64, w=64):
    """Bright land with one dark, green-over-NIR 'lake' in the middle."""
    x = np.full((len(config.BANDS), h, w), 0.3, dtype="float32")
    lake = np.zeros((h, w), bool)
    lake[20:40, 20:40] = True
    x[config.BAND_INDEX["B03"], lake] = 0.08
    x[config.BAND_INDEX["B08"], lake] = 0.02
    x[config.BAND_INDEX["B11"], lake] = 0.01
    return x, lake


def test_indices_find_water():
    x, lake = make_image()
    assert (indices.ndwi(x)[lake] > 0.5).all()
    assert (indices.mndwi(x)[~lake] == 0).all()
    pred = indices.predict(x, "ndwi", 0.2)
    assert (pred == lake).all()


def test_best_threshold_separates_classes():
    x, lake = make_image()
    t, iou = indices.best_threshold([indices.ndwi(x)], [lake])
    assert 0 <= t < 0.6 and iou == 1.0


def test_nan_pixels_are_ignored():
    x, lake = make_image()
    x[:, :10, :] = np.nan
    c = indices.threshold_counts([indices.ndwi(x)], [lake], [0.2])
    assert c[0].tolist() == [lake.sum(), 0, 0]
    assert not indices.predict(x, "ndwi", 0.2)[:10].any()


def test_scores():
    s = metrics.scores([8, 2, 2])
    assert s["iou"] == pytest.approx(8 / 12)
    assert s["f1"] == pytest.approx(16 / 20)
    assert s["precision"] == pytest.approx(0.8) and s["recall"] == pytest.approx(0.8)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 50, size=(40, 3))
    est = metrics.scores(tiles.sum(0))["iou"]
    lo, hi = metrics.bootstrap(tiles, n=300)["iou_ci"]
    assert lo <= est <= hi


def test_lake_detection_and_false_lakes():
    label = np.zeros((50, 50), bool)
    label[5:15, 5:15] = True      # 100 px, found
    label[30:40, 30:40] = True    # 100 px, missed
    pred = np.zeros_like(label)
    pred[5:15, 5:12] = True       # covers 70% of the first lake
    pred[40:50, 0:10] = True      # water where there is no lake
    rows = metrics.lake_detection(pred, label, min_px=50)
    assert sorted(rows) == [(100, False), (100, True)]
    assert metrics.false_lakes(pred, label, min_px=50) == 1


def test_recall_by_size_bins():
    rows = [(100, True), (100, False), (5000, True)]
    out = metrics.recall_by_size(rows, px_area_km2=1e-4)
    assert [b["lakes"] for b in out] == [2, 0, 1]
    assert out[0]["recall"] == 0.5 and out[2]["recall"] == 1.0


def test_remove_small():
    m = np.zeros((20, 20), bool)
    m[0:2, 0:2] = True
    m[10:20, 10:20] = True
    out = remove_small(m, 10)
    assert not out[0:2, 0:2].any() and out[10:20, 10:20].all()


def test_sliding_predict_matches_pixelwise_model():
    rng = np.random.default_rng(1)
    image = rng.random((6, 300, 410)).astype("float32")
    image[:, :5, :5] = np.nan
    fn = lambda b: 1 / (1 + np.exp(-np.nan_to_num(b[:, 1] - b[:, 3])))
    prob = sliding_predict(fn, image, tile=128, overlap=32, batch=4)
    expected = 1 / (1 + np.exp(-(image[1] - image[3])))
    ok = np.isfinite(expected)
    assert prob.shape == (300, 410)
    assert np.allclose(prob[ok], expected[ok], atol=1e-5)
    assert np.isnan(prob[:5, :5]).all()


def test_track_follows_a_growing_lake():
    a = np.zeros((40, 40), bool)
    a[10:20, 10:20] = True
    b = np.zeros_like(a)
    b[10:25, 10:20] = True        # same lake, larger
    b[30:40, 30:40] = True        # new lake
    table, ids = track({2017: a, 2018: b}, px_area_km2=1.0, min_px=10)
    table = table.sort_values(2018)
    assert len(table) == 2
    assert table[2017].tolist() == [0.0, 100.0]
    assert table[2018].tolist() == [100.0, 150.0]
    assert ids.max() == 2


def test_split_strips_and_buffer():
    assert assign_split(84.7) == "test"
    assert assign_split(87.2) == "test"
    assert assign_split(86.25) == "val"
    assert assign_split(85.6) == "train"
    assert assign_split(84.49) is None      # just outside a test strip
    assert assign_split(86.01) is None      # just inside the validation strip
    assert in_bbox(85.4, 28.4, config.MONITOR_BBOX)
