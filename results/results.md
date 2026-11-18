| Method | IoU (95% CI) | F1 | Precision | Recall | Lakes found | False lakes per 100 km2 |
|---|---|---|---|---|---|---|
| NDWI threshold | 0.490 (0.414-0.570) | 0.658 | 0.819 | 0.550 | 45.9% of 170 | 0.3 |
| MNDWI threshold | 0.039 (0.015-0.075) | 0.076 | 0.045 | 0.244 | 8.2% of 170 | 38.7 |
| U-Net (ResNet-34) | 0.669 (0.582-0.737) | 0.802 | 0.861 | 0.750 | 70.6% of 170 | 11.5 |

Test set: 160 tiles of 2.56 km x 2.56 km in two held-out longitude strips. Pixel scores are summed over all test pixels; intervals come from 1,000 bootstrap resamples of tiles. A lake counts as found when at least half of it is predicted as water; lakes smaller than 81 pixels (0.0081 km2) are not counted.

| Lake size (km2) | NDWI threshold | MNDWI threshold | U-Net (ResNet-34) |
|---|---|---|---|
| 0.0081-0.02 (64 lakes) | 39.1% | 3.1% | 51.6% |
| 0.02-0.1 (77 lakes) | 50.6% | 6.5% | 80.5% |
| >= 0.1 (29 lakes) | 48.3% | 24.1% | 86.2% |
