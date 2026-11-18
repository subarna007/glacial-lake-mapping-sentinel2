# Mapping glacial lakes in the Nepal Himalaya with Sentinel-2

On 8 July 2025 a flood came down the Lende river into the Bhotekoshi at Rasuwagadhi, destroyed the
Friendship Bridge to China and killed several people. It was traced to a lake on the surface of the
Purepu glacier in Gyirong, Tibet, about 35 km upstream. On 26 August 2026 the same valley flooded again; early
assessments point to an ice avalanche that dammed the Lende near the border and then broke through.
Both times people downstream had almost no warning, and both floods started in high, remote
headwaters that are rarely watched.

Sentinel-2 photographs every part of the Himalaya every five days for free. This project asks two questions:

1. **How well can a segmentation model find glacial lakes in Sentinel-2 images, compared with the
   usual water-index thresholds (NDWI, MNDWI)?** Tested on parts of the Himalaya the model never saw.
2. **What does the model show for the Lende and upper Bhotekoshi catchments** from 2017 to 2025, and
   around the 2025 and 2026 floods?

## Data

| What | Source |
|---|---|
| Images | Sentinel-2 Level-2A surface reflectance, bands B2, B3, B4, B8 (10 m) and B11, B12 (20 m, resampled to 10 m), from the Microsoft Planetary Computer |
| Lake outlines (labels) | Hi-MAG, the annual glacial lake inventory of High Mountain Asia 2008-2017 (Chen et al., 2021). Mapped from Landsat and checked by hand. I use the 2017 outlines. |
| Study area | 84.0-88.5 E, 27.6-29.6 N: the Nepal Himalaya and the Tibetan side of the border |

## Method

**Images.** For each location I take every Sentinel-2 scene from 15 September to 30 November 2017
(after the monsoon, when there is the least cloud and seasonal snow), mask cloud, cloud shadow and
cirrus with the scene classification layer, and take the per-pixel median. Scenes processed after
25 January 2022 carry a +1000 reflectance offset; it is removed so all years are comparable.

**Tiles.** The area is cut into 256 x 256 pixel tiles (2.56 km). I keep up to 900 tiles that contain
a Hi-MAG lake and half as many lake-free tiles from the same mountain terrain, so the model also
sees rivers, shadows, snow and ice that are *not* lakes.

**Split.** Tiles are split by longitude, not at random, because neighbouring tiles look alike and a
random split would overestimate accuracy:

| Split | Longitude |
|---|---|
| Test | 84.5-85.0 E and 87.0-87.5 E |
| Validation | 86.0-86.5 E |
| Training | everything else |

Tiles within about one tile of a strip edge are dropped. The Lende / upper Bhotekoshi area
(85.15-85.75 E, 28.20-28.65 N) is excluded from training altogether.

**Models.**
- *NDWI* and *MNDWI* thresholds. The threshold is chosen on the training and validation tiles.
- *U-Net* with a ResNet-34 encoder (ImageNet weights, first layer adapted to six bands), trained for
  40 epochs with cross-entropy plus Dice loss, random rotations and flips. The checkpoint and the
  probability threshold are chosen on the validation strip.

**Evaluation** on the test strips only:
- pixel IoU, F1, precision and recall, with 95% bootstrap intervals over tiles;
- the share of Hi-MAG lakes found (at least half of the lake predicted as water), by lake size;
- false lakes per 100 km² (predicted water bodies that touch no labelled lake).

Lakes smaller than 81 pixels (0.0081 km², the Hi-MAG minimum) are not counted in the lake-level scores.

**Monitoring.** The trained model is run over the Lende / upper Bhotekoshi box on one post-monsoon
composite per year (2017-2025). Each lake is followed through the years as one connected region of
the union of all yearly masks, so a lake that grows or splits keeps one id. Separately, every clear
single scene from May-August 2025 and July-October 2026 is checked to see whether lakes appear or
disappear around the two floods.

## Results

The full run has not finished yet; the tables and figures will be added here from `results/`.

## Limitations I expect

- **Labels are 30 m, images are 10 m.** Hi-MAG outlines come from Landsat, so lake edges in the
  labels are blocky. This limits the best IoU any model can reach against them.
- **Label and image dates differ.** Hi-MAG 2017 uses one Landsat scene per lake between July and
  November; my images are a September-November median. Lakes that changed within that window will
  look like errors.
- **Autumn composites miss short-lived lakes.** The 2025 flood came from a supraglacial lake that
  formed and drained within weeks in early summer. A once-a-year autumn map would not show it; that is
  why the single-scene check around each flood is part of the project.
- **Frozen lakes, shadow and turbid water** are the expected hard cases for every method.

## Repository

```
src/
  config.py      regions, years, bands, split
  labels.py      download Hi-MAG and keep the 2017 lakes in the study area
  imagery.py     Sentinel-2 search, cloud masking, median composites
  chips.py       training tiles and label rasters
  split.py       train / validation / test strips
  indices.py     NDWI, MNDWI and threshold search
  train.py       U-Net training
  evaluate.py    test-strip comparison, tables and figures
  lakes.py       sliding-window prediction, small-object removal, lake tracking
  monitor.py     Lende / upper Bhotekoshi time series and flood-window scenes
  metrics.py     IoU, F1, lake-level recall, bootstrap
tests/           checks for the parts that do not need imagery
notebooks/       the whole pipeline in one notebook
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests
python -m src.labels      # Hi-MAG 2017 lakes in the study area
python -m src.chips       # Sentinel-2 tiles (needs internet, takes a while)
python -m src.train       # needs a GPU
python -m src.evaluate
python -m src.monitor
```

## References

- Chen, F., Zhang, M., Guo, H., Allen, S., Kargel, J. S., Haritashya, U. K. and Watson, C. S. (2021).
  Annual 30 m dataset for glacial lakes in High Mountain Asia from 2008 to 2017. *Earth System Science
  Data*, 13, 741-766. https://doi.org/10.5194/essd-13-741-2021
- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation
  of open water features. *International Journal of Remote Sensing*, 17(7), 1425-1432.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water
  features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025-3033.
- Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: Convolutional networks for biomedical image
  segmentation. *MICCAI*.
- ICIMOD situation updates on the Rasuwa floods of July 2025 and August 2026. https://www.icimod.org
