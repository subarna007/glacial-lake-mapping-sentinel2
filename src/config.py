"""Regions, years and band settings used across the project."""

# All imagery is resampled to 10 m in UTM zone 45N, which covers 84-90 E.
CRS = "EPSG:32645"
RES = 10
TILE = 256  # pixels, so one tile is 2.56 km x 2.56 km

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Sentinel-2 bands used as model input (B11 and B12 are 20 m and are resampled to 10 m).
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_INDEX = {b: i for i, b in enumerate(BANDS)}

# Scene classification (SCL) classes treated as unusable:
# 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 thin cirrus.
# Snow/ice (11) is kept because frozen lake surfaces are often classed as snow.
SCL_MASK = [0, 1, 3, 8, 9, 10]

# From processing baseline 04.00 (25 Jan 2022) L2A reflectance has a +1000 offset.
OFFSET_DATE = "2022-01-25"

# Training labels: Hi-MAG, annual glacial lake inventory of High Mountain Asia 2008-2017
# (Chen et al., 2021, ESSD 13, 741-766), mapped from Landsat and checked by hand.
HIMAG_URL = "https://zenodo.org/records/4275164/files/Hi-MAG%20database.zip?download=1"
LABEL_YEAR = 2017

# Central Himalaya (Nepal and the Tibetan side of the border), lon/lat.
STUDY_BBOX = (84.0, 27.6, 88.5, 29.6)

# Post-monsoon window: least cloud and least seasonal snow in Nepal.
SEASON = ("09-15", "11-30")
MAX_SCENE_CLOUD = 40  # percent, per Sentinel-2 granule

# Spatial split by longitude strips so that train, validation and test lakes are far apart.
VAL_STRIPS = [(86.0, 86.5)]
TEST_STRIPS = [(84.5, 85.0), (87.0, 87.5)]
BOUNDARY_BUFFER_DEG = 0.03  # drop tiles this close to a strip edge (about one tile width)

# Bhotekoshi floods of 8 July 2025 and 26 August 2026 both came down the Lende (Lhende)
# river from Gyirong county, Tibet. This box covers the Lende and upper Bhotekoshi
# catchments above Rasuwagadhi and is never used for training.
MONITOR_BBOX = (85.15, 28.20, 85.75, 28.65)
MONITOR_YEARS = list(range(2017, 2026))

# Smallest lake counted in object-level results: 9 Landsat pixels, as in Hi-MAG (0.0081 km2).
MIN_LAKE_PX = 81

SEED = 42
