"""Which tiles go to training, validation and test."""
from . import config


def assign_split(lon, buffer=config.BOUNDARY_BUFFER_DEG):
    """'train', 'val' or 'test' from the tile's longitude; None for tiles next to a strip edge."""
    for name, strips in (("val", config.VAL_STRIPS), ("test", config.TEST_STRIPS)):
        for a, b in strips:
            if a <= lon < b:
                return name if (lon - a >= buffer and b - lon >= buffer) else None
            if a - buffer <= lon < a or b <= lon < b + buffer:
                return None
    return "train"


def in_bbox(lon, lat, bbox):
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]
