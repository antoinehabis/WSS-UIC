import sys
from pathlib import Path
print(Path(__file__).resolve().parent.parent)
sys.path.append(Path(__file__).resolve().parent.parent)

from config import *
from scribble_inside_shape import Scribble
import numpy as np
from openslide import OpenSlide
import cv2
import tifffile


for filename in os.listdir(path_slide_tumor_test):
    image = OpenSlide(os.path.join(path_slide_tumor_test, filename))
    mask = np.zeros(image.dimensions, dtype=np.uint8)
    s = Scribble(
        filename.split(".")[0], 0, "test", interpolation_method="cubic", show=False
    )

    ann = s.create_dataframe_annotations()
    coordinates = []
    for column in ann.columns[:1]:
        arr = ann[column].dropna()
        coordinates.append(np.array(list(arr)))
    coordinates = np.stack(coordinates).astype(np.int32)
    mask = cv2.fillPoly(mask.copy(), coordinates, 1).astype(bool)
    tifffile.imsave(os.path.join(path_slide_true_masks, filename), mask)
