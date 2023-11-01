import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from scribble_inside_shape import Scribble
import numpy as np
from openslide import OpenSlide
import cv2
import tifffile
from tqdm import tqdm
import pyvips

for filename in tqdm(os.listdir(path_slide_tumor_test)):
    image = OpenSlide(os.path.join(path_slide_tumor_test, filename))
    mask = np.zeros(np.flip(image.dimensions), dtype=np.uint8).copy()
    s = Scribble(filename.split(".")[0], 0, "test", show=False)
    ann = s.create_dataframe_annotations()
    coordinates = []
    for column in ann.columns:
        arr = ann[column].dropna()
        coordinates.append(np.array(list(arr)).astype(np.int32))
    mask = cv2.fillPoly(mask, coordinates, 1)
    image_binary = pyvips.Image.new_from_array(mask.astype(bool))
    image_binary.write_to_file(
        os.path.join(path_slide_true_masks, filename),
        pyramid=True,
        tile=True,
        bigtiff=True,
        subifd=False,
        compression="jpeg",
        tile_height=512,
        tile_width=512,
    )
