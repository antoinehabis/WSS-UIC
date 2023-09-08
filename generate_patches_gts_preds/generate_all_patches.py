import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import (
    ExtractorParameters,
    PatchExtractor,
)
import multiprocessing
from config import *
from glob import glob

num_processors = 16  # Number of processes that can be running at once
k = 0  # index of the split: can only run 16 Slides at a time so we have to split the Slides into 3 split

wsi_fn = glob(path_slide_tumor_test + "/*")[
    num_processors * k : num_processors * (k + 1)
]  # The corresponding 16 slides
patches = os.listdir(path_slide_tumor_test)[
    num_processors * k : num_processors * (k + 1)
]

folders = [u.split(".")[0] for u in patches]
abs_path_folders = [os.path.join(path_patches_test, u) for u in folders]


### This part create each folder in path_patches if they don't exist yet

for u in abs_path_folders:
    if not os.path.exists(u):
        os.makedirs(u)


output_dir = path_patches_test  # Define an output directory

# Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
# tissue.
parameters = ExtractorParameters(
    output_dir,  # Where the patches should be extracted to
    save_format=".png",  # Can be '.jpg', '.png', or '.tfrecord'
    sample_cnt=-1,  # Limit the number of patches to extract (-1 == all patches)
    patch_size=ps,
    stride=int(ps * (1 - ov)),  # Size of patches to extract (Height & Width)
    rescale_rate=128,  # Fold size to scale the thumbnail to (for faster processing)
    patch_filter_by_area=0.5,  # Amount of tissue that should be present in a patch
    extract_layer=0,  # OpenSlide Level
)


tissue_detector = TissueDetector("LAB_Threshold", threshold=85, training_files=None)

patch_extractor = PatchExtractor(
    tissue_detector,
    parameters,
    feature_map=None,  # See note below
    annotations=None,  # Object of Annotation Class (see other note below)
)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    pool = multiprocessing.Pool(processes=num_processors)
    pool.map(patch_extractor.extract, wsi_fn)
    pool.close()
