import sys
from pathlib import Path
import pathlib
from PIL import Image
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from config import *
from multiprocessing import Pool
from wsitools.patch_reconstruction.save_wsi_downsampled import SubPatches2BigTiff
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Code to generate the patches of the heatmap and to stitch them."
)
parser.add_argument(
    "-f",
    "--filename",
    help="Select the filename of the slide from which you want to create a uncertainty map",
    type=str,
)

args = parser.parse_args()

filename = args.filename
path_patches = os.path.join(path_patches_test, filename)
new_filename = filename.replace("_", "")
path_pp = os.path.join(path_prediction_patches_correction, new_filename)

if not os.path.exists(path_pp):
    os.makedirs(path_pp)
path_pf = os.path.join(
    path_prediction_features, filename, "predictions_correction.npy"
)

preds = np.squeeze(np.load(path_pf))
filenames = os.listdir(path_patches)
i_s = np.arange(len(filenames))

print(" \n loading predictions and creating patches...")

def create_segmap(args,preds = preds):
    filename, i = args
    value = preds[i]
    seg = np.zeros((ps, ps, 3)) + value
    plt.imsave(os.path.join(path_pp, filename), seg)


if __name__ == "__main__":

    pool = Pool(processes=16)
    pool.map(create_segmap, zip(filenames, i_s))
    pool.close()
    
    print("stitching patches together and creating heatmap...")

# if not os.path.exists(path_segmaps):
#     os.makedirs(path_segmaps)

# sub = SubPatches2BigTiff(
#     patch_dir=path_pp,
#     save_to=os.path.join(path_segmaps, filename + ".tif"),
#     ext=".jpg",
#     down_scale=4,
#     patch_size=(ps, ps),
#     xy_step=(int(ps * (1 - ov)), int(ps * (1 - ov))),
#     grayscale = True
# )

# sub.save()
