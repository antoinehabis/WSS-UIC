import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from multiprocessing import Pool
from wsitools.patch_reconstruction.save_wsi_downsampled import SubPatches2BigTiff
from uncertainty_metrics import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(
    description="Code to generate the patches of uncertainty map and to stitch them."
)
parser.add_argument(
    "-f",
    "--filename",
    help="Select the filename of the slide from which you want to create a uncertainty map",
    type=str,
)

parser.add_argument(
    "-u",
    "--uncertainty",
    help="Select the uncertainty metric you want to calculate:\n choose between mvr, entropy, std",
    type=str,
    default="entropy",
)
args = parser.parse_args()
filename = args.filename
uncertainty = args.uncertainty
path_patches = os.path.join(path_patches_test, filename)
new_filename = filename.replace("_", "")
path_pp = os.path.join(path_prediction_patches, new_filename)

if not os.path.exists(path_pp):
    os.makedirs(path_pp)

path_pf = os.path.join(
    os.path.join(path_prediction_features, filename), "predictions.npy"
)
preds = np.squeeze(np.load(path_pf))


##### SELECT THE UNCERTAINTY MEASURE
if uncertainty == "mvr":
    preds = compute_minority_vote_ratio(preds)
if uncertainty == "entropy":
    preds = compute_entropy(preds, patch_level=True)
if uncertainty == "std":
    preds = compute_std(preds, patch_level=True)
#####

filenames = os.listdir(path_patches)
i_s = np.arange(len(filenames))


def create_heatmap(args, colormap=plt.cm.PiYG):
    filename, i = args
    real_img = np.asarray(Image.open(os.path.join(path_patches, filename))).copy()
    value = preds[i]
    heatmap = 1 - (np.ones((ps, ps)) * value)
    colormapped_heatmap = (colormap(heatmap) * 255).astype(np.uint8)[:, :, :3]
    img_to_save = cv2.addWeighted(colormapped_heatmap, 0.6, real_img[:, :, :3], 0.4, 0)
    plt.imsave(os.path.join(path_pp, filename), img_to_save)


if __name__ == "__main__":
    print("saving patches predictions...")
    pool = Pool(processes=16)
    pool.map(create_heatmap, zip(filenames, i_s))

print("stitching patches together and creating heatmap...")

if not os.path.exists(path_uncertainty_maps):
    os.makedirs(path_uncertainty_maps)

sub = SubPatches2BigTiff(
    patch_dir=path_pp,
    save_to=os.path.join(path_uncertainty_maps, filename + ".tif"),
    ext="",
    down_scale=4,
    patch_size=(ps, ps),
    xy_step=(int(ps * (1 - ov)), int(ps * (1 - ov))),
)
sub.save()
