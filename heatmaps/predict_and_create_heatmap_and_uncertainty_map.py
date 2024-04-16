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

path_uncertainty = os.path.join(path_uncertainty_patches, new_filename)
path_prediction = os.path.join(path_prediction_patches, new_filename)

if not os.path.exists(path_prediction):
    os.makedirs(path_prediction)

if not os.path.exists(path_uncertainty):
    os.makedirs(path_uncertainty)

path_pf = os.path.join(
    os.path.join(path_prediction_features, filename),
    "predictions_correction_3_heatmap.npy",
)
preds = np.load(path_pf)

##### SELECT THE UNCERTAINTY MEASURE

if uncertainty == "mvr":
    uncertainties = compute_minority_vote_ratio(preds)
if uncertainty == "entropy":
    uncertainties = compute_entropy(preds, patch_level=True)
if uncertainty == "std":
    uncertainties = compute_std(preds, patch_level=True)
else:
    mean_predictions = np.mean(preds, axis=0)

#####
filenames = os.listdir(path_patches)
i_s = np.arange(len(filenames))


def create_heatmap(
    args,
    colormap_uncertainty=plt.cm.PiYG,
    colormap_heatmap=cv2.COLORMAP_JET,
    uncertainty=uncertainty,
):
    filename, i = args
    real_img = np.asarray(Image.open(os.path.join(path_patches, filename))).copy()
    uncertainty, mean = uncertainties[i], mean_predictions[i]
    if uncertainty != None:
        heatmap = 1 - (np.ones((ps, ps)) * uncertainty)

        colormapped_uncertainty_heatmap = (colormap_uncertainty(heatmap) * 255).astype(
            np.uint8
        )[:, :, :3]
        img_heatmap_save = cv2.addWeighted(
            colormapped_uncertainty_heatmap, 0.6, real_img[:, :, :3], 0.4, 0
        )
        plt.imsave(os.path.join(path_uncertainty, filename), img_heatmap_save)

    else:

        heatmap = 1 - (np.ones((ps, ps)) * mean)
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap_heatmap)
        img_heatmap_save = cv2.addWeighted(heatmap, 0.6, real_img[:, :, :3], 0.4, 0)
        plt.imsave(os.path.join(path_prediction, filename), img_heatmap_save)


if __name__ == "__main__":
    print("saving patches predictions...")
    pool = Pool(processes=32)
    pool.map(create_heatmap, zip(filenames, i_s))
    pool.close()

print("stitching patches together and creating heatmap/uncertainty map...")

if uncertainty != None:
    if not os.path.exists(path_uncertainty_maps):
        os.makedirs(path_uncertainty_maps)

    sub = SubPatches2BigTiff(
        patch_dir=path_uncertainty,
        save_to=os.path.join(path_uncertainty_maps, filename + ".tif"),
        ext="",
        down_scale=4,
        patch_size=(ps, ps),
        xy_step=(int(ps * (1 - ov)), int(ps * (1 - ov))),
    )
    sub.save()

else:
    sub = SubPatches2BigTiff(
        patch_dir=path_prediction,
        save_to=os.path.join(path_heatmaps, filename + ".tif"),
        ext="",
        down_scale=4,
        patch_size=(ps, ps),
        xy_step=(int(ps * (1 - ov)), int(ps * (1 - ov))),
    )
    sub.save()
