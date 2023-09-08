import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
from config import *
from multiprocessing import Pool
from wsitools.patch_reconstruction.save_wsi_downsampled import SubPatches2BigTiff
from uncertainty_metrics import *
import numpy as np
import cv2
import matplolib.pyplot as plt
from PIL import Image

filename = "test_001"

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

preds = compute_minority_vote_ratio(preds)
# preds = compute_entropy(preds, patch_level=True)
# preds = compute_std(preds, patch_level=True)

#####

filenames = os.listdir(path_patches)
i_s = np.arange(len(filenames))


def create_heatmap(args):
    filename, i = args
    real_img = np.asarray(Image.open(os.path.join(path_patches, filename)))
    value = preds[i]
    heatmap = 255 - (np.ones((ps, ps)) * value * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    img_to_save = cv2.addWeighted(heatmap, 0.6, real_img[:, :, :3], 0.4, 0)
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
