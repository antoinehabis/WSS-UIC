import sys
from pathlib import Path
import pathlib
import tifffile
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

if __name__ == "__main__":
    for filename in tqdm(os.listdir(path_slide_tumor_test)):
        filename = filename.split(".")[0]
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

        def create_segmap(args):
            filename, i = args
            value = preds[i]
            seg = np.zeros((ps, ps)) + value
            tifffile.imsave(os.path.join(path_pp, filename.split('.')[0]+'.tif'), seg)

        pool = Pool(processes=16)
        pool.map(create_segmap, zip(filenames, i_s))
        print("stitching patches together and creating heatmap...")

        if not os.path.exists(path_segmaps):
            os.makedirs(path_segmaps)

        sub = SubPatches2BigTiff(
            patch_dir=path_pp,
            save_to=os.path.join(path_segmaps, filename + ".tif"),
            ext=".tif",
            down_scale=4,
            patch_size=(ps, ps),
            xy_step=(int(ps * (1 - ov)), int(ps * (1 - ov))),
            grayscale = True
        )

        sub.save()
