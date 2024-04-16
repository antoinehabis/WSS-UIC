import sys
from pathlib import Path
import multiprocessing
import ctypes
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
import tifffile
import numpy as np
from tqdm import tqdm
from openslide import OpenSlide


filenames = os.listdir(path_slide_tumor_test)
# filenames = ['test_008.tif']

for filename in tqdm(filenames):
    
    filename = filename.split(".")[0]
    rename_dir = os.path.join(path_patches_test, filename)
    path_patches = os.path.join(path_patches_test, filename)
    path_pf = os.path.join(path_prediction_features, filename)

    if not os.path.exists(path_pf):
        os.makedirs(path_pf)

    path_mask = os.path.join(path_slide_true_masks, filename + ".tif")

    print("retrieving labels from mask ...")
    path_patches = os.path.join(path_patches_test, filename)
    img = OpenSlide(path_mask)
    print("finish loading image")

    files = os.listdir(path_patches)
    true_vals_base = multiprocessing.Array(ctypes.c_double, len(files))
    true_vals = np.ctypeslib.as_array(true_vals_base.get_obj())
    true_vals = true_vals.reshape(len(files))

    def fill_true_vals(args, true_vals = true_vals):
        index, filename = args
        split = filename.split("_")

        x = int(split[2])
        y = int(split[3].split(".")[0])

        patch = np.array(img.read_region(location=(x, y), level=0, size=(ps, ps)))[:,:,:-1]//255
        mean_ = np.mean(np.mean(patch,axis = -1))
        th = 0.1
        if mean_ > th:
            true_vals[index] = 1
        else:
            true_vals[index] = 0

    if __name__ == '__main__':

        pool = multiprocessing.Pool(processes=16)
        pool.map(fill_true_vals, zip(np.arange(len(files)), files))
        np.save(os.path.join(path_pf, "trues.npy"), true_vals)