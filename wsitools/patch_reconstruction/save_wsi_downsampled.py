from config import *

# from operator import itemgetter
# import pyvips
import subprocess
import re
import openslide
import multiprocessing
import math
from glob import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# save image patches back into a big tiff file
# file name should fellow this pattern uuid_x_y.jpg or uuid_x_y.png
# otherwise, you have to rewrite the function


def hann_2d_win(shape=(256, 256)):
    def hann2d(i, j):
        i_val = 1 - np.cos((2 * math.pi * i) / (shape[0] - 1))
        j_val = 1 - np.cos((2 * math.pi * j) / (shape[1] - 1))
        normalized = (i_val * j_val) / 4

        return normalized

    hann2d_win = np.fromfunction(hann2d, shape)
    return hann2d_win


class SubPatches2BigTiff:
    def __init__(
        self,
        patch_dir,
        save_to,
        ext=".tif",
        down_scale=8,
        patch_size=(256, 256),
        xy_step=(128, 128)):
        self.patch_dir = patch_dir
        self.save_to = save_to
        self.ext = ext
        self.patch_size = patch_size
        self.xy_step = xy_step
        self.filenames = sorted(glob(patch_dir + "/*" + ext))
        w, h, self.x_min, self.y_min = self.calculate_tiff_w_h()
        self.down_scale = down_scale
        print("Image W:%d/H:%d" % (int(w / self.down_scale), int(h / self.down_scale)))

        self.out_arr = (
                np.zeros((int(h / self.down_scale), int(w / self.down_scale), 3)) + 255
            )
        self.filter = hann_2d_win((512 // down_scale, 512 // down_scale))

    @staticmethod
    def shell_cmd(cmd):
        cmd = re.sub("\s+", " ", cmd).strip()
        cmd = cmd.split(" ")
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = m.communicate()
        exitCode = m.returncode
        return exitCode

    def calculate_tiff_w_h(self):
        locations = []

        for f in self.filenames:
            fn = os.path.split(f)[1]
            p = fn.split("_")
            p_3 = int(p[3].split(".")[0])
            locations.append([int(p[2]), p_3])
        patch_locs = np.array(locations)
        x_min = min(patch_locs[:, 0])
        x_max = max(patch_locs[:, 0])
        y_min = min(patch_locs[:, 1])
        y_max = max(patch_locs[:, 1])

        w = x_max - x_min + self.patch_size[0]
        h = y_max - y_min + self.patch_size[1]

        row_cnt = int(h / self.xy_step[1])
        column_cnt = int(w / self.xy_step[0])
        return w, h, x_min, y_min

    def insert_patch(self, f):
        x_r = int(self.patch_size[0] / self.down_scale)
        y_r = int(self.patch_size[1] / self.down_scale)

        try:
            img = plt.imread(f)
            img = (img * 255).astype(np.uint8)
            sub_arr = cv2.resize(img, dsize=(x_r, y_r), interpolation=cv2.INTER_CUBIC)[:,:,:-1]
            fn = os.path.split(f)[1]
            p = fn.split("_")
            x = int(p[2])
            y = int(p[3].split('.')[0])
            x_loc = int((x-self.x_min)/self.down_scale)
            y_loc = int((y-self.y_min)/self.down_scale)
            self.out_arr[y_loc:y_loc+y_r, x_loc:x_loc+x_r, :] += sub_arr * self.filter[:,:,None]
        except:
            pass

    def parallel_save(self):  # example: save("big.tiff")
        num_processors = 16
        multiprocessing.set_start_method("spawn")
        pool = multiprocessing.Pool(processes=num_processors)
        pool.map(self.insert_patch, self.filenames)
        pool.close()
    def save(self):
        cnt = 0
        print("Insert %d images patches" % len(self.filenames))
        for f in self.filenames:
            self.insert_patch(f)
            cnt += 1
            if cnt % 2000 == 0:
                print("Insert %d/%d images patches" % (cnt, len(self.filenames)))

        Image.fromarray(self.out_arr.astype(np.uint8), "RGB").save(self.save_to)
        # else:
        #     print(np.max(self.out_arr))
        #     Image.fromarray(self.out_arr.astype(np.uint8)).save(self.save_to)

    def get_thumbnail(self, thumbnail_fn):
        obj = openslide.open_slide(self.save_to)
        print("WSI loaded")

        thumbnail = obj.get_thumbnail(size=(1024, 1024)).convert("RGB")
        thumbnail.save(thumbnail_fn)
