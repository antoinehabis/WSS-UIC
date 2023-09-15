import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from config import *


def compute_std(x, patch_level=True):

    """
        Input: Monte_carlo predictions of the whole slide
            shape is (n_passes, nb_patches)
        Output:
            if patch_level = True
                return a modified uncertainty value for each patches of the slide:
                1. value close to 0 means high uncertainty but the predictions are under the threshold which means the patch is a potential FN.
                2. value close to 1 means high uncertainty but the predictions are above the threshold which means the patch is a potential FP.
                3. value close to 0.5 means low uncertainty.

    """
    if not (patch_level):
        x = x[:, np.mean(x, 0) > optimal_threshold]
    image_std = np.std(x, 0)
    max_std = 0.5
    pl = image_std / max_std

    if patch_level:
        pl = pl/np.max(pl)
        pl = ((pl * 2 * (np.mean(x, 0) > optimal_threshold).astype(int) - 1) +1)/2
        return pl

        return np.mean(pl)


def compute_entropy(x, patch_level=True):
        
    """
    Input: Monte_carlo predictions of the whole slide
        shape is (n_passes, nb_patches)
    Output:
        if patch_level = True
            return a modified uncertainty value for each patches of the slide:
            1. value close to 0 means high uncertainty but the predictions are under the threshold which means the patch is a potential FN.
            2. value close to 1 means high uncertainty but the predictions are above the threshold which means the patch is a potential FP.
            3. value close to 0.5 means low uncertainty.

    """

    if not (patch_level):
        x = x[:, np.mean(x, 0) > optimal_threshold]

    n_predictions = x.shape[0]
    bins = np.linspace(0, 1, n_predictions)
    bin_array = np.digitize(x, bins=bins) - 1
    count_array = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_predictions), 0, bin_array
    )
    prob_array = count_array / np.sum(count_array, 0)
    eps = 1e-8
    entropy_array = -prob_array * np.log2(np.clip(prob_array, eps, 1))
    pl = np.sum(entropy_array, axis=0) / np.log2(n_predictions)

    if patch_level:

        pl = pl / np.max(pl)
        pl = (pl * (2 * (np.mean(x, 0) > optimal_threshold).astype(int) - 1) +1)/2
        return pl
    else:
        return np.mean(pl)


def compute_minority_vote_ratio(x, threshold=optimal_threshold, patch_level=False):
    n_ = x.shape[0]
    binary_preds = (x >= threshold).astype(int)
    patch_tumor_pred_count = np.sum(binary_preds, axis=0)
    pl = ((patch_tumor_pred_count - (n_ // 2)) / (n_ // 2) + 1) / 2
    return pl
