import numpy as np 
optimal_threshold = 0.33

def compute_std(x,
                patch_level = True):
                
    if not(patch_level):
        x = x[:,np.mean(x,0)>optimal_threshold]
    image_std = np.std(x,0)
    max_std = np.std(np.concatenate([np.ones(10),np.zeros(10)]))
    pl = image_std/max_std

    if patch_level:
        return pl
    else: 
        return np.mean(pl)


def compute_entropy(x,
                    patch_level = True):

    if not(patch_level):
        x = x[:,np.mean(x,0)>optimal_threshold]

    n_predictions = x.shape[0]
    bins = np.linspace(0,1,n_predictions)
    bin_array = np.digitize(x,bins=bins) - 1
    count_array = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_predictions), 0, bin_array)
    prob_array = count_array/np.sum(count_array,0)
    eps = 1e-8
    entropy_array = -prob_array*np.log2(np.clip(prob_array,eps,1))
    pl = np.sum(entropy_array,axis = 0)/np.log2(n_predictions)

    if patch_level:
        return pl
    else: 
        return np.mean(pl)


def compute_minority_vote_ratio(x,
                                threshold=optimal_threshold,
                                patch_level = False):
    n_ = x.shape[0]
    binary_preds = (x >= threshold).astype(int)
    patch_tumor_pred_count = np.sum(binary_preds, axis=0)
    pl = ((patch_tumor_pred_count-(n_//2))/(n_//2) +1)/2
    return pl