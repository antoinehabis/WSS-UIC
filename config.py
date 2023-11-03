import os

path_camelyon = "/home/ahabis/sshfs/CAMELYON"


path_slide_tumor_train = os.path.join(path_camelyon, "train/tumor")
path_slide_tumor_test = os.path.join(path_camelyon, "test/tumor")
path_annotations_train = os.path.join(path_camelyon, "train/annotations")
path_annotations_test = os.path.join(path_camelyon, "test/annotations")


path_patches_scribbles_train = os.path.join(path_camelyon, "patches_scribbles_train")
path_patches_scribbles_test = os.path.join(path_camelyon, "patches_scribbles_test")
path_dataframe_train = os.path.join(path_camelyon, "dataframe_train.csv")
path_dataframe_test = os.path.join(path_camelyon, "dataframe_test.csv")
path_preds = "/home/ahabis/sshfs_zeus/CAMELYON"

path_patches_test = os.path.join(path_preds, "patches_test")
path_patches_mask_test = os.path.join(path_preds, "patches_masks")
path_prediction_features = os.path.join(path_preds, "features_predictions")
path_slide_true_masks = os.path.join(path_preds, "truemasks")
path_uncertainty_maps = os.path.join(path_preds, "uncertainty_maps")
path_heatmaps = os.path.join(path_preds, "heatmaps")
path_segmaps = os.path.join(path_preds, "segmaps")

path_metric_tables = os.path.join(path_preds, "metric_tables")
path_weights = os.path.join(path_preds, "weights")
path_prediction_patches = os.path.join(path_preds, 'patches_prediction')
path_prediction_patches_correction = os.path.join(path_preds, 'patches_prediction_correction')
path_uncertainty_patches = os.path.join(path_preds, 'patches_uncertainty')



optimal_threshold = 0.33
percentage_scribbled_regions = 0.1
ov = 0.5 #### overlap
ps = 512 #### patch_size
bs = 16 #### batch_size
n_passes = 20 ## monte_carlo predictions

test_set = [
    "test_040",
    "test_090",
    "test_026",
    "test_104",
    "test_027",
    "test_002",
    "test_029",
    "test_001",
    "test_061",
    "test_108",
    "test_110",
    "test_004",
    "test_092",
    "test_066",
    "test_122",
    "test_074",
    "test_079",
    
]
val_set = [
    "test_016",
    "test_068",
    "test_021",
    "test_075",
    "test_113",
    "test_094",
    "test_082",
    "test_105",
    "test_073",
    "test_071",
    "test_121",
    "test_064",
    "test_065",
    "test_069",
    "test_051",
    "test_008",
    "test_033",
    "test_030",
    "test_048",
    "test_052",
    "test_046",
    "test_099",
    "test_038",
    "test_013",
]