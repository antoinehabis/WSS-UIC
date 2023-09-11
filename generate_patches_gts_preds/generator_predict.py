import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.utils.data import Dataset, DataLoader
from config import *
from PIL import ImageFile, Image
import pandas as pd
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

filename = "test_001"
path_image = os.path.join(path_patches_test, filename)


class CustomImageDataset(Dataset):
    def __init__(self, path_image):
        self.path_image = path_image
        self.dataframe = pd.DataFrame(columns=["filename"])
        self.dataframe["filename"] = os.listdir(self.path_image)

    def __getitem__(self, idx):
        filename = self.dataframe.loc[idx]["filename"]
        label = float(filename.split("_")[-1].split(".")[0])
        image = np.asarray(Image.open(os.path.join(self.path_image, filename)))[
            :, :, :3
        ]
        image = np.transpose(image, (-1, 0, 1)) / 255

        return image, label

    def __len__(self):
        return self.dataframe.shape[0]


dataset_test = CustomImageDataset(path_image=path_image)

loader_test = DataLoader(
    batch_size=bs, dataset=dataset_test, num_workers=16, shuffle=False
)

dataloaders = {"test": loader_test}
