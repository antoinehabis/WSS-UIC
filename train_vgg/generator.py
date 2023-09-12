import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFile, Image
import pandas as pd
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomImageDataset(Dataset):
    def augmenter(self, image):
        k = np.random.choice([1, 2, 3])
        alea_shift1 = np.random.random()
        alea_shift2 = np.random.random()
        image = np.rot90(image, k=k, axes=(0, 1))

        if alea_shift1 > 0.5:
            image = np.flipud(image)
        if alea_shift2 > 0.5:
            image = np.fliplr(image)

        return image

    def __init__(self, path_image, augmenter_bool):
        self.path_image = path_image
        self.dataframe = pd.DataFrame(columns=["filename"])
        self.dataframe["filename"] = os.listdir(self.path_image)
        self.augmenter_bool = augmenter_bool

    def __getitem__(self, idx):
        filename = self.dataframe.loc[idx]["filename"]
        label = float(filename.split("_")[-1].split(".")[0])
        image = np.asarray(Image.open(os.path.join(self.path_image, filename)))
        if self.augmenter_bool:
            image = self.augmenter(image)
        image = np.transpose(image, (-1, 0, 1)) / 255

        return image, label

    def __len__(self):
        return self.dataframe.shape[0]


dataset_train = CustomImageDataset(
    path_image=path_patches_scribbles_train, augmenter_bool=True
)
dataset_test = CustomImageDataset(
    path_image=path_patches_scribbles_test, augmenter_bool=True
)

loader_train = DataLoader(
    batch_size=bs, dataset=dataset_train, num_workers=16, shuffle=True
)

loader_test = DataLoader(
    batch_size=bs, dataset=dataset_test, num_workers=16, shuffle=False
)

dataloaders = {"train": loader_train, "test": loader_test}
