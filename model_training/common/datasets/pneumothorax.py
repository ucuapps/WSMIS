import torch
import os
import pandas as pd
import pydicom
import numpy as np

from model_training.common.datasets.common import read_mask

__all__ = [
    "PneumothoraxClassificationTrainDataset",
    "PneumothoraxClassificationTestDataset",
    "PneumothoraxSegmentationDataset",
]


class PneumothoraxClassificationTrainDataset(torch.utils.data.Dataset):
    def __init__(self, images_list_path, transform, return_name=False):
        self.images_list_path = images_list_path
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.images_list = pd.read_csv(images_list_path)
        self.transform = transform
        self.return_name = return_name

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        name, label = self.images_list.iloc[idx]

        ds = pydicom.dcmread(os.path.join(self.parent_dir, name + ".dcm"))
        image = np.tile(ds.pixel_array[..., None], 3)
        image = self.transform(image, mask=None)
        if self.return_name:
            return (
                torch.FloatTensor(image).permute(2, 0, 1),
                torch.FloatTensor([label]),
                name,
            )

        return torch.FloatTensor(image).permute(2, 0, 1), torch.FloatTensor([label])


class PneumothoraxClassificationTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_list_path, transform):
        self.images_list_path = images_list_path
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.images_list = pd.read_csv(images_list_path)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        name, _ = self.images_list.iloc[idx]

        ds = pydicom.dcmread(os.path.join(self.parent_dir, name + ".dcm"))
        image = np.tile(ds.pixel_array[..., None], 3)
        image = self.transform(image, mask=None)

        return torch.FloatTensor(image).permute(2, 0, 1), name


class PneumothoraxSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_list_path, transform, masks_dir):
        self.images_list_path = images_list_path
        self.parent_dir = os.path.abspath(os.path.join(images_list_path, os.pardir))
        self.images_list = pd.read_csv(images_list_path)
        self.masks_dir = os.path.join(self.parent_dir, masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        name, label, *_ = self.images_list.iloc[idx]

        ds = pydicom.dcmread(os.path.join(self.parent_dir, name + ".dcm"))
        image = np.tile(ds.pixel_array[..., None], 3)
        if label == 0:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)[..., None]
        else:
            mask = read_mask(os.path.join(self.masks_dir, name + ".png"))
        image, mask_transformed = self.transform(image=image, mask=mask)

        return (
            image.permute(2, 0, 1),
            mask_transformed.squeeze(-1),
            name,
            mask.squeeze(-1),
        )
