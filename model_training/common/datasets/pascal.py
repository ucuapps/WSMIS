import torch
import os
import json
import numpy as np
import cv2

from .common import read_img, read_mask

__all__ = [
    "PascalSegmentationDataset",
    "PascalClassificationDataset",
    "PascalCRFSegmentationDataset",
]


class PascalSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_list_path,
        transform,
        verbose=True,
        image_set="train",
        masks_folder="segmentation",
        return_original_mask=False,
    ):
        self.images_list_path = images_list_path
        self.transform = transform
        self.verbose = verbose
        self.return_original_mask = return_original_mask

        assert image_set in ("train", "validation")
        with open(images_list_path, "r") as fp:
            annotations = json.load(fp)
            if image_set == "train":
                self.images_list = list(
                    map(lambda x: x["image"], annotations["train"])
                ) + list(map(lambda x: x["image"], annotations["trainaug"]))
            else:
                self.images_list = list(map(lambda x: x["image"], annotations))

        parent_dir = os.path.abspath(os.path.join(images_list_path, os.path.pardir))
        self.images_dir = os.path.join(parent_dir, "images")
        self.segmentations_dir = os.path.join(parent_dir, masks_folder)

        with open(os.path.join(parent_dir, "class_to_index.json"), "r") as fp:
            self.class_to_index = json.load(fp)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        basename = self.images_list[idx]

        image = read_img(os.path.join(self.images_dir, basename + ".jpg"))
        mask = read_mask(os.path.join(self.segmentations_dir, basename) + ".png")

        return_values = list(map(self.__set_channel_first, self.transform(image, mask)))
        return_values[1].squeeze_(0)  # squeeze class dimension for mask
        return_values.append(basename)
        if self.return_original_mask:
            return_values.append(mask.squeeze().astype(np.int64))

        return return_values

    @staticmethod
    def __set_channel_first(image):
        return image.permute(2, 0, 1)


class PascalClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_list_path,
        transform,
        verbose=True,
        image_set="train",
        return_name=False,
        return_size=False,
    ):
        self.images_list_path = images_list_path
        self.transform = transform
        self.verbose = verbose
        self.image_set = image_set
        self.return_name = return_name
        self.return_size = return_size

        assert image_set in ("train", "validation")
        self.annotations = {}

        with open(images_list_path, "r") as fp:
            self.annotations[image_set] = json.load(fp)
            if image_set == "train":
                self.images_list = list(
                    map(lambda x: x, self.annotations[image_set]["train"])
                ) + list(map(lambda x: x, self.annotations[image_set]["trainaug"]))
            else:
                self.images_list = list(map(lambda x: x, self.annotations[image_set]))

        parent_dir = os.path.abspath(os.path.join(images_list_path, os.path.pardir))
        self.images_dir = os.path.join(parent_dir, "images")

        with open(os.path.join(parent_dir, "class_to_index.json"), "r") as fp:
            self.class_to_index = json.load(fp)

        self.num_classes = len(self.class_to_index.values()) - 1  # except background

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        basename = self.images_list[idx]["image"]
        label = np.vectorize(self.class_to_index.get)(self.images_list[idx]["classes"])
        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[label - 1] = 1

        image = read_img(os.path.join(self.images_dir, basename + ".jpg"))
        shape = torch.tensor(image.shape[:2])
        image = self.transform(image, None).permute(2, 0, 1)

        return_values = [image, label_encoded]
        if self.return_name:
            return_values.append(self.images_list[idx]["image"])
        if self.return_size:
            return_values.append(shape)

        return return_values


class PascalCRFSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_list_path,
        transform,
        image_set="train",
        masks_folder="segmentation",
        scale_factor=8,
    ):
        self.images_list_path = images_list_path
        self.transform = transform

        assert image_set in ("train", "validation")
        with open(images_list_path, "r") as fp:
            annotations = json.load(fp)
            if image_set == "train":
                self.images_list = annotations["train"] + annotations["trainaug"]
            else:
                self.images_list = annotations

        parent_dir = os.path.abspath(os.path.join(images_list_path, os.path.pardir))
        self.images_dir = os.path.join(parent_dir, "images")
        self.segmentations_dir = os.path.join(parent_dir, masks_folder)

        with open(os.path.join(parent_dir, "class_to_index.json"), "r") as fp:
            self.class_to_index = json.load(fp)
        self.scale_factor = scale_factor
        self.num_classes = len(self.class_to_index) - 1  # except background class

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        basename = self.images_list[idx]["image"]
        label = np.vectorize(self.class_to_index.get)(self.images_list[idx]["classes"])
        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[label - 1] = 1

        image = read_img(os.path.join(self.images_dir, basename + ".jpg"))
        mask = read_mask(os.path.join(self.segmentations_dir, basename) + ".png")

        return_values = list(map(self.__set_channel_first, self.transform(image, mask)))
        return_values[1].squeeze_(0)  # squeeze class dimension for mask
        return_values.append(basename)

        return_values.append(label_encoded)

        h, w = return_values[0].shape[-2:]
        return_values.append(
            cv2.resize(
                image,
                (w // self.scale_factor, h // self.scale_factor),
                interpolation=cv2.INTER_CUBIC,
            )
        )
        return return_values

    @staticmethod
    def __set_channel_first(image):
        return image.permute(2, 0, 1)
