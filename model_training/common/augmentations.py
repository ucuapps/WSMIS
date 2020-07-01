from typing import Dict

import cv2
import albumentations as albu
import numpy as np
import torch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Batch of images of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_copy = tensor.clone().transpose(1, 0)
        for t, m, s in zip(tensor_copy, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_copy.transpose(1, 0)


output_format = {
    "none": lambda array: array,
    "float": lambda array: torch.FloatTensor(array),
    "long": lambda array: torch.LongTensor(array),
    "byte": lambda array: torch.ByteTensor(array),
}

normalization = {
    "none": lambda array: array,
    "float": lambda array: array / 255.0,
    "default": lambda array: albu.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(image=array)["image"],
    "pneumothorax": lambda array: albu.Normalize(
        mean=[0.490, 0.490, 0.490], std=[0.230, 0.230, 0.230]
    )(image=array)["image"],
    "binary": lambda array: np.array(array > 0, np.float32),
}

denormalization = {
    "none": lambda array: array,
    "float": lambda array: array,
    "default": lambda array: UnNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(array),
    "pneumothorax": lambda array: UnNormalize(
        mean=[0.490, 0.490, 0.490], std=[0.230, 0.230, 0.230]
    )(array),
}

augmentations = {
    "strong": albu.Compose(
        [
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(
                shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4
            ),
            albu.ElasticTransform(),
            albu.GaussNoise(),
            albu.OneOf(
                [
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                    albu.MedianBlur(),
                ],
                p=0.5,
            ),
            albu.OneOf([albu.RGBShift(), albu.HueSaturationValue()], p=0.5),
        ]
    ),
    "weak": albu.Compose([albu.HorizontalFlip()]),
    "pneumothorax": albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([albu.RandomBrightnessContrast(), albu.RandomGamma(),], p=0.3),
            albu.OneOf(
                [
                    albu.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.3,
            ),
        ]
    ),
    "horizontal_flip": albu.HorizontalFlip(p=0.5),
    "none": albu.Compose([]),
}

size_augmentations = {
    "none": lambda size: albu.NoOp(),
    "resize": lambda size: albu.Resize(
        height=size, width=size, interpolation=cv2.INTER_AREA
    ),
    "center": lambda size: albu.CenterCrop(size, size),
    "crop_or_resize": lambda size: albu.OneOf(
        [albu.RandomCrop(size, size), albu.Resize(height=size, width=size)], p=1
    ),
}


def get_transforms(config: Dict):
    size = config["size"]
    scope = config.get("augmentation_scope", "none")
    size_transform = config.get("size_transform", "none")

    images_normalization = config.get("images_normalization", "default")
    masks_normalization = config.get("masks_normalization", "binary")

    images_output_format_type = config.get("images_output_format_type", "float")
    masks_output_format_type = config.get("masks_output_format_type", "float")

    aug = albu.Compose([augmentations[scope], size_augmentations[size_transform](size)])

    def process(image, mask):
        r = aug(image=image, mask=mask)
        transformed_image = output_format[images_output_format_type](
            normalization[images_normalization](r["image"])
        )
        if r["mask"] is not None:
            transformed_mask = output_format[masks_output_format_type](
                normalization[masks_normalization](r["mask"])
            )

            return transformed_image, transformed_mask

        else:
            return transformed_image

    return process
