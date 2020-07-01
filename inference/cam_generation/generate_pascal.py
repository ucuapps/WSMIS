import yaml
import torch
import os

from inference.cam_generation.cam_grub_cut import ActivationExtractor
from model_training.common.datasets import PascalClassificationDataset
from model_training.common.augmentations import get_transforms

with open(os.path.join(os.path.dirname(__file__), "config", "cam.yaml")) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config["train"]["transform"])
val_transform = get_transforms(config["val"]["transform"])

train_ds = PascalClassificationDataset(
    config["train"]["input_path"],
    transform=train_transform,
    image_set="train",
    return_name=True,
    return_size=True,
)
val_ds = PascalClassificationDataset(
    config["val"]["input_path"],
    transform=val_transform,
    image_set="validation",
    return_name=True,
    return_size=True,
)

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=12
)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=config["batch_size"], shuffle=True, num_workers=12
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")

extractor = ActivationExtractor(config, train_dl, val_dl, device)
extractor.extract()
