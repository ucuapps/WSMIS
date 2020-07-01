import yaml
import torch
import os

from model_training.common.datasets import PascalSegmentationDataset
from model_training.common.augmentations import get_transforms
from inference.segmentation.evaluator import Evaluator

with open(
    os.path.join(os.path.dirname(__file__), "config", "eval.yaml")
) as config_file:
    config = yaml.full_load(config_file)

train_config_path = os.path.join(config["experiment_path"], "config.yaml")
with open(os.path.join(train_config_path)) as train_config_file:
    train_config = yaml.full_load(train_config_file)

transform = get_transforms(train_config["val"]["transform"])

train_ds = PascalSegmentationDataset(
    train_config["train"]["path"],
    transform=transform,
    image_set="train",
    masks_folder="out_masks/cam",
    return_original_mask=True,
)
val_ds = PascalSegmentationDataset(
    train_config["val"]["path"],
    transform=transform,
    image_set="validation",
    masks_folder="../VOCdevkit/VOC2012/SegmentationClass",
    return_original_mask=True,
)

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=12
)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=config["batch_size"], shuffle=True, num_workers=12
)

evaluator = Evaluator(config, train_dl, val_dl)
config["results"] = evaluator.evaluate()

# save result with evaluation config to the experiment directory
with open(os.path.join(config["experiment_path"], "results.yaml"), "w") as fp:
    yaml.dump(config, fp)
