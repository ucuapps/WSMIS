import yaml
import torch
import numpy as np
import os
from torchsampler import ImbalancedDatasetSampler

from model_training.common.trainer import Trainer
from model_training.common.datasets import PneumothoraxSegmentationDataset
from model_training.common.augmentations import get_transforms

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(
    os.path.join(os.path.dirname(__file__), "config", "pneumothorax.yaml")
) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config["train"]["transform"])
val_transform = get_transforms(config["val"]["transform"])

train_ds = PneumothoraxSegmentationDataset(
    config["train"]["path"],
    transform=train_transform,
    masks_dir=config["train"]["masks"],
)
val_ds = PneumothoraxSegmentationDataset(
    config["val"]["path"], transform=val_transform, masks_dir=config["val"]["masks"]
)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    num_workers=12,
    drop_last=True,
    sampler=ImbalancedDatasetSampler(
        train_ds, callback_get_label=lambda ds, idx: int(ds.images_list.iloc[idx][1])
    ),
)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=config["batch_size"], shuffle=True, num_workers=12
)

trainer = Trainer(config, train_dl, val_dl)
trainer.train()
