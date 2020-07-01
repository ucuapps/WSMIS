import yaml
import torch
import os
import numpy as np
import random
from torchsampler import ImbalancedDatasetSampler

from model_training.common.trainer import Trainer
from model_training.common.augmentations import get_transforms
from model_training.common.datasets import PneumothoraxClassificationTrainDataset

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(
    os.path.join(os.path.dirname(__file__), "config", "pneumothorax.yaml")
) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config["train"]["transform"])
val_transform = get_transforms(config["val"]["transform"])

train_ds = PneumothoraxClassificationTrainDataset(
    config["train"]["path"], transform=train_transform
)
val_ds = PneumothoraxClassificationTrainDataset(
    config["val"]["path"], transform=val_transform
)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    num_workers=12,
    sampler=ImbalancedDatasetSampler(
        train_ds, callback_get_label=lambda ds, idx: int(ds.images_list.iloc[idx][1])
    ),
)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=config["batch_size"], drop_last=True, num_workers=12
)

print("Train: ", len(train_dl), "  Val: ", len(val_dl))

trainer = Trainer(config, train_dl, val_dl)
trainer.train()
