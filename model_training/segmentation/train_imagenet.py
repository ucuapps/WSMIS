import yaml
import torch
import numpy as np
import os

from model_training.common.trainer import Trainer
from model_training.common.datasets import ImageNetSegmentation
from model_training.common.augmentations import get_transforms

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(os.path.dirname(__file__), 'config', 'imagenet.yaml')) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config['train']['transform'])
val_transform = get_transforms(config['val']['transform'])

train_ds = ImageNetSegmentation(config['train']['path'], transform=train_transform,
                                masks_path=config['train']['mask_path'])
val_ds = ImageNetSegmentation(config['val']['path'], transform=val_transform, masks_path=config['val']['mask_path'])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12,
                                       drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12,
                                     drop_last=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device {device}')

trainer = Trainer(config, train_dl, val_dl, device)
trainer.train()
