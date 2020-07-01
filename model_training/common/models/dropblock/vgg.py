"""
Based on torchvision vgg
@author: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .dblock import DropBlock2D, LinearScheduler
import math
import copy

model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.extra_linear = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x, epoch=1, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = self.extra_linear(x.squeeze(-1).squeeze(-1))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_params_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if "extra" in name:
                if "weight" in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if "weight" in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


class VGG_dropblock(nn.Module):
    def __init__(self, model, drop_prob, block_size):
        super(VGG_dropblock, self).__init__()
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16_bn"]), strict=False)

        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

        new_features = nn.Sequential()
        l = 0
        for i, layer in enumerate(list(model.features)):
            new_features.add_module(str(l), layer)
            l += 1
            if l == 3:  # Conv2d-2,   l == 9 -  Conv2d-9
                l += 1
                new_features.add_module(str(l), self.dropblock)  # dropblock
                l += 1

        self.features = new_features
        self.extra_convs = copy.deepcopy(model.extra_convs)
        self.extra_linear = copy.deepcopy(model.extra_linear)
        del model  # clear memmory

    def forward(self, x):
        x = self.features(x)
        x = self.extra_convs(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = self.extra_linear(x.squeeze(-1).squeeze(-1))
        return x

    def get_params_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if "extra" in name:
                if "weight" in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if "weight" in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


class VGG_dropblock_Scheduler(nn.Module):
    def __init__(self, model, drop_prob, block_size):
        super(VGG_dropblock_Scheduler, self).__init__()
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16_bn"]), strict=False)

        self.dropblock_scheduler = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=10,
        )
        new_features = nn.Sequential()
        l = 0
        for i, layer in enumerate(list(model.features)):
            new_features.add_module(str(l), layer)
            l += 1
            if l == 3:  # Conv2d-2,   l == 9 -  Conv2d-9
                l += 1
                new_features.add_module(str(l), self.dropblock_scheduler)  # dropblock
                l += 1

        self.features = new_features
        self.extra_convs = copy.deepcopy(model.extra_convs)
        self.extra_linear = copy.deepcopy(model.extra_linear)
        del model  # clear memmory

    def forward(self, x):
        self.dropblock_scheduler.step()

        x = self.features(x)
        x = self.extra_convs(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = self.extra_linear(x.squeeze(-1).squeeze(-1))
        return x

    def get_params_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if "extra" in name:
                if "weight" in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if "weight" in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "N":
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "D1": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "N",
        512,
        512,
        512,
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg16_dropblock(pretrained=True, drop_prob=0.5, block_size=5, **kwargs):
    model_vgg_gap = VGG(make_layers(cfg["D1"], batch_norm=True), **kwargs)
    model = VGG_dropblock(model_vgg_gap, drop_prob, block_size)
    return model


def vgg16_dropblock_scheduler(pretrained=True, drop_prob=0.5, block_size=5, **kwargs):
    model_vgg_gap = VGG(make_layers(cfg["D1"], batch_norm=True), **kwargs)
    model = VGG_dropblock_Scheduler(model_vgg_gap, drop_prob, block_size)
    return model
