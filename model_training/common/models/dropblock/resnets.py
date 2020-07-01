import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, BasicBlock
from .dblock import DropBlock2D, LinearScheduler

# from dropblock import DropBlock2D, LinearScheduler
import torch.nn.functional as F
import copy

model_urls = {
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


class ResNet50_DropBlock(nn.Module):
    def __init__(self, drop_prob, block_size, num_classes=2, pretrained=False):
        super(ResNet50_DropBlock, self).__init__()
        conv_modules = list(
            models.resnet50(
                pretrained=pretrained, replace_stride_with_dilation=[False, True, True]
            ).children()
        )[:8]
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        conv_modules.append(self.dropblock)  # after 4th group
        conv_modules.insert(7, self.dropblock)  # after 3rd group

        self.backbone = nn.Sequential(*conv_modules)
        self.extra_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
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


class ResNet50_DropBlock_Scheduler(nn.Module):
    def __init__(self, drop_prob, block_size, num_classes=2, pretrained=False):
        super(ResNet50_DropBlock_Scheduler, self).__init__()
        conv_modules = list(
            models.resnet50(
                pretrained=pretrained, replace_stride_with_dilation=[False, True, True]
            ).children()
        )[:8]
        self.dropblock_scheduler = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=10,
        )
        conv_modules.append(self.dropblock_scheduler)  # after 4th group
        conv_modules.insert(7, self.dropblock_scheduler)  # after 3rd group
        self.backbone = nn.Sequential(*conv_modules)
        self.extra_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        self.dropblock_scheduler.step()

        x = self.backbone(x)
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


class ResNet50_GAP_Dropblock(nn.Module):
    def __init__(self, drop_prob, block_size, num_classes=2, pretrained=False):
        super(ResNet50_GAP_Dropblock, self).__init__()
        conv_modules = list(
            models.resnet50(
                pretrained=pretrained, replace_stride_with_dilation=[False, True, True]
            ).children()
        )[:7]

        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

        conv_modules.append(self.dropblock)  # after 4th group
        conv_modules.insert(7, self.dropblock)  # after 3rd group
        self.backbone = nn.Sequential(*conv_modules)

        self.extra_convs = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.extra_linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
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


class ResNet50_GAP_Dropblock_Scheduler(nn.Module):
    def __init__(self, drop_prob, block_size, num_classes=2, pretrained=False):
        super(ResNet50_GAP_Dropblock_Scheduler, self).__init__()
        conv_modules = list(
            models.resnet50(
                pretrained=pretrained, replace_stride_with_dilation=[False, True, True]
            ).children()
        )[:7]

        self.dropblock_scheduler = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=10,
        )
        conv_modules.append(self.dropblock_scheduler)  # after 4th group
        conv_modules.insert(7, self.dropblock_scheduler)  # after 3rd group
        self.backbone = nn.Sequential(*conv_modules)

        self.extra_convs = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.extra_linear = nn.Linear(512, num_classes)

    def forward(self, x):
        self.dropblock_scheduler.step()

        x = self.backbone(x)
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


def resnet50_gap_dropblock(
    num_classes, drop_prob, block_size, pretrained=True, weights_path=None
):
    model = ResNet50_GAP_Dropblock(
        drop_prob, block_size, num_classes=num_classes, pretrained=pretrained
    )
    return model


def resnet50_gap_dropblock_scheduler(
    num_classes, drop_prob, block_size, pretrained=True, weights_path=None
):
    model = ResNet50_GAP_Dropblock_Scheduler(
        drop_prob, block_size, num_classes=num_classes, pretrained=pretrained
    )
    return model


def resnet50_dropblock(
    num_classes, drop_prob, block_size, pretrained=True, weights_path=None
):
    model = ResNet50_DropBlock(
        drop_prob, block_size, num_classes=num_classes, pretrained=pretrained
    )
    return model


def resnet50_dropblock_scheduler(
    num_classes, drop_prob, block_size, pretrained=True, weights_path=None
):
    model = ResNet50_DropBlock_Scheduler(
        drop_prob, block_size, num_classes=num_classes, pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = resnet50_dropblock(
        num_classes=20, pretrained=True, drop_prob=0.1, block_size=5
    )
    summary(model, input_size=(3, 352, 352), batch_size=10, device="cpu")
