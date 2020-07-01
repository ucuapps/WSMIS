import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, BasicBlock

resnet50_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
resnet34_url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetGAP(nn.Module):
    """Model that uses ResNet50 conv layers as backbone followed by Global Average Polling"""

    def __init__(self, n_classes=200):
        super(ResNetGAP, self).__init__()
        conv_modules = list(models.resnet50(pretrained=True).children())[:7]
        self.backbone = nn.Sequential(*conv_modules)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(1024, n_classes)

    def forward(self, x, return_maps=False):
        """
        Forward path of ResNetGAP model.

        Args:
            x (torch.tensor): 4d input volume of size (B, C, H, W)
            return_maps (bool): boolean flag whether to return activation maps from last con layer
        Returns:
            torch.tensor: logits for classification task
            torch.tensor (optional): activation maps from last con layer
        """
        maps = self.backbone(x)
        x = self.gap(maps)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        if return_maps:
            return x, maps

        return x


class ResNet34(nn.Sequential):
    def __init__(self, num_classes):
        model_modules = list(
            models.resnet34(pretrained=False, num_classes=num_classes).named_children()
        )
        # model_modules[0] = (
        # 'conv1_single_channel', nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        model_modules.insert(-1, ("flatten", Flatten()))
        super().__init__(OrderedDict(model_modules))

    def get_params_groups(self):
        feature_extractor_params = []
        for name, module in self.named_children():
            if name != "fc":
                feature_extractor_params.extend(module.parameters())
            else:
                fc_params = module.parameters()
        return [feature_extractor_params, fc_params]


def resnet34(pretrained, num_classes, weights_path):
    model = ResNet34(num_classes if num_classes != 2 else 1)

    if pretrained:
        if weights_path is not None:
            state_dict = torch.load(weights_path)["model"]
        else:
            state_dict = load_state_dict_from_url(resnet34_url, progress=True)
            if num_classes != 1000:  # if not default configuratipn:
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


class ResNet50(nn.Sequential):
    def __init__(self, num_classes, replace_stride_with_dilation):
        model_modules = list(
            models.resnet50(
                pretrained=False,
                num_classes=num_classes,
                replace_stride_with_dilation=replace_stride_with_dilation,
            ).named_children()
        )
        model_modules.insert(-1, ("flatten", Flatten()))
        super().__init__(OrderedDict(model_modules))

    def get_params_groups(self):
        feature_extractor_params = []
        for name, module in self.named_children():
            if name != "fc":
                feature_extractor_params.extend(module.parameters())
            else:
                fc_params = module.parameters()
        return [feature_extractor_params, fc_params]


def resnet50(pretrained, num_classes, replace_stride_with_dilation, weights_path):
    model = ResNet50(
        num_classes if num_classes != 2 else 1, replace_stride_with_dilation
    )

    if pretrained:
        if weights_path is not None:
            state_dict = torch.load(weights_path)["model"]
        else:
            state_dict = load_state_dict_from_url(resnet50_url, progress=True)
            if num_classes != 1000:  # if not default configuratipn:
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = resnet50(True, 200, [False, True, True], None).cuda()
    print(model)
