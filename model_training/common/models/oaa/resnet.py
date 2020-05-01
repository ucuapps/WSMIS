import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet50, self).__init__()
        conv_modules = list(
            models.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True]).children())[:8]

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
            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


class OAAResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(OAAResNet50, self).__init__()
        conv_modules = list(
            models.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True]).children())[:8]

        self.backbone = nn.Sequential(*conv_modules)
        self.extra_convs = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
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
            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def resnet50_oaa(num_classes, pretrained=True, weights_path=None):
    return OAAResNet50(num_classes=num_classes, pretrained=pretrained)


def resnet50_full(num_classes, pretrained=True, weights_path=None):
    return ResNet50(num_classes=num_classes, pretrained=pretrained)


# def resnet50_oaa(num_classes, pretrained=True, weights_path=None):
#     model = OAAResNet50(num_classes=num_classes)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#     return model

# for child in model.children():
#     for param in child.parameters():
#         param.requires_grad = True


if __name__ == '__main__':
    from torchsummary import summary

    model = resnet50_oaa(num_classes=20)
    summary(model, input_size=(3, 352, 352), batch_size=10, device='cpu')
