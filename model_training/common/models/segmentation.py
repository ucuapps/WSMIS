import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50
from deep_labv3plus_pytorch.network.modeling import deeplabv3plus_resnet50


class MultiScaleDeeplabV3Resnet50(nn.Module):
    def __init__(self, num_classes, scales=(0.5, 0.75), pretrained=False, pretrained_backbone=True):
        super(MultiScaleDeeplabV3Resnet50, self).__init__()
        self.model = deeplabv3_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone
        )
        self.num_classes = num_classes
        self.scales = scales

    def forward(self, x):
        batch_size, _, *shape = x.shape
        out = torch.empty(
            size=(1 + len(self.scales), batch_size, self.num_classes, *shape),
            device=x.device)

        out[0] = self.model(x)['out']
        for i, scale in enumerate(self.scales, start=1):
            x_scaled = F.interpolate(x, size=(int(shape[0] * scale), int(shape[1] * scale)), mode='bilinear',
                                     align_corners=False)
            out_scaled = self.model(x_scaled)['out']
            out[i] = F.interpolate(out_scaled, size=shape, mode='bilinear', align_corners=False)

        return out.max(dim=0).values


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.model = deeplabv3plus_resnet50(num_classes=num_classes if num_classes != 2 else 1,
                                            pretrained_backbone=True)

    def forward(self, x):
        return {'out': self.model(x)}

    def get_params_groups(self):
        return (list(self.model.backbone.parameters()),
                list(self.model.classifier.parameters()))


class UNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super(UNet, self).__init__()
        self.model = smp.Unet(encoder, classes=num_classes if num_classes != 2 else 1,
                              encoder_weights='imagenet')

    def forward(self, x):
        return {'out': self.model(x)}

    def get_params_groups(self):
        return (
            list(self.model.encoder.parameters()),
            list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())
        )
