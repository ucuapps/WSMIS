from .cam_generation import ResNetGAP, resnet50, resnet34
from .oaa.vgg import vgg16
from .segmentation import DeepLabV3Plus, UNet
from torchvision.models.segmentation import deeplabv3_resnet50
from .oaa.resnet import resnet50_oaa, resnet50_full
from .dropblock.resnets import resnet50_dropblock, resnet50_dropblock_scheduler


def get_network(model_config):
    """
    Create model form configuration
    Args:
        model_config (dict): dictionary of model config
    Return:
        model (torch.nn.Module): model created from config
    """
    arch = model_config['arch']

    if arch == 'deeplabv3_resnet50':
        return deeplabv3_resnet50(pretrained=model_config['pretrained'], num_classes=model_config['classes'])
    elif arch == 'deeplabv3plus_resnet50':
        return DeepLabV3Plus(num_classes=model_config['classes'])
    elif arch == 'resnet50':
        return resnet50(pretrained=model_config['pretrained'], num_classes=model_config['classes'],
                        replace_stride_with_dilation=[False, True, True],
                        weights_path=model_config.get('weights_path', None))
    elif arch == 'resnet34':
        return resnet34(pretrained=model_config['pretrained'], num_classes=model_config['classes'],
                        weights_path=model_config.get('weights_path', None))
    elif arch == 'resnet_gap':
        return ResNetGAP(n_classes=model_config['classes'])
    elif arch == 'vgg_gap':
        return vgg16(pretrained=model_config['pretrained'], num_classes=model_config['classes'])
    elif arch == 'resnet_oaa':
        return resnet50_oaa(pretrained=model_config['pretrained'], num_classes=model_config['classes'],
                            weights_path=model_config.get('weights_path', None))
    elif arch == 'resnet50_oaa':
        return resnet50_oaa(pretrained=model_config['pretrained'], num_classes=model_config['classes'],
                            weights_path=model_config.get('weights_path', None))
    elif arch == 'resnet50_full':
        return resnet50_full(pretrained=model_config['pretrained'], num_classes=model_config['classes'],
                             weights_path=model_config.get('weights_path', None))
    elif arch == 'resnet50_dropblock':
        return resnet50_dropblock(num_classes=model_config['classes'], pretrained=model_config['pretrained'],
                                  drop_prob=model_config['regularize']['parameters']['drop_prob'],
                                  block_size=model_config['regularize']['parameters']['block_size'])
    elif arch == 'resnet50_dropblock_scheduler':
        return resnet50_dropblock_scheduler(num_classes=model_config['classes'], pretrained=model_config['pretrained'],
                                            drop_prob=model_config['regularize']['parameters']['drop_prob'],
                                            block_size=model_config['regularize']['parameters']['block_size'])
    elif arch == 'unet':
        return UNet(encoder=model_config['encoder'], num_classes=model_config['classes'])
    else:
        raise ValueError(f'Model architecture [{arch}] not recognized.')
