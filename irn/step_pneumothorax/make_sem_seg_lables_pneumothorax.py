import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import tqdm

import voc12.dataloader
from misc import torchutils, indexing
from model_training.common.augmentations import get_transforms

cudnn.enabled = True


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = 2

    transform_config = {
        'size': 1024,
        'augmentation_scope': 'none',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
        'size_transform': 'resize'
    }
    transform = get_transforms(transform_config)

    dataset = voc12.dataloader.PneumothoraxMSDataset('/datasets/LID/Pneumothorax/train/val.csv',
                                                     transform=transform, scales=(1.0, 0.75, 0.5, 0.25))

    data_loader = DataLoader(dataset,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(1):
        model.cuda()

        for iter, pack in tqdm.tqdm(enumerate(data_loader)):
            img_name = pack['name'][0]
            if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.npy')):
                continue
            orig_img_size = np.array([512, 512])
            x = torch.cat([pack['img'][2], pack['img'][2].flip(-1)], dim=0)

            edge, dp = model(x.cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times,
                                            radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0],
                    :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            np.save(os.path.join(args.sem_seg_out_dir, img_name + '.npy'), rw_up.cpu().numpy())

    torch.cuda.empty_cache()
