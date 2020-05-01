import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
from model_training.common.augmentations import get_transforms

import importlib


def run(args):
    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(
        path_index)

    transform_config = {
        'augmentation_scope': 'horizontal_flip',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
        'masks_normalization': 'none',
        'masks_output_format_type': 'byte',
        'size': 512,
        'size_transform': 'resize'
    }
    transform = get_transforms(transform_config)

    train_dataset = voc12.dataloader.PneumothoraxAffinityDataset(
        '/datasets/LID/Pneumothorax/train/train_all_positive.csv',
        transform=transform,
        indices_from=path_index.src_indices,
        indices_to=path_index.dst_indices,
    )

    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1 * args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model.cuda(1), device_ids=['cuda:1', 'cuda:2'])
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img']
            bg_pos_label = pack['aff_bg_pos_label'].cuda(1, non_blocking=True)
            fg_pos_label = pack['aff_fg_pos_label'].cuda(1, non_blocking=True)
            neg_label = pack['aff_neg_label'].cuda(1, non_blocking=True)

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)

            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (
                          avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'),
                          avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()

    transform_config = {
        'augmentation_scope': 'none',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
        'size': 512,
        'size_transform': 'resize'
    }
    transform = get_transforms(transform_config)

    infer_dataset = voc12.dataloader.PneumothoraxImageDataset(
        '/datasets/LID/Pneumothorax/train/train_all_positive.csv',
        transform=transform
    )
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img']

            aff, dp = model(img, False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.module.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    print('done.')

    torch.save(model.module.state_dict(), args.irn_weights_name)
    torch.cuda.empty_cache()
