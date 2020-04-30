"""Evaluate Image-Net from given prediction masks"""

import os
import cv2
import tqdm
import json
import yaml

import tabulate
from inference.segmentation.metrics import IoUNumpy

experiment_path = '/home/ostap/logs/camgrad_edge_inference/resnet50-2020-03-03-21-48-31/'
gt_path = '/datasets/LID/LID_track1/val_gt'


with open(os.path.join(experiment_path, 'config.yaml')) as config_file:
    config = yaml.full_load(config_file)


out_path = config['data']['out_path']
images_list_path = config['data']['path']
index_to_class_path = config['data']['index_to_class_path']

with open(index_to_class_path) as fp:
    index_to_class = {int(k): v for k, v in json.load(fp).items()}
    index_to_class[0] = 'background'


metric = IoUNumpy(classes=len(index_to_class))

with open(images_list_path) as fp:
    for name in tqdm.tqdm(fp):
        name = name.strip() + '.png'
        output = cv2.imread(os.path.join(out_path, name), 0)
        target = cv2.imread(os.path.join(gt_path, name), 0)
        metric.add(output, target)

with open(os.path.join(experiment_path, 'results.txt'), 'w') as fp:
    metric_value, metric_per_class = metric.get()
    fp.write(f'IoU metric: {metric_value}\n')

    metric_per_class = {index_to_class[i]: metric_per_class_value for i, metric_per_class_value in
                        enumerate(metric_per_class)}
    fp.write(tabulate.tabulate(metric_per_class.items(), headers=('Name', "Score")) + '\n')
