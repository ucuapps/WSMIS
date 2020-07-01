import os
import cv2
import tqdm
import json
import yaml

import tabulate
from inference.segmentation.metrics import IoUNumpy

gt_path = "/datasets/LID/VOCdevkit/VOC2012/SegmentationClass"
class_to_index = "/datasets/LID/PascalVOC2012SegmentationAug/class_to_index.json"
images_list_path = "/datasets/LID/PascalVOC2012SegmentationAug/val.json"
pred_path = "/datasets/LID/PascalVOC2012SegmentationAug/out_masks/cam_grad_pp"

with open(class_to_index) as fp:
    class_to_index = json.load(fp)
    index_to_class = {v: k for k, v in class_to_index.items()}

metric = IoUNumpy(classes=len(class_to_index))

with open(images_list_path) as fp:
    images_list = json.load(fp)
if isinstance(images_list, dict):
    images_list = images_list["train"] + images_list["trainaug"]

for name in tqdm.tqdm(images_list):
    name = name["image"] + ".png"
    output = cv2.imread(os.path.join(pred_path, name), 0)
    target = cv2.imread(os.path.join(gt_path, name), 0)
    metric.add(output, target)

metric_value, metric_per_class = metric.get()
print(f"IoU metric: {metric_value}\n")
metric_per_class = {
    index_to_class[i]: metric_per_class_value
    for i, metric_per_class_value in enumerate(metric_per_class)
}
print(tabulate.tabulate(metric_per_class.items(), headers=("Name", "Score")) + "\n")
