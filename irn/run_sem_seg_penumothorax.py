import sys
import os

sys.path.append("lid_segmentation")
import step.make_sem_seg_lables_pneumothorax as make_sem_seg_lables_pneumothorax


class args:
    irn_network = "net.resnet50_irn"
    num_workers = os.cpu_count() // 2
    irn_weights_name = "sess/res50_irn_pneumothorax.pth"

    cam_out_dir = "/datasets/LID/Pneumothorax/train/out_masks/cam"
    beta = 10
    exp_times = 8
    sem_seg_bg_thres = 0.25
    sem_seg_out_dir = "/datasets/LID/Pneumothorax/train/out_masks/sem_seg"


make_sem_seg_lables_pneumothorax.run(args)
