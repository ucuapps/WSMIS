import sys
import os

sys.path.append("lid_segmentation")
import step.train_irn_pneumothorax as train_irn_pneumo


class args:
    irn_crop_size = 512
    irn_network = "net.resnet50_irn"
    num_workers = os.cpu_count() // 2
    irn_batch_size = 24
    irn_num_epoches = 10
    irn_learning_rate = 0.1
    irn_weight_decay = 1e-4
    irn_weights_name = "sess/res50_irn_pneumothorax.pth"


train_irn_pneumo.run(args)
