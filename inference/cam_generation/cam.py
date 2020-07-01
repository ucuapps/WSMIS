import torch
import tqdm
import os
import torch.nn.functional as F
import cv2

from model_training.common.models import get_network


class ActivationExtractor:
    """Extract activation maps from pretrained model"""

    def __init__(self, config, train_dl, val_dl, device):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device

        self.model = get_network(config["model"])
        state_dict = torch.load(config["model"]["weights_path"])
        self.model.load_state_dict(state_dict["model"])
        self.model = self.model.to(device)
        self.model.eval()

        self.maps_weights = (
            self.model.linear.weight
        )  # shape: [num_classes, K] K - number of kernel filters
        self.interpolation_mode = config["interpolation"]

    def extract(self):
        self._run(self.train_dl, prefix="train")
        self._run(self.val_dl, prefix="val")

    @torch.no_grad()
    def _run(self, dl, prefix="train"):
        for X, y, image_names in tqdm.tqdm(dl):
            X, y = X.to(self.device), y.to(self.device)
            height, width = X.shape[2:]
            y_pred, maps = self.model(X, return_maps=True)
            if not os.path.exists(self.config[prefix]["output_path"]):
                os.makedirs(self.config[prefix]["output_path"])
            for name, activation_maps, classes in zip(image_names, maps, y):
                self._process_single_image(
                    activation_maps, classes, name, (height, width), prefix=prefix
                )

    def _process_single_image(self, activation_maps, classes, name, size, prefix):
        """
        Save segmentation map for single image
        Args:
            activation_maps (torch.tensor): activation of last conv layer [K, H_s, W_s]
            classes (torch.tensor): ground-truth labels for image [num_classes]
            name (str): name of segmap to save
            size (tuple(int, int)): size of segmap to save
            prefix (str): flag whether to save into train or val set
        """

        maps_weights = self.maps_weights[classes == 1]  # shape: [true_classes, K]
        seg_maps = torch.tensordot(
            maps_weights, activation_maps, dims=((1,), (0,))
        )  # shape: [true_classes, H_s, W_s]

        save_path = os.path.join(self.config[prefix]["output_path"], name + ".png")
        seg_maps = F.interpolate(
            seg_maps[None], size, mode=self.interpolation_mode, align_corners=False
        )[0]

        class_labels = (
            torch.where(classes == 1)[0].type(torch.float32) + 1
        )  # e.g. [2, 7, 20]
        seg_maps_max, seg_maps_indices = seg_maps.max(dim=0)
        threshold = (seg_maps_max.max() - seg_maps_max.min()) * self.config[
            "background_threshold"
        ] + seg_maps_max.min()

        idx = seg_maps_max > threshold
        seg_maps_max[~idx] = 0
        seg_maps_max[idx] = class_labels[seg_maps_indices][idx]

        cv2.imwrite(save_path, seg_maps_max.type(torch.uint8).cpu().numpy())
