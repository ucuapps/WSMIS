import torch
import tqdm
import os
import torch.nn.functional as F
import cv2
import numpy as np

from model_training.common.models import get_network
from model_training.common.augmentations import denormalization


def linidx_take(val_arr, z_indices):
    # Get number of columns and rows in values array
    _, nC, nR = val_arr.shape

    # Get linear indices and thus extract elements with np.take
    idx = nC * nR * z_indices + nR * np.arange(nR)[:, None] + np.arange(nC)
    return np.take(val_arr, idx)  # Or val_arr.ravel()[idx]


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

        self.maps_weights = getattr(
            self.model, config["weights_layer"]
        ).weight  # shape: [num_classes, K] K - number of kernel filters
        self.interpolation_mode = config["interpolation"]
        self.use_predicted_labels = config["use_predicted_labels"]
        self.denorm = denormalization["default"]

        self.maps = None
        getattr(self.model, config["maps_layer"]).register_forward_hook(
            self.save_maps_forward
        )

    def save_maps_forward(self, module, input_tensor, output_tensor):
        self.maps = output_tensor.detach()

    def extract(self):
        self._run(self.train_dl, prefix="train")
        self._run(self.val_dl, prefix="val")

    @torch.no_grad()
    def _run(self, dl, prefix="train"):
        if not os.path.exists(self.config[prefix]["output_path"]):
            os.makedirs(self.config[prefix]["output_path"])

        for X, y, image_names, shapes in tqdm.tqdm(dl):
            X, y = X.to(self.device), y.to(self.device)
            height, width = X.shape[2:]
            y_pred = self.model(X)

            if self.use_predicted_labels:
                _, y_pred_max_idx = y_pred.max(dim=1)
                y_pred = (y_pred > 0).type(torch.int64)
                no_class_idx = torch.where(torch.sum(y_pred, dim=1) == 0)[0]
                y_pred[no_class_idx, y_pred_max_idx[no_class_idx]] = 1
                y = y_pred

            X = self.denorm(X).permute(0, 2, 3, 1)
            for image, name, activation_maps, classes, shape, in zip(
                X, image_names, self.maps, y, shapes
            ):
                self._process_single_image(
                    image,
                    activation_maps,
                    classes,
                    name,
                    (height, width),
                    shape,
                    prefix=prefix,
                )

    def _process_single_image(
        self, image, activation_maps, classes, name, size, original_shape, prefix
    ):
        """
        Save segmentation map for single image
        Args:
            activation_maps (torch.tensor): activation of last conv layer [K, H_s, W_s]
            classes (torch.tensor): ground-truth labels for image [num_classes]
            name (str): name of segmap to save
            size (tuple(int, int)): size of segmap to save
            prefix (str): flag whether to save into train or val set
        """
        image = (image * 255).type(torch.uint8).cpu().numpy()

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

        seg_maps = seg_maps.cpu().numpy()
        final_maps = np.zeros_like(seg_maps, dtype=np.uint8)
        for i, (seg_map, label), in enumerate(zip(seg_maps, class_labels)):
            final_maps[i] = self.grub_cut_mask(image, seg_map, label)

        max_idx = np.argmax(seg_maps, axis=0)
        seg_maps_max = linidx_take(final_maps, max_idx)
        seg_maps_max = cv2.resize(
            seg_maps_max,
            (original_shape[1].item(), original_shape[0].item()),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(save_path, seg_maps_max)

    @staticmethod
    def grub_cut_mask(image, seg_map, label):
        x1, x2, x3 = np.percentile(seg_map, [15, 70, 99.5])
        new_mask = np.zeros(seg_map.shape, dtype=np.uint8)

        new_mask[seg_map > x3] = cv2.GC_FGD
        new_mask[seg_map <= x3] = cv2.GC_PR_FGD
        new_mask[seg_map <= x2] = cv2.GC_PR_BGD
        new_mask[seg_map <= x1] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        mask, _, _ = cv2.grabCut(
            image, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
        )
        return np.where(
            (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, label.item()
        ).astype("uint8")
