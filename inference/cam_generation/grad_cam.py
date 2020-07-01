"""CAM_GRAD/CAM-GRAD++ implementations"""

import torch
import torch.nn.functional as F

from model_training.common.models import get_network


class CamGrad:
    def __init__(self, config, device, return_one_always=True):
        self.model = get_network(config["model"])
        self.model.load_state_dict(
            torch.load(config["model"]["weights_path"], map_location=device)["model"]
        )
        self.config = config
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.prediction_threshold = config.get("prediction_threshold", None)
        self.set_target_layer_hook(config["model"]["target_layer"])
        self.return_one_always = return_one_always

        self.activation_maps = None
        self.grad = None

        self.func_on_target = lambda x: x

    def set_target_layer_hook(self, target_layer):
        modules_path = target_layer.split(".")
        module = self.model
        for subpath in modules_path:
            for name, current_module in module.named_children():
                if name == subpath:
                    module = current_module
                    break
            else:
                raise ValueError(
                    f"Module path {target_layer} is not valid for current module."
                )

        module.register_forward_hook(self.save_output)
        module.register_backward_hook(self.save_grad)

    def save_output(self, module, input_tensor, output_tensor):
        """Forward hook that saves output of target layer"""
        self.activation_maps = output_tensor.squeeze().detach()

    def save_grad(self, module, input_grad, output_grad):
        """Backward hook that saves gradients of output of target layer"""
        self.grad = output_grad[0].squeeze().detach()

    def forward(self, x, y=None):
        """
        Args:
            x: input tensor of size [1, C, H, W]
            y: target of shape [1, N] or None
        Returns:
            class_maps: list of maps corresponding to different classes
            y: if input was None returns predicted classes else y from input
        """
        assert x.size(0) == 1

        logits_pred = self.model(x).squeeze(0)
        if y is None:
            y = torch.where(torch.sigmoid(logits_pred) > self.prediction_threshold)[0]
            if y.shape[0] == 0 and self.return_one_always:
                y = torch.argmax(logits_pred, dim=0, keepdim=True)
        else:
            y = torch.where(y)[1]

        class_maps = []
        for label in y:
            self.model.zero_grad()
            self.func_on_target(logits_pred[label]).backward()

            weights = self.get_maps_weights()  # [K, ]

            class_map = torch.tensordot(
                weights, self.activation_maps, dims=((0,), (0,))
            )
            # class_map = F.interpolate(class_map[None, None], x.shape[2:], mode='nearest').squeeze()
            class_maps.append(class_map.detach())

        return torch.stack(class_maps), y

    @torch.no_grad()
    def get_maps_weights(self):
        return torch.mean(self.grad, dim=(1, 2))


class CamGradPlusPlus(CamGrad):
    def __init__(self, config, device, return_one_always=True):
        super().__init__(config, device, return_one_always=return_one_always)
        self.func_on_target = lambda x: torch.exp(x)

    @torch.no_grad()
    def get_maps_weights(self, gamma=1e-8):
        squared_grad = self.grad ** 2
        cubed_grad = squared_grad * self.grad

        alpha = squared_grad / (
            2 * squared_grad
            + torch.sum(cubed_grad * self.activation_maps, dim=(1, 2), keepdim=True)
            + gamma
        )
        return torch.sum(alpha * torch.relu(self.grad), dim=(1, 2))


def get_cam_grad_extractor(config, device):
    name = config["extraction_method"]
    if name == "grad-cam":
        return CamGrad(config, device)
    elif name == "grad-cam++":
        return CamGradPlusPlus(config, device)
    else:
        raise ValueError(f"Unrecognized mask generation method {name}.")


if __name__ == "__main__":
    model_config = {
        "arch": "resnet50",
        "pretrained": False,
        "classes": 201,
    }
    config = {
        "model": model_config,
        "target_layer": "layer4.1.conv3",
        "prediction_threshold": 0.5,
    }

    model = get_network(model_config).cuda()
    for name, module in model.named_children():
        print("Direct child name: ", name)
        for n, m in module.named_children():
            print("\t", n)
    print(list(model.named_modules()))

