import torch
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from ..models import get_network
from ..metrics import get_metric
from ..losses import get_loss
from ..augmentations import denormalization

__all__ = ["ModelAdapter"]


class ModelAdapter:
    def __init__(self, config, log_path):
        self.device = config["devices"][0]

        self.log_path = log_path
        self.get_loss_function(config["model"]["loss"])
        metrics_names = config["model"]["metrics"]
        self.metrics = OrderedDict(
            [
                (
                    metric_name,
                    get_metric(metric_name, config["model"]["classes"], self.device),
                )
                for metric_name in metrics_names
            ]
        )
        self.main_metric = metrics_names[0] if len(metrics_names) > 0 else "loss"

        self.denormalize = denormalization[
            config["val"]["transform"]["images_normalization"]
        ]
        self.epoch = 0
        self.mode = "train"
        self.writer = SummaryWriter(self.log_path)

        self.model = get_network(config["model"])
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=config["devices"])

    def get_loss_function(self, losses_config):
        if isinstance(losses_config, dict):
            self.criterion = {
                losses_config["name"]: (get_loss(losses_config).to(self.device), 1.0)
            }
        elif isinstance(losses_config, list):
            self.criterion = {
                x["name"]: (get_loss(x).to(self.device), x.get("weight", 1.0))
                for x in losses_config
            }

    def set_epoch(self, epoch):
        assert epoch > 0
        self.epoch = epoch
        return self

    def forward(self, data):
        X = data[0]
        return self.model(X)

    def add_metrics(self, y_pred, data):
        """Calculate metrics on given models output and targets"""
        y_true = data[1].to(self.device)
        for metric in self.metrics.values():
            metric.add(y_pred, y_true)

    def get_metrics(self):
        rv = OrderedDict(
            [
                (metric_name, metric.get())
                for metric_name, metric in self.metrics.items()
            ]
        )

        for metric in self.metrics.values():
            metric.reset()

        return rv

    def get_loss(self, y_pred, data):
        """Calculate loss given models output and targets"""
        y_true = data[1].to(self.device)
        loss = 0
        for criterion, weight in self.criterion.values():
            loss += weight * criterion(y_pred, y_true)
        return loss

    def make_tensorboard_grid(self, batch_sample):
        """Make grid of model inputs and outputs"""
        raise NotImplementedError()

    def write_to_tensorboard(self, epoch, train_loss, val_loss, batch_sample):
        # write train and val losses
        for scalar_prefix, loss in zip(("Train", "Validation"), (train_loss, val_loss)):
            self.writer.add_scalar(f"{scalar_prefix}_Loss", loss, epoch)

        for metric in self.metrics.values():
            metric.write_to_tensorboard(self.writer, epoch)

        images_grid = self.make_tensorboard_grid(batch_sample)
        if images_grid is not None:
            self.writer.add_image("Images", images_grid, epoch)

    def zero_grad(self):
        self.model.module.zero_grad()

    def train(self):
        self.mode = "train"
        return self.model.module.train()

    def eval(self):
        self.mode = "val"
        return self.model.module.eval()

    def get_params_groups(self):
        return self.model.module.get_params_groups()

    def parameters(self):
        return self.model.module.parameters()

    def state_dict(self):
        return self.model.module.state_dict()

    def on_training_end(self):
        pass
