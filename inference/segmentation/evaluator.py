import torch
import torch.nn.functional as F
import tqdm
import os

from model_training.common.models import get_network
from model_training.common.metrics import get_metric


class Evaluator:
    """Evaluate the model based on given metrics"""

    def __init__(self, config, train_dl, val_dl):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = config['devices'][0]

        model_path = os.path.join(config['experiment_path'], config['model']['weights_path'])
        self.model = get_network(config['model'])
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.metrics = {metric_name: get_metric(metric_name, config['model']['classes'], self.device)
                        for metric_name in config['model']['metrics']}

    def evaluate(self):
        # return {
        return {'train': self.__run(self.train_dl),
                'val': self.__run(self.val_dl)}

    @torch.no_grad()
    def __run(self, dl):
        for metric_fn in self.metrics.values():
            metric_fn.reset()

        for X, _, names, y_orig in tqdm.tqdm(dl):
            X, y_orig = X.to(self.device), y_orig.to(self.device)
            y_pred = self.model(X)['out']
            y_pred = F.interpolate(y_pred, y_orig.size()[1:], mode='nearest')

            for metric_fn in self.metrics.values():
                metric_fn.add(y_pred, y_orig)

        return {metric_name: metric_fn.get() for metric_name, metric_fn in self.metrics.items()}
