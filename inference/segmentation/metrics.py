import numpy as np
import itertools

from model_training.common.metrics import Metric


class IoUNumpy(Metric):
    def __init__(self, classes=2):
        self.classes = classes
        self.reset()

    def add(self, output, target):
        output, target = output.reshape(-1), target.reshape(-1)
        valid_idx = target != 255
        target[~valid_idx] = 0
        for i, j in itertools.product(np.unique(target), np.unique(output)):
            self.conf_matrix[i, j] += np.sum(
                (target[valid_idx] == i) & (output[valid_idx] == j)
            )

    def get(self):
        conf_matrix = self.conf_matrix.astype(np.float64)
        true_positives = np.diagonal(conf_matrix)
        false_positives = np.sum(conf_matrix, 0) - true_positives
        false_negatives = np.sum(conf_matrix, 1) - true_positives
        metric_per_class = true_positives / (
            true_positives + false_negatives + false_positives
        )
        return np.mean(metric_per_class), metric_per_class

    def reset(self):
        self.conf_matrix = np.zeros((self.classes, self.classes), dtype=np.int64)

    def write_to_tensorboard(self, writer, epoch):
        pass
