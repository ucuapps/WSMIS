import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

__EPSILON = 1e-6


class Metric:
    def add(self, y_pred, y_true):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def write_to_tensorboard(self, writer, epoch):
        raise NotImplementedError


class MultiLabelAccuracy(Metric):
    NAME = "multilabel_accuracy"
    THRESHOLD = 0.5

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_true = y_true.type(torch.bool)
        self.correct_count += torch.sum(
            (torch.sigmoid(y_pred) > self.THRESHOLD) == y_true, dim=0
        )
        self.total_count += y_pred.size(0)

    def get(self):
        self.accuracies = self.correct_count / self.total_count
        return torch.mean(self.accuracies)

    def reset(self):
        self.correct_count = torch.zeros(self.num_classes).to(self.device)
        self.total_count = torch.tensor(0)

    def write_to_tensorboard(self, writer, epoch):
        fig, ax = plt.subplots(1, 1)
        ax.hist(
            np.arange(1, self.num_classes + 1),
            weights=self.accuracies.cpu().numpy(),
            bins=self.num_classes,
        )
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Accuracy")
        writer.add_figure(self.NAME, fig, epoch)


class BinaryF1(Metric):
    NAME = "f1"
    THRESHOLD = 0.5

    def __init__(self):
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_pred = (torch.sigmoid(y_pred) > self.THRESHOLD).view(y_pred.shape[0], -1)
        y_true = y_true.type(torch.bool).view(y_true.shape[0], -1)
        self.true_positives += torch.sum(y_pred & y_true)
        self.false_positives += torch.sum(y_pred & ~y_true)
        self.false_negatives += torch.sum(~y_pred & y_true)
        self.true_negatives += torch.sum(~y_pred & ~y_true)

    def get(self):
        self.positive_score = (
            2
            * self.true_positives
            / (2 * self.true_positives + self.false_negatives + self.false_positives)
        )
        self.negative_score = (
            2
            * self.true_negatives
            / (2 * self.true_negatives + self.false_negatives + self.false_positives)
        )
        self.precision = self.true_positives / (
            self.true_positives + self.false_positives
        )
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.jaccard = self.true_positives / (
            self.true_positives + self.false_positives + self.false_negatives
        )
        return self.positive_score

    def reset(self):
        self.true_positives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.true_negatives = 0.0

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(f"{self.NAME}-positive", self.positive_score, epoch)
        writer.add_scalar(f"{self.NAME}-negative", self.negative_score, epoch)
        writer.add_scalar(f"precision", self.precision, epoch)
        writer.add_scalar(f"jaccard", self.jaccard, epoch)
        writer.add_scalar(f"recall", self.recall, epoch)


class Accuracy(Metric):
    NAME = "accuracy"
    THRESHOLD = 0.5

    def __init__(self):
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred) > self.THRESHOLD
        y_true = y_true.type(torch.bool)
        self.correct += torch.sum(y_pred == y_true)
        self.total += y_pred.shape[0]

    def get(self):
        self.score = self.correct / self.total
        return self.score

    def reset(self):
        self.correct = 0.0
        self.total = 0.0

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(f"{self.NAME}", self.score, epoch)


class MultiLabelF1(Metric):
    NAME = "multilabel_f1"
    THRESHOLD = 0.5

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred) > self.THRESHOLD
        y_true = y_true.type(torch.bool)

        self.true_positives += torch.sum(y_pred & y_true, dim=0)
        self.false_positives += torch.sum(y_pred & ~y_true, dim=0)
        self.false_negatives += torch.sum(~y_pred & y_true, dim=0)

    def get(self):
        self.scores = (
            2
            * self.true_positives
            / (2 * self.true_positives + self.false_negatives + self.false_positives)
        )
        return torch.mean(self.scores)

    def reset(self):
        self.true_positives = torch.zeros(self.num_classes).to(self.device)
        self.false_positives = torch.zeros(self.num_classes).to(self.device)
        self.false_negatives = torch.zeros(self.num_classes).to(self.device)

    def write_to_tensorboard(self, writer, epoch):
        fig, ax = plt.subplots(1, 1)
        ax.hist(
            np.arange(1, self.num_classes + 1),
            weights=self.scores.cpu().numpy(),
            bins=self.num_classes,
        )
        ax.set_xlabel("Class ID")
        ax.set_ylabel("F1")
        writer.add_figure(self.NAME, fig, epoch)

        writer.add_scalar(f"Mean-{self.NAME}", torch.mean(self.scores), epoch)


class HammingScore(Metric):
    NAME = "hamming_score"
    THRESHOLD = 0.5

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        self.incorrect_count += torch.sum((y_pred > self.THRESHOLD) != y_true)
        self.total_count += y_pred.shape[0] * y_pred.shape[1]

    def get(self):
        self.score = 1 - self.incorrect_count.type(torch.float) / self.total_count
        return self.score

    def reset(self):
        self.incorrect_count = torch.tensor(0)
        self.total_count = torch.tensor(0)

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(self.NAME, self.score, epoch)


class IoUMetric(Metric):
    NAME = "meanIoU"

    def __init__(self, classes, device, ignore_value=255):
        self.classes = classes
        self.device = device
        self.ignore_value = ignore_value
        self.reset()

    def add(self, output, target):
        output = torch.argmax(output, dim=1).view(-1)
        target = target.view(-1)
        valid_idx = target != self.ignore_value
        target[~valid_idx] = 0

        for i, j in itertools.product(torch.unique(target), torch.unique(output)):
            self.conf_matrix[i, j] += torch.sum(
                (target[valid_idx] == i) & (output[valid_idx] == j)
            )

    def get(self):
        conf_matrix = self.conf_matrix.float()
        true_positives = torch.diagonal(conf_matrix)
        false_positives = torch.sum(conf_matrix, 0) - true_positives
        false_negatives = torch.sum(conf_matrix, 1) - true_positives

        iou_per_class = true_positives / (
            true_positives + false_negatives + false_positives
        )
        self.score = torch.mean(iou_per_class).item()
        return self.score

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.classes, self.classes), dtype=torch.int64
        ).to(self.device)

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(self.NAME, self.score, epoch)


def get_metric(metric_name, num_classes, device):
    if metric_name == "f1":
        return BinaryF1()
    if metric_name == "accuracy":
        return Accuracy()
    if metric_name == "hamming":
        return HammingScore(num_classes, device)
    if metric_name == "mlaccuracy":
        return MultiLabelAccuracy(num_classes, device)
    if metric_name == "mlf1":
        return MultiLabelF1(num_classes, device)
    if metric_name == "iou":
        return IoUMetric(num_classes, device)
    else:
        raise ValueError(f"Metric [{metric_name}] not recognized.")
