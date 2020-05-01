import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def add_value(self, value):
        self.values.append(value)

    def get_value(self):
        return np.mean(self.values)

    def reset(self):
        self.values = []

    def update_best_model(self):
        """Return flag whether to save the model based on the main metric"""
        current_metric = self.get_main_metric()
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            return True
        return False
