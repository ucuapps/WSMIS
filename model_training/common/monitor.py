class TrainingMonitor:
    available_methods = ["epochs"]

    def __init__(self, monitor_method, interval):
        assert monitor_method in self.available_methods

        self.method = monitor_method
        self.counter = 0
        self.interval = interval
        self.loss = None

    def reset(self):
        if self.method == "epochs":
            self.counter = 0

    def update(self, add_value=1):
        if self.method == "epochs":
            self.counter += add_value

    def update_best_model(self, loss) -> bool:
        if self.loss is None or loss < self.loss:
            self.loss = loss
            return True
        return False

    def should_save_checkpoint(self) -> bool:
        if self.method == "epochs":
            return self.counter >= self.interval

    @classmethod
    def from_config(cls, monitor_config):
        return cls(monitor_method=monitor_config['method'], interval=monitor_config['interval'])
