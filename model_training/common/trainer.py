import torch
import torch.optim as optim
import tqdm
import os
import yaml
import wandb

from datetime import datetime as datetime
from glog import logger

from .adapters import get_model_adapter
from .average_meter import AverageMeter
from .monitor import TrainingMonitor
from deep_labv3plus_pytorch.utils.scheduler import PolyLR
import inspect
import importlib


class Trainer:
    """
    TODO: checkpoint loading
    """

    def __init__(self, config, train_dl, val_dl):
        # set dataset sizes into config; needed for attention accumulation
        config['train_size'], config['val_size'] = len(train_dl.dataset), len(val_dl.dataset)
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.monitor = TrainingMonitor.from_config(self.config['monitor'])
        if not os.path.exists(os.path.join(config["log_path"], config["task"])):
            os.makedirs(os.path.join(config["log_path"], config["task"]))

    def __module_mapping(self, module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping

    def train(self):
        self._init_params()

        for epoch in range(1, self.epochs + 1):
            self.monitor.update()
            self.model_adapter.set_epoch(epoch)

            train_loss = self._run_epoch(epoch)
            val_loss, metrics, batch_sample = self._validate()
            self.scheduler.step(epoch=epoch)
            # self.scheduler.step(metrics[self.model_adapter.main_metric] if self.model_adapter.main_metric != 'loss'
            #                     else val_loss)
            if self.monitor.should_save_checkpoint():
                self.monitor.reset()
                self._save_checkpoint(file_prefix=f'model_epoch_{epoch}')
            self._set_checkpoint(val_loss)

            logger.info(f"\nEpoch: {epoch}; train loss = {train_loss}; validation loss = {val_loss}")

            self.model_adapter.write_to_tensorboard(epoch, train_loss, val_loss, batch_sample)

        self.model_adapter.on_training_end()

    def _save_checkpoint(self, file_prefix):
        torch.save(
            {
                'model': self.model_adapter.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss': self.config['model']['loss'],
                'scheduler': self.scheduler.state_dict()
            },
            os.path.join(self.log_path, '{}.h5'.format(file_prefix)))

    def _set_checkpoint(self, loss):
        """ Saves model weights in the last checkpoint.
        Also, model is saved as the best model if model has the best loss
        """
        if self.monitor.update_best_model(loss):
            self._save_checkpoint(file_prefix='model_best')

        self._save_checkpoint(file_prefix='model_last')

    def _init_params(self):
        experiment_name = f'{self.config["model"]["arch"]}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        self.log_path = os.path.join(
            self.config['log_path'],
            self.config['task'],
            experiment_name
        )
        os.mkdir(self.log_path)
        with open(os.path.join(self.log_path, 'config.yaml'), 'w') as fp:
            yaml.dump(self.config, fp)
        wandb.init(name=experiment_name, project='lid', entity='ucu_lab', group=self.config['task'],
                   sync_tensorboard=True)

        self.model_adapter = get_model_adapter(self.config, self.log_path)
        self.epochs = self.config["num_epochs"]
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.loss_counter = AverageMeter()

    def _run_epoch(self, epoch):
        self.model_adapter.train()
        self.loss_counter.reset()

        lr = list(map(lambda x: x['lr'], self.optimizer.param_groups))

        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f'Epoch {epoch}, lr {lr}')

        for data in self.train_dl:
            self.model_adapter.zero_grad()
            y_pred = self.model_adapter.forward(data)
            loss = self.model_adapter.get_loss(y_pred, data)
            loss.backward()
            self.optimizer.step()

            self.loss_counter.add_value(loss.item())
            status_bar.update()
            status_bar.set_postfix(loss=loss.item())

        status_bar.close()
        return self.loss_counter.get_value()

    def _validate(self):
        self.model_adapter.eval()
        self.loss_counter.reset()
        status_bar = tqdm.tqdm(total=len(self.val_dl))

        with torch.no_grad():
            for data in self.val_dl:
                y_pred = self.model_adapter.forward(data)

                loss = self.model_adapter.get_loss(y_pred, data)
                self.model_adapter.add_metrics(y_pred, data)

                self.loss_counter.add_value(loss.item())
                status_bar.update()
                status_bar.set_postfix(loss=loss.item())

        status_bar.close()
        prediction_samples = {'data': data, 'y_pred': y_pred}

        return (self.loss_counter.get_value(),
                self.model_adapter.get_metrics(),
                prediction_samples)

    def _get_scheduler(self):
        """ Creates scheduler for a given optimizer from Trainer config

            Returns:
                torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
        """
        scheduler_config = self.config['scheduler']
        if scheduler_config['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                             mode=scheduler_config['mode'],
                                                             patience=scheduler_config['patience'],
                                                             factor=scheduler_config['factor'],
                                                             min_lr=scheduler_config['min_lr'])
        elif scheduler_config['name'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                  step_size=scheduler_config['step_size'],
                                                  gamma=scheduler_config['gamma'])
        elif scheduler_config['name'] == 'poly':
            scheduler = PolyLR(self.optimizer, max_iters=scheduler_config['max_iters'],
                               power=scheduler_config['power'], min_lr=scheduler_config['min_lr'])
        else:
            raise ValueError(f"Scheduler [{scheduler_config['name']}] not recognized.")
        return scheduler

    def _get_optimizer(self):
        """ Creates model optimizer from Trainer config

            Returns:
                torch.optim.optimizer.Optimizer: model optimizer
        """
        optimizer_config = self.config['optimizer']
        lr_list = optimizer_config['parameters']['lr']
        if isinstance(lr_list, list):
            param_groups = self.model_adapter.get_params_groups()
            if not len(param_groups) == len(lr_list):
                raise ValueError(
                    f'Length of lr list ({len(lr_list)}) must match number of parameter groups ({len(param_groups)})')
            param_lr = [{'params': group, 'lr': lr_value} for group, lr_value in zip(param_groups, lr_list)]
        elif isinstance(lr_list, float):
            param_lr = [{'params': self.model_adapter.parameters(), 'lr': lr_list}]
        del optimizer_config['parameters']['lr']

        mapping = self.__module_mapping('torch.optim')
        mapping.update(self.__module_mapping('model_training.common.optimizers'))
        optimizer = mapping[optimizer_config['name']](param_lr, **optimizer_config['parameters'])

        return optimizer