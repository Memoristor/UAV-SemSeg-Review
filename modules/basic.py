# coding=utf-8

from glob import glob
from torch import nn
import torch
import os

__all__ = ['BasicModule']


class BasicModule(object):
    """
    The basic module for training or testing.

    Params:
        model: nn.Module. The ConvNet model for training or testing.
        args: parser.parse_args. The other custom arguments.
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args

        # Module prepare steps
        self.prepare_dir()
        self.prepare_device()
        self.prepare_optimizer()
        self.prepare_model()

        # Init stat epoch if needs retrain
        if self.args.retrain is True:
            self.start_epoch = 0

    def prepare_dir(self):
        """Prepare and make needed directories"""
        self.model_name = self.model.__class__.__name__

        self.output_root = os.path.join(self.args.output_root, self.args.dataset, self.model_name)
        self.output_dir = f'lr_{self.args.init_lr}_wd_{self.args.weight_decay}_bs_{self.args.batch_size}_sd_{self.args.seed}'
        
        self.save_weights = os.path.join(self.output_root, 'weights', self.output_dir)
        self.save_logs = os.path.join(self.output_root, 'logs', self.output_dir)
        self.save_records = os.path.join(self.output_root, 'records', self.output_dir)
        
        os.makedirs(self.save_weights, exist_ok=True)
        os.makedirs(self.save_logs, exist_ok=True)
        os.makedirs(self.save_records, exist_ok=True)

    def prepare_device(self):
        """Prepare and make needed device"""
        # Get `device` and `device ids`
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = 'cuda'  # default 'cuda:0'
                self.device_ids = [i for i in range(torch.cuda.device_count())]
            else:
                self.device = 'cuda'  # default 'cuda:0'
                self.device_ids = [0]
        else:
            self.device = 'cpu'  # default 'cpu'
            self.device_ids = None

    def prepare_optimizer(self):
        """Prepare and make needed optimizer"""
        # Init learning rate
        self.learning_rate = self.args.init_lr

        # Chose optimizer for training
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = None
            raise AttributeError(f'Expected optimizer: SGD, Adam, AdamW, but input type: {self.args.optimizer}')

    def prepare_model(self):
        """Prepare model by `self.device` and `self.device_ids`"""
        # Init model and reload model if saved weight is available
        self.start_epoch = self.load_model(os.path.join(self.save_weights, self.args.resume)) + 1

        # Set model to proper GPU
        if self.device_ids is not None:
            if len(self.device_ids) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            self.model.to(torch.device(self.device))

    def save_model(self, path, epoch):
        """
        Save model, note that epoch, model_state, optimizer state will be saved.

        Params:
            path: str. The path for saving model.
            epoch: int. The number epoch the model has been trained.
        """
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """
        Load model, note that epoch, model_state, optimizer state will be loaded.

        Params:
            path: str. The file path of saved model parameters.

        Return:
            return the epoch of saved model.
        """
        if not os.path.exists(path):
            print(f'[{self.model_name}] Can not reload parameters from `{path}`, file not exist')
            return -1

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        epoch = checkpoint['epoch']
        print(f'[{self.model_name}] Reloaded parameters from `{path}`, epoch {epoch + 1}')

        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]

        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(torch.device(self.device))
        return epoch

    def init_model(self, ignore_key=None):
        """
        Weight initialization of ${self.model}

        Params:
            ignore_key: list. If `ignore_key` appears in the named parameters, the parameter
                will not be not initialized
        """
        stat_dict = {}
        if ignore_key is not None:
            print(f'[{self.__class__.__name__}] Parameters which name contains {ignore_key} will be ignored when init model')
            # Pick up parameters which should be ignored
            for k, v in self.model.named_parameters():
                for i in ignore_key:
                    if i in k:
                        stat_dict[k] = v
            print(f'[{self.__class__.__name__}] {len(list(stat_dict.keys()))} parameters has been ignored when init model')

        # Init all parameters
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # Reload ignored parameters
        self.model.load_state_dict(stat_dict, strict=False)
