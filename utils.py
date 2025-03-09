import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Union, List
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from timm.data.transforms import str_to_interp_mode
import importlib
import ast
from submodules.probe3d.evals.utils.metrics import evaluate_depth

def parse_unknown_args(unknown_args):
    """
    Parse unknown arguments into a nested dictionary.
    Assumes unknown arguments are of the form '--key.subkey=value'.
    """
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    unknown_args_dict = {}
    for arg in unknown_args:
        # Split argument into nested keys and value based on '=' delimiter
        keys, value = arg.split('=', 1)
        keys = keys.lstrip('-').split('.')  # Split nested keys
        current_level = unknown_args_dict
        # Traverse the nested dictionary and create missing levels if necessary
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        # Convert numerical values to integers or floats where appropriate
        if value.isdigit():
            value = int(value)
        elif is_float(value):
            value = float(value)
        # detect boolean values
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif '[' in value.lower():
            value = ast.literal_eval(value)
        elif not value or value == 'None':
            value = None
        # Assign the value to the innermost key
        current_level[keys[-1]] = value
    return unknown_args_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_transform_wo_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    interpolation = data_config['interpolation']
    tf = []
    #if input_size[0] == input_size[1]:
    tf += [transforms.Resize(input_size, interpolation=str_to_interp_mode(interpolation))]
    #else:
    #    tf += [ResizeKeepRatio(input_size)]
    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)


def binary_accuracy(output, target):
    with torch.no_grad():
        output = output > 0
        return torch.sum(output==target).item()/len(target)*100
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)

    with torch.no_grad():
        if len(output.shape) == 1:
            # binary case
            output = output > 0
            return torch.sum(output==target)/batch_size*100
        else:
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res[0]


def euclidean_distance_with_logits(output, target, target_range=[-1.5,1.5]):
    with torch.no_grad():
        # scale output range to target range
        output = torch.sigmoid(output)
        output = output * (target_range[1] - target_range[0]) + target_range[0]
        mean_distance = torch.sqrt(torch.sum((output - target) ** 2, dim=1)).mean()
    return mean_distance


class Accuracy:
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.name = 'acc'
        self.is_higher_better = True
    def __call__(self, output, target):
        return accuracy(output, target, self.topk)
    def get_preds(self, output):
        if len(output.shape) == 1:
            return torch.sigmoid(output)
        else:
            return torch.argmax(torch.softmax(output, dim=1), dim=1) # (B,)


class EuclideanDistanceWithLogits:
    def __init__(self, target_range=[-1.5,1.5]):
        self.target_range = target_range
        self.name = 'eucdist'
        self.is_higher_better = False
    def __call__(self, output, target):
        return euclidean_distance_with_logits(output, target, self.target_range)
    def get_preds(self, output):
        output = torch.sigmoid(output)
        output = output * (self.target_range[1] - self.target_range[0]) + self.target_range[0]
        return output


class HorizonErrorWithLogits:
    def __init__(self, target_scales=[1, 0.625]):
        self.target_scales = target_scales
        self.name = 'horizon_error'
        self.is_higher_better = False
    def __call__(self, output, target):
        output = torch.tanh(output)
        output = output * torch.tensor(self.target_scales).to(output.device).unsqueeze(0)
        yl_pred = output[:,0]
        yr_pred = output[:,0] + output[:,1]
        yl_true = target[:,0]
        yr_true = target[:,0] + target[:,1]
        errorl = torch.abs(yl_pred - yl_true) 
        errorr = torch.abs(yr_pred - yr_true)
        error = torch.amax(torch.stack([errorl, errorr], dim=1), dim=1) # (B,)
        return error.mean()
    def get_preds(self, output):
        output = torch.tanh(output)
        output = output * torch.tensor(self.target_scales).to(output.device).unsqueeze(0)
        return output


class MSEWithLogitsLoss(torch.nn.Module):
    def __init__(self, target_range=[-1.5,1.5]):
        super().__init__()
        self.target_range = target_range
        self.loss = torch.nn.MSELoss()

    def forward(self, preds, labels):
        # scale output range to target range
        preds = torch.sigmoid(preds)
        preds = preds * (self.target_range[1] - self.target_range[0]) + self.target_range[0]
        return self.loss(preds, labels)


class MultiMSEWithLogitsLoss(torch.nn.Module):
    def __init__(self, target_scales=[1, 0.625]):
        super().__init__()
        self.target_scales = target_scales
        self.loss = torch.nn.MSELoss()

    def forward(self, preds, labels):
        # scale output range to target range
        preds = torch.tanh(preds)
        preds = preds * torch.tensor(self.target_scales).to(preds.device).unsqueeze(0)
        return self.loss(preds, labels)    


class MultiCueMetric(torch.nn.Module):
    def __init__(
            self,
            metric_config=None,
            cues=['occlusion', 'lightshadow', 'perspective', 'size', 'texturegrad', 'elevation'], 
    ):
        super().__init__()
        assert len(metric_config) == len(cues), "Different number of cues and loss functions!"

        self.cues = cues

        self.cue_indices = {}
        left, right = 0, 1 # exclusive
        for cue in cues:
            if cue in ['perspective', 'elevation']:
                right += 1
            self.cue_indices[cue] = (left, right)
            left = right
            right += 1

        self.metrics = {
            cue: instantiate_from_config(metric_config[cue]) for cue in cues
        }
        
    def forward(self, preds, labels, cue_idx):
        '''
        Args:
            preds -- (B,num_heads) model output
            labels -- (B,?) label for current batch
            cue_idx -- (B,) idx of current task
        '''
        cue_idx = cue_idx[0].item() # all data in cur batch are from one task
        cue = self.cues[cue_idx]
        metric = self.metrics[cue]
        left, right = self.cue_indices[cue]
        return metric(preds[:, left:right].squeeze(), labels)
    
    def get_preds(self, preds, cue_idx):
        cue_idx = cue_idx[0].item() # all data in cur batch are from one task
        cue = self.cues[cue_idx]
        metric = self.metrics[cue]
        left, right = self.cue_indices[cue]
        return metric.get_preds(preds[:, left:right].squeeze())

class DepthMetric:
    def __init__(self):
        self.name = 'd1'#['d1', 'd2', 'd3', 'rmse']
        self.is_higher_better = True # using d1
        self.name_list = ['d1', 'd2', 'd3', 'rmse']
    def __call__(self, output, target, reduce_mean=False):
        return evaluate_depth(output, target, image_average=reduce_mean)


class WarmupCosineSchedule(object):
    '''
    https://github.com/facebookresearch/jepa/blob/main/src/utils/schedulers.py
    '''
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return new_lr

    def get_last_lr(self):
        return self._last_lr

'''
https://github.com/santurini/cosine-annealing-linear-warmup/tree/main
'''
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: List[float] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
            ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow is not None:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs is not None:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(self.max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(self.min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]