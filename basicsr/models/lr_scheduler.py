# ------------------------------------------------------------------------
# Copyright 2024 Felipe Torres
# ------------------------------------------------------------------------
# This code is modified from:
# - HINet (https://github.com/megvii-model/HINet)
# 
# More details about license and acknowledgement are in LICENSE.
# ------------------------------------------------------------------------
import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    This scheduler adjusts the learning rate of the optimizer based on 
    specified milestones and restart points. At each milestone, the learning 
    rate is decreased by a factor of `gamma`. At each restart point, the 
    learning rate is reset and scaled by the corresponding weight.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer.
        milestones (list): Iterations at which the learning rate will be decreased.
        gamma (float): Factor by which the learning rate will be decreased at each milestone. Default: 0.1.
        restarts (list): Iterations at which the learning rate will be restarted. Default: [0].
        restart_weights (list): Weights to apply to the learning rate at each restart iteration. Default: [1].
        last_epoch (int): The index of the last epoch. This is used internally by _LRScheduler. Default: -1.
    """
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=(0, ),
                 restart_weights=(1, ),
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

class LinearLR(_LRScheduler):
    """ Linear learning rate scheduler.

    This scheduler adjusts the learning rate of the optimizer linearly 
    from the initial learning rate to zero over a specified total number 
    of iterations.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer.
        total_iter (int): Total number of iterations over which the learning rate will be decreased linearly.
        last_epoch (int): The index of the last epoch. This is used internally by _LRScheduler. Default: -1.
    """
    def __init__(self,
                 optimizer,
                 total_iter,
                 last_epoch=-1):
        self.total_iter = total_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter
        weight = (1 - process)
        # print('get lr ', [weight * group['initial_lr'] for group in self.optimizer.param_groups])
        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]
    

class LinearDecreaseWithPlateauLR(_LRScheduler):
    """
    Linear learning rate scheduler with a plateau phase.

    This scheduler decreases the learning rate linearly from start_lr to end_lr 
    over a specified total number of iterations, starting from a given iteration.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer.
        total_iter (int): Total number of iterations over which the learning rate will be decreased.
        start_iter (int): Iteration at which to start decreasing the learning rate.
        start_lr (float, optional): Initial learning rate. If None, uses the optimizer's initial learning rate. Default: None.
        end_lr (float, optional): Final learning rate. Default: 1E-7.
    """
    def __init__(self, optimizer, total_iter, start_iter, start_lr=None, end_lr=1E-7):
        self.total_iterations = total_iter
        self.start_lr = start_lr if start_lr is not None else optimizer.param_groups[0]['lr']
        self.end_lr = end_lr
        self.start_iteration = start_iter
        self.learning_rates = []
        super(LinearDecreaseWithPlateauLR, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.start_iteration:
            lr = self.start_lr
        else:
            # Linearly decrease learning rate from start_lr to end_lr over total_iterations
            alpha = min((self.last_epoch - self.start_iteration) / (self.total_iterations - self.start_iteration), 1)
            lr = self.start_lr * (1 - alpha) + self.end_lr * alpha
        self.learning_rates.append(lr)
        return [lr for _ in self.optimizer.param_groups]


class VibrateLR(_LRScheduler):
    """
    Learning rate scheduler with a vibrating pattern.

    This scheduler adjusts the learning rate based on a vibrating pattern 
    determined by the current epoch and total iterations.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer.
        total_iter (int): Total number of iterations over which the learning rate will be adjusted.
        last_epoch (int): The index of the last epoch. This is used internally by _LRScheduler. Default: -1.
    """
    def __init__(self,
                 optimizer,
                 total_iter,
                 last_epoch=-1):
        self.total_iter = total_iter
        super(VibrateLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter

        f = 0.1
        if process < 3 / 8:
            f = 1 - process * 8 / 3
        elif process < 5 / 8:
            f = 0.2

        T = self.total_iter // 80
        Th = T // 2

        t = self.last_epoch % T

        f2 = t / Th
        if t >= Th:
            f2 = 2 - f2

        weight = f * f2

        if self.last_epoch < Th:
            weight = max(0.1, weight)

        # print('f {}, T {}, Th {}, t {}, f2 {}'.format(f, T, Th, t, f2))
        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    This scheduler adjusts the learning rate using a cosine annealing schedule
    with periodic restarts. The learning rate is reset at each restart point
    and scaled by the corresponding weight.

    Example config:
        periods = [10, 10, 10, 10]
        restart_weights = [1, 0.5, 0.5, 0.5]
        eta_min = 1e-7

    It has four cycles, each with 10 iterations. At the 10th, 20th, and 30th 
    iterations, the scheduler will restart with the weights specified in 
    restart_weights.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer.
        periods (list): Period for each cosine annealing cycle.
        restart_weights (list): Restart weights at each restart iteration. Default: [1].
        eta_min (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of the last epoch. This is used internally by _LRScheduler. Default: -1.
    """
    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
