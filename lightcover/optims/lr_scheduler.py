import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        max_lr: float,
        warmup_step: int,
        fix_step: int,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gap_lr = max_lr - min_lr
        self.warmup_step = warmup_step
        self.fix_step = fix_step
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = max(1, self.last_epoch)
        new_lrs = [self._cosine_annealing(lr, step) for lr in self.base_lrs]
        return new_lrs

    def _cosine_annealing(self, base_lr, step):
        if step < self.warmup_step:
            lr = self.min_lr + self.gap_lr * (step / self.warmup_step)

        elif step >= self.warmup_step and step < self.fix_step:
            s = (step - self.warmup_step) / (self.fix_step - self.warmup_step)
            lr = self.min_lr + 0.5 * self.gap_lr * (1 + math.cos(math.pi * s))

        else:
            lr = self.min_lr

        return base_lr * lr


class NoamScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        model_size: int,
        warmup_steps: int,
    ):
        self.warmup_steps = warmup_steps
        self.normalize = model_size ** (-0.5)
        super(NoamScheduler, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = max(1, self.last_epoch)
        new_lrs = [self._noam_annealing(lr, step) for lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, base_lr, step):
        return (
            base_lr
            * self.normalize
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
