
from torch.optim.lr_scheduler import _LRScheduler
import math

class FlatplusAnneal(_LRScheduler):
    def __init__(self, optimizer, max_iter, step_size=0.7, eta_min=0, last_epoch=-1):
        self.flat_range = int(max_iter * step_size)
        self.T_max = max_iter - self.flat_range
        self.eta_min = 0
        super(FlatplusAnneal, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.flat_range:
            return [base_lr for base_lr in self.base_lrs]
        else:
            cr_epoch = self.last_epoch - self.flat_range
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (cr_epoch / self.T_max)))/ 2 for base_lr in self.base_lrs]

class FlatplusAnnealTeste(_LRScheduler):
    def __init__(self, optimizer, max_iter, step_size=0.7, eta_min=0, last_epoch=-1, stabilize_epoch=70):
        self.flat_range = int(max_iter * step_size)
        self.T_max = max_iter - self.flat_range
        self.eta_min = eta_min
        self.stabilize_epoch = stabilize_epoch
        super(FlatplusAnnealTeste, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 30:
            # No increase in the first 30 epochs, return base_lr
            return [base_lr for base_lr in self.base_lrs]
        elif 30 <= self.last_epoch < 60:
            # Increase LR after 30 epochs
            return [base_lr * 1.2 for base_lr in self.base_lrs]  # Aumento de 20% como exemplo
        elif 60 <= self.last_epoch < self.stabilize_epoch:
            # Increase LR after 60 epochs
            return [base_lr * 1.5 for base_lr in self.base_lrs]  # Aumento de 50% como exemplo
        else:
            # Continue with the increased LR after epoch 70 (keep it high)
            return [base_lr * 1.5 for base_lr in self.base_lrs]  # Mantém o aumento de 50%