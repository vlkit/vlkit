# iteration-wise learning rate control for pytorch
# lr_scheduler = CosAnnealingLR(epoch_size=5005, epochs=120, max_lr=0.1, warmup_epochs=5)
# Note that you should call lr_scheduler.step() before EACH ITERATION rather than each epoch!
import math
import numpy as np


class CosAnnealingLR(object):

    def __init__(self, epoch_size, epochs, max_lr, min_lr=1e-4, warmup_epochs=0, last_epoch=-1):
        """
        epoch_size: number of iterations in a epoch (length of dataloader);
        epochs: total optimization epochs;
        max_lr, min_lr: maximal and minimal learning rates;
        """

        max_iters = epoch_size * epochs
        warmup_iters = epoch_size * warmup_epochs
        assert max_lr >= 0
        assert warmup_iters >= 0
        assert max_iters >= 0 and max_iters >= warmup_iters

        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.last_epoch = last_epoch

        assert self.last_epoch >= -1
        self.iter_counter = (self.last_epoch+1) * epoch_size

        self.lr = 0
    
    def restart(self, max_lr=None):

        if max_lr:
            self.max_lr = max_lr
        self.iter_counter = 0

    def step(self):

        self.iter_counter += 1

        if self.warmup_iters > 0 and self.iter_counter <= self.warmup_iters:

            self.lr = float(self.iter_counter / self.warmup_iters) * self.max_lr

        else:

            self.lr = (1 + math.cos((self.iter_counter-self.warmup_iters) / \
                                    (self.max_iters - self.warmup_iters) * math.pi)) / 2 * self.max_lr

        return self.lr

class MultiStepLR(object):

    def __init__(self, epoch_size, milestones, gamma=None, gammas=None, base_lr=0.1, warmup_epochs=0, last_epoch=-1):

        if (gamma is not None and gammas is not None) or \
           (gamma is None and gammas is None):
            raise ValueError("You should specify either specify gamma or gammas.")

        if gamma is not None:
            gammas = [gamma] * len(milestones)

        assert isinstance(milestones, list)
        assert isinstance(gammas, list)
        assert len(milestones) == len(gammas)

        self.warmup_iters = warmup_epochs * epoch_size
        self.epoch_size = epoch_size
        self.base_lr = base_lr
        self.lr = base_lr
        self.milestones = milestones
        self.gammas = gammas
        self.last_epoch = last_epoch

        assert self.last_epoch >= -1
        self.iter_counter = (self.last_epoch+1) * epoch_size

        self.milestone_counter = 0

    def step(self):

        self.iter_counter += 1

        if self.warmup_iters > 0 and self.iter_counter <= self.warmup_iters:
            self.lr = float(self.iter_counter / self.warmup_iters) * self.base_lr

        else:
            if self.milestone_counter < len(self.milestones) and \
               self.iter_counter == self.milestones[self.milestone_counter] * self.epoch_size:

                self.lr = self.lr * self.gammas[self.milestone_counter]
                self.milestone_counter += 1

        return self.lr

