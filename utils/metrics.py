from typing import Any

import numpy as np
import torch

from torchmetrics import Metric

class TimePerStep(Metric):
    def __init__(self, dist_sync_on_step=False, accumulate_grad_batches=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.accumulate_grad_batches= accumulate_grad_batches
    
    def update(self, time):
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time).float().to(self.time.device)
        self.time += time
        self.total += 1
    
    def compute(self):
        return self.time / (self.total / self.accumulate_grad_batches)


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        # preds = logits[:, :target.max()+1].argmax(dim=-1) # in case that the vocab_size is much larger
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape
        
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total