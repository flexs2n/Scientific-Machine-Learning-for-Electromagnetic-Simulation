import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import Tensor
from typing import List


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def mse(self, x, y):
        num_examples = x.size()[0]
        mse_loss = torch.mean((x.view(num_examples, -1) - y.view(num_examples, -1)) ** 2, dim=1)
        if self.reduction:
            if self.size_average:
                return torch.mean(mse_loss)
            else:
                return torch.sum(mse_loss)
        return mse_loss

    def __call__(self, x, y):
        return self.mse(x, y)  # Switch to MSE instead of relative loss