from torch.nn import Module
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
import torch

from torch import Tensor
from typing import Callable, Optional
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self,alpha=0.0, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def one_hot(self, targets, n_classes):
        targets = torch.empty((targets.size(0), n_classes), device=targets.device, requires_grad=False) \
        .fill_(self.alpha / n_classes) \
        .scatter_(1, targets.unsqueeze(1), 1. - self.alpha + (self.alpha / n_classes))
        return targets

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, inputs, targets):
        targets = self.one_hot(targets, inputs.size(-1))
        log_preds = F.log_softmax(inputs, -1)
        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))