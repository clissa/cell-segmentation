#  #!/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Luca Clissa
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#  #http://www.apache.org/licenses/LICENSE-2.0
#  #Unless required by applicable law or agreed to in writing, software
#  #distributed under the License is distributed on an "AS IS" BASIS,
#  #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  #See the License for the specific language governing permissions and
#  #limitations under the License.
from fastai.vision.all import *


class DiceLoss:
    "Dice loss for segmentation (copy-pasted from fastai 2.5.1)"

    def __init__(self, axis=1, smooth=1e-6, reduction="sum", square_in_union=False):
        self.name = 'DiceLoss'
        store_attr()

    def __call__(self, pred, targ):
        targ = self._one_hot(targ, pred.shape[self.axis])
        pred, targ = TensorBase(pred), TensorBase(targ)
        assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred ** 2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        return ((1 - dice_score).flatten().mean() if self.reduction == "mean"
                else (1 - dice_score).flatten().sum())

    @staticmethod
    def _one_hot(x, classes, axis=1):
        "Creates one binay mask per class"
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)

    def decodes(self, x): return x.argmax(dim=self.axis)


class FocalLoss(Module):
    "Focal loss for segmentation (copy-pasted from fastai 2.5.1)"

    y_int = True

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        self.name = 'FocalLoss'
        store_attr()

    def forward(self, inp: torch.Tensor, targ: torch.Tensor):
        ce_loss = F.cross_entropy(
            inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class CombinedLoss:
    """Add Dice to Focal with weight alpha.

    refs: https://docs.fast.ai/losses.html
    """

    def __init__(self, axis=1, smooth=1., alpha=1.):
        self.name = 'CombinedLoss'
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x): return x.argmax(dim=self.axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)
