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

"""
Created on Tue May  7 10:42:13 2019
@author: Luca Clissa
"""
# OS related
import os
import sys
import argparse
from pathlib import Path

from fastai.vision.all import *
from fastai.interpret import *  # SegmentationDataLoaders
from fastai.metrics import foreground_acc, JaccardCoeff, Dice

from fluocells.config import DATA_PATH_r, REPO_PATH

print(f'\nWorking directory: {os.getcwd()}')

import wandb
from fastai.callback.wandb import *

parser = argparse.ArgumentParser(description='Setup and run a W&B sweep for hyperparamenters comparison.')
parser.add_argument('proj_name', type=str, help='Name of the W&B project')
parser.add_argument('--alias', type=str, default='latest', help="Alias for the W&B artifact. Default: 'latest'")
parser.add_argument('-ds', '--dataset', type=str, default='red',
                    help="Name of the dataset to be uploaded. Values: 'red'(default)|'yellow'")
parser.add_argument('-art_name', '--artifact_name', type=str, default='',
                    help="Name of the W&B artifact. The default will create a name as f'fuocells-{dataset}',"
                         "; f'{artifact_name}-{dataset}' otherwise")
parser.add_argument('--crops', type=str, default='', help="Crops dataset folder. Default '', i.e., no crops")
args = parser.parse_args()


# TODO: some conceptual modifications are auspicable:
#   i) create module for custom classes and helper functions to import from
#   ii) reformulate this script so to take custom configuration file

### CUSTOM LOSSES
class DiceLoss:
    "Dice loss for segmentation"

    def __init__(self, axis=1, smooth=1e-6, reduction="sum", square_in_union=False):
        store_attr()
        self.name = 'DiceLoss'

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
    "Dice and Focal combined"

    def __init__(self, axis=1, smooth=1., alpha=1.):
        self.name = 'CombinedLoss'
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x): return x.argmax(dim=self.axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)


### HYPERPARAMS SWEEP

sweep_config = {
    "name": 'test-sweep-alpha',
    "method": "random",
    "metric": {"name": "dice", "goal": "maximize"},
    "parameters": {
        "epochs": {
            "values": [30, 50, 100]
        },
        "resize1": {'value': 512},
        "resize2": {'value': 224},
        "batch_size": {'values': [4, 8, 12]},
        "encoder": {'value': 'resnet18'},
        "pretrained": {'values': [True, False]},
        "lr": {
            "distribution": 'uniform',
            "min": 0.0001,
            "max": 0.1
        },
        "one_cycle": {'value': False},
        "dls_workers": {'values': [0, 4, 8]},
        "optimizer": {'values': ['Adam', 'RAdam', 'SGD', 'QHAdam']},
        "loss_func": {'values': ['DiceLoss', 'FocalLoss', 'CombinedLoss']},
        # "model": {'value': resnet18},
    }
}

PREFIX = f"{args.artifact_name}-{args.dataset}" if args.artifact_name else f"fluocells-{args.dataset}"
if args.crops:
    PREFIX = f"{PREFIX}-{args.crops}"


# PREFIX = 'fluocells-red'

# hyperparameter_defaults = dict(
#     resize1=512,
#     resize2=224,
#     batch_size=8,
#     epochs=20,
#     encoder=model.__name__,
#     pretrained=True,  # use pre-trained model and train only last layers
#     #     dropout = 0.5,
#     one_cycle=False,  # "1cycle" policy -> https://arxiv.org/abs/1803.09820
#     lr=0.001,
#     # loss_func=loss_func.name,
#     dls_workers=0,
#     optimizer=Adam,
#     loss_func=CombinedLoss(),
#     model= resnet18,
# )


def _get_train_val_names(pTr, pVal):
    return get_image_files(pTr / 'images') + get_image_files(pVal / 'images')


def _make_dataloader(run, cfg=None):
    """Download dataset artifact and setup dataloaders according to configuration parameters. Return dls: DataLoaders"""

    # download artifact and set paths
    artifact_ref = f"{PREFIX}:{args.alias}"
    train_artifact_ref = f"{PREFIX}_train_data_60:{'latest'}"  # fluocells-red_train_data_60:v0
    val_artifact_ref = f"{PREFIX}_val_data_16:{'latest'}"  # fluocells-red_val_data_16:v0
    # find the most recent ("latest") version of the full raw data
    train_ds = run.use_artifact(train_artifact_ref)
    val_ds = run.use_artifact(val_artifact_ref)
    # download locally (if not present)
    train_path = train_ds.download(root=REPO_PATH / 'dataset' / f"{train_artifact_ref.split('_')[0]}-split" / 'train')
    val_path = val_ds.download(root=REPO_PATH / 'dataset' / f"{val_artifact_ref.split('_')[0]}-split" / 'valid')

    # config = namedtuple("config", hyperparameter_defaults.keys())(*hyperparameter_defaults.values())
    def label_func(p):
        return Path(str(p).replace('images', 'masks'))

    pre_tfms = [
        #     IntToFloatTensor(div_mask=255),
        Resize(cfg.resize1)
    ]
    tfms = [
        IntToFloatTensor(div_mask=255),  # need masks in [0, 1] format
        *aug_transforms(
            size=cfg.resize2,
            max_lighting=0.1, p_lighting=0.5,
            min_zoom=0.9, max_zoom=1.1,
            max_warp=0, max_rotate=15.0)
    ]
    splitter = GrandparentSplitter(train_name='train', valid_name='valid')
    # train_fnames = get_image_files(train_path / 'images')
    # val_fnames = get_image_files(val_path / 'images')

    dls = SegmentationDataLoaders.from_label_func(
        train_path.parent, bs=cfg.batch_size, fnames=_get_train_val_names(train_path, val_path), label_func=label_func,
        splitter=splitter,  # RandomSplitter(0.2, 42),
        item_tfms=pre_tfms, batch_tfms=tfms,
        num_workers=cfg.dls_workers,
    )
    return dls


def _make_learner(dls, config=None):
    """Use the input dataloaders and configuration to setup a unet_learner with desired parameters. Return learn:
    Learner and updates config.lr if None"""

    print('inside learner', config)
    model = globals()[config.encoder]
    optimizer = globals()[config.optimizer]
    loss_func = globals()[config.loss_func]

    learn = unet_learner(dls, arch=model,
                         loss_func=loss_func(),
                         opt_func=optimizer,
                         # accuracy],
                         metrics=[Dice(), JaccardCoeff(), foreground_acc],
                         #                      cbs=EarlyStoppingCallback(monitor='dice', min_delta=0, patience=2),
                         cbs=[ActivationStats(
                             with_hist=True, every=4), CSVLogger()],
                         path=REPO_PATH / 'trainings', model_dir='models',
                         pretrained=config.pretrained,
                         n_out=2
                         )  # .to_fp16()
    # learn.fine_tune(6)

    learn.model_dir = learn.model_dir + "/" + learn.loss_func.name
    print(
        f'Logs save path: {learn.path}\nModel save path: {learn.path / learn.model_dir}')

    if config.lr is None:
        lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(
            suggest_funcs=(minimum, steep, valley, slide))
        config.lr = max(lr_valley, lr_steep)
        print(
            f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}\nSlide "
            f"interval:\t{lr_slide:.2e}")
    # else:
    #     print(f"Learning rate: {config.lr}")
    print(f"Using LR={config.lr:.6}")
    return learn


def train(config=None):
    with wandb.init(project='fluocells', entity='lclissa', config=config, job_type='sweep_train') as run:
        config = wandb.config
        dls = _make_dataloader(run, config)
        learn = _make_learner(dls, config=config)

        lr = config.lr
        model_save_name = f"{config.encoder}_{config.loss_func}_lr{lr:.6}"
        # save_cb = SaveModelWithEpochCallback(fname=model_save_name, at_end=True)
        save_cb = SaveModelCallback(monitor='valid_loss', fname=model_save_name, at_end=True, with_opt=True)

        if config.one_cycle:
            learn.fit(int(config.epochs), lr,
                      cbs=[
                          WandbCallback(log='all',
                                        log_dataset=False,
                                        log_model=True
                                        ),
                          save_cb
                      ])

        else:
            learn.fit_one_cycle(int(config.epochs), lr,
                                cbs=[
                                    WandbCallback(log='all',
                                                  log_dataset=False,
                                                  log_model=True
                                                  ),
                                    save_cb
                                ])

        # log_model(path, name=None, metadata=config, description='trained model')

        print('Finished Training')


def _fit_sweep(proj_name, sweep_config, func, entity='lclissa', count=10):
    sweep_id = wandb.sweep(sweep_config, project=proj_name)
    wandb.agent(sweep_id, function=func, entity=entity, count=count)


if __name__ == '__main__':
    _fit_sweep(args.proj_name, sweep_config, func=train)
