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
__all__ = ['_get_train_val_names', '_get_wb_datasets', '_make_dataloader', '_make_learner']

from pathlib import Path
from fastai.vision.all import *
from fluocells.config import REPO_PATH
from fluocells.losses import *


def _get_train_val_names(pTr, pVal):
    """Return list with stacked train/val image paths"""
    return get_image_files(pTr / 'images') + get_image_files(pVal / 'images')


def _get_wb_datasets(run, prefix="fluocells-red", alias='latest'):
    """Download train/val datasets artifact. Return (train_path, val_path): paths to datasets"""

    # download artifact and set paths
    # fluocells-red_train_data_60:v0
    train_artifact_ref = f"{prefix}_train_data:{alias}"
    # fluocells-red_val_data_16:v0
    val_artifact_ref = f"{prefix}_val_data:{alias}"
    # find the most recent ("latest") version of the full raw data
    train_ds = run.use_artifact(train_artifact_ref)
    val_ds = run.use_artifact(val_artifact_ref)
    # download locally (if not present)
    train_path = train_ds.download(
        root=REPO_PATH / 'dataset' / f"{train_artifact_ref.split('_')[0]}-split" / 'train')
    val_path = val_ds.download(
        root=REPO_PATH / 'dataset' / f"{val_artifact_ref.split('_')[0]}-split" / 'valid')
    return train_path, val_path


def _make_dataloader(train_path, val_path, tfms=[], pre_tfms=[], cfg=None):
    """Download dataset artifact and setup dataloaders according to configuration parameters. Return dls: DataLoaders"""

    def label_func(p):
        return Path(str(p).replace('images', 'masks'))

    if isinstance(cfg, dict):
        cfg = namedtuple("WBConfig", cfg.keys())(*cfg.values())

    splitter = GrandparentSplitter(train_name='train', valid_name='valid')

    dls = SegmentationDataLoaders.from_label_func(
        train_path.parent, bs=cfg.batch_size, fnames=_get_train_val_names(train_path, val_path), label_func=label_func,
        splitter=splitter,  # RandomSplitter(0.2, 42),
        item_tfms=pre_tfms, batch_tfms=tfms,
        num_workers=cfg.dls_workers,
    )
    return dls


# def _make_dataloader(run, cfg=None, prefix="fluocells-red", alias='latest'):
#     """Download dataset artifact and setup dataloaders according to configuration parameters. Return dls: DataLoaders"""
#
#     # download artifact and set paths
#     train_artifact_ref = f"{prefix}_train_data:{alias}"  # fluocells-red_train_data_60:v0
#     val_artifact_ref = f"{prefix}_val_data:{alias}"  # fluocells-red_val_data_16:v0
#     # find the most recent ("latest") version of the full raw data
#     train_ds = run.use_artifact(train_artifact_ref)
#     val_ds = run.use_artifact(val_artifact_ref)
#     # download locally (if not present)
#     train_path = train_ds.download(root=REPO_PATH / 'dataset' / f"{train_artifact_ref.split('_')[0]}-split" / 'train')
#     val_path = val_ds.download(root=REPO_PATH / 'dataset' / f"{val_artifact_ref.split('_')[0]}-split" / 'valid')
#
#     # cfg = namedtuple("config", hyperparameter_defaults.keys())(*hyperparameter_defaults.values())
#     def label_func(p):
#         return Path(str(p).replace('images', 'masks'))
#
#     pre_tfms = [
#         #     IntToFloatTensor(div_mask=255),
#         Resize(cfg.resize1)
#     ]
#     tfms = [
#         IntToFloatTensor(div_mask=255),  # need masks in [0, 1] format
#         *aug_transforms(
#             size=cfg.resize2,
#             max_lighting=0.1, p_lighting=0.5,
#             min_zoom=0.9, max_zoom=1.1,
#             max_warp=0, max_rotate=15.0)
#     ]
#     splitter = GrandparentSplitter(train_name='train', valid_name='valid')
#     # train_fnames = get_image_files(train_path / 'images')
#     # val_fnames = get_image_files(val_path / 'images')
#
#     dls = SegmentationDataLoaders.from_label_func(
#         train_path.parent, bs=cfg.batch_size, fnames=_get_train_val_names(train_path, val_path), label_func=label_func,
#         splitter=splitter,  # RandomSplitter(0.2, 42),
#         item_tfms=pre_tfms, batch_tfms=tfms,
#         num_workers=cfg.dls_workers,
#     )
#     return dls


def _make_learner(dls, cfg=None):
    """Use the input dataloaders and configuration to setup a unet_learner with desired parameters. Return learn:
    Learner and updates cfg.learning_rate if None"""

    print('inside learner', cfg)
    model = globals()[cfg.encoder]
    optimizer = globals()[cfg.optimizer]
    loss_func = globals()[cfg.loss_func]()

    learn = unet_learner(dls, arch=model,
                         loss_func=loss_func,
                         opt_func=optimizer,
                         # accuracy],
                         metrics=[Dice(), JaccardCoeff(), foreground_acc],
                         #                      cbs=EarlyStoppingCallback(monitor='dice', min_delta=0, patience=2),
                         cbs=[ActivationStats(
                             with_hist=True, every=4), CSVLogger()],
                         path=REPO_PATH / 'trainings', model_dir='models',
                         pretrained=cfg.pretrained,
                         n_out=2
                         )  # .to_fp16()
    # learn.fine_tune(6)

    learn.model_dir = learn.model_dir + "/" + learn.loss_func.name
    print(
        f'Logs save path: {learn.path}\nModel save path: {learn.path / learn.model_dir}')

    if cfg.learning_rate is None:
        lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(
            suggest_funcs=(minimum, steep, valley, slide))
        cfg.learning_rate = max(lr_valley, lr_steep)
        print(
            f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}\nSlide "
            f"interval:\t{lr_slide:.2e}")
    # else:
    #     print(f"Learning rate: {cfg.learning_rate}")
    print(f"Using LR={cfg.learning_rate:.6}")
    return learn
