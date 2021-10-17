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
__all__ = ['_get_train_val_names', '_get_wb_datasets', '_make_dataloader', '_make_learner', '_train_learner_with_args',
           '_resize', '_random_resized_crop', '_zoom', '_rotate', '_warp', '_brightness', '_contrast', '_saturation',
           '_hue']

import random
from pathlib import Path
from itertools import product

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


def _get_fitter_name(method_str):
    return method_str.split('.')[1].split(' ')[0]


def _train_learner_with_args(learn, one_cycle=False, multi_gpu=False, **kwargs):
    """Wrapper for training configurations depending on one cycle policy and gpus. Training params are passed as kwargs."""

    fit_func = getattr(learn, "fit") if one_cycle else getattr(learn, "fit_one_cycle")
    print(f"\nPerforming fit using {_get_fitter_name(fit_func.__str__())} and {'multi' if multi_gpu else 'single'} gpu")
    if multi_gpu:
        with learn.distrib_ctx():
            fit_func(**kwargs)
    else:
        fit_func(**kwargs)
    return learn


def _batch_ex(bs, pilimg):
    timg = TensorImage(array(pilimg)).permute(2, 0, 1).float() / 255.
    return TensorImage(timg[None].expand(bs, *timg.shape).clone())


def _random_coord(min_c=0.3, max_c=0.7):
    """Pick a random coordinate for the center of augmentation transforms"""
    return random.uniform(min_c, max_c)


def _resize(img, sizes=[512], methods=['Crop', 'Pad', 'Squish'], pad_mode=['Border', 'Reflection']):
    tfms_dict = {}
    for args in product(sizes, methods, pad_mode):
        s, m, p = args
        tfmd = Resize(size=s, method=m, pad_mode=p)(img)
        tfms_dict[f"Size={s}, Method={m}, Padding={p}"] = tfmd
    return tfms_dict


def _random_resized_crop(img, sizes=[512]):
    tfms_dict = {}
    for s in sizes:
        tfmd = RandomResizedCrop(size=s, min_scale=0.6, max_scale=1.2, ratio=(0.7, 1.3))(img)
        tfms_dict[f"Size={s}"] = tfmd
    return tfms_dict


def _zoom(img, scales=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5], mode=['bilinear', 'bicubic'], pad_mode=['border', 'reflection']):
    tfms_dict = {}
    for args in product(mode, pad_mode):
        m, p = args
        z = Zoom(p=1., draw=scales, draw_x=_random_coord(), draw_y=_random_coord(), mode=m, pad_mode=p, size=512)
        b = _batch_ex(len(scales), img)
        tfms = z(b)
        for s, t in zip(scales, tfms):
            tfms_dict[f"Scale={s}, Mode={m}, Padding={p}"] = t
    return tfms_dict


def _rotate(img, angles=[-25, -10, 0, 10, 25], mode=['bilinear', 'bicubic'], pad_mode=['border', 'reflection']):
    tfms_dict = {}
    for args in product(mode, pad_mode):
        m, p = args
        z = Rotate(p=1., draw=angles, mode=m, pad_mode=p, size=512)
        b = _batch_ex(len(angles), img)
        tfms = z(b)
        for s, t in zip(angles, tfms):
            tfms_dict[f"Angle={s}, Mode={m}, Padding={p}"] = t
    return tfms_dict


def _warp(img, scales=[-0.4, -0.2, 0., 0.2, 0.4], wtype=['horizontal', 'vertical']):
    tfms_dict = {}
    v_warp = Warp(p=1., draw_y=scales, draw_x=0., size=512)
    h_warp = Warp(p=1., draw_x=scales, draw_y=0., size=512)
    b = _batch_ex(len(scales), img)
    tfms = v_warp(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Scale={s}, Type=vertical"] = t
    tfms = h_warp(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Scale={s}, Type=horizontal"] = t
    return tfms_dict


def _brightness(img, scales=[0.3, 0.4, 0.5, 0.7, 0.8]):
    tfms_dict = {}
    z = Brightness(p=1., draw=scales)
    b = _batch_ex(len(scales), img)
    tfms = z(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Magnitude={s}"] = t
    return tfms_dict


def _contrast(img, scales=[0.65, 0.8, 1., 1.25, 1.55]):
    tfms_dict = {}
    z = Contrast(p=1., draw=scales)
    b = _batch_ex(len(scales), img)
    tfms = z(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Magnitude={s}"] = t
    return tfms_dict


def _saturation(img, scales=[0.9, 0.95, 1., 1.05, 1.1]):
    tfms_dict = {}
    z = Saturation(p=1., draw=scales)
    b = _batch_ex(len(scales), img)
    tfms = z(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Magnitude={s}"] = t
    return tfms_dict


def _hue(img, scales=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]):
    tfms_dict = {}
    z = Hue(p=1., draw=scales)
    b = _batch_ex(len(scales), img)
    tfms = z(b)
    for s, t in zip(scales, tfms):
        tfms_dict[f"Magnitude={s}"] = t
    return tfms_dict
