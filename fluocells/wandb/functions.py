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
__all__ = ['augmentation', 'batch_size_VS_resize', 'dataloader_VS_loss', 'train']

import os
import wandb
from pathlib import Path
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.distributed import *
from fluocells.config import TRAIN_PATH, VAL_PATH, REPO_PATH
from fluocells.losses import CombinedLoss
from fluocells.wandb.utils import *
from fluocells.utils import *

GPUS = os.getenv('GPUS', default=None)


def augmentation(config=None):
    import random
    from fluocells.config import TRAIN_PATH, VAL_PATH
    with wandb.init(project='fluocells', config=config, job_type='sweep_augmentation') as run:
        config = wandb.config

        # train_path, val_path = rank0_first(_get_wb_datasets, run=run, prefix='fluocells-red', alias='latest')
        fnames = _get_train_val_names(TRAIN_PATH, VAL_PATH)
        seed = config.seed
        n_samples = config.n_samples
        random.seed(seed)
        sample_paths = random.sample(fnames, n_samples)

        section_dict = {
            'resize': 'Resize',
            'random_resized_crop': 'Resize',
            'zoom': 'Perspective',
            'warp': 'Perspective',
            'rotate': 'Perspective',
            'brightness': 'Colourspace',
            'contrast': 'Colourspace',
            'saturation': 'Colourspace',
            'hue': 'Colourspace',
        }

        # TODO:
        # - implement configurable augmentation pipeline
        # - setup W&B logging
        aug_dict = {}  # dict(original={},transforms={},)
        tfm = globals()[f"_{config.tfm}"]
        for img_path in sample_paths:
            img = PILImage.create(img_path)
            tfms = tfm(img)
            # log one section per each image composed by original image + augmented versions
            wandb.log({f"{section_dict[config.tfm]}/{config.tfm.title()}": [
                                                                               wandb.Image(img, caption=img_path.name,
                                                                                           grouping=img_path.name)] + [
                                                                               wandb.Image(tfmd,
                                                                                           caption=cpt,
                                                                                           grouping=img_path.name) for
                                                                               cpt, tfmd in tfms.items()]
                       }
                      )


def _get_params(net, trainable=False):
    """Return network total parameters (default) or trainable only."""
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters()) if trainable else net.parameters()
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


import time


def batch_size_VS_resize(config=None) -> dict:
    """Run one epoch of training with a given configuration of batch size and resize shape.
    Return a dict with Learner and collected metrics"""
    pre_tfms = [Resize(config.resize)]
    gpu_id = get_less_used_gpu(gpus=GPUS, debug=False)
    torch.cuda.set_device(f"cuda:{gpu_id}")

    print('Initializing DataLoaders')
    start_time = time.process_time()
    dls = _make_dataloader(TRAIN_PATH, VAL_PATH, pre_tfms=pre_tfms, config=config)

    print('Initializing Learner')
    try:
        encoder = globals()[config.encoder]
    except:
        encoder = resnet18
    learn = unet_learner(dls, arch=encoder, n_out=2, loss_func=DiceLoss())

    print('Start training')
    try:
        learn.fit(n_epoch=1, lr=0.001)
        exec_time = time.process_time() - start_time
    except RuntimeError:
        print('WARNING: the run was ended due to Cuda Out Of Memory error --> releasing memory and exiting')
        exec_time = None
        free_memory(['learn'], debug=False)

    if getattr(config, 'log', None):
        # TODO: properly configure metrics dictionary with metrics to be tracked by W&B
        wandb.define_metric('Execution time')
        n_params = _get_params(learn, trainable=False)
        n_trainable_params = _get_params(learn, trainable=True)
        metrics = {'Execution time': exec_time, 'Batch': config.batch_size, 'Shape': config.resize,
                   'Total parameters': n_params, 'Trainable Parameters': n_trainable_params}
        # print(metrics)
    else:
        metrics = None
    # freeing memory
    # learn.zero_grad(set_to_none=True)
    # free_memory(['learn'], debug=True)
    # print('End of training:')
    # get_less_used_gpu()

    return {'metrics': metrics}
    # return {'learn': learn, 'metrics': metrics}


def dataloader_VS_loss(config=None) -> dict:
    """Run few epochs of training depending on configuration and track loss as function of batch size and resize shape.
    Return a dict with Learner and collected metrics"""
    pre_tfms = [Resize(config.resize)]
    gpu_id = get_less_used_gpu(gpus=GPUS, debug=False)
    torch.cuda.set_device(f"cuda:{gpu_id}")

    print('Initializing DataLoaders')
    dls = _make_dataloader(TRAIN_PATH, VAL_PATH, pre_tfms=pre_tfms, config=config)

    print('Initializing Learner')
    try:
        encoder = globals()[config.encoder]
    except:
        encoder = resnet18
    loss_func = globals()[config.loss_func]
    loss_func = partial(loss_func, axis=1) if config.loss_func == 'CrossEntropyLossFlat' else loss_func
    learn = unet_learner(dls, arch=encoder, n_out=2, loss_func=loss_func(),
                         metrics=[Dice(), JaccardCoeff(), foreground_acc],
                         path=REPO_PATH / 'trainings', model_dir='models',
                         pretrained=config.pretrained
                         )

    print('Start training')
    # try:
    # learning rate
    # lr = learn.lr_find()  # valley

    # callbacks
    # wandb_cb = WandbCallback(log=None, log_preds=False, log_dataset=False, log_model=False, )

    min_delta = 0.01
    monitor = 'valid_loss'
    earlystop_cb = EarlyStoppingCallback(monitor=monitor, comp=None, min_delta=min_delta, patience=3,
                                         reset_on_fit=True)
    savebest_cb = SaveModelCallback(monitor=monitor, min_delta=min_delta)

    # training
    learn.fit(n_epoch=config.epochs, lr=0.001, cbs=[
        # wandb_cb,
        earlystop_cb,
        savebest_cb
    ])

    # except RuntimeError:
    #     print('WARNING: the run was ended due to Cuda Out Of Memory error --> releasing memory and exiting')
    #     valid_loss = None
    #     free_memory(['learn'], debug=False)

    if getattr(config, 'log', None):
        wandb.define_metric('Validation Loss')
        wandb.define_metric('Dice')
        wandb.define_metric('Jaccard Coefficient')
        wandb.define_metric('Foreground Accuracy')
        valid_loss, dice, jacc, fg_acc = learn.validate()
        metrics = {'Batch': config.batch_size, 'Shape': config.resize, 'Encoder': encoder.__name__,
                   'Validation Loss': valid_loss,
                   'Dice': dice, 'Jaccard Coefficient': jacc, 'Foreground Accuracy': fg_acc}
    else:
        metrics = None
    return {'metrics': metrics}


def train(config=None, dataset='fluocells-red', alias='latest'):
    with wandb.init(config=config, job_type='sweep_augmentation') as run:
        config = wandb.config
        train_path, val_path = rank0_first(_get_wb_datasets, run=run, prefix='fluocells-red', alias='latest')
        # old version _get_wb_datasets(run, prefix='fluocells-red', alias='latest')
        pre_tfms = [
            #     IntToFloatTensor(div_mask=255),
            Resize(config.resize1)
        ] if config.resize1 else []
        tfms = [
            IntToFloatTensor(div_mask=255),  # need masks in [0, 1] format
            *aug_transforms(
                size=config.resize2,
                max_lighting=0.1, p_lighting=0.5,
                min_zoom=0.9, max_zoom=1.1,
                max_warp=0, max_rotate=15.0)
        ]
        dls = _make_dataloader(train_path, val_path, pre_tfms=pre_tfms, tfms=tfms, config=config)
        # dls = _make_dataloader(run, config, prefix=dataset, alias=alias)
        learn = _make_learner(dls, config=config)
        lr = config.lr
        model_save_name = f"{config.encoder}_{config.loss_func}_lr{lr:.6}"
        # save_cb = SaveModelWithEpochCallback(fname=model_save_name, at_end=True)
        save_cb = SaveModelCallback(monitor='valid_loss', fname=model_save_name, at_end=True, with_opt=True)
        if config.to_log == 'None': config.to_log = None
        wandb_cb = WandbCallback(log=config.to_log,
                                 log_dataset=False,
                                 log_model=config.log_model
                                 )
        cbs = [wandb_cb, save_cb]
        learn = _train_learner_with_args(learn, config.one_cycle, config.multi_gpu, n_epoch=config.epochs, lr_max=lr,
                                         cbs=cbs)
        print('Finished Training')
        return learn
