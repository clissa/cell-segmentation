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
import wandb
from pathlib import Path
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.distributed import *
from fluocells.config import TRAIN_PATH, VAL_PATH
from fluocells.losses import DiceLoss
from fluocells.wandb.utils import *
from fluocells.utils import *


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
    gpu_id = get_less_used_gpu()
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

    if config.log:
        # TODO: properly configure metrics dictionary with metrics to be tracked by W&B
        n_params = _get_params(learn, trainable=False)
        n_trainable_params = _get_params(learn, trainable=True)
        metrics = {'runtime': exec_time, 'batch_size': config.batch_size, 'shape': (config.resize, config.resize),
                   'n_params': n_params, 'n_trainable_params': n_trainable_params}
    else:
        metrics = None
    # freeing memory
    # learn.zero_grad(set_to_none=True)
    # free_memory(['learn'], debug=True)
    # print('End of training:')
    # get_less_used_gpu()

    return {'metrics': metrics}
    # return {'learn': learn, 'metrics': metrics}


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
