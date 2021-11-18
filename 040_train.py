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
from fastai.distributed import *
from fastai.callback.wandb import *
import wandb

from fluocells.wandb.utils import *

USE_DS_ARTIFACT = False

default_config = dict(
    # wandb info
    project='fluocells', entity='lclissa',
    # resources
    gpu_id=0, dls_workers=0,
    # hyperparameters: pre-proc
    batch_size=32, pre_tfms={'Resize': {'size': 512, 'pad_mode': 'Reflection'}}, tfms={},
    # hyperparameters: learn
    pretrained=False, encoder='resnet18', lr=None, epochs=50, one_cycle=False, loss_func='DiceLoss', optimizer='Adam',
    # logging
    log_model=True, to_log='all', descr='experiment',
)


def train(config=default_config):
    """Setup pipeline for training starting from specified configuration."""
    from fluocells.config import TRAIN_PATH, VAL_PATH

    # retrieve data path download or default
    if USE_DS_ARTIFACT:
        with wandb.init(project='fluocells', job_type='download') as run:
            TRAIN_PATH, VAL_PATH = _get_wb_datasets(run)

    pre_tfms = [_compose_tfms_from_config(k, v) for k, v in config.pre_tfms.items()]
    tfms = [_compose_tfms_from_config(k, v) for k, v in config.tfms.items()]

    dls = _make_dataloader(TRAIN_PATH, VAL_PATH, pre_tfms=pre_tfms, tfms=tfms, config=config)
    learn = _make_learner(dls, config=config)

    lr = config.lr
    model_save_name = f"{config.encoder}_{config.loss_func}_lr{lr:.8}"
    # save_cb = SaveModelWithEpochCallback(fname=model_save_name, at_end=True)
    save_cb = SaveModelCallback(monitor='valid_loss', fname=model_save_name, at_end=True, with_opt=True)
    wandb_cb = WandbCallback(log=config.to_log, log_dataset=False, log_model=config.log_model)

    fit_func = getattr(learn, 'fit_one_cycle') if config.one_cycle else getattr(learn, 'fit_one_cycle')

    fit_func(int(config.epochs), lr,
             cbs=[
                 wandb_cb,
                 save_cb
             ])

    print('Finished Training')
    return learn


def train_wandb(config=None):
    """Setup pipeline for training starting from specified configuration logging on W&B."""
    with wandb.init(project='fluocells', config=config, group=config['descr'], job_type='train'):
        learn = train(wandb.config)
    return learn


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test multi-gpu training')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--first_resize', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--one_cycle', type=bool, default=False)
    parser.add_argument('--loss_func', type=str, default='DiceLoss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_model', type=bool, default=True)
    parser.add_argument('--to_log', type=str, default='all')
    parser.add_argument('--descr', type=str, default='experiment')

    args = parser.parse_args()

    # set GPU (default 0)
    torch.cuda.set_device(f'cuda:{args.gpu_id}')

    # update size for first resize
    default_config['pre_tfms']['Resize']['size'] = args.first_resize
    del args.first_resize

    # integrate CLI args with default W&B config
    config = _update_config(args, default_config)

    train_wandb(config=config)
    print('Outside!!!!!')
