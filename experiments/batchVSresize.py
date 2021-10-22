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
# from pathlib import Path
# print(Path().cwd())
# import sys
# sys.path.append('./')
from fluocells.config import TRAIN_PATH, VAL_PATH
from fastai.vision.all import Resize, resnet18, unet_learner
from fluocells.losses import DiceLoss
from fluocells.utils.wandb import _make_dataloader, wandb_parser, _init_config
import argparse

# The following code contains comments that involve a tentative implementation using mutually exclusive args groups:
# adaptation from template at https://newbedev.com/does-argparse-python-support-mutually-exclusive-groups-of-arguments
# import conflictsparse # not available
# import itertools

# parser = conflictsparse.ConflictsOptionParser(parents=[wandb_parser]) # in case conflictsparse was available
parser = argparse.ArgumentParser(parents=[wandb_parser])
group = parser.add_argument_group('experiment configuraton')
bs = group.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=2)
rsz = group.add_argument('-rsz', '--resize', dest='resize', type=int, default=512)
log = group.add_argument('--log', action="store_true", default=False)
cfg = group.add_argument('-cfg', '--config', dest='config', type=str, default=None,
                         help="Relative path to the configuration file. Note: only `yaml` files are supported.")
parser.add_argument('--count', type=int, default=50, help="Number of iterations for the W&B agent")
# manual_config = (bs, rsz)
# config_file = cfg
# exclusives = itertools.product(manual_config, config_file)
# for exclusive_grp in exclusives:
#     parser.register_conflict(exclusive_grp)


def batch_size_VS_resize(config):
    """batch size VS resize"""

    pre_tfms = [Resize(config.resize)]

    print('Initializing DataLoaders')
    dls = _make_dataloader(TRAIN_PATH, VAL_PATH, pre_tfms=pre_tfms, config=config)

    print('Initializing Learner')
    model = resnet18
    learn = unet_learner(dls, arch=model, n_out=2, loss_func=DiceLoss())

    print('Start training')
    learn.fit(n_epoch=1, lr=0.001)

    if config.log:
        # TODO: properly configure metrics dictionary with metrics to be tracked by W&B
        metrics = 'Done'
    else:
        metrics = None
    return {'learn': learn, 'metrics': metrics}

if __name__ == '__main__':
    # initialization for testing
    # args = parser.parse_args(['-bs=8', '-rsz=224'])
    args = parser.parse_args()

    if args.config is None:
        config = _init_config(parser, args)
    print('Setup with config:\n', config)
    if args.log:
        # TODO: execute through wandb decorator
        # @wandb_session
        # batch_size_VS_resize(config)
        pass
    else:
        batch_size_VS_resize(config)
