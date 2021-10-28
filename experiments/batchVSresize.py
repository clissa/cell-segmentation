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
import os
import argparse
from fluocells.wandb.utils import wandb_parser, wandb_session, _init_config

# The following code contains comments that involve a tentative implementation using mutually exclusive args groups:
# adaptation from template at https://newbedev.com/does-argparse-python-support-mutually-exclusive-groups-of-arguments
# import conflictsparse # not available
# import itertools

# parser = conflictsparse.ConflictsOptionParser(parents=[wandb_parser]) # in case conflictsparse was available
parser = argparse.ArgumentParser(parents=[wandb_parser])
group = parser.add_argument_group('experiment configuraton')
bs = group.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=2)
rsz = group.add_argument('-rsz', '--resize', dest='resize', type=int, default=512)
gpus = group.add_argument('--gpus', dest='gpus', type=str)
log = group.add_argument('--log', action='store_true', default=False)
cfg = group.add_argument('-cfg', '--config', dest='config', type=str, default=None,
                         help="Relative path to the configuration file. Note: only `yaml` files are supported.")

# manual_config = (bs, rsz)
# config_file = cfg
# exclusives = itertools.product(manual_config, config_file)
# for exclusive_grp in exclusives:
#     parser.register_conflict(exclusive_grp)

args = parser.parse_args()

if args.gpus:
    os.environ['GPUS'] = args.gpus

if __name__ == '__main__':
    # initialization for testing
    # args = parser.parse_args(['-bs=8', '-rsz=224'])
    from fluocells.wandb.functions import batch_size_VS_resize

    if args.config is None:
        config = _init_config(parser, args)
    print('Setup with config:\n', config)
    if args.log:
        # wandb decorator
        batch_size_VS_resize = wandb_session(batch_size_VS_resize)
        batch_size_VS_resize(config=config)
    else:
        batch_size_VS_resize(config)
