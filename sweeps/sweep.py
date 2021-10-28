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
from pathlib import Path
import wandb
import yaml
import argparse

from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.distributed import *
from fluocells.wandb.utils import wandb_session, wandb_parser
from fluocells.wandb.functions import *

parser = argparse.ArgumentParser(parents=[wandb_parser])
group = parser.add_argument_group('sweep configuraton')
group.add_argument('--config', type=str, default='cfg_workers.yaml',
                   help="Relative path to the configuration file. Note: only `yaml` files are supported. Default: `cfg_workers.yaml`")
group.add_argument('--function', type=str, default='train',
                   help="Function to use for the sweep. It must match one of the funcitons defined in the script. Default: `train`")
group.add_argument('--count', type=int, default=50, help="Number of iterations for the W&B agent. Defalut: 50")
args = parser.parse_args()


# sweep helper
def _fit_sweep(proj_name, sweep_config, func, entity='lclissa', count=10):
    sweep_id = wandb.sweep(sweep_config, project=proj_name)
    wandb.agent(sweep_id, function=func, entity=entity, count=count)


NEED_DECORATOR = ['batch_size_VS_resize', 'dataloader_VS_loss']
if __name__ == '__main__':
    cwd = Path.cwd()
    p_config = cwd / args.config

    # check config format
    assert str(p_config).endswith(
        'yaml'), f"Unsupported format for configuration:\n{p_config}\n\nPlease provide a config file in `yaml` format."

    print(f'\nReading configuration file:\n{p_config}\n\n')
    # load config file
    with open(p_config) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    function = globals()[args.function]
    # decorate if needed
    if args.function in NEED_DECORATOR:
        function = wandb_session(function)
    _fit_sweep(args.proj_name, sweep_config=config, func=function, count=args.count)
