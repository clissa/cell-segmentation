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
from fluocells.wandb.utils import _get_wb_datasets, _get_train_val_names, wandb_parser, wandb_session
from fluocells.augmentation import *
from fluocells.wandb.functions import augmentation
import argparse

parser = argparse.ArgumentParser(parents=[wandb_parser])
group = parser.add_argument_group('experiment configuraton')
cfg = group.add_argument('-cfg', '--config', dest='config', type=str, default=None,
                         help="Relative path to the configuration file. Note: only `yaml` files are supported.")

if __name__ == '__main__':
    from pathlib import Path

    cwd = Path.cwd()
    p_config = cwd / args.config

    # check config format
    assert str(p_config).endswith(
        'yaml'), f"Unsupported format for configuration:\n{p_config}\n\nPlease provide a config file in `yaml` format."

    print(f'\nReading configuration file:\n{p_config}\n\n')
    # load config file
    with open(p_config) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    augmentation(config=config)
