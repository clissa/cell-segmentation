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
#
#  """
#  Created on 6/16/21, 5:18 PM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 6/16/21, 3:57 PM
#  @author: Luca Clissa
#
#
#  Run using image_processing environment
#  """

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Tue May  7 10:42:13 2019
@author: Luca Clissa
"""
import argparse

from fluocells.config import DATA_PATH_g, DATA_PATH_y, DATA_PATH_r
from fluocells.utils.data import compute_masks_stats

parser = argparse.ArgumentParser(description='Crop original images in square patches of customisable size.')
parser.add_argument('dataset', metavar='marker color', type=str,
                    help='Marker color (green, red or yellow)')
parser.add_argument('version', metavar='dataset version name', type=str, help='Version tag')

args = parser.parse_args()
if args.dataset == 'red':
    MASKS_PATH = DATA_PATH_r / f'{args.version}/masks'
elif args.dataset == 'yellow':
    MASKS_PATH = DATA_PATH_y / f'{args.version}/masks'
elif args.dataset == 'green':
    MASKS_PATH = DATA_PATH_g / 'raw_data_green/all_masks/images'
else:
    raise ValueError("Invalid argument `dataset`. Supported values are `red` and `yellow`.")

_ = compute_masks_stats(MASKS_PATH)
