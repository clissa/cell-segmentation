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
#  Created on 6/8/21, 10:54 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """

from pathlib import Path
import fluocells as fluo

REPO_PATH = Path(fluo.__path__[0]).parent
DATA_PATH_r = REPO_PATH / 'dataset/red'
DATA_PATH_y = REPO_PATH / 'dataset/yellow'
DATA_PATH_g = REPO_PATH / 'dataset/green'

IMG_PATH_r = DATA_PATH_r / 'original/images'
IMG_PATH_y = DATA_PATH_y / 'original/images'
IMG_PATH_g = DATA_PATH_g / 'original/images'

IMG_PATH_unlabelled_r = DATA_PATH_r / 'unlabelled'
IMG_PATH_unlabelled_y = DATA_PATH_y / 'unlabelled'
IMG_PATH_unlabelled_g = DATA_PATH_g / 'unlabelled'

# DATA_DEBUG_PATH = DATA_PATH / 'debug'
# DATA_DEBUG_PATH.mkdir(exist_ok=True, parents=True)
