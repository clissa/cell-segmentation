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
#
#  """
#  Created on 5/18/21, 10:33 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 5/18/21, 10:31 AM
#  @author: Luca Clissa
#
#
#  """

from pathlib import Path

HOME = Path.home()
if HOME.name == 'luca':
    workdir = 'PycharmProjects'
else:
    workdir = 'workspace'
REPO_PATH = HOME / workdir / 'cells'
DATA_PATH_r = REPO_PATH / 'dataset/red'
DATA_PATH_y = REPO_PATH / 'dataset/yellow'
DATA_PATH_g = REPO_PATH / 'dataset/green'

# DATA_DEBUG_PATH = DATA_PATH / 'debug'
# DATA_DEBUG_PATH.mkdir(exist_ok=True, parents=True)
