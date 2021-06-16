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
#  Created on 6/16/21, 5:15 PM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 6/14/21, 12:10 PM
#  @author: Luca Clissa
#
#
#  Run using image_processing environment
#  """
"""
Created on Tue May  7 10:42:13 2019
@author: Luca Clissa
"""
from fluocells.config import DATA_PATH_g
from fluocells.utils.data import get_name2num_map

IMG_PATH = DATA_PATH_g / 'v1.0/images'
IMG_NUM_PATH = DATA_PATH_g / 'raw_data_green/all_images/images'
IMG_NUM_TEST = DATA_PATH_g / 'raw_data_green/test/all_images/images'

names_list = list(IMG_PATH.iterdir())
numbers_list = list(IMG_NUM_PATH.iterdir()) + list(IMG_NUM_TEST.iterdir())
names_map, test_images = get_name2num_map(names_list, numbers_list, is_close_thresh=0.01, similarity_thresh=0.90,
                                          resize=True)

print(f'Missing: {len(test_images)};', f'\tFound: {len(names_map)}')

import pandas as pd

names_df = pd.DataFrame.from_dict(names_map, orient='index').reset_index()
names_df.columns = ['original_name', 'name_number']
names_df.set_index('name_number', inplace=True)
names_df.to_csv(DATA_PATH_g / 'names_map.csv')

# missing
missing = ['Mar20bS2C2R2_DMl_200x_g.png']

### test
from fluocells.config import DATA_PATH_g

IMG_PATH = DATA_PATH_g / 'v1.0/images'
IMG_NUM_PATH = DATA_PATH_g / 'raw_data_green/all_images/images'

name_img_list = ['Mar24bS2C2R2_VLPAGl_200x_g.png',
                 'Mar24bS2C1R3_VLPAGl_200x_g.png',
                 'Mar23bS1C5R2_LHr_200x_g.png',
                 'Mar23bS1C5R3_LHr_200x_g.png',
                 'Mar23bS1C6R1_DMl_200x_g.png',
                 'Mar27bS1C1R3_VLPAGr_200x_g.png',
                 'Mar27bS1C3R1_LHr_200x_g.png',
                 'Mar27bS1C3R1_LHl_200x_g.png',
                 'Mar27bS1C2R3_LHr_200x_g.png',
                 'Mar27bS1C2R2_LHr_200x_g.png',
                 'Mar27bS1C2R1_LHr_200x_g.png',
                 'Mar27bS1C2R1_LHl_200x_g.png',
                 'Mar23bS1C2R1_VLPAGr_200x_g.png',
                 'Mar23bS1C2R2_VLPAGl_200x_g.png',
                 'Mar23bS1C5R3_DMr_200x_g.png',
                 ]
num_img_list = ['campione107.TIF',
                'campione96.TIF',
                'campione191.TIF',
                'campione184.TIF',
                'campione183.TIF',
                'campione173.TIF',
                'campione174.TIF',
                'campione176.TIF',
                'campione177.TIF',
                'campione175.TIF',
                'campione179.TIF',
                'campione178.TIF',
                'campione182.TIF',
                'campione198.TIF',
                'campione203.TIF',
                ]
names_list = [IMG_PATH / name for name in name_img_list]
numbers_list = [IMG_NUM_PATH / name for name in num_img_list]

# names_map, test_images = get_name2num_map(names_list, numbers_list, is_close_thresh=0.01, similarity_thresh=0.95,
#                                           resize=True)
res = get_name2num_map_v2(names_list, numbers_list, is_close_thresh=0.01, similarity_thresh=0.95,
                          resize=True)

pairs = {name: number for name, number in zip(name_img_list, num_img_list)}

for k, v in pairs.items():
    assert v == names_map[k]
