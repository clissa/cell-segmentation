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
#  Created on 6/14/21, 4:11 PM
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

import pandas as pd
import skimage.io
import skimage.measure
import skimage.transform
from tqdm import tqdm

from fluocells.config import DATA_PATH_g, DATA_PATH_y, DATA_PATH_r

parser = argparse.ArgumentParser(description='Crop original images in square patches of customisable size.')
parser.add_argument('dataset', metavar='marker color', type=str,
                    help='Marker color (green, red or yellow)')
parser.add_argument('version', metavar='dataset version name', type=str, help='Version tag')

args = parser.parse_args()
if args.dataset == 'red':
    IMG_PATH = DATA_PATH_r / f'{args.version}/images'
    MASKS_PATH = DATA_PATH_r / f'{args.version}/masks'
elif args.dataset == 'yellow':
    IMG_PATH = DATA_PATH_y / f'{args.version}/images'
    MASKS_PATH = DATA_PATH_y / f'{args.version}/masks'
elif args.dataset == 'green':
    IMG_PATH = DATA_PATH_g / f'{args.version}/images'
    MASKS_PATH = DATA_PATH_g / 'raw_data_green/all_masks/images'
else:
    raise ValueError("Invalid argument `dataset`. Supported values are `red` and `yellow`.")

# build annotations df
annotation_df = pd.DataFrame({}, columns=['img_name', 'count', 'id_obj', 'dot'])
if args.dataset == 'green':
    annotation_types = ['dot']
    names_map = pd.read_csv(DATA_PATH_g / 'names_map.csv')
    resize = True
    iterator = zip(names_map['original_name'], names_map['name_number'])
    iterations = names_map.shape[0]
else:
    annotation_types = ['dot', 'bounding_box', 'polygon']
    resize = False
    iterator = [p.name for p in MASKS_PATH.iterdir()]
    iterations = len(iterator)


def poly_coords(obj, mask_height, mask_width):
    import numpy as np
    coordinates = []
    # create black image an attach the object patch
    obj_mask = np.zeros([mask_height, mask_width], dtype=np.uint8)
    obj_mask[obj._slice] = obj.image

    # extend coordinates with all the contours segments of the object
    contours = skimage.measure.find_contours(
        obj_mask, fully_connected='high', positive_orientation='high')
    for contour in contours:
        coordinates.extend([(p[1], p[0]) for p in contour])

    # downsample contour coordinates to 40 points
    idx = np.round(np.linspace(0, len(coordinates) - 1, 40)).astype(int)
    coordinates = np.array(coordinates).round(2)
    coordinates = coordinates[idx].tolist()
    return coordinates


def bbox_coords(bbox):
    return [(bbox[1], bbox[0]), (bbox[3], bbox[2])]


cols = ['img_name', 'count', 'id_obj'] + annotation_types
annotation_df = pd.DataFrame({}, columns=cols)
for item in tqdm(iterator, total=iterations):
    # for name_img, num_img in tqdm(zip(name_img_list, num_img_list), total=len(name_img_list)):
    if len(item) == 2:
        img_name, mask_name = item
    else:
        img_name, mask_name = item, item
    if resize:
        img = skimage.io.imread(IMG_PATH / img_name)
        mask = skimage.io.imread(MASKS_PATH / mask_name, as_gray=True)
        mask_orig = skimage.transform.resize(mask, img.shape[:-1])
    else:
        mask_orig = skimage.io.imread(MASKS_PATH / mask_name, as_gray=True)
    label, n_objs = skimage.measure.label(mask_orig, return_num=True)

    for id_obj, obj in enumerate(skimage.measure.regionprops(label)):
        if len(annotation_types) == 1:
            annotations = [obj.centroid]
        else:
            annotations = [obj.centroid, bbox_coords(obj.bbox),
                           poly_coords(obj, mask_orig.shape[0], mask_orig.shape[1])]
        # initialize record
        record = pd.DataFrame([[img_name, n_objs, id_obj] + annotations],
                              columns=cols)
        annotation_df = pd.concat([annotation_df, record])

annotation_df.to_csv(IMG_PATH.parent.parent / 'labels.csv', index=False)

# # test retrieved annotations df
# import matplotlib.pyplot as plt
# import ast
# ann_df = pd.read_csv(DATA_PATH_r / 'labels.csv',converters={"dot": ast.literal_eval, "bounding_box":
# ast.literal_eval, "polygon": ast.literal_eval})
# ann_df.bounding_box[0]
#
# def draw_rectangle(ax, bbox):
#     bottom_left, top_right = bbox
#     bx = (bottom_left[0], bottom_left[0], top_right[0], top_right[0], bottom_left[0])
#     by = (bottom_left[1], top_right[1], top_right[1], bottom_left[1], bottom_left[1])
#     ax.plot(bx, by, '-C0',  linewidth=.8)
#
# for i in range(5):
#     img_id = i
#     im_name = ann_df.img_name.unique()[img_id]
#     imred = skimage.io.imread(DATA_PATH_r / 'v1.0/images' / im_name)
#     fig, ax = plt.subplots()
#     ax.imshow(imred)
#     im_df = ann_df.loc[ann_df.img_name==im_name]
#     for r in im_df.itertuples():
#         draw_rectangle(ax, r.bounding_box)
#     plt.show()
#
# def draw_poly(ax, poly):
#     bx = [p[0] for p in poly] + [poly[0][0]]
#     by = [p[1] for p in poly] + [poly[0][1]]
#     ax.plot(bx, by, '-C0',  linewidth=.8)
#
# for i in range(5):
#     img_id = i
#     im_name = ann_df.img_name.unique()[img_id]
#     imred = skimage.io.imread(DATA_PATH_r / 'v1.0/images' / im_name)
#     fig, ax = plt.subplots()
#     ax.imshow(imred)
#     im_df = ann_df.loc[ann_df.img_name==im_name]
#     for r in im_df.itertuples():
#         draw_poly(ax, r.polygon)
#     plt.show()
