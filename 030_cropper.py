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
#  Created on 5/17/21, 4:49 PM
#  @author: Luca Clissa
#
#
#  Run using image_processing environment
#  """

import argparse

import numpy as np
import pandas as pd
import skimage
import skimage.io

from fluocells.config import DATA_PATH_r, DATA_PATH_y

parser = argparse.ArgumentParser(description='Crop original images in square patches of customisable size.')
parser.add_argument('dataset', metavar='marker color', type=str,
                    help='Marker color (either red or yellow)')
parser.add_argument('--v_imgs', metavar='images version name', type=str, help='Version tag', default='original')
parser.add_argument('--v_masks', metavar='masks version name', type=str, help='Version tag', default='v1.0')
parser.add_argument('--crop_size', metavar='Crop size', type=int, help='Size of the crop', default=512)

args = parser.parse_args()


def cropper(img, crop_size=512):
    if len(img.shape) == 2:  # 2D image
        height, width = img.shape
    else:
        height, width, channels = img.shape
    vert_ncrop = np.ceil(height / crop_size)
    horiz_ncrop = np.ceil(width / crop_size)

    vert_excess_pixels = vert_ncrop * crop_size - height
    horiz_excess_pixels = horiz_ncrop * crop_size - width

    vert_shift = vert_excess_pixels / (vert_ncrop - 1)
    horiz_shift = horiz_excess_pixels / (horiz_ncrop - 1)

    vert_starting_pixels = [int(np.floor(px - vert_shift * (i + 1))) for i, px in
                            enumerate(range(crop_size, height, crop_size))]
    vert_starting_pixels.insert(0, 0)
    horiz_starting_pixels = [int(np.floor(px - horiz_shift * (i + 1))) for i, px in
                             enumerate(range(crop_size, width, crop_size))]
    horiz_starting_pixels.insert(0, 0)

    crop_list = []
    for v in vert_starting_pixels:
        for h in horiz_starting_pixels:
            if len(img.shape) == 2:  # 2D image
                crop = img[v:v + crop_size, h:h + crop_size]
            else:
                crop = img[v:v + crop_size, h:h + crop_size, :]
            crop_list.append(crop)
    return crop_list


def crop_images(img_path, mask_path, crop_size=512):
    from tqdm import tqdm
    # setup output folder
    crops_path = img_path.parent.parent / f'crops_{crop_size}/images'
    crops_path.mkdir(parents=True, exist_ok=True)
    crops_mask_path = mask_path.parent.parent / f'crops_{crop_size}/masks'
    crops_mask_path.mkdir(parents=True, exist_ok=True)

    image_names = [p.name for p in img_path.iterdir()]
    crops_map_df = pd.DataFrame({'image_name': [], 'crop_id': []})

    for idx_image, p in tqdm(enumerate(img_path.iterdir()), total=len(image_names)):

        # crop original image
        img = skimage.io.imread(p)
        crops = cropper(img, crop_size)
        for idx_crop, crop in enumerate(crops):
            outname = crops_path / f'{idx_image * len(crops) + idx_crop}.png'
            skimage.io.imsave(outname, crop, check_contrast=False)
            crops_map_df.loc[idx_image * len(crops) + idx_crop] = [p.name, idx_image * len(crops) + idx_crop]
        # crop mask
        img = skimage.io.imread(mask_path / p.name)
        crops = cropper(img, crop_size)
        for idx_crop, crop in enumerate(crops):
            outname = crops_mask_path / f'{idx_image * len(crops) + idx_crop}.png'
            skimage.io.imsave(outname, crop, check_contrast=False)

    return crops_map_df


if __name__ == '__main__':
    if args.dataset == 'red':
        IMG_PATH = DATA_PATH_r / f'{args.v_imgs}/images'
        MASKS_PATH = DATA_PATH_r / f'{args.v_masks}/masks'
    elif args.dataset == 'yellow':
        IMG_PATH = DATA_PATH_y / f'{args.v_imgs}/images'
        MASKS_PATH = DATA_PATH_y / f'{args.v_masks}/masks'
    else:
        raise ValueError("Invalid argument `dataset`. Supported values are `red` and `yellow`.")

    crops_df = crop_images(IMG_PATH, MASKS_PATH, args.crop_size)
    crops_df.to_csv(IMG_PATH.parent / 'crops_map.csv')

    targets = [p for p in MASKS_PATH.iterdir()]
    masks_with_cells = 0
    for p in MASKS_PATH.iterdir():
        m = skimage.io.imread(p)
        if len(np.unique(m)) > 1:
            masks_with_cells += 1

    print(
        f'Masks containing cells: {masks_with_cells} out of {len(targets)} ('
        f'{np.round(masks_with_cells / len(targets), 2)}%)')
