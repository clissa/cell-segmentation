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
#  Created on 6/16/21, 5:09 PM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """

import numpy as np
import pandas as pd
import skimage
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm

pd.options.display.max_columns = 8


def check_noise_in_masks(masks_path, debug_path):
    """
    Read ground-truth masks and save separate images for each object detected with skimage.measure.label()
    :param masks_path: Pathlib.Path() to masks folder
    :param debug_path: Pathlib.Path() where to store single objects (created if don't exist)
    :return:
    """
    # from scipy import ndimage
    for p in masks_path.iterdir():
        # read tif image
        mask = skimage.io.imread(p, as_gray=True)
        # mask[mask>0] = 255

        # find objects
        # scipy_label, n_objs = ndimage.label(mask)
        # objects = ndimage.find_objects(scipy_label)
        skimage_label = label(mask, connectivity=1, return_num=False)

        # initialize save_path
        save_path = debug_path / p.name
        # save_path = save_path.parent / "pippo3"
        save_path.mkdir(exist_ok=True, parents=True)
        # print(f"Creating directory {save_path}\n for debugging noisy objects.")
        # save objects separately
        for idx, obj in enumerate(regionprops(skimage_label)):
            outname = str(save_path / f"blob{idx}.png")
            plt.imsave(fname=outname, arr=skimage_label[obj._slice], cmap='gray')
        # for idx, obj in enumerate(objects):
        #     plt.imsave(fname=f"blob{idx}.jpeg", arr=scipy_label[obj], cmap='gray')
    return


def fix_mask_filename(filename):
    return filename.replace('_mask', "").replace('_MASK', "").replace('_y', '')


def remove_noise_from_masks(min_obj_size, connectivity, masks_path, out_path):
    for p in masks_path.iterdir():
        mask = skimage.io.imread(p, as_gray=True)
        # binary
        mask[mask > 0] = 255
        # remove small objects
        label = skimage.measure.label(mask, connectivity=connectivity)
        remove_small_objects(label, min_size=min_obj_size, connectivity=connectivity, in_place=True)
        label[label > 0] = 255

        # fix filename and change format
        out_path = out_path if out_path.name == "masks" else out_path / 'masks'
        out_path.mkdir(parents=True, exist_ok=True)
        filename = p.name.split('.')[0] + '.png'
        outname = str(out_path / fix_mask_filename(filename))
        plt.imsave(fname=outname, arr=label, cmap='gray')
    return


def compute_masks_stats(masks_path):
    """
    Read ground-truth masks and compute metrics for cell counts and shapes
    :param masks_path: Pathlib.Path() to masks folder
    :return:
    """
    stats_df = pd.DataFrame(data=None,
                            columns=['img_name', 'n_cells', 'cell_id', 'area', 'min_axis_length', 'max_axis_length',
                                     'equivalent_diameter', 'feret_diameter_max',
                                     # 'min_intensity', 'mean_intensity', 'max_intensity'
                                     ])
    paths = list(masks_path.iterdir())
    for idx_image, p in tqdm(enumerate(paths), total=len(paths)):
        mask = skimage.io.imread(p, as_gray=True)
        skimage_label, n_objs = label(mask, connectivity=1, return_num=True)
        for idx_obj, obj in enumerate(regionprops(skimage_label)):  # , coordinates='xy')):
            stats_df.loc[stats_df.shape[0] + 1] = [
                p.name.split('.')[0], n_objs, idx_obj, obj.area, obj.minor_axis_length, obj.major_axis_length,
                obj.equivalent_diameter, obj.feret_diameter_max,
                # obj.min_intensity, obj.mean_intensity, obj.max_intensity
            ]

    stats_df.round(4).to_csv(masks_path.parent / 'stats_df.csv', index=False)
    return stats_df


def get_yellow_filenames(source_paths):
    paths_set = set()
    names_set = set()
    for source in source_paths:
        source_paths_list = []
        for folder in source.iterdir():
            if (folder.is_dir() & (folder.name != 'all_images')):
                for filepath in folder.iterdir():
                    filename = filepath.name.split('.')[0]
                    # check if yellow image
                    if filename.endswith('_y'):
                        # check if already filename already in list
                        if filename not in names_set:
                            names_set = names_set.union(set([filename]))
                            source_paths_list.append(filepath)
        paths_set = paths_set.union(set(source_paths_list))
    return list(paths_set), names_set


def copy_yellow_originals(paths_list, out_path):
    for p in tqdm(paths_list, total=len(paths_list)):
        image = skimage.io.imread(p)

        # fix filename and change format
        if out_path.name == 'images':
            out_path = out_path
        else:
            out_path = out_path / 'images'
        out_path.mkdir(parents=True, exist_ok=True)
        filename = p.name.split('.')[0] + '.png'
        outname = str(out_path / fix_mask_filename(filename))
        skimage.io.imsave(fname=outname, arr=image, check_contrast=False)
    return


# TODO: not working with yellow images (because of image ranges?). Consider whether to extend it
def get_name2num_map(paths_name_list, paths_number_list, is_close_thresh=0.1, similarity_thresh=0.95, resize=False,
                     shape=(512, 512, 3)):
    """
    Retrieve mapping between original and numbered filenames by pixelwise comparison based on np.isclose and
    similarity threshold.
    :param paths_name_list: list of paths to images with original names 
    :param paths_number_list: list of paths to images with numbered names
    :param is_close_thresh: threshold for np.isclose()
    :param similarity_thresh: percentage threshold for similarity
    :param resize: (bool) whether to resize original images (default: False)
    :param shape: tuple (H, W, C) indicating the shape for resizing if `resize=True` (default=(512, 512, 3))
    :return: 
    """
    paths_name_list = paths_name_list.copy()
    paths_number_list = paths_number_list.copy()
    names_map = {}
    test_images = []
    for p_name in tqdm(paths_name_list, total=len(paths_name_list)):
        image_name = skimage.io.imread(p_name)
        if resize:
            image_name = skimage.transform.resize(image_name, shape)  # [0,1] format
            scale = 255.
        else:
            scale = 1.
        for p_number in paths_number_list:
            image_number = skimage.io.imread(p_number) / scale  # [0, 1] format
            try:
                # custom np.allclose(): the atol threshold can be tuned and a similarity threshold is added.
                # Experiments suggest that `is_close_thresh`=0.01 leads to similarity>0.97 with matching image
                # while >~0.6 with the others
                similarity = np.sum(np.isclose(image_name, image_number, atol=is_close_thresh)) / np.prod(shape)
                same_image = similarity > similarity_thresh
                # same_image = np.allclose(image_name, image_number)
            except ValueError:
                similarity = np.sum(
                    np.isclose(image_name[:, :, :3], image_number[:, :, :3], atol=is_close_thresh)) / np.prod(shape)
                same_image = similarity > similarity_thresh
                # same_image = np.allclose(image_name[:, :, :3], image_number[:, :, :3])
            if same_image:
                names_map[p_name.name] = p_number.name
                paths_number_list.remove(p_number)
                continue
        if names_map.get(p_name.name, None) is None:
            test_images.append(p_name.name)
    return names_map, test_images


# TODO: create customisable pre-processing script to clean the masks + compute stats
if __name__ == '__main__':
    from fluocells.config import DATA_PATH_r, DATA_PATH_y

    # red
    DATA_PATH_r = DATA_PATH_r / 'original'
    IMG_PATH = DATA_PATH_r / 'images'
    MASKS_PATH = DATA_PATH_r / 'masks'
    DATA_DEBUG_PATH = DATA_PATH_r / 'debug'
    DATA_DEBUG_PATH.mkdir(exist_ok=True, parents=True)
    DATA_PATH_new_version = DATA_PATH_r.parent / 'v1.0'

    remove_noise_from_masks(100, 1, MASKS_PATH, DATA_PATH_new_version)
    check_noise_in_masks(DATA_PATH_new_version / 'masks', DATA_PATH_new_version / 'debug')
    _ = compute_masks_stats(DATA_PATH_new_version / 'masks')

    # yellow
    # BASE_PATH = Path('/media/luca/Elements')
    # SOURCE1 = BASE_PATH / 'Immagini Immuno_Fisiologia'
    # SOURCE2 = BASE_PATH / 'INFN_conteggio gialli_06-2019/test_new_images'
    # all_paths, names = get_yellow_filenames([SOURCE1, SOURCE2])
    # DATA_PATH_y = Path('/home/luca/PycharmProjects/cells') / 'dataset/yellow/original'
    # copy_yellow_originals(all_paths, DATA_PATH_y)
    #
    DATA_PATH_y = DATA_PATH_y / 'original'
    _ = compute_masks_stats(DATA_PATH_y / 'masks')