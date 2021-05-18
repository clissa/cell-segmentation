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
#  Created on 5/18/21, 11:49 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 5/18/21, 11:48 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 5/18/21, 11:47 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """

import datetime
from pathlib import Path

import numpy as np
import pytz
import skimage
import skimage.draw
import skimage.io
import skimage.measure


def lsname2orig(ls_name: str, endings: list = ['_r1', '_r', '_200x']):
    """
    Drop hash from Label Studio (LS) filename so to match files in the dataset.

    :param ls_name: LS annotation filename, i.e. ['annotation']['file_upload']
    :param endings: list of possible endings, default order guarantees dealing with both red and yellow datasets
    :return: LS filename truncated at first ending found + extension, or LS filename
    """
    orig_name = ls_name
    for ending in endings:
        if ls_name.find(ending) != -1:
            orig_name = ls_name[:ls_name.find(ending) + len(ending)] + ls_name[ls_name.find('.'):]
            break
    return orig_name


def lspoly2mask(points: list, h: int, w: int):
    """
    Transform object's coordinates from Label Studio (LS) polygon format to pixel coordinates for the actual mask.

    :param points: list of points coordinates in the LS format (i.e.: [[y1, x1], [y2, x2], ...], where x*,
    y* are expressed as percentages of the image shape
    :param h: image height
    :param w: image width
    :return: reconstructed binary mask [0, 255]
    """
    converted_points = [[h * p[1] / 100.0, w * p[0] / 100.0] for p in points]
    return converted_points


def lstask2mask(task_annotation: list, img_path=None):
    """
    Convert Label Studio (LS) image annotation to binary mask.

    :param task_annotation: list containing LS annotations for given task, i.e. ['annotations'][0]['result']
    :param img_path: pathlib.Path where to read original image when is not possible to infer height and width from
    annotation (default: None)
    :return: reconstructed binary mask [0, 255] or None if impossible to reconstruct
    """
    # TODO: create config script with data paths for red/yellow

    if len(task_annotation) >= 1:
        img_h = task_annotation[0]['original_height']
        img_w = task_annotation[0]['original_width']

        reco_mask = np.zeros([img_h, img_w], dtype=np.uint8)
        image_shape = (img_h, img_w)

        for obj in task_annotation:
            try:
                obj['value']['points']
            except KeyError:
                return obj['value']['points']
            polygon = np.array(lspoly2mask(obj['value']['points'], img_h, img_w))
            obj_mask = skimage.draw.polygon2mask(image_shape, polygon)
            reco_mask += obj_mask
    elif image_path is not None:
        # when no annotations are present then read the to get its shape
        img = skimage.io.imread(img_path)
        reco_mask = np.zeros(img.shape, dtype=np.uint8)
    else:
        return None

    # transform 0-1 binary to 0-255 range
    reco_mask[reco_mask > 0] = 255
    return reco_mask


def mask2lspoly(contour: list, h: int, w: int):
    """
    Transform object's contour coordinates in the mask to Label Studio (LS) polygon format for the json annotation.

    :param contour: list of points coordinates (i.e.: [[y1, x1], [y2, x2], ...], where x*, y* are expressed as pixel
    in the mask
    :param h: image height
    :param w: image width
    :return:
    """
    converted_points = [[100 * p[1] / w, 100 * p[0] / h] for p in contour]
    return converted_points


annotator_dict = {
    "id": 1, "email": "annotation_robot@unibo.it",
    "first_name": "Mr.", "last_name": "Annotator"
}


def format_annotation(p: Path, mask: np.array, task_id: int, proj_id: int = 2, annotator_dict: dict = annotator_dict,
                      data_path: str = "data/upload"):
    """
    Return dictionary of annotations from binary mask in a format compatible with Label Studio.

    :param p: path to image
    :param mask: binary mask
    :param task_id: task id for Label Studio
    :param proj_id: id of Label Studio project
    :param annotator_dict: dictionary with info about the annotator in the Label Studio project
    :return:
    """
    import time

    t1 = time.time()
    tmstmp_start = datetime.datetime.now(tz=pytz.utc).astimezone().isoformat()

    mask_width = mask.shape[1]
    mask_height = mask.shape[0]

    # annotations will be displayed all with id=0
    annotation_id = 1
    output_dict = {
        "id": task_id,
        "annotations": [],
        "predictions": [],
        "file_upload": p.name,
        "data": {"image": f"{data_path}/{p.name}"},
        "meta": {},
        "created_at": tmstmp_start,
        "updated_at": tmstmp_start,
        "project": proj_id
    }

    result = []

    # iterate over objects to get unique contours for each object
    label, n_objs = skimage.measure.label(mask, return_num=True)
    for id_obj, obj in enumerate(skimage.measure.regionprops(label)):
        LS_coordinates = []
        # add contour only if area is greater than 100 pixels
        if obj.area > 100:
            # create black image an attach the object patch
            obj_mask = np.zeros([mask_height, mask_width], dtype=np.uint8)
            obj_mask[obj._slice] = obj.image

            # extend LS_coordinates with all the contours segments of the object
            contours = skimage.measure.find_contours(
                obj_mask, fully_connected='high', positive_orientation='high')
            for contour in contours:
                LS_coordinates.extend(mask2lspoly(
                    contour, mask_height, mask_width))

            # downsample contour coordinates to 40 points
            idx = np.round(np.linspace(0, len(LS_coordinates) - 1, 40)).astype(int)
            LS_coordinates = np.array(LS_coordinates).round(2)
            LS_coordinates = LS_coordinates[idx].tolist()

            # populate object annotation dictionary
            contour_obj = {
                "value": {
                    "points": LS_coordinates,
                    "polygonlabels": [
                        "Cell"
                    ]
                },
                "original_width": mask_width,
                "original_height": mask_height,
                "image_rotation": 0,
                "id": str(id_obj),
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels"
            }

            result.append(contour_obj)

    # tmstmp_stop = datetime.datetime.now(tz=pytz.utc).astimezone().isoformat()
    t2 = time.time()

    annotations = {
        "id": annotation_id,
        "state": {},
        "result": result,
        "was_cancelled": False,
        "ground_truth": True,
        "created_at": tmstmp_start,
        "updated_at": tmstmp_start,
        "lead_time": np.round(t2 - t1, 3),
        "prediction": {},
        "result_count": n_objs,
        "task": task_id,
        "completed_by": annotator_dict
    }

    output_dict["annotations"].append(annotations)
    return output_dict
