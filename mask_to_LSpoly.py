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
#  Created on 5/18/21, 12:37 PM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 5/18/21, 12:33 PM
#  @author: Luca Clissa
#
#
#  Run using image_processing environment
#  """
import argparse
import json
from pathlib import Path

import skimage.io
from tqdm import tqdm

from fluocells.utils.conversion import format_annotation

parser = argparse.ArgumentParser(description='Convert json annotations in Label Studio polygon format to binary masks.')
parser.add_argument('masks_path', metavar='masks path', type=str,
                    help='Path to the json file containing Label Studio annotations')
parser.add_argument('--proj_id', metavar='folder', type=str, help='ID Label Studio project')
parser.add_argument('--annotator_id', metavar='Annotator information', type=str,
                    help='ID annotator in Label Studio project', default="")
parser.add_argument('--data_path', metavar='Data where images are fecthed by Label Studio', type=str,
                    help='Path where Label Studio fetches the data from', default="data/upload")

args = parser.parse_args()

MASKS_PATH = Path(args.masks_path)
if args.annotator_id != "":
    annotator_dict = {
        "id": 3, "email": "annotation_robot@unibo.it",
        "first_name": "Mr.", "last_name": "Annotator"
    }
else:
    annotator_dict = args.annotator_id

# build annotations from masks
output_json = []

for task_id, p in tqdm(enumerate(MASKS_PATH.iterdir()), total=len([p for p in MASKS_PATH.iterdir()])):
    mask = skimage.io.imread(p, as_gray=True)
    annotation_dict = format_annotation(p, mask, task_id, args.proj_id, annotator_dict, args.data_path)
    output_json.append(annotation_dict)

LABEL_PATH = MASKS_PATH.parent / 'labels'
LABEL_PATH.mkdir(parents=True, exist_ok=True)

with open(LABEL_PATH / 'annotations_local.json', 'w') as fp:
    json.dump(output_json, fp)
