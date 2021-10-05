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
"""
Created on Tue May  7 10:42:13 2019
@author: Luca Clissa
"""
import argparse
# OS related
import sys
from pathlib import Path

# custom
import wandb
from fluocells.config import IMG_PATH_r, IMG_PATH_y

# add repo root path to pythonpath
sys.path.insert(0, str(Path.cwd().parent))

parser = argparse.ArgumentParser(
    description='Create a W&B project and upload labelled and unlabelled images for given dataset.')
parser.add_argument('proj_name', type=str,
                    help='Name of the W&B project')
parser.add_argument('-ds', '--dataset', type=str, default='red',
                    help="Name of the dataset to be uploaded. Values: 'red'(default)|'yellow'")
parser.add_argument('-art_name', '--artifact_name', type=str, default='',
                    help="Name of the W&B artifact. The default will create a name as f'fuocells-{dataset}',"
                         "; f'{artifact_name}-{dataset}' otherwise")
parser.add_argument('--alias', type=str, default='latest', help="Alias for the W&B artifact. Default: 'latest'")
parser.add_argument('--crops', type=str, default='',
                    help="Name of the crops folder if the target dataset contains crops. Default: '', i.e., "
                         "full size images")
args = parser.parse_args()
# retrieve data from local source
IMG_PATH = globals()[f"IMG_PATH_{args.dataset[0]}"]
if args.crops:
    ds_dict = {
        'images': IMG_PATH.parent.parent / args.crops / 'images',
        'masks': IMG_PATH.parent.parent / args.crops / 'masks'
    }
else:
    ds_dict = {
        'unlabelled': IMG_PATH.parent.parent / 'unlabelled',
        'images': IMG_PATH,
        'masks': IMG_PATH.parent.parent / 'v1.0' / 'masks'
    }


def main():
    # create a run in W&B
    with wandb.init(project=args.proj_name, job_type="upload") as run:
        # create an artifact for all the raw data
        PREFIX = f"{args.artifact_name}-{args.dataset}" if args.artifact_name else f"fluocells-{args.dataset}"
        ds = wandb.Artifact(PREFIX, type="raw_data")

        # loop through folders
        for folder_name, folder_path in ds_dict.items():
            ds.add_dir(folder_path, name=folder_name)
        if args.crops:
            ds.add_file(folder_path.parent / 'crops_map.csv')

        # save artifact to W&B
        run.log_artifact(ds, aliases=args.alias)


if __name__ == '__main__':
    main()
