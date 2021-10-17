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

# add repo root path to pythonpath
sys.path.insert(0, str(Path.cwd().parent))

# custom
import wandb
import random
from fluocells.config import REPO_PATH

parser = argparse.ArgumentParser(description='Create a data split for an existing W&B.')
parser.add_argument('proj_name', type=str, help='Name of the W&B project')
parser.add_argument('--alias', type=str, default='latest', help="Alias for the W&B artifact. Default: 'latest'")
parser.add_argument('-ds', '--dataset', type=str, default='red',
                    help="Name of the dataset to be uploaded. Values: 'red'(default)|'yellow'")
parser.add_argument('-art_name', '--artifact_name', type=str, default='',
                    help="Name of the W&B artifact. The default will create a name as f'fuocells-{dataset}',"
                         "; f'{artifact_name}-{dataset}' otherwise")
parser.add_argument('--crops', type=str, default='', help="Crops dataset folder. Default '', i.e., no crops")
parser.add_argument('--seed', type=int, default=2, help="Random seed for data split")
args = parser.parse_args()


def _get_img_mask_path_pairs(img_mask_dir: Path):
    """Return a list with pairs of image/mask Path."""
    return list(zip(list((img_mask_dir / 'images').iterdir()), list((img_mask_dir / 'masks').iterdir())))


def _get_unsplitted(orig_list, split):
    """Return list elements that were not included in the previous split."""
    return list(set(orig_list) - set(split))


def _add_file(artifact, path):
    """Add file at given path to the artifact."""
    artifact.add_file(path, name='/'.join(path.parts[-2:]))


PREFIX = f"{args.artifact_name}-{args.dataset}" if args.artifact_name else f"fluocells-{args.dataset}"
if args.crops:
    PREFIX = f"{PREFIX}-{args.crops}"

def main():
    with wandb.init(project=args.proj_name, job_type="data_split") as run:
        # TODO: check if this breks when alias==`latest` is updated
        artifact_ref = f"{PREFIX}:{args.alias}"
        # find the most recent ("latest") version of the full raw data
        ds = run.use_artifact(artifact_ref)
        # download locally (if not present)
        data_dir = ds.download(root=REPO_PATH / 'dataset' / artifact_ref)
        img_mask_pairs = _get_img_mask_path_pairs(data_dir)

        # create balanced train/val/test splits with proportions ~ 70/20/10 %
        # each count is the number of images per label
        n_crops = 12 if args.crops else 1
        n_imgs_split = [60, 16, 8] if args.dataset == 'red' else [200, 55, 28]
        n_imgs_split = [x * n_crops for x in n_imgs_split]
        DATA_SPLITS = {"train": n_imgs_split[0], "val": n_imgs_split[1], "test": n_imgs_split[2]}
        random.seed(args.seed)
        artifacts = {}
        # wrap artifacts in dictionary for convenience
        for split, count in DATA_SPLITS.items():
            artifacts[split] = wandb.Artifact("_".join([PREFIX, split, "data"]),
                                              "_".join([split, "data"]))

            # get split
            split_data = random.sample(img_mask_pairs, count)
            # update unsplitted data list
            img_mask_pairs = _get_unsplitted(img_mask_pairs, split_data)

            #  add "count" images per class
            for img_mask_pair in split_data:
                # add image
                _add_file(artifact=artifacts[split], path=img_mask_pair[0])
                # add mask
                _add_file(artifact=artifacts[split], path=img_mask_pair[1])
        # save all three artifacts to W&B
        for split, artifact in artifacts.items():
            run.log_artifact(artifact)


if __name__ == '__main__':
    main()
