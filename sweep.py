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
from pathlib import Path
import wandb
import yaml
import argparse

from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.distributed import *
from fluocells.utils.wandb import *

parser = argparse.ArgumentParser(description='Setup and run a W&B sweep for hyperparamenters comparison.')
parser.add_argument('--proj_name', type=str, help='Name of the W&B project', default='fluocells')
parser.add_argument('--alias', type=str, default='latest', help="Alias for the W&B artifact. Default: 'latest'")
parser.add_argument('-ds', '--dataset', type=str, default='red',
                    help="Name of the dataset to be uploaded. Values: 'red'(default)|'yellow'")
parser.add_argument('-art_name', '--artifact_name', type=str, default='',
                    help="Name of the W&B artifact. The default will create a name as f'fuocells-{dataset}',"
                         "; f'{artifact_name}-{dataset}' otherwise")
parser.add_argument('--crops', type=str, default='', help="Crops dataset folder. Default '', i.e., no crops")
parser.add_argument('--config', type=str, default='cfg_workers.yaml',
                    help="Relative path to the configuration file. Note: only `yaml` files are supported. Default: `cfg_workers.yaml`")
parser.add_argument('--function', type=str, default='train',
                    help="Function to use for the sweep. It must match one of the funcitons defined in the script. Default: `train`")
parser.add_argument('--count', type=int, default=50, help="Number of iterations for the W&B agent. Defalut: 50")
args = parser.parse_args()

DATASET = f"{args.dataset}-{args.crops}" if args.crops else args.dataset
print('Dataset name:', DATASET)


# sweep functions
def train(config=None, dataset=DATASET, alias=args.alias):
    with wandb.init(config=config, job_type='sweep_augmentation') as run:
        config = wandb.config
        train_path, val_path = rank0_first(_get_wb_datasets, run=run, prefix='fluocells-red', alias='latest')
        # old version _get_wb_datasets(run, prefix='fluocells-red', alias='latest')
        pre_tfms = [
            #     IntToFloatTensor(div_mask=255),
            Resize(config.resize1)
        ] if config.resize1 else []
        tfms = [
            IntToFloatTensor(div_mask=255),  # need masks in [0, 1] format
            *aug_transforms(
                size=config.resize2,
                max_lighting=0.1, p_lighting=0.5,
                min_zoom=0.9, max_zoom=1.1,
                max_warp=0, max_rotate=15.0)
        ]
        dls = _make_dataloader(train_path, val_path, pre_tfms=pre_tfms, tfms=tfms, cfg=config)
        # dls = _make_dataloader(run, config, prefix=dataset, alias=alias)
        learn = _make_learner(dls, cfg=config)
        lr = config.lr
        model_save_name = f"{config.encoder}_{config.loss_func}_lr{lr:.6}"
        # save_cb = SaveModelWithEpochCallback(fname=model_save_name, at_end=True)
        save_cb = SaveModelCallback(monitor='valid_loss', fname=model_save_name, at_end=True, with_opt=True)
        if config.to_log == 'None': config.to_log = None
        wandb_cb = WandbCallback(log=config.to_log,
                                 log_dataset=False,
                                 log_model=config.log_model
                                 )
        cbs = [wandb_cb, save_cb]
        learn = _train_learner_with_args(learn, config.one_cycle, config.multi_gpu, n_epoch=config.epochs, lr_max=lr,
                                         cbs=cbs)
        print('Finished Training')
        return learn


def augmentation(config=None, dataset=DATASET, alias=args.alias):
    with wandb.init(config=config, job_type='sweep_augmentation') as run:
        import random

        config = wandb.config
        # train_path, val_path = (Path('/home/luca/PycharmProjects/cell-segmentation/dataset/fluocells-red-split/train'),
        #                         Path('/home/luca/PycharmProjects/cell-segmentation/dataset/fluocells-red-split/valid'))
        train_path, val_path = rank0_first(_get_wb_datasets, run=run, prefix='fluocells-red', alias='latest')
        fnames = _get_train_val_names(train_path, val_path)
        seed = config.seed
        n_samples = config.n_samples
        random.seed(seed)
        sample_paths = random.sample(fnames, n_samples)

        section_dict = {
            'resize': 'Resize',
            'random_resized_crop': 'Resize',
            'zoom': 'Perspective',
            'warp': 'Perspective',
            'rotate': 'Perspective',
            'brightness': 'Colourspace',
            'contrast': 'Colourspace',
            'saturation': 'Colourspace',
            'hue': 'Colourspace',
        }

        # TODO:
        # - implement configurable augmentation pipeline
        # - setup W&B logging
        aug_dict = {}  # dict(original={},transforms={},)
        tfm = globals()[f"_{config.tfm}"]
        for img_path in sample_paths:
            img = PILImage.create(img_path)
            tfms = tfm(img)
            # log one section per each image composed by original image + augmented versions
            wandb.log({f"{section_dict[config.tfm]}/{config.tfm.title()}": [
                                                                               wandb.Image(img, caption=img_path.name,
                                                                                           grouping=img_path.name)] + [
                                                                               wandb.Image(tfmd,
                                                                                           caption=cpt,
                                                                                           grouping=img_path.name) for
                                                                               cpt, tfmd in tfms.items()]
                       }
                      )


# sweep helper
def _fit_sweep(proj_name, sweep_config, func, entity='lclissa', count=10):
    sweep_id = wandb.sweep(sweep_config, project=proj_name)
    wandb.agent(sweep_id, function=func, entity=entity, count=count)


if __name__ == '__main__':
    cwd = Path.cwd()
    p_config = cwd / args.config

    # check config format
    assert str(p_config).endswith(
        'yaml'), f"Unsupported format for configuration:\n{p_config}\n\nPlease provide a config file in `yaml` format."

    print(f'\nReading configuration file:\n{p_config}\n\n')
    # load config file
    with open(p_config) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    function = globals()[args.function]
    _fit_sweep(args.proj_name, sweep_config=config, func=function, count=args.count)
