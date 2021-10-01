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
#  Created on 7/7/21, 2:43 PM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
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
# OS related
import sys
from pathlib import Path

# Data manipulation
import pandas as pd

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30
pd.options.display.float_format = '{:,.4f}'.format

# add repo root path to pythonpath
sys.path.insert(0, str(Path.cwd().parent))

import fastai
from fastai.vision.all import *
from albumentations.augmentations.transforms import ToGray

from fluocells.config import REPO_PATH

print(
    f"Using versions:\n\nPyTorch: {torch.__version__}\nfastai: {fastai.__version__}")

parser = argparse.ArgumentParser(
    description='Run self-supervised learning with specified model architecture. Optionally choose whether to start '
                'from pretrained model or not.')
parser.add_argument('arch', metavar='model architecture', type=str,
                    help='Architecture to use for self-supervised training. To be chosen among the ones available in '
                         'fastai.')
parser.add_argument('--pretrained', metavar='pretrained bool', type=bool,
                    help='Whether to start from pretrained weight or train from scratch', default=True)
parser.add_argument('-bs', '--batch_size', default=32,
                    help='batch size to use in the dataloader')
parser.add_argument('-w', '--num_workers', default=8,
                    help='num_workers to use in the dataloader')
parser.add_argument('-r1', '--resize1', default=512,
                    help='first resize shape')
parser.add_argument('-r2', '--resize2', default=224,
                    help='second resize shape')
parser.add_argument('-e', '--epochs', default=6,
                    help='training epochs')

args = parser.parse_args()

# hyperparams

# augmentation
resize1 = args.resize1
resize2 = args.resize2
max_lighting, p_lighting = 0.1, 0.5
min_zoom, max_zoom = 0.9, 1.1
max_warp, max_rotate = 0, 15.0

# dataloader
bs = args.batch_size
n_workers = args.num_workers


# utils

class AlbumentationsToGray(Transform):
    def __init__(self, aug): self.aug = aug

    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        # ToGray returns 3 channels with same grey image repeated,
        # this causes conflicts with model training:
        # [b x (3 chan x image size) VS (b x 3 classes) x 1]
        aug_img = PILImage.create(aug_img[:, :, 0], mode='L')
        return aug_img


# read dataset df
DATA_PATH = REPO_PATH / 'dataset'
df_path = DATA_PATH / 'self_supervised_df.csv'
selfsuper_df = pd.read_csv(df_path)

print(
    f'\nReading dataset stored at:\n{df_path}\n\nDataset preview:\n')
print(selfsuper_df.head())

# setup augmentation pipeline
RGB2Grey = AlbumentationsToGray(ToGray(p=1))

tfms = [
    *aug_transforms(
        size=resize2,
        max_lighting=max_lighting, p_lighting=p_lighting,
        min_zoom=min_zoom, max_zoom=max_zoom,
        max_warp=max_warp, max_rotate=max_rotate)
]

# initialize dataloader
SSL = DataBlock(blocks=(ImageBlock(PILImageBW), CategoryBlock),
                splitter=ColSplitter(),
                get_x=lambda o: f'{DATA_PATH}/' + o.fname,
                get_y=lambda o: o.label,
                item_tfms=[Resize(resize1)],
                batch_tfms=tfms,
                )

dls = SSL.dataloaders(selfsuper_df, bs=bs, num_workers=n_workers)

# # initialize dataloader
# dls = ImageDataLoaders.from_df(selfsuper_df, DATA_PATH, valid_col='is_valid',  label_col='label',
#                                item_tfms=[RGB2Grey, Resize(resize1)], batch_tfms=tfms,
#                                bs=bs)

# initialize learner

# hyperparams
epochs = args.epochs

loss = CrossEntropyLossFlat()
metrics = [error_rate, accuracy]
early_stopping_patience = 10
arch = getattr(fastai.vision.models, args.arch)
# , raise(ValueError("Specified architecture not available. Please check spelling is in accordance to
# `fastai.vision.models` names."))
n_in = 3 if args.pretrained else 1

learn = cnn_learner(dls, arch,
                    loss_func=loss,
                    metrics=metrics,
                    cbs=[EarlyStoppingCallback(monitor='accuracy', patience=early_stopping_patience),
                         CSVLogger(fname=f'history_{args.arch}')],
                    pretrained=args.pretrained,
                    n_in=n_in,
                    path=REPO_PATH
                    )

# find optimal learning rate and train
lr = learn.lr_find(show_plot=False)
print('Learning rate:', lr)

import time

start_time = time.time()
learn.fit_one_cycle(epochs, lr.valley * 10)
end_time = time.time()
avg_time_min = (end_time - start_time) / 60 / epochs

for ep in range(len(learn.recorder.values)):
    learn.recorder.values[ep].insert(0, ep + 1)
    learn.recorder.values[ep].append(f"{avg_time_min:.2}mins")
train_stats = pd.DataFrame(learn.recorder.values, columns=learn.recorder.metric_names)
STATS_PATH = learn.path / learn.model_dir / 'train_stats'
STATS_PATH.mkdir(exist_ok=True)

model_outname = f"{args.arch}_{'pretrain' if args.pretrained else 'no-pretrain'}_membership"
train_stats.to_csv(STATS_PATH / (model_outname + '.csv'))
save_path = learn.save(model_outname)

print('Model saved at:\n\n', save_path)
