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
__all__ = ['_get_train_val_names', '_get_wb_datasets', '_make_dataloader', '_make_learner', '_train_learner_with_args',
           '_compose_tfms_from_config', '_update_config', 'wandb_parser', '_init_config', 'wandb_session']

import random
from pathlib import Path

from fastai.vision.all import *
from fluocells.config import REPO_PATH
from fluocells.losses import *
import argparse
from functools import wraps
from inspect import getfullargspec, isfunction
from itertools import starmap

wandb_parser = argparse.ArgumentParser(
    description='Parent parser to initialize arguments related to W&B project and data.', add_help=False)
group = wandb_parser.add_argument_group('wandb')
group.add_argument('--proj_name', type=str, help='Name of the W&B project', default='fluocells')
group.add_argument('--alias', type=str, default='latest', help="Alias for the W&B artifact. Default: 'latest'")
group.add_argument('-ds', '--dataset', type=str, default='red',
                   help="Name of the dataset to be uploaded. Values: 'red'(default)|'yellow'")
group.add_argument('-art_name', '--artifact_name', type=str, default='',
                   help="Name of the W&B artifact. The default will create a name as f'fuocells-{dataset}',"
                        "; f'{artifact_name}-{dataset}' otherwise")
group.add_argument('--crops', type=str, default='', help="Crops dataset folder. Default '', i.e., no crops")


def _get_train_val_names(pTr, pVal):
    """Return list with stacked train/val image paths"""
    return get_image_files(pTr / 'images') + get_image_files(pVal / 'images')


def _get_wb_datasets(run, prefix="fluocells-red", alias='latest'):
    """Download train/val datasets artifact. Return (train_path, val_path): paths to datasets"""
    # download artifact and set paths
    # fluocells-red_train_data_60:v0
    train_artifact_ref = f"{prefix}_train_data:{alias}"
    # fluocells-red_val_data_16:v0
    val_artifact_ref = f"{prefix}_val_data:{alias}"
    # find the most recent ("latest") version of the full raw data
    train_ds = run.use_artifact(train_artifact_ref)
    val_ds = run.use_artifact(val_artifact_ref)
    # download locally (if not present)
    train_path = train_ds.download(
        root=REPO_PATH / 'dataset' / f"{train_artifact_ref.split('_')[0]}-split" / 'train')
    val_path = val_ds.download(
        root=REPO_PATH / 'dataset' / f"{val_artifact_ref.split('_')[0]}-split" / 'valid')
    return train_path, val_path


def _make_dataloader(train_path, val_path, tfms=[], pre_tfms=[], config=None):
    """Download dataset artifact and setup dataloaders according to configuration parameters. Return dls: DataLoaders"""

    def label_func(p):
        return Path(str(p).replace('images', 'masks'))

    if isinstance(config, dict):
        config = namedtuple("WBConfig", config.keys())(*config.values())

    splitter = GrandparentSplitter(train_name='train', valid_name='valid')

    try:
        n_workers = config.dls_workers
    except:
        n_workers = 0

    dls = SegmentationDataLoaders.from_label_func(
        train_path.parent, bs=config.batch_size, fnames=_get_train_val_names(train_path, val_path),
        label_func=label_func,
        splitter=splitter,  # RandomSplitter(0.2, 42),
        item_tfms=pre_tfms, batch_tfms=tfms,
        num_workers=n_workers,
    )
    return dls


def _make_learner(dls, config=None):
    """Use the input dataloaders and configuration to setup a unet_learner with desired parameters. Return learn:
    Learner and updates config.lr if None"""

    model = globals()[config.encoder]
    optimizer = globals()[config.optimizer]
    loss_func = globals()[config.loss_func]()

    learn = unet_learner(dls, arch=model,
                         loss_func=loss_func,
                         opt_func=optimizer,
                         metrics=[Dice(), JaccardCoeff(), foreground_acc],
                         #                      cbs=EarlyStoppingCallback(monitor='dice', min_delta=0, patience=2),
                         cbs=[ActivationStats(
                             with_hist=True, every=4), CSVLogger()],
                         path=REPO_PATH / 'trainings', model_dir='models',
                         pretrained=config.pretrained,
                         n_out=2
                         )  # .to_fp16()

    learn.model_dir = learn.model_dir + "/" + learn.loss_func.name
    print(
        f'Logs save path: {learn.path}\nModel save path: {learn.path / learn.model_dir}')

    if config.lr is None:
        lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(
            suggest_funcs=(minimum, steep, valley, slide))
        config.lr = max(lr_valley, lr_steep)
        print(
            f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}\nSlide "
            f"interval:\t{lr_slide:.2e}")
    # else:
    #     print(f"Learning rate: {config.lr}")
    print(f"Using LR={config.lr:.6}")
    return learn


def _get_fitter_name(method_str):
    return method_str.split('.')[1].split(' ')[0]


def _train_learner_with_args(learn, one_cycle=False, multi_gpu=False, **kwargs):
    """Wrapper for training configurations depending on one cycle policy and gpus. Training params are passed as kwargs."""

    fit_func = getattr(learn, "fit_one_cycle") if one_cycle else getattr(learn, "fit")
    print(f"\nPerforming fit using {_get_fitter_name(fit_func.__str__())} and {'multi' if multi_gpu else 'single'} gpu")
    if multi_gpu:
        with learn.distrib_ctx():
            fit_func(**kwargs)
    else:
        fit_func(**kwargs)
    return learn


def _compose_tfms_from_config(tfm: str, args: dict):
    """Transform config dictionary with key, values items to fastai transform"""
    import fastai.vision.augment as aug
    return getattr(aug, tfm)(**args)


def _update_config(CLI_args, default_config):
    """Update default config with user defined arguments from command line"""
    for k in dir(CLI_args):
        if not k.startswith('__'):
            default_config[k] = getattr(CLI_args, k)
    return default_config


def autoassign(*names, **kwargs):
    """
    Adapted to newer python version from https://code.activestate.com/recipes/551763/
    autoassign(function) -> method
    autoassign(*argnames) -> decorator
    autoassign(exclude=argnames) -> decorator

    allow a method to assign (some of) its arguments as attributes of
    'self' automatically.  E.g.

    >>> class Foo(object):
    ...     @autoassign
    ...     def __init__(self, foo, bar): pass
    ...
    >>> breakfast = Foo('spam', 'eggs')
    >>> breakfast.foo, breakfast.bar
    ('spam', 'eggs')

    To restrict autoassignment to 'bar' and 'baz', write:

        @autoassign('bar', 'baz')
        def method(self, foo, bar, baz): ...

    To prevent 'foo' and 'baz' from being autoassigned, use:

        @autoassign(exclude=('foo', 'baz'))
        def method(self, foo, bar, baz): ...
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l: filter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and isfunction(names[0]):
        f = names[0]
        sieve = lambda l: l
    else:
        names, f = set(names), None
        sieve = lambda l: filter(lambda nv: nv[0] in names, l)

    def decorator(f):

        fargnames, _, _, fdefaults, _, _, _ = getfullargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(zip(reversed(fargnames), reversed(fdefaults))))

        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(zip(fargnames, args)))
            assigned.update(sieve(kwargs.items()))
            for _ in starmap(assigned.setdefault, defaults): pass
            self.__dict__.update(assigned)
            return f(self, *args, **kwargs)

        return decorated

    return f and decorator(f) or decorator


def _get_action_group(parser, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            return group


def _init_config(parser, args):
    """Helper function to initialize a configuration object with CLI args specified by the user"""
    experiment_config_group = _get_action_group(parser, 'experiment configuraton')
    attr_to_store = [action.dest for action in experiment_config_group._group_actions[:-1]]

    class Configurator:
        @autoassign(*attr_to_store)
        def __init__(self, **kwargs):
            pass

        def __repr__(self):
            incipit = "Configurator object:\n"
            attrs = [f"{attr}: {getattr(self, attr)}" for attr in dir(self) if not attr.startswith('__')]
            return incipit + '\n'.join(attrs)

    config = Configurator(**vars(args))
    return config


def wandb_session(f):
    @wraps(f)
    def run_session(config=None):
        import wandb
        from fluocells.utils import free_memory
        with wandb.init(project='fluocells', config=config, job_type='experiment',
                        group=f.__name__.replace('_', ' ').title()) as run:
            config = wandb.config
            res_dict = f(config)
            wandb.log(res_dict['metrics'])
            free_memory(['res_dict'], debug=False)
        return
    return run_session
