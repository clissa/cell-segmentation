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
__all__ = ["save_pkl", "load_pkl", "state_dict_Kformat", "copy_weights_k2pt", "get_layer_name"]

import functools
import pickle
from collections import OrderedDict

import torch


def save_pkl(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        return d


def load_model(arch: str = 'c-ResUnet', mode: str = 'eval'):
    from fluocells.models import c_resunet
    # arch = 'c-ResUnet_noWM'
    model = c_resunet(arch=arch, n_features_start=4,
                      n_out=1, pretrained=True)
    # for m in model.modules():
    #     for child in m.children():
    #         if type(child) == nn.BatchNorm2d:
    #             child.track_running_stats = False
    #             child.running_mean = None
    #             child.running_var = None
    if mode == 'eval':
        model.eval()
    return model


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def tfm_keras_weights(keras_w):
    k_w = keras_w.copy()
    if k_w.ndim == 4:  # convolution filter
        torch_w = torch.from_numpy(k_w.transpose((3, 2, 0, 1)))
    elif k_w.ndim == 1:  # convolution bias; batchnorm weight/gamma and bias/beta
        torch_w = torch.from_numpy(k_w)
    else:
        raise ValueError(
            f'Unexpected shape {k_w.shape} has dimension {k_w.ndim} instead of 1 or 4.')
    return torch_w


def copy_weights_k2pt(model, k_dict, pt_dict, freeze=True):
    for pt_key, k_weight in zip(pt_dict.keys(), k_dict.values()):
        if freeze:
            with torch.no_grad():
                rsetattr(model, f'{pt_key}.data', tfm_keras_weights(k_weight))
        else:
            rsetattr(model, f'{pt_key}.data', tfm_keras_weights(k_weight))


def state_dict_Kformat(d):
    """
    Return state_dict without PyTorch-specific layers. This makes it comparable with Keras weight format
    :param d: pytorch model state_dict
    :return: state_dict in Keras-like format
    """
    fixed = OrderedDict(
        {k: v for k, v in d.items() if not 'num_batches_tracked' in k})
    return fixed
