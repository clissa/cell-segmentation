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
__all__ = ['ResUnet', 'c_resunet']

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
from fluocells.config import MODELS_PATH


class ResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=1):
        super(ResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        # colorspace transformation
        self.colorspace = nn.Conv2d(
            3, 1, kernel_size=1, padding=0)

        # block 1
        self.c1 = ConvBlock(1, n_features_start)
        self.p1 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 2
        self.c2 = ConvResNetBlock(n_features_start, 2 * n_features_start)
        self.p2 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 3
        self.c3 = ConvResNetBlock(2 * n_features_start, 4 * n_features_start)
        self.p3 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 4: BRIDGE START
        self.c4 = ConvResNetBlock(4 * n_features_start, 32 *
                                  n_features_start, kernel_size=5, padding=2)

        # block 5: BRIDGE END
        self.c5 = ResNetBlock(8 * n_features_start, 32 *
                              n_features_start, kernel_size=5, padding=2)

        # block 6
        self.c6 = UpResNetBlock(n_in=8 * n_features_start,
                                n_out=4 * n_features_start)

        # block 7
        self.c7 = UpResNetBlock(
            4 * n_features_start, 2 * n_features_start)

        # block 8
        self.c8 = UpResNetBlock(
            2 * n_features_start, n_features_start)

        # output
        self.output = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        c0 = self.colorspace(x)
        c1 = self.c1(c0)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        c5 = self.c5(c4, c4)
        c6 = self.c6(c5, c3)
        c7 = self.c7(c6, c2)
        c8 = self.c8(c7, c1)
        output = self.output(c8)

        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet(
        arch: str,
        n_features_start: int,
        n_out: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnet(n_features_start, n_out)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        weights_path = MODELS_PATH / f"{arch}_state_dict.pkl"
        print('loading pretrained Keras weights from', weights_path)
        keras_weights = load_pkl(weights_path)
        Klike_state_dict = state_dict_Kformat(model.state_dict())
        assert len(keras_weights) == len(Klike_state_dict)
        copy_weights_k2pt(model, keras_weights, Klike_state_dict)
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #         model.load_state_dict(state_dict)
    return model


def c_resunet(arch='c-ResUnet', n_features_start: int = 16, n_out: int = 1, pretrained: bool = False,
              progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet(arch=arch, n_features_start=n_features_start, n_out=n_out, pretrained=pretrained,
                    progress=progress, **kwargs)
