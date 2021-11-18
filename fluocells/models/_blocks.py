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
__all__ = ['_get_ltype', 'Add', 'Concatenate', 'ConvBlock', 'ConvResNetBlock', 'ResNetBlock', 'UpResNetBlock',
           'Heatmap']

from fastai.vision.all import *


# Utils
def _get_ltype(layer):
    name = str(layer.__class__).split("'")[1]
    return name.split('.')[-1]


# Blocks
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.add

    def forward(self, x1, x2):
        return self.add(x1, x2)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.cat = partial(torch.cat, dim=dim)

    def forward(self, x):
        return self.cat(x)


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(n_in),
            nn.ELU(),
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out),
            nn.ELU(),
            nn.Conv2d(n_out, n_out, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.block(x)


class ConvResNetBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
        super(ConvResNetBlock, self).__init__()
        self.conv_block = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.short_connect = nn.Conv2d(n_in, n_out, kernel_size=1, padding=0)
        self.resnet_block = Add()

    def forward(self, x):
        conv_block = self.conv_block(x)
        short_connect = self.short_connect(x)
        resnet_block = self.resnet_block(conv_block, short_connect)
        return resnet_block


class ResNetBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv_block = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.resnet_block = Add()

    def forward(self, x1, x2):
        conv_block = self.conv_block(x1)
        resnet_block = self.resnet_block(conv_block, x2)
        return resnet_block


class UpResNetBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResNetBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            n_in, n_out, kernel_size=2, stride=2, padding=0)
        self.concat = Concatenate(dim=concat_dim)
        self.conv_block = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.up_resnet_block = Add()

    def forward(self, x, long_connect):
        short_connect = self.up_conv(x)
        concat = self.concat([short_connect, long_connect])
        up_resnet_block = self.up_resnet_block(
            self.conv_block(concat), short_connect)
        return up_resnet_block


class Heatmap(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(Heatmap, self).__init__()
        self.conv_block = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv_block(x))
