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
import numpy as np
from fastai.vision.all import *
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fluocells.utils.data import Batch, label_func, get_input_image
from fluocells.models._utils import load_model

model = load_model('c-ResUnet')


def plot_heatmap(img: torch.Tensor, mask: torch.Tensor, heatmap: torch.Tensor, fig=None, axes=None, show=False):
    """
    Plot original image with mask's contours besides predicted heatmap
    :param img: imgut image in torch format (B, C, H, W)
    :param mask: corresponding ground-truth mask
    :param heatmap: predicted heatmap
    :return:
    """
    img = img.to('cpu').permute(0, 2, 3, 1)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # original image + true objects
    axes[0].imshow(np.squeeze(img),
                   cmap=plt.cm.RdBu, aspect="auto")
    axes[0].contour(mask, [0.5], linewidths=1.2, colors='w')
    axes[0].set_title('Original image and mask')

    # heatmap
    im = axes[1].imshow(heatmap, cmap='jet', aspect="auto")  # y-axis oriented as img
    # im = axes[1].pcolormesh(np.flipud(heatmap), cmap='jet')  # y-axis flipped upside-down
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # get colorbar to set params
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    # cbar.ax.ticks=([0, 0.2, 0.4, 0.6, 0.8, 0.9])
    axes[1].set_title('Predicted heatmap')
    if show:
        plt.show()
    return fig


def plot_heatmap_from_folder(img_folder: Path, suptitle: bool = True, device: str = 'cpu', show=True):
    """
    Loop through images folder and plot original with maks's contours besides predicted heatmap.
    :param img_folder: path where the images to be plotted are stored
    :param suptitle: whether to plot the image name as suptitle
    :param device: device to use for predictions
    :return:
    """
    model.to(device)
    for idx, img_path in enumerate(img_folder.iterdir()):

        # get input image and mask
        img = get_input_image(img_path, device)
        mask_path = label_func(img_path)
        mask = np.asarray(load_image(mask_path, mode='L'))

        # predictions
        with torch.no_grad():
            print('Feeding:', img.shape, img.dtype, repr(img.min()), repr(img.max()))
            heatmap = np.squeeze(model(img)).to('cpu')

        # plot
        fig = plot_heatmap(img, mask, heatmap, show=False)
        if suptitle: fig.suptitle(img_path.name)
        if show:
            plt.show()
