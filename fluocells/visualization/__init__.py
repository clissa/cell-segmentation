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
from fastai.vision.all import *
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fluocells.utils.data import label_func, get_input_image
from fluocells.models._utils import load_model, get_features

model = load_model('c-ResUnet')


def plot_heatmap(img: torch.Tensor, mask: torch.Tensor, heatmap: torch.Tensor, fig=None, axes=None, show=False):
    """
    Plot original image with mask's contours besides predicted heatmap
    :param img: imgut image in torch format (B, C, H, W)
    :param mask: corresponding ground-truth mask
    :param heatmap: predicted heatmap
    :param show: wheter to show the output
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
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

    # get colorbar to set params
    cbar = fig.colorbar(im, cax=cbar_ax)
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
    :param show: wheter to show the output
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


def plot_featuremap(img: torch.Tensor, features_dict: dict, layers: list, show=False) -> plt.Figure:
    """
    Plots a grid with all of the features computed by the layers for the input img. Returns the figure
    :param img: input image in torch format
    :param features_dict: dictionary containing layer names as keys and correspondent activations as values (according to the output of `get_features`
    :param layers: list of layers
    :param show: wheter to show the output
    :return:
    """
    fig_shape_dict = {1: (1, 1, (8, 8)),
                      16: (4, 4, (12, 12)),
                      32: (4, 8, (16, 16)),
                      64: (8, 8, (20, 20)),
                      128: (16, 8, (24, 24))
                      }

    # TODO: implement plotting for multiple layers in one call
    # for l in layers:
    #     features = features_dict[l]
    l = layers[0]
    features = features_dict[l]
    n_rows, n_cols, figsize = fig_shape_dict[features.shape[1]]
    row_spacing, col_spacing = 0.01, 0.01

    # GRIDSPEC
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.suptitle(
        f"{l}, feature size: {np.squeeze(features[:, 0, :, :]).shape}")
    gs = plt.GridSpec(n_rows, n_cols,
                      wspace=0, hspace=0,
                      height_ratios=[1 - col_spacing * n_rows] * n_rows,
                      left=0.01, right=0.9, top=0.95, bottom=0.05,
                      width_ratios=[1 - row_spacing * n_cols] * n_cols,
                      figure=fig,
                      )

    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(gs[i, j])
            feat_map = np.squeeze(
                features[:, n_cols * i + j, :, :]).to('cpu')
            im = ax.imshow(feat_map, cmap='jet')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    ### axes not available from gs, so need to proceed from figure
    # divider = make_axes_locatable(axes[1])
    # cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    fig.subplots_adjust(right=0.9)  # , left=0.05, top=0.95, bottom=0.05)
    cbar_ax = fig.add_axes([0.91, 0.05, 0.03, 0.9])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    # cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    if show:
        plt.show()
    return fig


def plot_featuremap_from_folder(img_folder: Path, layers: list, device: str = 'cpu', suptitle=True, show=True):
    """
    Plots a grid with all of the features computed by the layers for all the images in img_folder.
    :param img_folder: input image in torch format
    :param layers: list of layers
    :param device: whether to use the model on cpu or gpu (either 'cpu' or 'cuda', respectively)
    :param suptitle: whether to modify figure's suptitle
    :param show: wheter to show the output
    :return:
    """
    model.to(device)
    for idx, img_path in enumerate(img_folder.iterdir()):
        # get input image and mask
        img = get_input_image(img_path, device)
        mask_path = label_func(img_path)
        mask = np.asarray(load_image(mask_path, mode='L'))

        # original image + true objects
        # img_rgb = img.to('cpu').permute(0, 2, 3, 1)
        # fig = plt.figure(figsize=(12, 16))
        # plt.imshow(np.squeeze(img_rgb), cmap=plt.cm.RdBu,aspect="auto")
        # plt.contour(mask, [0.5], linewidths=1.2, colors='w')
        # plt.title(img_path.name)
        # plt.show()

        # predictions
        features = get_features(img, model, layers)
        fig = plot_featuremap(img, features, layers, show=False)
        if suptitle:
            layer_suptitle = f", {layers[0]}, feature size: {np.squeeze(features[layers[0]][:, 0, :, :]).shape}"
            fig.suptitle(img_path.name + layer_suptitle)
        if show:
            plt.show()
