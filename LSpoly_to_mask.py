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
#  Created on 5/18/21, 11:47 AM
#  @author: Luca Clissa
#
#
#  Run using fastai/image_processing environment
#  """
#
#  """
#  Created on 5/18/21, 10:37 AM
#  @author: Luca Clissa
#
#
#  Run using image_processing environment
#  """
if __name__ == '__main':
    import argparse
    import json
    from pathlib import Path
    import skimage
    from tqdm import tqdm
    from fluocells.utils.conversion import lstask2mask, lsname2orig

    parser = argparse.ArgumentParser(
        description='Convert json annotations in Label Studio polygon format to binary masks.')
    parser.add_argument('annotations_path', metavar='ann_path', type=str,
                        help='Path to the json file containing Label Studio annotations')
    parser.add_argument('--out_folder', metavar='folder', type=str,
                        help='Output folder')
    args = parser.parse_args()

    IMG_PATH = Path('/home/luca/PycharmProjects/cells/dataset/yellow/v1.0/images')

    # read annotations
    with open(args.annotations_path, 'r') as f:
        annotations = json.load(f)

    # configure output path
    outpath = Path(args.out_folder)
    outpath.mkdir(exist_ok=True, parents=True)
    print(f"Output folder created at: {outpath}")

    # reconstruct masks from annotations
    for annotation in tqdm(annotations, total=len(annotations)):
        # TODO: how to deal with multiple annotations? is it even possible to have such case?
        assert len(annotation['annotations']) == 1

        task_annotation = annotation['annotations'][0]['result']

        # reconstruct binary mask
        img_name = lsname2orig(annotation['file_upload'])
        img_path = IMG_PATH / img_name
        mask = lstask2mask(task_annotation, img_path=img_path)

        if mask is not None:
            # save output
            outname = outpath / img_name
            skimage.io.imsave(outname, skimage.img_as_ubyte(mask), check_contrast=False)
        else:
            print(f'Cannot find image: {img_name}. Skipped.')
