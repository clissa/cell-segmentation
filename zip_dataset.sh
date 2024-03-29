#!/bin/bash
#
# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Luca Clissa
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CTRL=${3:-pass}
DEFAULT="preproc"

if [[ "$CTRL" == "$DEFAULT" ]];
then
  # generate annotation dfp
  echo "Generating annotations in multiple formats"
  python get_annotations_df.py $1 $2

  # compute objects stats
  echo "Computing objects stats"
  python get_stats_df.py $1 $2
fi

# zip dataset
cd dataset
zip $1_$2.zip $1/$2/masks/* $1/unlabelled/* $1/original/images/* $1/$2/stats_df.csv $1/labels.csv $1/$2/README.md
