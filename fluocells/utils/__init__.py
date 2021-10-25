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
__all__ = ['get_less_used_gpu', 'free_memory']
from torch import cuda


def get_less_used_gpu():
    """Inspect cached/reserved and allocated memory on all gpus and return the id of the less used device"""
    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in range(cuda.device_count()):
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    print('Current allocated memory:', cur_allocated_mem)
    print('Current reserved memory:', cur_cached_mem)
    print('Maximum allocated memory:', max_allocated_mem)
    print('Maximum reserved memory:', max_cached_mem)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    print(min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu()

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu()
