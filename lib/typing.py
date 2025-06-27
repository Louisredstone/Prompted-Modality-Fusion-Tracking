import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np

from torch import Tensor
from numpy import ndarray
from typing import Generic, TypeVar, Tuple
from typing import Literal, Any

'''
Actually this file is not so useful. Due to the problematic VSCode and its Pylance plugin, it is not very possible to write correct stub .pyi files or type hinters below for numpy arrays and torch tensors. Therefore, current version of this file only provides a non-strict type hinting for TorchTensor and NPArray. It doesn't work well when you try to hover on a variable or use it as hint for return type or parameter type of a function, but it still provides some information about the context.
'''

Dims = TypeVar("Dims", int, tuple[int,...])
DType = TypeVar("DType", type, np.dtype, torch.dtype)
Device = TypeVar("Device", str, torch.device)

class TorchTensor(Generic[Dims, DType, Device], Tensor):
    @classmethod
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        
        for i, param in enumerate(params):
            if not isinstance(param, int): break
        dims = params[:i]
        info = params[i:]
        dtype = torch.float32
        device = "cpu"
        if len(info) ==0: pass
        elif len(info) == 1:
            if isinstance(info[0], str): device = info[0]
            elif isinstance(info[0], type): dtype = info[0]
            else: raise TypeError("Invalid type of the second argument")
        elif len(info) == 2:
            assert isinstance(info[0], type) and isinstance(info[1], str), f"Wrong type at the tail of the tuple: {info}, should be (type, str)"
        else:
            raise ValueError("Invalid number of arguments")
        
        return super(TorchTensor, cls).__class_getitem__((dims, dtype, device),)

class NPArray(ndarray):
    @classmethod
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple): params = (params,)
        if isinstance(params[-1], type):
            dtype = params[-1]
            dims = params[:-1]
        elif isinstance(params[0], type):
            dtype = params[0]
            dims = params[1:]
        else:
            dtype = Any
            dims = params
        if len(dims) == 0:
            return ndarray[Any, dtype]
        else:
            return ndarray[tuple[*(Literal[i] for i in dims)], dtype]
    
if __name__ == '__main__':
    print(TorchTensor[5, 6, int, "gpu"])