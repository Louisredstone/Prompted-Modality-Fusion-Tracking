from .tensor import TensorDict, TensorList

import os

def is_subpath(path, parent_path):
    """
    Check if path is a subpath of parent_path.
    """
    try:
        return os.path.commonpath([path, parent_path]) == parent_path
    except ValueError:
        return False
    
from .object_accessor import ObjectAccessor