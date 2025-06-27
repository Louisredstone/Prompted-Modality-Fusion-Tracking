import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os
import cv2

from numpy import ndarray, uint8, prod
from abc import ABC, abstractmethod
from typing import Optional, Union
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from time import sleep

from ...utils import ObjectAccessor


# constants
# load mode
EAGER = 'eager'
LAZY = 'lazy'
DISK = 'disk'
DISABLED = 'disabled'
# typing
N = 'n_batch'
W = 'width'
H = 'height'
C = 'channel'
# frame status
NOT_LOADED = 0
LOADING = 1
LOADED = 2

class BaseImageSequence(ABC):
    _lock = Lock()
    def __init__(self, path: Optional[str] =None, name: Optional[str] =None, load_mode: str = DISK):
        '''
        Args:
            path: str, path to the dataset
            name: str, name of the dataset
            mode: str, mode of loading: 'eager', 'lazy', 'disk'
                disk: load data from disk every time
                eager: load all data into (shared) memory # There is a bug here, need to fix it.
                lazy: load data on demand, then keep in (shared) memory
            
            NOTE: 'mode' only affects images. Other light-weight data, like bboxes, will always be loaded eagerly.
        '''
        # parameters check
        if name==None and path == None:
            raise ValueError("Either name or path should be provided.")
        load_mode = load_mode.lower()
        assert load_mode in [EAGER, LAZY, DISK, DISABLED], f"Invalid mode: {load_mode}"
        # prepare attributes
        self.path: str = path
        self.name: str = name if name else os.path.basename(path)
        self.load_mode: str = load_mode
        self._width: int = None
        self._height: int = None
        self.bboxes: ndarray[(N, 6), uint8] = None
        self._valid: ndarray[(N,), bool] = None
        self._unblocked: ndarray[(N,), bool] = None
        self.init_filepaths()
        self.init_bboxes()
        self.init_valid()
        self.init_unblocked()
        if self.load_mode in [EAGER, LAZY]:
            # self._lock = Lock()
            self._shared_memory_created: bool = False
            frame_cache_shape: tuple[int, int, int, int] = self.shape
            with self._lock:
                try:
                    self._shared_memory: SharedMemory = SharedMemory(name=f'{self.__class__.__name__}.{self.name}', create=False, size=1 + prod(frame_cache_shape))
                except FileNotFoundError:
                    self._shared_memory: SharedMemory = SharedMemory(name=f'{self.__class__.__name__}.{self.name}', create=True, size=1 + prod(frame_cache_shape))
                    self._shared_memory_created = True
            self._frames_status: ndarray[1, uint8] = np.frombuffer(self._shared_memory.buf, dtype=uint8, count=1, offset=0)
            self._frame_cache: ndarray[(N, '?', H, W), uint8] = np.frombuffer(self._shared_memory.buf, dtype=uint8, count=prod(frame_cache_shape), offset=1).reshape(frame_cache_shape)
        if self.load_mode == EAGER:
            self.ensure_load_frames_to_cache()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._frame_cache
        del self._valid
        del self._unblocked
        if self.load_mode in [EAGER, LAZY]:
            self._shared_memory.close()
            if self._shared_memory_created:
                self._shared_memory.unlink()
    
    def __len__(self):
        return self.bboxes.shape[0]
    
    def __getitem__(self, idx):
        return ImageSequenceAccessor(self, indices=idx)
    
    def __iter__(self):
        for idx in range(len(self)): yield self[idx]
    
    ### PROPERTIES ###
    
    @property
    def frames(self) -> ndarray[(N, C, H, W),]:
        return self._get_frames()
    
    def _get_frames(self, indices=None) -> ndarray[(N, C, H, W),]:
        if self.load_mode == DISABLED:
            raise ValueError("Cannot load frames in disabled mode.")
        elif self.load_mode == DISK:
            logger.verbose(f'loading frames of `{self.name}` from disk, indices: {indices}')
            if indices is None: indices = range(len(self))
            elif not isinstance(indices, (list, tuple, np.ndarray)): indices = [indices]
            return np.array([self.load_frame(idx) for idx in indices])
        else:
            self.ensure_load_frames_to_cache()
            return self._frame_cache if indices is None else self._frame_cache[indices, :]
    
    @property
    def frames_status(self) -> uint8:
        return self._frames_status[0]
    @frames_status.setter
    def frames_status(self, value: uint8) -> None:
        with self._lock:
            self._frames_status[0] = value
    
    @property
    def shape(self) -> tuple[int, int, int, int]:
        '''Return the shape of the data, (N, C, H, W).'''
        raise NotImplementedError()
    
    @property
    def width(self) -> int:
        if self._width is None:
            self._width = self.load_frame(0).shape[2]
        return self._width
    
    @property
    def height(self) -> int:
        if self._height is None:
            self._height = self.load_frame(0).shape[1]
        return self._height
    
    @property
    def bboxes_ltwh(self) -> ndarray[(N, 4),]:
        return self.bboxes[:, :4]
    
    @property
    def bboxes_ltrb(self) -> ndarray[(N, 4),]:
        return self.bboxes[:, [0, 1, 4, 5]]
    
    @property
    def valid(self) -> ndarray[(N,), bool]:
        return self._valid
    
    @property
    def unblocked(self) -> ndarray[(N,), bool]:
        return self._unblocked

    ### INITIALIZERS ###
    
    def init_filepaths(self) -> None:
        '''Initialize filepaths.
        Note: we do not prepare a member like `self._filepath`. You may use any other ways to prepare filepaths when you design a subclass.'''
        pass
    
    @abstractmethod
    def init_bboxes(self) -> None:
        '''Initialize groundtruth bounding boxes.
        Recommended inplementation:
        self._gt_bboxes = self.parse_bboxes_from_file(os.path.join(self.path, 'groundtruth.txt'))
        '''
        raise NotImplementedError()

    def init_valid(self) -> None:
        '''Initialize valid flags.'''
        bboxes_ltwh = self.bboxes_ltwh
        bboxes_width, bboxes_height = bboxes_ltwh[:, 2], bboxes_ltwh[:, 3]
        self._valid = (bboxes_width > 0) & (bboxes_height > 0)
        
    def init_unblocked(self) -> None:
        '''Initialize unblocked flags.'''
        self._unblocked = self._valid
    
    def init_tags(self) -> None:
        '''Initialize tags.'''
        pass

    def ensure_load_frames_to_cache(self) -> None:
        '''Load images to self._frame_cache if not loaded.
        Need to mention that we use lazy loading.
        '''
        # if self._frames_loaded: return
        if self.load_mode == DISABLED: raise ValueError("Cannot load frames in disabled mode.")
        if self.load_mode == DISK: raise ValueError("Cannot load frames in disk mode.")
        if self.frames_status == LOADED: return
        if self.frames_status == LOADING:
            logger.info('Frames are loading by other process. Waiting...')
            while(self.frames_status == LOADING):
                logger.verbose(f'Waiting for frames to be loaded ({self.name}).')
                sleep(5)
            assert self.frames_status == LOADED, "Frames are not loaded."
            return
        # self.frames_status == NOT_LOADED
        self.frames_status = LOADING
        logger.info(f'loading images of `{self.name}`')
        with self._lock:
            for idx in range(len(self)):
                self._frame_cache[idx, :] = self.load_frame(idx)
        # self._frames_loaded = True
        self.frames_status = LOADED
        logger.debug(f'images loaded ({self.name})')

    ### CONST METHODS ###

    def column(self, *args):
        '''This method is slow, not recommended.'''
        if 'image' in args: self.ensure_load_frames_to_cache()
        return pd.DataFrame(data=[getattr(self, arg) for arg in args], columns=args)

    def parse_bboxes_from_file(self, filepath: str, delimiter: str =',', format: str = 'ltwh') -> np.ndarray[(N, 6),]:
        if format == 'ltwh':
            gt_bboxes_ltwh = np.loadtxt(filepath, delimiter=delimiter)
            gt_bboxes_ltwhrb = np.concatenate((gt_bboxes_ltwh, 
                                            gt_bboxes_ltwh[:, [0]] + gt_bboxes_ltwh[:, [2]],
                                            gt_bboxes_ltwh[:, [1]] + gt_bboxes_ltwh[:, [3]]), axis=1)
        elif format == 'ltrb':
            gt_bboxes_ltrb = np.loadtxt(filepath, delimiter=delimiter)
            gt_bboxes_ltwhrb = np.concatenate((gt_bboxes_ltrb[:, 0:2], 
                                            gt_bboxes_ltrb[:, [2]] - gt_bboxes_ltrb[:, [0]],
                                            gt_bboxes_ltrb[:, [3]] - gt_bboxes_ltrb[:, [1]],
                                            gt_bboxes_ltrb[:, 2:4]), axis=1)
        else:
            raise ValueError(f"Invalid format: {format}")
        return gt_bboxes_ltwhrb

    def parse_image_from_file(self, image_filepath: str, retries=3, interval=1) -> np.ndarray[(3, H, W), uint8]:
        logger.verbose(f'parsing image from `{image_filepath}`')
        for attempt in range(retries):
            img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
            if img is not None: break
            logger.warning(f'Failed to load image from `{image_filepath}`. Retrying in {interval} seconds. (attempt {attempt+1}/{retries})')
            sleep(interval)
        if img is None:
            # raise ValueError(f'Failed to load image from `{image_filepath}` after {retries} attempts.')
            logger.warning(f'Failed to load image from `{image_filepath}` after {retries} attempts. Returning Empty Image.')
            return np.zeros((3, self.height, self.width), dtype=uint8)
        HWC_image: ndarray[(H, W, 3), uint8] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        CHW_image: ndarray[(3, H, W), uint8] = np.transpose(HWC_image, (2, 0, 1))
        return CHW_image
    
    @abstractmethod
    def load_frame(self, idx: int) -> np.ndarray[(C, H, W), uint8]:
        raise NotImplementedError()

class ImageSequenceAccessor(ObjectAccessor):
    def __init__(self, obj: BaseImageSequence, indices: Union[int, slice, list, tuple]):
        super().__init__(obj, indices)
        
    def __getattr__(self, key: Union[int, slice, list, tuple]):
        if key == 'frames':
            seq: BaseImageSequence = self.__object__
            return seq._get_frames(self.__indices__)
        else:
            return super().__getattr__(key)