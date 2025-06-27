import logging
logger = logging.getLogger(__name__)

import numpy as np

from numpy import ndarray, uint8
from abc import abstractmethod
from typing import Optional

from .base_image_sequence import BaseImageSequence
from .base_image_sequence import EAGER, LAZY, DISK

class RGB_AUX_ImageSequence(BaseImageSequence):
    '''
    Image Sequence with rgb and auxiliary images
    Auxiliary modal can be depth, event, tir, etc.
    '''
    def __init__(self, path: Optional[str] =None, name: Optional[str] =None, mode: str =DISK):
        self._rgb_image_filepaths: list[str] = None
        self._aux_image_filepaths: list[str] = None
        super().__init__(path=path, name=name, load_mode=mode)
    
    ### PROPERTIES ###
    
    @property
    def aux_type(self) -> str:
        '''The auxiliary image type, e.g. 'depth', 'event', 'tir', etc.'''
        raise NotImplementedError()
    
    @property
    def rgb_images(self) -> ndarray[('N', 3, 'H', 'W'),]:
        if self.load_mode == DISK:
            return np.array([self.parse_image_from_file(filepath) for filepath in self._rgb_image_filepaths])
        elif self.load_mode in [EAGER, LAZY]:
            self.ensure_load_frames_to_cache()
            return self._frame_cache[:, :3, :, :]
        else:
            raise ValueError('Invalid mode: {}'.format(self.load_mode))
    
    @property
    def aux_images(self) -> ndarray[('N', 3, 'H', 'W'),]:
        if self.load_mode == DISK:
            return np.array([self.parse_image_from_file(filepath) for filepath in self._aux_image_filepaths])
        elif self.load_mode in [EAGER, LAZY]:
            self.ensure_load_frames_to_cache()
            return self._frame_cache[:, 3:, :, :]
        else:
            raise ValueError('Invalid mode: {}'.format(self.load_mode))

    @property
    def frames(self)  -> ndarray[('N', 6, 'H', 'W'),]:
        if self.load_mode == DISK:
            return np.array([self.load_frame(idx) for idx in range(len(self))])
        elif self.load_mode in [EAGER, LAZY]:
            self.ensure_load_frames_to_cache()
            return self._frame_cache
        else:
            raise ValueError('Invalid mode: {}'.format(self.load_mode))
    
    @property
    def shape(self) -> tuple[int, int, int, int]:
        return len(self), 6, self.height, self.width
    
    ### INITIALIZERS ###
    
    def init_filepaths(self) -> None:
        self.init_rgb_image_filepaths()
        self.init_auxiliary_image_filepaths()
        assert len(self._rgb_image_filepaths) == len(self._aux_image_filepaths), \
            f'Number of rgb and auxiliary images should be the same. Path: {self.path}'
    
    @abstractmethod
    def init_rgb_image_filepaths(self) -> None:
        '''Initialize rgb images in self.df.
        You shall set self._rgb_image_filepaths to a list of filepaths in this function.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def init_auxiliary_image_filepaths(self) -> None:
        '''Initialize auxiliary images in self.df.
        You shall set self._aux_image_filepaths to a list of filepaths in this function.
        '''
        raise NotImplementedError()
    
    ### CONST METHODS ###
    
    def load_frame(self, idx: int) -> ndarray[(6, 'H', 'W'), uint8]:
        '''Load a single frame of the sequence.'''
        rgb_image_filepath = self._rgb_image_filepaths[idx]
        aux_image_filepath = self._aux_image_filepaths[idx]
        rgb_image = self.parse_image_from_file(rgb_image_filepath)
        aux_image = self.parse_image_from_file(aux_image_filepath)
        frame = np.concatenate([rgb_image, aux_image], axis=0)
        return frame