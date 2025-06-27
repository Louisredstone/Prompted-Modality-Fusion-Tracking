import logging
logger = logging.getLogger(__name__)

import os
import pandas as pd

from os.path import join as pjoin
from typing import Union

from .base_img_seq_data_resource import BaseImageSequenceDataResource, RGB_AUX_ImageSequence


class RGBT234Sequence(RGB_AUX_ImageSequence):
    def __init__(self, path=None, name=None, **kwargs):
        super(RGBT234Sequence, self).__init__(path=path, name=name, **kwargs)
        
    @property
    def aux_type(self):
        return 'tir'
        
    def init_rgb_image_filepaths(self) -> None:
        rgb_images_dir = pjoin(self.path, 'visible')
        rgb_image_filenames = sorted(os.listdir(rgb_images_dir))
        rgb_image_filepaths = [pjoin(rgb_images_dir, filename) for filename in rgb_image_filenames]
        self._rgb_image_filepaths = rgb_image_filepaths
    
    def init_auxiliary_image_filepaths(self):
        aux_images_dir = pjoin(self.path, 'infrared')
        aux_image_filenames = sorted(os.listdir(aux_images_dir))
        aux_image_filepaths = [pjoin(aux_images_dir, filename) for filename in aux_image_filenames]
        self._aux_image_filepaths = aux_image_filepaths
    
    def init_bboxes(self):
        self.bboxes = self.parse_bboxes_from_file(pjoin(self.path, 'visible.txt'))

class RGBT234(BaseImageSequenceDataResource):
    def __init__(self, root, split: Union[str, None, list[str]]):
        super(RGBT234, self).__init__('RGBT234', root, split)
        
    def init_sequences(self):
        logger.info("Initializing RGBT234 sequences...")
        split_subdirs = {
            'test': '.'
        }
        from tqdm import tqdm
        for split in self.splits:
            subdir = split_subdirs[split]
            _, dirs, _ = next(os.walk(pjoin(self.root, subdir)))
            for seq_dirname in tqdm(dirs):
                seq_dir = pjoin(self.root, subdir, seq_dirname)
                seq = RGBT234Sequence(path=seq_dir, name=seq_dirname)
                self.sequences.append(seq)
    
    @property
    def valid_splits(self) -> set[str]:
        return {'test'}