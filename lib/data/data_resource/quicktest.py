import logging
logger = logging.getLogger(__name__)

import os

from os.path import join as pjoin

from .base_img_seq_data_resource import BaseImageSequenceDataResource, RGB_AUX_ImageSequence


class QuickTestSequence(RGB_AUX_ImageSequence):
    def __init__(self, path=None, name=None):
        super(QuickTestSequence, self).__init__(path=path, name=name)
        
    @property
    def aux_type(self) -> str:
        return 'tir'
        
    def init_rgb_image_filepaths(self):
        images_dir = pjoin(self.path, 'visible')
        image_filenames = sorted(os.listdir(images_dir))
        image_filepaths = [pjoin(images_dir, filename) for filename in image_filenames]
        # self.df['image_filepath'] = image_filepaths
        self._rgb_image_filepaths = image_filepaths
    
    def init_auxiliary_image_filepaths(self):
        aux_images_dir = pjoin(self.path, 'infrared')
        aux_image_filenames = sorted(os.listdir(aux_images_dir))
        aux_image_filepaths = [pjoin(aux_images_dir, filename) for filename in aux_image_filenames]
        # self.df['aux_image_filepath'] = aux_image_filepaths
        self._aux_image_filepaths = aux_image_filepaths
    
    def init_bboxes(self):
        # gt_bboxes = pd.read_csv(pjoin(self.path, 'visible.txt'), delimiter=',', header=None)
        # self.df[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']] = gt_bboxes
        # self.df['bbox_right'] = self.df['bbox_left'] + self.df['bbox_width']
        # self.df['bbox_bottom'] = self.df['bbox_top'] + self.df['bbox_height']
        # gt_bboxes_ltwh = np.loadtxt(pjoin(self.path, 'visible.txt'), delimiter=',')
        # gt_bboxes_ltwhrb = np.concatenate((gt_bboxes_ltwh, 
        #                                    gt_bboxes_ltwh[0] + gt_bboxes_ltwh[2],
        #                                    gt_bboxes_ltwh[1] + gt_bboxes_ltwh[3]), axis=1)
        # self._gt_bboxes = gt_bboxes_ltwhrb
        self.bboxes = self.parse_bboxes_from_file(pjoin(self.path, 'visible.txt'))

    # def ensure_load_auxiliary_images(self):
    #     raise NotImplementedError('This Exception is only for Debugging')

class QuickTest(BaseImageSequenceDataResource):
    # def __init__(self, root, split: Union[str, None, list[str]]):
    def __init__(self, root, split):
        super(QuickTest, self).__init__('QuickTest', root, split)
        
    def init_sequences(self):
        logger.debug('Initializing QuickTest sequences (we use lasher)')
        split_subdirs = {
            'train': 'TrainingSet/trainingset',
            'test': 'TestingSet/testingset'
        }
        from tqdm import tqdm
        for split in self.splits:
            subdir = split_subdirs[split]
            _, dirs, _ = next(os.walk(pjoin(self.root, subdir)))
            dirs = dirs[:3] # only use first 3 sequences for quick test
            for seq_dirname in tqdm(dirs):
                seq_dir = pjoin(self.root, subdir, seq_dirname)
                seq = QuickTestSequence(path=seq_dir, name=seq_dirname)
                self.sequences.append(seq)