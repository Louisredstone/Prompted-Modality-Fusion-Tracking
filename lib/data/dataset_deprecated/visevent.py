import os
import os.path
import torch
import numpy as np
import pandas
import csv
import cv2
from collections import OrderedDict
from .base_image_sequence_dataset import BaseImageSequenceDataset
from ..utils import jpeg4py_loader_w_failsafe, smart_loader
from ...train.admin import env_settings
# from ..dataset.depth_utils import get_x_frame

class VisEvent(BaseImageSequenceDataset):
    """ VisEvent dataset.
    """

    def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=smart_loader): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the VisEvent dataset. (this script is copied from lasot dataset and modified)
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        if root is None: root = env_settings().VisEvent_dir
        super().__init__('VisEvent', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        if split not in ['train', 'val', 'test']:
            # Need to mention that 'train' split is actually a part of VisEvent official trainset. like:
            # (official) trainset  | testset
            # (code)     train val | test
            # TODO: DO NOT USE manually splitted traini-val dataset, but randomly split the official trainset during training.
            raise ValueError('Invalid split: {}'.format(split))
        self.sequence_list = self._build_sequence_list()
        self.seq_frame_list = {seq: sorted([filename.split('.')[0] for filename in os.listdir(os.path.join(self.root, seq, 'vis_imgs')) if filename.endswith('.bmp')]) for seq in self.sequence_list}

        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

    def _build_sequence_list(self):

        # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # file_path = os.path.join(ltr_path, 'assets', 'VisEvent_%s.txt'%self.split)
        # file_path = os.path.join(self.root, f'{self.split}list.txt')
        # with open(file_path, 'r') as f:
        #     sequence_list = [line.strip() for line in f.readlines()]
        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        # return sequence_list
        root, dirs, files = next(os.walk(self.root))
        return dirs

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return 'VisEvent'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 10.0) & (bbox[:, 3] > 10.0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return depth image path
        '''
        # return os.path.join(seq_path, 'vis_imgs', f'frame{frame_id:04}.bmp') , os.path.join(seq_path, 'event_imgs', f'frame{frame_id:04}.bmp') # frames start from 1
        seq_name = seq_path.split('/')[-1]
        return os.path.join(seq_path, 'vis_imgs', f'{self.seq_frame_list[seq_name][frame_id]}.bmp'), os.path.join(seq_path, 'event_imgs', f'{self.seq_frame_list[seq_name][frame_id]}.bmp') # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - colormap from depth image
            - 3xD = [depth, depth, depth], 255
            - rgbcolormap
            - rgb3d
            - color
            - raw_depth
        '''
        color_path, depth_path = self._get_frame_path(seq_path, frame_id)
        img = self.get_x_frame(color_path, depth_path, dtype=self.dtype, 
                            #    clip=True
                               )

        return img

    def _get_class(self, seq_path):
        # raw_class = seq_path.split('/')[-2]
        # return raw_class
        return self.split

    def get_class_name(self, seq_id):
        depth_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(depth_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_x_frame(self, color_path, event_path, dtype='rgbrgb', 
                    # clip=False
                    ):
        ''' read RGB and event images
        '''
        if color_path:
            rgb = cv2.imread(color_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = None

        if event_path:
            dp = cv2.imread(event_path, -1)

            # if clip:
            #     max_depth = min(np.median(dp) * 3, 10000)
            #     dp[dp > max_depth] = max_depth
        else:
            dp = None

        if dtype == 'color':
            img = rgb

        elif dtype == 'raw_x':
            img = dp

        elif dtype == 'colormap':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

        elif dtype == '3x':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            img = cv2.merge((dp, dp, dp))

        elif dtype == 'normalized_x':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            img = np.asarray(dp, dtype=np.uint8)

        elif dtype == 'rgbcolormap':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
            img = cv2.merge((rgb, colormap))  # (h,w,6)

        elif dtype == 'rgb3x':
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            dp = cv2.merge((dp, dp, dp))
            img = cv2.merge((rgb, dp))

        elif dtype == 'rgbrgb':
            dp = cv2.cvtColor(dp, cv2.COLOR_BGR2RGB)
            #print(rgb.shape)
            #print(dp.shape)
            if rgb.shape != dp.shape: 
                rows, cols = rgb.shape[:2] #获取sky的高度、宽度
                dp = cv2.resize(dp,(cols,rows),interpolation=cv2.INTER_CUBIC) #放大图像
            

            img = cv2.merge((rgb, dp))

        else:
            print('No such dtype !!! ')
            raise NotImplementedError()
            img = None

        return img