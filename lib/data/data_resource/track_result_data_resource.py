import logging
logger = logging.getLogger(__name__)

import os
import re
import pandas as pd
import numpy as np

from os.path import join as pjoin
from typing import Union

from .base_img_seq_data_resource import BaseImageSequenceDataResource
from .base_image_sequence import BaseImageSequence, DISABLED
from ...utils.misc import detect_delimiter_of_line

class TrackResultSequence(BaseImageSequence): 
    def __init__(self, path = None, name = None):
        # path: *.csv file path
        if path is None:
            raise ValueError("Path should not be None")
        filename = os.path.basename(path)
        if not filename.endswith('.csv'):
            raise ValueError(f"Invalid file type: {filename}")
        if not name:
            name = filename.split('.')[0]
        super().__init__(path, name, load_mode=DISABLED)
        
    def init_bboxes(self):
        df = pd.read_csv(self.path)
        gt_bboxes_ltwhrb = df[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'bbox_right', 'bbox_bottom']].values.astype(np.float32)
        self.bboxes = gt_bboxes_ltwhrb

    def load_frame(self, idx: int):
        raise NotImplementedError("TrackResultSequence does not support loading frames")

class TrackResultDataResource(BaseImageSequenceDataResource): # TODO: migrate to ProMFT_TrackResultDataResource if neccessary.
    def __init__(self, root):
        name = os.path.basename(root)
        super().__init__(name, root, 'result')
    
    def init_sequences(self) -> None:
        root, dirs, files = next(os.walk(self.root))
        for file in files:
            if file.endswith('.csv'):
                seq_name = file.split('.')[0]
                seq_path = pjoin(self.root, file)
                seq = TrackResultSequence(seq_path, seq_name)
                self.sequences.append(seq)
    
    @property
    def valid_splits(self) -> set[str]:
        return {'result'}
    
class ProMFT_TrackResultSequence(BaseImageSequence):
    def __init__(self, path = None, name = None):
        # path: *.csv file path
        if path is None:
            raise ValueError("Path should not be None")
        filename = os.path.basename(path)
        if not filename.endswith('.csv'):
            raise ValueError(f"Invalid file type: {filename}")
        if not name:
            name = filename.split('.')[0]
        super().__init__(path, name, load_mode=DISABLED)
        
    def init_bboxes(self):
        df = pd.read_csv(self.path)
        gt_bboxes_ltwhrb = df[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'bbox_right', 'bbox_bottom']].values.astype(np.float32)
        self.bboxes = gt_bboxes_ltwhrb

    def load_frame(self, idx: int):
        raise NotImplementedError("TrackResultSequence does not support loading frames")

class ProMFT_TrackResultDataResource(BaseImageSequenceDataResource):
    """
    TrackResultDataResource for ProMFT
    dir/
        seq1.csv
        seq2.csv
       ...
    """
    def __init__(self, root):
        name = os.path.basename(root)
        super().__init__(name, root, 'result')
    
    def init_sequences(self) -> None:
        root, dirs, files = next(os.walk(self.root))
        for file in files:
            if file.endswith('.csv'):
                seq_name = file.split('.')[0]
                seq_path = pjoin(self.root, file)
                seq = ProMFT_TrackResultSequence(seq_path, seq_name)
                self.sequences.append(seq)
    
    @property
    def valid_splits(self) -> set[str]:
        return {'result'}
    
class ViPT_TrackResultSequence_Dir(BaseImageSequence):
    """
    seq/
        seq_001_confidence.value
        seq_001_time.value
        seq_001.txt
    """
    def __init__(self, path = None, name = None):
        # path: *.csv file path
        if path is None:
            raise ValueError("Path should not be None")
        seq_name = os.path.basename(path.rstrip('/'))
        if not name: name = seq_name
        super().__init__(path, name, load_mode=DISABLED)
    
    def init_filepaths(self):
        fileInfo = dict()
        files = os.listdir(self.path)
        seq_name = self.name
        for file in files:
            match = re.match(seq_name+r'_(\d+)(_confidence\.value|_time\.value|\.txt)', file)
            if not match:
                raise ValueError(f"Invalid file name: {file}")
            id_ = int(match.group(1))
            type_ = match.group(2)
            type_ = 'gt' if type_ == '.txt' else 'confidence' if type_ == '_confidence.value' else 'time'
            if id_ not in fileInfo: fileInfo[id_] = dict()
            fileInfo[id_][type_] = file
        if len(fileInfo) != 1:
            raise NotImplementedError("Multiple sequence ids found in the same directory, not supported yet.")
        for id_ in fileInfo:
            if len(fileInfo[id_]) != 3:
                raise ValueError(f"Invalid file number for id {id_}: {len(fileInfo[id_])}")
        self.fileInfo = fileInfo
    
    def init_bboxes(self):
        id_ = list(self.fileInfo.keys())[0]
        txt_file = self.fileInfo[id_]['gt']
        txt_path = pjoin(self.path, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if len(lines) <= 1:
            raise ValueError(f"Invalid gt file: {txt_path}")
        # First line is '1', indicating given template, thus should be skipped during result analysis. We put a place holder here to ensure the index is consistent with the other data sources.
        delimiter = detect_delimiter_of_line(lines[1])
        if delimiter is None: raise ValueError(f"Invalid delimiter of gt file: {txt_path}")
        from io import StringIO
        arr = np.genfromtxt(StringIO("\n".join(lines[1:])), delimiter=delimiter, dtype=np.float32)
        bboxes_ltwh = np.concatenate(([[np.nan]*4], arr), axis=0)
        # df = pd.read_csv(txt_path)
        # gt_bboxes_ltwhrb = df[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'bbox_right', 'bbox_bottom']].values.astype(np.float32)
        bboxes_ltwhrb = np.concatenate([bboxes_ltwh, bboxes_ltwh[:,:2]+bboxes_ltwh[:,2:]], axis=1)
        self.bboxes = bboxes_ltwhrb

    def load_frame(self, idx: int):
        raise NotImplementedError("TrackResultSequence does not support loading frames")

class ViPT_TrackResultSequence_Txt(BaseImageSequence):
    """
    seq.txt
    """
    def __init__(self, path = None, name = None):
        # path: *.txt file path
        if path is None:
            raise ValueError("Path should not be None")
        seq_name = os.path.splitext(os.path.basename(path.rstrip('/')))[0]
        if not name: name = seq_name
        super().__init__(path, name, load_mode=DISABLED)
    
    def init_bboxes(self):
        txt_path = self.path
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        delimiter = detect_delimiter_of_line(lines[0])
        if delimiter is None: raise ValueError(f"Invalid delimiter of gt file: {txt_path}")
        gt_bboxes_ltwh = np.genfromtxt(txt_path, delimiter=delimiter, dtype=np.float32)
        gt_bboxes_ltwhrb = np.concatenate([gt_bboxes_ltwh, gt_bboxes_ltwh[:,:2]+gt_bboxes_ltwh[:,2:]], axis=1)
        self.bboxes = gt_bboxes_ltwhrb

    def load_frame(self, idx: int):
        raise NotImplementedError("TrackResultSequence does not support loading frames")

class ViPT_TrackResultDataResource(BaseImageSequenceDataResource):
    """
    TrackResultDataResource for ViPT
    dir/
        seq1/
            seq1_001_confidence.value
            seq1_001_time.value
            seq1_001.txt
        seq2/
            ...
        ...
    (This '001' might be the number of experimental repeats, default is 1.)
    """
    def __init__(self, root):
        name = os.path.basename(root)
        super().__init__(name, root, 'result')
    
    def init_sequences(self) -> None:
        root, dirs, files = next(os.walk(self.root))
        for file in files:
            if file.endswith('.txt'):
                seq_name = file.split('.')[0]
                seq_path = pjoin(self.root, file)
                seq = ViPT_TrackResultSequence_Txt(seq_path, seq_name)
                self.sequences.append(seq)
        for dir in dirs:
            seq_path = pjoin(self.root, dir)
            seq = ViPT_TrackResultSequence_Dir(seq_path, dir)
            self.sequences.append(seq)
    
    @property
    def valid_splits(self) -> set[str]:
        return {'result'}