import logging
logger = logging.getLogger(__name__)

import numpy as np

from abc import ABC, abstractmethod

from ..data import BBox

# 未实装

class BaseMetric(ABC):
    def __init__(self):
        pass
    
    def __call__(self, pred, gt):
        pass
    
    def preprocess(self, value):
        pass
    
    @abstractmethod
    def compute_metric(self, pred, gt):
        is_one_image = False
        is_one_sequence = False
        is_dataset = False
        if isinstance(pred, BBox) and isinstance(gt, BBox):
            is_one_image = True
        elif isinstance(pred, list) and isinstance(gt, list):
            if len(pred) == len(gt) and all(isinstance(p, BBox) and isinstance(g, BBox) for p, g in zip(pred, gt)):
                is_one_sequence = True
            elif all(isinstance(p, list) and isinstance(g, list) and len(p) == len(g) and all(isinstance(pp, BBox) and isinstance(gg, BBox) for pp, gg in zip(p, g)) for p, g in zip(pred, gt)):
                is_dataset = True
        is_one_image =  isinstance(pred, BBox)\
                    and isinstance(gt, BBox)
        is_one_sequence =   isinstance(pred, list)\
                        and isinstance(gt, list)\
                        and len(pred) == len(gt)\
                        and all(isinstance(p, BBox)\
                            and isinstance(g, BBox)\
                            for p, g in zip(pred, gt))
        is_dataset =    isinstance(pred, list)\
                    and isinstance(gt, list)\
                    and len(pred) == len(gt)\
                    and all(isinstance(p, list)\
                        and isinstance(g, list)\
                        and len(p) == len(g)\
                        and all(isinstance(pp, BBox)\
                            and isinstance(gg, BBox)\
                            for pp, gg in zip(p, g))\
                        for p, g in zip(pred, gt))
        if not (is_one_image or is_one_sequence or is_dataset):
            raise ValueError("Input type not supported.")
        if is_one_image:
            return self.compute_metric_single(pred, gt)
        elif is_one_sequence:
            return self.compute_metric_sequence(pred, gt)
        elif is_dataset: # multiple sequences
            return self.compute_metric_dataset(pred, gt)
        
    @abstractmethod
    def compute_metric_single(self, pred, gt):
        raise NotImplementedError()
    
    @abstractmethod
    def compute_metric_sequence(self, pred_seq, gt_seq):
        return [self.compute_metric_single(pred, gt) for pred, gt in zip(pred_seq, gt_seq)]
    
    @abstractmethod
    def compute_metric_dataset(self, pred_dataset, gt_dataset):
        return [self.compute_metric_sequence(pred_seq, gt_seq) for pred_seq, gt_seq in zip(pred_dataset, gt_dataset)]
    
    def preprocess(self, bbox_per_img_pred, bbox_per_img_gt):
        '''
        预处理矩形框坐标, 包括归一化和裁剪.
        '''
        seq_length = bbox_per_img_gt.shape[0]

        if bbox_per_img_pred.shape[0] != seq_length:
            bbox_per_img_pred = bbox_per_img_pred[:seq_length, :]
            # 如果检测的长度不等于标注的长度, 则截断检测结果到标注的长度.

        bbox_per_img_pred = bbox_per_img_pred.copy() # 避免修改原结果.
        
        bbox_per_img_pred[0, :] = bbox_per_img_gt[0, :]
        # 第0帧的检测结果是由GT初始化的, 所以不参与评估. 用GT初始化第0帧的检测结果.
        
        for i in range(1, seq_length):
            # 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
            bbox_pred = bbox_per_img_pred[i, :]
            bbox_gt = bbox_per_img_gt[i, :]
            if (np.isnan(bbox_pred).any()\
                    or not np.isreal(bbox_pred).all()\
                    or bbox_pred[2] <= 0\
                    or bbox_pred[3] <= 0)\
                and not np.isnan(bbox_gt).any():
                bbox_per_img_pred[i, :] = bbox_per_img_pred[i-1, :]
                # 如果检测结果不合法 (比如检测框的宽或高为0), 则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.
        
        return bbox_per_img_pred, bbox_per_img_gt