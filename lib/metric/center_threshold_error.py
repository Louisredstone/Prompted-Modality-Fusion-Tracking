import logging
logger = logging.getLogger(__name__)

import numpy as np

# from . import BaseThresholdMetric

# 未实装

class CenterError:
    def __init__(self):
        self.norm_dst = False # 是否对矩形框坐标进行归一化处理
    
    def __call__(self, bboxes_pred, bboxes_gt):
        '''
        bboxes_pred: [BBox] x n_frames. tracker的检测结果
        bboxes_gt: [BBox] x n_frames. 标注的矩形框, groundtruth
        '''
        # Preprocessing
        seq_length = len(bboxes_gt)

        if bboxes_pred.shape[0] != seq_length:
            bboxes_pred = bboxes_pred[:seq_length, :]

        for i in range(1, seq_length):
            # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
            rect_tracker = bboxes_pred[i, :]
            rect_anno = bboxes_gt[i, :]
            if (np.isnan(rect_tracker).any() or not np.isreal(rect_tracker).all() or rect_tracker[2] <= 0 or rect_tracker[3] <= 0) and not np.isnan(rect_anno).any():
                bboxes_pred[i, :] = bboxes_pred[i-1, :]
                # Reader memo: 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.

        res2 = bboxes_pred.copy()
        res2[0, :] = bboxes_gt[0, :]

        center_GT = np.column_stack((bboxes_gt[:, 0] + (bboxes_gt[:, 2] - 1) / 2,
                                    bboxes_gt[:, 1] + (bboxes_gt[:, 3] - 1) / 2))
        # Reader memo: groundtruth框中心坐标

        center = np.column_stack((res2[:, 0] + (res2[:, 2] - 1) / 2,
                                res2[:, 1] + (res2[:, 3] - 1) / 2))
        # Reader memo: tracker框中心坐标

        if self.norm_dst:
            center[:, 0] /= bboxes_gt[:, 2]
            center[:, 1] /= bboxes_gt[:, 3]
            center_GT[:, 0] /= bboxes_gt[:, 2]
            center_GT[:, 1] /= bboxes_gt[:, 3]

        err_center = np.sqrt(np.sum((center - center_GT) ** 2, axis=1))

        # index = anno > 0
        # idx = np.all(index, axis=1)
        idx = np.all(bboxes_gt > 0, axis=1)
        # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
        # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
        # 这回答了之前关于丢失目标的疑问.

        overlap_rate = calc_overlap_rate(res2[idx, :], bboxes_gt[idx, :])

        err_overlap = -np.ones(len(idx))
        err_overlap[idx] = overlap_rate
        err_center[~idx] = -1

        return err_overlap, err_center