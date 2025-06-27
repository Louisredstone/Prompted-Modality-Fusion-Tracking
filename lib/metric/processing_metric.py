import numpy as np

from . import BaseMetric

# 未实装

class CenterErrorMetric(BaseMetric):
    def __init__(self, norm_dst=False):
        '''
        norm_dst: 是否对矩形框坐标进行归一化处理
        '''
        self.norm_dst = norm_dst
    
    def compute_metric_sequence(self, bbox_per_img_pred, bbox_per_img_gt):
        '''
        results: tracker的检测结果, shape=(seq_length, 4)
        rect_anno: 标注的矩形框, groundtruth, shape=(seq_length, 4)
        '''
        seq_length = bbox_per_img_gt.shape[0]

        bbox_per_img_pred, bbox_per_img_gt = self.preprocess(bbox_per_img_pred, bbox_per_img_gt)
        # 预处理矩形框坐标, 包括归一化和裁剪.

        center_per_img_gt = np.column_stack((bbox_per_img_gt[:, 0] + (bbox_per_img_gt[:, 2] - 1) / 2,
                                    bbox_per_img_gt[:, 1] + (bbox_per_img_gt[:, 3] - 1) / 2))
        # groundtruth框中心坐标

        center_per_img_pred = np.column_stack((bbox_per_img_pred[:, 0] + (bbox_per_img_pred[:, 2] - 1) / 2,
                                bbox_per_img_pred[:, 1] + (bbox_per_img_pred[:, 3] - 1) / 2))
        # tracker预测结果的中心坐标

        if self.norm_dst:
            center_per_img_pred[:, 0] /= bbox_per_img_gt[:, 2]
            center_per_img_pred[:, 1] /= bbox_per_img_gt[:, 3]
            center_per_img_gt[:, 0] /= bbox_per_img_gt[:, 2]
            center_per_img_gt[:, 1] /= bbox_per_img_gt[:, 3]

        center_err_per_img = np.sqrt(np.sum((center_per_img_pred - center_per_img_gt) ** 2, axis=1))

        idx = np.all(bbox_per_img_gt > 0, axis=1)
        # 此处idx是指标注矩形框中有目标的帧, 即idx=True的位置.
        # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.

        center_err_per_img[~idx] = -1

        return center_err_per_img

class OverlapErrorMetric(BaseMetric):
    def __init__(self, norm_dst=False):
        '''
        norm_dst: 是否对矩形框坐标进行归一化处理
        '''
        self.norm_dst = norm_dst
    
    def calc_overlap_rate(self, A, B):
        '''
        Calculate overlap of two rectangles.
        A: (N, 4) ndarray of float
        B: (N, 4) ndarray of float
        return: (N,) ndarray of overlap (float 0~1) between A and B
        '''
        leftA = A[:, 0]
        bottomA = A[:, 1]
        # rightA = leftA + A[:, 2] - 1
        # topA = bottomA + A[:, 3] - 1
        rightA = leftA + A[:, 2]
        topA = bottomA + A[:, 3]

        leftB = B[:, 0]
        bottomB = B[:, 1]
        # rightB = leftB + B[:, 2] - 1
        # topB = bottomB + B[:, 3] - 1
        rightB = leftB + B[:, 2]
        topB = bottomB + B[:, 3]

        # area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB) + 1)) * \
        #       (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB) + 1))
        area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB))) * \
            (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB)))
        areaA = A[:, 2] * A[:, 3]
        areaB = B[:, 2] * B[:, 3]
        overlap_rate = area_overlap / (areaA + areaB - area_overlap)
        return overlap_rate

    def compute_metric_sequence(self, bbox_per_img_pred, bbox_per_img_gt):
        '''
        results: tracker的检测结果, shape=(seq_length, 4)
        rect_anno: 标注的矩形框, groundtruth, shape=(seq_length, 4)
        '''
        seq_length = bbox_per_img_gt.shape[0]

        bbox_per_img_pred, bbox_per_img_gt = self.preprocess(bbox_per_img_pred, bbox_per_img_gt)
        # 预处理矩形框坐标, 包括归一化和裁剪.

        center_per_img_gt = np.column_stack((bbox_per_img_gt[:, 0] + (bbox_per_img_gt[:, 2] - 1) / 2,
                                    bbox_per_img_gt[:, 1] + (bbox_per_img_gt[:, 3] - 1) / 2))
        # groundtruth框中心坐标

        center_per_img_pred = np.column_stack((bbox_per_img_pred[:, 0] + (bbox_per_img_pred[:, 2] - 1) / 2,
                                bbox_per_img_pred[:, 1] + (bbox_per_img_pred[:, 3] - 1) / 2))
        # tracker预测结果的中心坐标

        if self.norm_dst:
            center_per_img_pred[:, 0] /= bbox_per_img_gt[:, 2]
            center_per_img_pred[:, 1] /= bbox_per_img_gt[:, 3]
            center_per_img_gt[:, 0] /= bbox_per_img_gt[:, 2]
            center_per_img_gt[:, 1] /= bbox_per_img_gt[:, 3]

        center_err_per_img = np.sqrt(np.sum((center_per_img_pred - center_per_img_gt) ** 2, axis=1))

        idx = np.all(bbox_per_img_gt > 0, axis=1)
        # 此处idx是指标注矩形框中有目标的帧, 即idx=True的位置.
        # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.

        def calc_overlap_rate(A, B):
            '''
            Calculate overlap of two rectangles.
            A: (N, 4) ndarray of float
            B: (N, 4) ndarray of float
            return: (N,) ndarray of overlap (float 0~1) between A and B
            '''
            leftA = A[:, 0]
            bottomA = A[:, 1]
            # rightA = leftA + A[:, 2] - 1
            # topA = bottomA + A[:, 3] - 1
            rightA = leftA + A[:, 2]
            topA = bottomA + A[:, 3]

            leftB = B[:, 0]
            bottomB = B[:, 1]
            # rightB = leftB + B[:, 2] - 1
            # topB = bottomB + B[:, 3] - 1
            rightB = leftB + B[:, 2]
            topB = bottomB + B[:, 3]

            # area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB) + 1)) * \
            #       (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB) + 1))
            area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB))) * \
                (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB)))
            areaA = A[:, 2] * A[:, 3]
            areaB = B[:, 2] * B[:, 3]
            overlap_rate = area_overlap / (areaA + areaB - area_overlap)
            return overlap_rate

        overlap_rate_per_img = calc_overlap_rate(bbox_per_img_pred[idx, :], bbox_per_img_gt[idx, :])

        # err_overlap = -np.ones(len(idx))
        overlap_err_per_img = -np.ones(seq_length)
        overlap_err_per_img[idx] = overlap_rate_per_img

        return overlap_err_per_img