# Warning: this file is just for temporary use, will be removed in the future.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..data.data_resource import BaseImageSequenceDataResource, BaseImageSequence

def get_plot_style():
    plot_styles = [
        {'color': np.array([1, 0, 0]), 'lineStyle': '-'},
        {'color': np.array([0, 1, 0]), 'lineStyle': '-'},
        {'color': np.array([0, 0, 1]), 'lineStyle': '-'},
        {'color': np.array([0, 0, 0]), 'lineStyle': '-'},
        {'color': np.array([1, 0, 1]), 'lineStyle': '-'},
        {'color': np.array([0, 1, 1]), 'lineStyle': '-'},
        {'color': np.array([0.5, 0.5, 0.5]), 'lineStyle': '-'},
        {'color': np.array([136, 0, 21]) / 255, 'lineStyle': '-'},
        {'color': np.array([255, 127, 39]) / 255, 'lineStyle': '-'},
        {'color': np.array([0, 162, 232]) / 255, 'lineStyle': '-'},
        {'color': np.array([163, 73, 164]) / 255, 'lineStyle': '-'},
        {'color': np.array([191, 144, 0]) / 255, 'lineStyle': '-'},
        {'color': np.array([47, 85, 151]) / 255, 'lineStyle': '-'},
        {'color': np.array([146, 208, 80]) / 255, 'lineStyle': '-'},
        {'color': np.array([102, 153, 255]) / 255, 'lineStyle': '-'},
        {'color': np.array([51, 153, 102]) / 255, 'lineStyle': '-'},
        {'color': np.array([1, 0, 0]), 'lineStyle': '--'},
        {'color': np.array([0, 1, 0]), 'lineStyle': '--'},
        {'color': np.array([0, 0, 1]), 'lineStyle': '--'},
        {'color': np.array([0, 0, 0]), 'lineStyle': '--'},
        {'color': np.array([1, 0, 1]), 'lineStyle': '--'},
        {'color': np.array([0, 1, 1]), 'lineStyle': '--'},
        {'color': np.array([0.5, 0.5, 0.5]), 'lineStyle': '--'},
        {'color': np.array([136, 0, 21]) / 255, 'lineStyle': '--'},
        {'color': np.array([255, 127, 39]) / 255, 'lineStyle': '--'},
        {'color': np.array([0, 162, 232]) / 255, 'lineStyle': '--'},
        {'color': np.array([163, 73, 164]) / 255, 'lineStyle': '--'},
        {'color': np.array([191, 144, 0]) / 255, 'lineStyle': '--'},
        {'color': np.array([47, 85, 151]) / 255, 'lineStyle': '--'},
        {'color': np.array([146, 208, 80]) / 255, 'lineStyle': '--'},
        {'color': np.array([102, 153, 255]) / 255, 'lineStyle': '--'},
        {'color': np.array([51, 153, 102]) / 255, 'lineStyle': '--'},
        {'color': np.array([1, 0, 0]), 'lineStyle': '-.'},
        {'color': np.array([0, 1, 0]), 'lineStyle': '-.'},
        {'color': np.array([0, 0, 1]), 'lineStyle': '-.'},
        {'color': np.array([0, 0, 0]), 'lineStyle': '-.'},
        {'color': np.array([1, 0, 1]), 'lineStyle': '-.'},
        {'color': np.array([0, 1, 1]), 'lineStyle': '-.'},
        {'color': np.array([0.5, 0.5, 0.5]), 'lineStyle': '-.'},
        {'color': np.array([136, 0, 21]) / 255, 'lineStyle': '-.'},
        {'color': np.array([255, 127, 39]) / 255, 'lineStyle': '-.'},
        {'color': np.array([0, 162, 232]) / 255, 'lineStyle': '-.'},
        {'color': np.array([163, 73, 164]) / 255, 'lineStyle': '-.'},
        {'color': np.array([191, 144, 0]) / 255, 'lineStyle': '-.'},
        {'color': np.array([47, 85, 151]) / 255, 'lineStyle': '-.'},
        {'color': np.array([146, 208, 80]) / 255, 'lineStyle': '-.'},
        {'color': np.array([102, 153, 255]) / 255, 'lineStyle': '-.'},
        {'color': np.array([51, 153, 102]) / 255, 'lineStyle': '-.'}
    ]
    return plot_styles

def get_sequences(type, pred_dir):
    import os

    # dataset_name = {
    #     'test_set': 'testing_set.txt',
    #     'extension_test_set': 'extension_testing_set.txt',
    #     'all': 'all_dataset.txt'
    # }.get(type)

    # if not dataset_name:
    #     raise ValueError("Error in evaluation dataset type! Either 'testing_set', 'extension_test_set', or 'all'.")

    # if not os.path.exists(dataset_name):
    #     raise FileNotFoundError(f"{dataset_name} is not found!")

    # with open(dataset_name, 'r') as fid:
    #     sequences = [line.strip() for line in fid if line.strip()]

    print('[DEBUG] Using DepthTrack_test dataset for evaluation.')

    # txt_files = os.listdir(pred_dir)
    # sequences = [os.path.splitext(txt_file)[0] for txt_file in txt_files if txt_file.endswith('.txt')]

    # return sequences
    root, dirs, files = os.walk(pred_dir).__next__()
    return dirs

def get_trackers():
    trackers = [
        # {'name': 'CAT', 'publish': '1111'},
        # {'name': 'CMR', 'publish': '1111'},
        # {'name': 'DAFNet', 'publish': '1111'},
        # {'name': 'DAPNet', 'publish': '1111'},
        # {'name': 'DMCNet', 'publish': '1111'},
        # {'name': 'FANet', 'publish': '1111'},
        # {'name': 'MaCNet', 'publish': '1111'},
        # {'name': 'MANet', 'publish': '1111'},
        # {'name': 'mfDiMP', 'publish': '1111'},
        # {'name': 'MANet++', 'publish': '1111'},
        # {'name': 'SGT', 'publish': '1111'},
        # {'name': 'SGT++', 'publish': '1111'}
        {'name': 'ProMFT', 'publish': '1111'}
    ]
    return trackers

def calc_overlap_rate(rectA, rectB):
    '''
    Calculate overlap of two rectangles.
    rectA: (N, 4) ndarray of float
    rectB: (N, 4) ndarray of float
    return: (N,) ndarray of overlap (float 0~1) between A and B
    
    Note: 
    rect = (left, bottom, width, height)
    bbox = (left, top, right, bottom)
    '''
    leftA = rectA[:, 0]
    topA = rectA[:, 1]
    # rightA = leftA + A[:, 2] - 1
    # topA = bottomA + A[:, 3] - 1
    rightA = leftA + rectA[:, 2]
    bottomA = topA + rectA[:, 3] # topest = 0, bottomest > 0

    leftB = rectB[:, 0]
    topB = rectB[:, 1]
    # rightB = leftB + B[:, 2] - 1
    # topB = bottomB + B[:, 3] - 1
    rightB = leftB + rectB[:, 2]
    bottomB = topB + rectB[:, 3]

    # area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB) + 1)) * \
    #       (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB) + 1))
    area_overlap = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB))) * \
          (np.maximum(0, np.minimum(bottomA, bottomB) - np.maximum(topA, topB)))
    areaA = rectA[:, 2] * rectA[:, 3]
    areaB = rectB[:, 2] * rectB[:, 3]
    overlap_rate = area_overlap / (areaA + areaB - area_overlap)
    return overlap_rate

def calculate_seq_overlap_center(rects_pred, rects_gt, norm_dst):
    '''
    results: tracker的检测结果, shape=(seq_length, 4)
    rect_anno: 标注的矩形框, groundtruth, shape=(seq_length, 4)
    norm_dst: 暂不知晓, 似乎是是否对矩形框坐标进行归一化处理
    
    Warning: this function will modify the input rects_pred, so consider making a copy before calling it.
    
    DEPRECATED WARNING!!!
    '''
    seq_length = rects_gt.shape[0]

    if rects_pred.shape[0] != seq_length:
        rects_pred = rects_pred[:seq_length, :]

    for i in range(1, seq_length):
        # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
        rect_pred = rects_pred[i, :]
        rect_gt = rects_gt[i, :]
        if (np.isnan(rect_pred).any() or not np.isreal(rect_pred).all() or rect_pred[2] <= 0 or rect_pred[3] <= 0) and not np.isnan(rect_gt).any():
            rects_pred[i, :] = rects_pred[i-1, :]
            # Reader memo: 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.

    rects_pred[0, :] = rects_gt[0, :]

    center_GT = np.column_stack((rects_gt[:, 0] + (rects_gt[:, 2] - 1) / 2,
                                 rects_gt[:, 1] + (rects_gt[:, 3] - 1) / 2))
    # Reader memo: groundtruth框中心坐标

    center = np.column_stack((rects_pred[:, 0] + (rects_pred[:, 2] - 1) / 2,
                             rects_pred[:, 1] + (rects_pred[:, 3] - 1) / 2))
    # Reader memo: tracker框中心坐标

    if norm_dst:
        center[:, 0] /= rects_gt[:, 2]
        center[:, 1] /= rects_gt[:, 3]
        center_GT[:, 0] /= rects_gt[:, 2]
        center_GT[:, 1] /= rects_gt[:, 3]

    center_error = np.sqrt(np.sum((center - center_GT) ** 2, axis=1))

    # index = anno > 0
    # idx = np.all(index, axis=1)
    idx = np.all(rects_gt > 0, axis=1)
    # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
    # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
    # 这回答了之前关于丢失目标的疑问.

    overlap_rate = -np.ones(len(idx))
    overlap_rate[idx] = calc_overlap_rate(rects_pred[idx, :], rects_gt[idx, :])
    center_error[~idx] = -1

    return overlap_rate, center_error

# def eval_tracker(seqs, trackers, eval_type, name_tracker_all, tmp_mat_path, dataset_dir, pred_dir, norm_dst):
def eval_trackers(seqs, trackers, name_tracker_all, tmp_mat_path, dataset_dir, pred_dir, norm_dst):
    num_tracker = len(trackers)
    overlap_rate_thresholds = np.arange(0, 1.05, 0.05)
    center_error_thresholds = np.arange(0, 51)
    TP_thresholds = np.arange(0, 1.05, 0.05)
    FP_thresholds = np.arange(0, 1.05, 0.05)
    TR_thresholds = np.arange(0, 1.05, 0.05)
    FR_thresholds = np.arange(0, 1.05, 0.05)
    if norm_dst:
        center_error_thresholds = center_error_thresholds / 100

    overlap_rate_plots = np.zeros((num_tracker, len(seqs), len(overlap_rate_thresholds)))
    center_error_plots = np.zeros((num_tracker, len(seqs), len(center_error_thresholds)))

    for i, s in enumerate(seqs):  # for each sequence
        rects_gt = np.loadtxt(os.path.join(dataset_dir, f'{s}/groundtruth.txt'), delimiter=',') # FIXME: convert to Dataset class in the future

        for k, t in enumerate(trackers):  # evaluate each tracker
            # res_file = os.path.join(rp_all, f'{t["name"]}_tracking_result/{s}.txt')
            res_file = os.path.join(pred_dir, f'{s}.txt') # DEBUG
            if not os.path.exists(res_file):
                print(f"File {res_file} not found, skipping...")
                continue

            rects_pred = np.loadtxt(res_file)
            print(f'evaluating {t["name"]} on {s} ...')

            if rects_pred.size == 0:
                # continue # ???
                break

            # overlap_rate, center_error = calculate_seq_overlap_center(res, anno, norm_dst)
            
            seq_length = rects_gt.shape[0]

            if rects_pred.shape[0] != seq_length:
                rects_pred = rects_pred[:seq_length, :]

            for i in range(1, seq_length):
                # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
                rect_tracker = rects_pred[i, :]
                rect_anno = rects_gt[i, :]
                if (np.isnan(rect_tracker).any() or not np.isreal(rect_tracker).all() or rect_tracker[2] <= 0 or rect_tracker[3] <= 0) and not np.isnan(rect_anno).any():
                    rects_pred[i, :] = rects_pred[i-1, :]
                    # Reader memo: 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.

            rects_pred[0, :] = rects_gt[0, :]
            # 预处理完毕

            center_gt = np.column_stack((rects_gt[:, 0] + (rects_gt[:, 2] - 1) / 2,
                                        rects_gt[:, 1] + (rects_gt[:, 3] - 1) / 2))
            # Reader memo: groundtruth框中心坐标

            center_pred = np.column_stack((rects_pred[:, 0] + (rects_pred[:, 2] - 1) / 2,
                                    rects_pred[:, 1] + (rects_pred[:, 3] - 1) / 2))
            # Reader memo: tracker框中心坐标

            if norm_dst:
                center_pred[:, 0] /= rects_gt[:, 2]
                center_pred[:, 1] /= rects_gt[:, 3]
                center_gt[:, 0] /= rects_gt[:, 2]
                center_gt[:, 1] /= rects_gt[:, 3]

            center_error = np.sqrt(np.sum((center_pred - center_gt) ** 2, axis=1))

            # index = anno > 0
            # idx = np.all(index, axis=1)
            idx = np.all(rects_gt > 0, axis=1)
            # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
            # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
            # 这回答了之前关于丢失目标的疑问.

            overlap_rate = -np.ones(len(idx))
            overlap_rate[idx] = calc_overlap_rate(rects_pred[idx, :], rects_gt[idx, :])
            center_error[~idx] = -1
            
            for t_idx, threshold in enumerate(overlap_rate_thresholds):
                overlap_rate_plots[k, i, t_idx] = np.sum(overlap_rate > threshold) / len(rects_gt)

            for t_idx, threshold in enumerate(center_error_thresholds):
                center_error_plots[k, i, t_idx] = np.sum(center_error <= threshold) / len(rects_gt)

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)

    # dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_{eval_type}.npz')
    dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_rate.npz')
    np.savez(dataName1, ave_success_rate_plot=overlap_rate_plots, name_tracker_all=name_tracker_all)

    # dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_error_{eval_type}.npz')
    dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_center_error.npz')
    np.savez(dataName2, ave_success_rate_plot=center_error_plots, name_tracker_all=name_tracker_all)
    
    return overlap_rate_plots, center_error_plots

def eval_tracker(data_resource: BaseImageSequenceDataResource, tracker: str, tmp_mat_path, pred_dir, norm_dst):
    '''
    return: overlap_rate_plot, center_error_plot
        overlap_rate_plot: shape=(num_seq, num_thresholds)
        center_error_plot: shape=(num_seq, num_thresholds)
    '''
    overlap_rate_thresholds = np.arange(0, 1.05, 0.05)
    center_error_thresholds = np.arange(0, 51)
    precision_thresholds = np.arange(0, 1.05, 0.05)
    recall_thresholds = np.arange(0, 1.05, 0.05)
    f_score_thresholds = np.arange(0, 1.05, 0.05)
    
    if norm_dst:
        center_error_thresholds = center_error_thresholds / 100

    overlap_rate_curve_of_sequences = np.zeros((len(data_resource), len(overlap_rate_thresholds)))
    center_error_curve_of_sequences = np.zeros((len(data_resource), len(center_error_thresholds)))
    precision_curve_of_sequences = np.zeros((len(data_resource), len(precision_thresholds)))
    recall_curve_of_sequences = np.zeros((len(data_resource), len(recall_thresholds)))
    f_score_curve_of_sequences = np.zeros((len(data_resource), len(f_score_thresholds)))

    for i_seq, seq in enumerate(data_resource):  # for each sequence
        seq: BaseImageSequence
        # rects_gt = seq.column('bbox_left', 'bbox_top', 'bbox_width', 'bbox_height').to_numpy()
        rects_gt = seq.bboxes_ltwh
        # rects_gt = np.loadtxt(os.path.join(dataset_dir, f'{seq}/groundtruth.txt'), delimiter=',') # FIXME: convert to Dataset class in the future
        # import cv2
        # height, width, n_channel = cv2.imread(os.path.join(dataset_dir, f'{seq}/color/00000001.jpg')).shape # FIXME
        # height, width, n_channel = seq.images[0].shape # CHECK IT
        width, height = seq.width, seq.height

        # res_file = os.path.join(rp_all, f'{t["name"]}_tracking_result/{s}.txt')
        # pred_file = os.path.join(pred_dir, f'{seq.name}.txt') # DEBUG
        pred_file = os.path.join(pred_dir, f'{seq.name}.csv')
        if not os.path.exists(pred_file):
            print(f"File {pred_file} not found, skipping...")
            continue

        # rects_pred = np.loadtxt(pred_file)
        df_pred = pd.read_csv(pred_file)
        rects_pred = df_pred[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']].to_numpy()
        print(f'evaluating {tracker} on {seq.name} ...')

        if rects_pred.size == 0:
            # continue # ???
            break

        # overlap_rate, center_error = calculate_seq_overlap_center(rects_pred, rects_gt, norm_dst)

        seq_length = rects_gt.shape[0]

        if rects_pred.shape[0] != seq_length:
            rects_pred = rects_pred[:seq_length, :]

        def find_first_true_index(arr)-> int: return int(np.argmax(arr))
        first_valid_index = find_first_true_index(seq.valid)
        # first_valid_index: usually 0, but sometimes > 0.

        for i_frame in range(first_valid_index + 1, seq_length):
            # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
            rect_pred = rects_pred[i_frame, :]
            rect_gt = rects_gt[i_frame, :]
            if (np.isnan(rect_pred).any() or not np.isreal(rect_pred).all() or rect_pred[2] <= 0 or rect_pred[3] <= 0) and not np.isnan(rect_gt).any():
                rects_pred[i_frame, :] = rects_pred[i_frame-1, :]
                # 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.
                # 此外, 有些sequence从第一帧开始是invalid, 因此需要跳过这些帧.

        # rects_pred[0, :] = rects_gt[0, :]
        rects_pred[:first_valid_index+1, :] = rects_gt[:first_valid_index+1, :]

        center_gt = np.column_stack((rects_gt[:, 0] + (rects_gt[:, 2] - 1) / 2,
                                    rects_gt[:, 1] + (rects_gt[:, 3] - 1) / 2))
        # Reader memo: groundtruth框中心坐标

        center_pred = np.column_stack((rects_pred[:, 0] + (rects_pred[:, 2] - 1) / 2,
                                rects_pred[:, 1] + (rects_pred[:, 3] - 1) / 2))
        # Reader memo: tracker框中心坐标

        if norm_dst:
            center_pred[:, 0] /= rects_gt[:, 2]
            center_pred[:, 1] /= rects_gt[:, 3]
            center_gt[:, 0] /= rects_gt[:, 2]
            center_gt[:, 1] /= rects_gt[:, 3]

        center_error = np.sqrt(np.sum((center_pred - center_gt) ** 2, axis=1))

        # index = anno > 0
        # idx = np.all(index, axis=1)
        idx = np.all(rects_gt > 0, axis=1)
        # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
        # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
        # 这回答了之前关于丢失目标的疑问.
        
        bboxes_pred = rects_pred.copy()
        bboxes_pred[:, 2] += bboxes_pred[:, 0]
        bboxes_pred[:, 3] += bboxes_pred[:, 1]
        bboxes_gt = rects_gt.copy()
        bboxes_gt[:, 2] += bboxes_gt[:, 0]
        bboxes_gt[:, 3] += bboxes_gt[:, 1]
        
        area_overlap = (np.maximum(
                            0, 
                            np.minimum(bboxes_pred[:, 2], bboxes_gt[:, 2]) - np.maximum(bboxes_pred[:, 0], bboxes_gt[:, 0])
                            # min(right_pred, right_gt) - max(left_pred, left_gt)
                        )) * (np.maximum(
                            0, 
                            np.minimum(bboxes_pred[:, 3], bboxes_gt[:, 3]) - np.maximum(bboxes_pred[:, 1], bboxes_gt[:, 1])
                            # min(bottom_pred, bottom_gt) - max(top_pred, top_gt)
                        ))
        area_pred = rects_pred[:, 2] * rects_pred[:, 3]
        area_gt = rects_gt[:, 2] * rects_gt[:, 3]
        area_all = height * width # scalar
        
        overlap_rate = area_overlap / (area_pred + area_gt - area_overlap)
        overlap_rate[~idx] = -1
        center_error[~idx] = -1

        # calculate confusion matrix
        true_positive = area_overlap
        false_positive = area_pred - area_overlap
        false_negative = area_gt - area_overlap
        true_negative = area_all - area_pred - area_gt + area_overlap
        true_positive_rate = true_positive / (true_positive + false_negative) # might encounter zero division
        false_positive_rate = false_positive / (false_positive + true_negative) # might encounter zero division
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

        for t_idx, threshold in enumerate(overlap_rate_thresholds):
            overlap_rate_curve_of_sequences[i_seq, t_idx] = np.sum(overlap_rate > threshold) / len(rects_gt)

        for t_idx, threshold in enumerate(center_error_thresholds):
            center_error_curve_of_sequences[i_seq, t_idx] = np.sum(center_error <= threshold) / len(rects_gt)
        
        for t_idx, threshold in enumerate(precision_thresholds):
            precision_curve_of_sequences[i_seq, t_idx] = np.sum(precision > threshold) / len(rects_gt)
            
        for t_idx, threshold in enumerate(recall_thresholds):
            recall_curve_of_sequences[i_seq, t_idx] = np.sum(recall > threshold) / len(rects_gt)
            
        for t_idx, threshold in enumerate(f_score_thresholds):
            f_score_curve_of_sequences[i_seq, t_idx] = np.sum(f_score > threshold) / len(rects_gt)

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)

    # dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_{eval_type}.npz')
    dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_alg_overlap_rate.npz')
    np.savez(dataName1, ave_success_rate_plot=overlap_rate_curve_of_sequences, tracker=tracker)

    # dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_error_{eval_type}.npz')
    dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_alg_center_error.npz')
    np.savez(dataName2, ave_success_rate_plot=center_error_curve_of_sequences, tracker=tracker)
    
    return overlap_rate_curve_of_sequences, center_error_curve_of_sequences, precision_curve_of_sequences, recall_curve_of_sequences, f_score_curve_of_sequences

def plot_draw_save(num_tracker, plot_style, plots, idx_seq_set, rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf):
    plt.figure(figsize=(10, 6))
    for i in range(num_tracker):
        # 确保索引有效
        if plots is None:
            raise ValueError("ave_success_rate_plot is None")
        if plot_style is None:
            raise ValueError("plot_style is None")

        ax = plt.subplot(1, 1, 1)
        ax.plot(threshold_set, plots[i, idx_seq_set, :].mean(axis=0), color=plot_style[i]['color'], linestyle=plot_style[i]['lineStyle'], label=name_tracker_all[i])

    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.title(title_name)
    plt.legend(loc='best')
    plt.grid(True)

    if save_fig_suf == 'both':
        suffixes = ['png', 'eps']
    else:
        suffixes = [save_fig_suf]

    for suffix in suffixes:
        fig_path = os.path.join(save_fig_path, f'{fig_name}.{suffix}')
    plt.savefig(fig_path)
    plt.close()

def calculate_metrics(work_dir, data_resource: BaseImageSequenceDataResource, pred_dir, save_fig_suf='both', metric_types=['PR', 'SR', 'Pr', 'Re', 'F']):
    '''
    dataset_name: 数据集名称
    dataset_dir: 数据集路径
    pred_dir: 预测结果路径, 例如work_dirs/evaluate/20241206-105725-evaluate/results/DepthTrack_test
    save_fig_suf: 图表后缀，'png' 或 'eps' 或 'both'
    metrics: 评估指标列表
    
    metric命名规则: xxx_rate越大越好, xxx_error越小越好. 不过, 经过AUC (Area Under the Curve) 转换, 一般来说AUC值越大越好.
    '''
    print('Info: this function is transferred from LasHeR_matlab_toolkit.m, but we use it on depthtrack evaluation (for now).')
    
    assert save_fig_suf in ['png', 'eps', 'both'], "save_fig_path should be 'png' or 'eps' or 'both'"

    tmp_mat_path = os.path.join(work_dir, 'tmp_mat')
    save_fig_path = os.path.join(work_dir, f'res_fig.{save_fig_suf}')
    # config_file_path = os.path.join(work_dir, 'config.yaml')

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    # att_name = [
    #     'Scale Variation', 'Fast Motion', 'Object Disappear', 'Illumination Variation',
    #     'Camera Motion', 'Motion Blur', 'Background Clutter', 'Similar Object',
    #     'Deformation', 'Partial Occlusion', 'Full Occlusion', 'Abrupt Motion',
    #     'Tiny Object', 'Low Illumination'
    # ]
    # att_fig_name = [
    #     'SV', 'FM', 'OD', 'IV', 'CM', 'MB', 'BC', 'SO', 'DEF', 'PO', 'FO', 'AM', 'TO', 'LI'
    # ]

    evaluation_dataset_type = 'all'  # 评估数据集类型 # FIXME

    norm_dst = False  # 是否使用归一化

    # trackers = get_trackers()
    tracker = 'ProMFT'
    # sequences = get_sequences(evaluation_dataset_type, data_resource)

    center_error_thresholds = np.arange(0, 51)
    if norm_dst:
        center_error_thresholds = center_error_thresholds / 100
    overlap_rate_thresholds = np.arange(0, 1.05, 0.05)
    precision_thresholds = np.arange(0, 1.05, 0.05)
    recall_thresholds = np.arange(0, 1.05, 0.05)
    f_score_thresholds = np.arange(0, 1.05, 0.05)

    # 评估跟踪器性能
    overlap_rate_curve_of_sequences, center_error_curve_of_sequences, precision_curve_of_sequences, recall_curve_of_sequences, f_score_curve_of_sequences = eval_tracker(data_resource, tracker, tmp_mat_path, pred_dir, norm_dst)

    def AUC(x, y, max_y=None):
        # self-defined AUC function (Area Under the Curve)
        x = np.array(x)
        y = np.array(y)
        assert x.ndim==1, "x should be a 1-D array"
        assert y.ndim in [1,2], "y should be a 1-D or 2-D array"
        assert x.shape[0]==y.shape[-1], "x and y should have the same length"
        idx = np.argsort(x)
        x = x[idx]
        y = y[:, idx] if y.ndim==2 else y[idx]
        axis = 1 if y.ndim==2 else 0
        if not max_y: max_y = np.max(y, axis=axis)
        if y.ndim==1:
            area = np.sum([max(y[i], y[i+1])*(x[i+1]-x[i]) for i in range(len(x)-1)])
            return area / (max_y*(x[-1]-x[0]))
        else: # y.ndim==2
            area = (np.max([y[:, :-1], y[:,1:]], axis=0)*(x[1:]-x[:-1])).sum(axis=1)
            return area / (max_y*(x[-1]-x[0]))
    
    # 绘制整体性能图表
    results = {}
    for metric_type in metric_types:
        if metric_type in ['Center Error', 'Precision Rate', 'PR']:
            # AUC of **average center error among all sequences, determined by threshold**
            # Each frame has a center error;
            # Each sequence has a center error - threshold curve, showing the rate of frames with center error less than or equal to the threshold;
            # A dataset has an average center error - threshold curve. Given a threshold, the value of the curve is the average of values from all sequences at that threshold.
            # MENTION: DO NOT CONFUSE Precision Rate (PR) with Precision (Pr). They are different.
            results['PR'] = AUC(center_error_thresholds, center_error_curve_of_sequences, max_y=1).mean()
        elif metric_type in ['Overlap Rate', 'Success Rate', 'SR']:
            # AUC of **average overlap rate among all sequences, determined by threshold**
            # Each frame has an overlap rate;
            # Each sequence has an overlap rate - threshold curve, showing the rate of frames with overlap rate greater than or equal to the threshold;
            # A dataset has an average overlap rate - threshold curve. Given a threshold, the value of the curve is the average of values from all sequences at that threshold.
            results['SR'] = AUC(overlap_rate_thresholds, overlap_rate_curve_of_sequences, max_y=1).mean()
        elif metric_type in ['Precision', 'Pr']:
            results['Precision'] = AUC(precision_thresholds, precision_curve_of_sequences, max_y=1).mean()
        elif metric_type in ['Recall', 'Recall Rate', 'Re']:
            results['Recall'] = AUC(recall_thresholds, recall_curve_of_sequences, max_y=1).mean()
        elif metric_type in ['F-score', 'F1-score', 'F-measure', 'F']:
            results['F-score'] = AUC(f_score_thresholds, f_score_curve_of_sequences, max_y=1).mean()
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    return results

def get_plots(work_dir, dataset_name, dataset_dir, pred_dir, save_fig_suf='both', plots=[]):
    '''
    dataset_name: 数据集名称
    dataset_dir: 数据集路径
    pred_dir: 预测结果路径, 例如work_dirs/evaluate/20241206-105725-evaluate/results/DepthTrack_test
    save_fig_suf: 图表后缀，'png' 或 'eps' 或 'both'
    plots: 图表类别, 'success' 或 'precision' 或 'error' 或 'overlap'
    '''
    print('Info: this function is transferred from LasHeR_matlab_toolkit.m, but we use it on depthtrack evaluation (for now).')
    
    assert save_fig_suf in ['png', 'eps', 'both'], "save_fig_path should be 'png' or 'eps' or 'both'"

    tmp_mat_path = os.path.join(work_dir, 'tmp_mat')
    save_fig_path = os.path.join(work_dir, f'res_fig.{save_fig_suf}')
    # config_file_path = os.path.join(work_dir, 'config.yaml')

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    # att_name = [
    #     'Scale Variation', 'Fast Motion', 'Object Disappear', 'Illumination Variation',
    #     'Camera Motion', 'Motion Blur', 'Background Clutter', 'Similar Object',
    #     'Deformation', 'Partial Occlusion', 'Full Occlusion', 'Abrupt Motion',
    #     'Tiny Object', 'Low Illumination'
    # ]
    # att_fig_name = [
    #     'SV', 'FM', 'OD', 'IV', 'CM', 'MB', 'BC', 'SO', 'DEF', 'PO', 'FO', 'AM', 'TO', 'LI'
    # ]

    evaluation_dataset_type = 'all'  # 评估数据集类型

    norm_dst = False  # 是否使用归一化

    trackers = get_trackers()
    sequences = get_sequences(evaluation_dataset_type, pred_dir)
    plot_style = get_plot_style()

    num_seq = len(sequences)
    num_tracker = len(trackers)

    # 加载跟踪器信息
    name_tracker_all = [tracker['name'] for tracker in trackers]

    # 参数设置
    metric_type_set = ['error', 'overlap']
    eval_type = 'OPE'
    ranking_type = 'threshold'  # 可以改为 'AUC' 来绘制成功图

    rank_num = 48

    threshold_set_error = np.arange(0, 51)
    if norm_dst:
        threshold_set_error = threshold_set_error / 100
    threshold_set_overlap = np.arange(0, 1.05, 0.05)

    # 评估跟踪器性能
    ave_success_rate_plot, ave_success_rate_plot_err = eval_tracker(sequences, trackers, name_tracker_all, tmp_mat_path, dataset_dir, pred_dir, norm_dst)

    # 打印 ave_success_rate_plot 的值和形状
    print(f"ave_success_rate_plot shape: {ave_success_rate_plot.shape}")
    print(f"ave_success_rate_plot: {ave_success_rate_plot}")

    # 绘制整体性能图表
    for metric_type in metric_type_set:
        if metric_type == 'error':
            threshold_set = threshold_set_error
            rank_idx = 21 # unused
            x_label_name = 'Center error threshold'
            y_label_name = 'Precision'
        elif metric_type == 'overlap':
            threshold_set = threshold_set_overlap
            rank_idx = 11 # unused
            x_label_name = 'Overlap threshold'
            y_label_name = 'Success rate'

        if metric_type == 'error' and ranking_type == 'AUC':
            continue

        plot_type = f"{metric_type}_{eval_type}"
        title_name = f"{'Normalized ' if norm_dst else ''}Precision plots of {eval_type} on SOT" if metric_type == 'error' else f"Success plots of {eval_type} on SOT"
        fig_name = f"{plot_type}_{ranking_type}"

        # plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, np.arange(num_seq), rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
        plot_draw_save(num_tracker, plot_style, ave_success_rate_plot if metric_type == 'overlap' else ave_success_rate_plot_err, np.arange(num_seq), rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
        
        
if __name__ == '__main__':
    # DEBUG
    from lib.data.data_resource.depthtrack import DepthTrack
    data_resource = DepthTrack('datasets/depthtrack/test') # BROKEN
    metrics = calculate_metrics('work_dirs/evaluate/20241216-234421-evaluate', data_resource, 'work_dirs/evaluate/20241216-234421-evaluate/results/DepthTrack_test')
    print(metrics)
    
    
    