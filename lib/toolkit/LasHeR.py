import os
import numpy as np
import matplotlib.pyplot as plt

def config_plot_style():
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

def config_sequence(type, path_anno):
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

    txt_files = os.listdir(path_anno)
    sequences = [os.path.splitext(txt_file)[0] for txt_file in txt_files if txt_file.endswith('.txt')]

    return sequences

def config_tracker():
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

def calc_seq_err_robust(res, anno, norm_dst):
    '''
    results: tracker的检测结果, shape=(seq_length, 4)
    rect_anno: 标注的矩形框, groundtruth, shape=(seq_length, 4)
    norm_dst: 暂不知晓, 似乎是是否对矩形框坐标进行归一化处理
    '''
    seq_length = anno.shape[0]

    if res.shape[0] != seq_length:
        res = res[:seq_length, :]

    for i in range(1, seq_length):
        # Reader memo: 此处从1开始, 因为第0帧的检测结果是由GT初始化的, 所以不参与评估.
        rect_tracker = res[i, :]
        rect_anno = anno[i, :]
        if (np.isnan(rect_tracker).any() or not np.isreal(rect_tracker).all() or rect_tracker[2] <= 0 or rect_tracker[3] <= 0) and not np.isnan(rect_anno).any():
            res[i, :] = res[i-1, :]
            # Reader memo: 这里的处理方式是，如果检测结果不合法（比如检测框的宽或高为0），则用上一帧的检测结果代替. 但这可能引发问题, 因为有时丢失目标是正常的, 目标物体可能受到遮挡. 因此, 这里需要进一步讨论.

    res2 = res.copy()
    res2[0, :] = anno[0, :]

    center_GT = np.column_stack((anno[:, 0] + (anno[:, 2] - 1) / 2,
                                 anno[:, 1] + (anno[:, 3] - 1) / 2))
    # Reader memo: groundtruth框中心坐标

    center = np.column_stack((res2[:, 0] + (res2[:, 2] - 1) / 2,
                             res2[:, 1] + (res2[:, 3] - 1) / 2))
    # Reader memo: tracker框中心坐标

    if norm_dst:
        center[:, 0] /= anno[:, 2]
        center[:, 1] /= anno[:, 3]
        center_GT[:, 0] /= anno[:, 2]
        center_GT[:, 1] /= anno[:, 3]

    err_center = np.sqrt(np.sum((center - center_GT) ** 2, axis=1))

    # index = anno > 0
    # idx = np.all(index, axis=1)
    idx = np.all(anno > 0, axis=1)
    # Reader memo: 这里的idx是指标注矩形框中有目标的帧, 即idx=True的位置.
    # 换言之, 如果有些帧的gt显示没有目标/目标遮挡/其他不合法的情况, 则这些帧的检测结果也不参与评估.
    # 这回答了之前关于丢失目标的疑问.

    overlap_rate = calc_overlap_rate(res2[idx, :], anno[idx, :])

    err_overlap = -np.ones(len(idx))
    err_overlap[idx] = overlap_rate
    err_center[~idx] = -1

    return err_overlap, err_center

def eval_tracker(seqs, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst):
    num_tracker = len(trackers)
    threshold_set_overlap = np.arange(0, 1.05, 0.05)
    threshold_set_error = np.arange(0, 51)
    if norm_dst:
        threshold_set_error = threshold_set_error / 100

    ave_success_rate_plot = np.zeros((num_tracker, len(seqs), len(threshold_set_overlap)))
    ave_success_rate_plot_err = np.zeros((num_tracker, len(seqs), len(threshold_set_error)))

    for i, s in enumerate(seqs):  # for each sequence
        anno = np.loadtxt(os.path.join(path_anno, f'{s}.txt'), delimiter=',')

        for k, t in enumerate(trackers):  # evaluate each tracker
            # res_file = os.path.join(rp_all, f'{t["name"]}_tracking_result/{s}.txt')
            res_file = os.path.join(rp_all, f'{s}.txt') # DEBUG
            if not os.path.exists(res_file):
                print(f"File {res_file} not found, skipping...")
                continue

            res = np.loadtxt(res_file)
            print(f'evaluating {t["name"]} on {s} ...')

            if res.size == 0:
                # continue # ???
                break

            err_coverage, err_center = calc_seq_err_robust(res, anno, norm_dst)

            for t_idx, threshold in enumerate(threshold_set_overlap):
                ave_success_rate_plot[k, i, t_idx] = np.sum(err_coverage > threshold) / len(anno)

            for t_idx, threshold in enumerate(threshold_set_error):
                ave_success_rate_plot_err[k, i, t_idx] = np.sum(err_center <= threshold) / len(anno)

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)

    dataName1 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_overlap_{eval_type}.npz')
    np.savez(dataName1, ave_success_rate_plot=ave_success_rate_plot, name_tracker_all=name_tracker_all)

    dataName2 = os.path.join(tmp_mat_path, f'aveSuccessRatePlot_{num_tracker}alg_error_{eval_type}.npz')
    np.savez(dataName2, ave_success_rate_plot=ave_success_rate_plot_err, name_tracker_all=name_tracker_all)
    
    return ave_success_rate_plot, ave_success_rate_plot_err

def plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf):
    # 打印调试信息
    print(f"ave_success_rate_plot shape: {ave_success_rate_plot.shape}")
    print(f"ave_success_rate_plot: {ave_success_rate_plot}")

    plt.figure(figsize=(10, 6))
    for i in range(num_tracker):
        # 确保索引有效
        if ave_success_rate_plot is None:
            raise ValueError("ave_success_rate_plot is None")
        if plot_style is None:
            raise ValueError("plot_style is None")

        ax = plt.subplot(1, 1, 1)
        ax.plot(threshold_set, ave_success_rate_plot[i, idx_seq_set, :].mean(axis=0), color=plot_style[i]['color'], linestyle=plot_style[i]['lineStyle'], label=name_tracker_all[i])

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

def run(work_dir, path_anno='work_dirs/evaluate/groundtruth/DepthTrack_test/', save_fig_suf='eps'):
    '''
    work_dir: 工作目录
    path_anno: 注释路径
    save_fig_suf: 图表后缀，'png' 或 'eps' 或 'both'
    '''
    print('Info: this function is transferred from LasHeR_matlab_toolkit.m, but we use it on depthtrack evaluation (for now).')
    
    assert save_fig_path in ['png', 'eps', 'both'], "save_fig_path should be 'png' or 'eps' or 'both'"

    rp_all = os.path.join(work_dir, 'results/DepthTrack_test') # FIXME
    tmp_mat_path = os.path.join(work_dir, 'tmp_mat')
    save_fig_path = os.path.join(work_dir,'res_fig')

    if not os.path.exists(tmp_mat_path):
        os.makedirs(tmp_mat_path)
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    att_name = [
        'Scale Variation', 'Fast Motion', 'Object Disappear', 'Illumination Variation',
        'Camera Motion', 'Motion Blur', 'Background Clutter', 'Similar Object',
        'Deformation', 'Partial Occlusion', 'Full Occlusion', 'Abrupt Motion',
        'Tiny Object', 'Low Illumination'
    ]
    att_fig_name = [
        'SV', 'FM', 'OD', 'IV', 'CM', 'MB', 'BC', 'SO', 'DEF', 'PO', 'FO', 'AM', 'TO', 'LI'
    ]

    evaluation_dataset_type = 'all'  # 评估数据集类型

    norm_dst = False  # 是否使用归一化

    trackers = config_tracker()
    sequences = config_sequence(evaluation_dataset_type, path_anno)
    plot_style = config_plot_style()

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
    ave_success_rate_plot, ave_success_rate_plot_err = eval_tracker(sequences, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst)

    # 打印 ave_success_rate_plot 的值和形状
    print(f"ave_success_rate_plot shape: {ave_success_rate_plot.shape}")
    print(f"ave_success_rate_plot: {ave_success_rate_plot}")

    # 绘制整体性能图表
    for metric_type in metric_type_set:
        if metric_type == 'error':
            threshold_set = threshold_set_error
            rank_idx = 21
            x_label_name = 'Location error threshold'
            y_label_name = 'Precision'
        elif metric_type == 'overlap':
            threshold_set = threshold_set_overlap
            rank_idx = 11
            x_label_name = 'Overlap threshold'
            y_label_name = 'Success rate'

        if metric_type == 'error' and ranking_type == 'AUC':
            continue

        plot_type = f"{metric_type}_{eval_type}"
        title_name = f"{'Normalized ' if norm_dst else ''}Precision plots of {eval_type} on SOT" if metric_type == 'error' else f"Success plots of {eval_type} on SOT"
        fig_name = f"{plot_type}_{ranking_type}"

        # plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, np.arange(num_seq), rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
        plot_draw_save(num_tracker, plot_style, ave_success_rate_plot if metric_type == 'overlap' else ave_success_rate_plot_err, np.arange(num_seq), rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)