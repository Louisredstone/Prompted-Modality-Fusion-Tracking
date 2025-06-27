import sys
sys.path.append('.')

from utils import *

import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Run tracker performance evaluation.')
parser.add_argument('--work-dir', type=str, required=True, help='Work directory.')
parser.add_argument('--path-anno', type=str, default='work_dirs/evaluate/groundtruth/DepthTrack_test/', help='Path of annotations.')
# parser.add_argument('--rp-all', type=str, required=True, help='Path of tracking results.')
parser.add_argument('--save-fig-suf', type=str, default='eps', help='Figure suffix, png or eps.')

args = parser.parse_args()

work_dir = args.work_dir
path_anno = args.path_anno
# rp_all = args.rp_all
rp_all = os.path.join(work_dir, 'results/DepthTrack_test')
save_fig_suf = args.save_fig_suf
tmp_mat_path = os.path.join(work_dir, 'tmp_mat')
save_fig_path = os.path.join(work_dir,'res_fig')

if not os.path.exists(tmp_mat_path):
    os.makedirs(tmp_mat_path)
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

# tmp_mat_path = './tmp_mat/'  # 路径保存临时结果
# # path_anno = './annos/'  # 注释路径
# # path_att = './annos/att/'  # 属性路径
# path_anno = '../work_dirs/evaluate/groundtruth/DepthTrack_test/'
# # rp_all = './tracking_results/'  # 跟踪结果路径
# rp_all = '../work_dirs/evaluate/20241126-234934-evaluate/results/DepthTrack_test'
# save_fig_path = './res_fig/'  # 结果图表路径
# save_fig_suf = 'eps'  # 图表后缀，'png' 或 'eps'

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