import numpy as np
import os
from .calc_seq_err_robust import calc_seq_err_robust
import matplotlib.pyplot as plt

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