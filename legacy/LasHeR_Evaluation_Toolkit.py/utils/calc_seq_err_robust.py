import numpy as np
from .calc_rect_int import calc_rect_int

def calc_seq_err_robust(results, rect_anno, norm_dst):
    seq_length = rect_anno.shape[0]

    if results.shape[0] != seq_length:
        results = results[:seq_length, :]

    for i in range(1, seq_length):
        r = results[i, :]
        r_anno = rect_anno[i, :]
        if (np.isnan(r).any() or not np.isreal(r).all() or r[2] <= 0 or r[3] <= 0) and not np.isnan(r_anno).any():
            results[i, :] = results[i-1, :]

    rect_mat = results.copy()
    rect_mat[0, :] = rect_anno[0, :]

    center_GT = np.column_stack((rect_anno[:, 0] + (rect_anno[:, 2] - 1) / 2,
                                 rect_anno[:, 1] + (rect_anno[:, 3] - 1) / 2))

    center = np.column_stack((rect_mat[:, 0] + (rect_mat[:, 2] - 1) / 2,
                             rect_mat[:, 1] + (rect_mat[:, 3] - 1) / 2))

    if norm_dst:
        center[:, 0] /= rect_anno[:, 2]
        center[:, 1] /= rect_anno[:, 3]
        center_GT[:, 0] /= rect_anno[:, 2]
        center_GT[:, 1] /= rect_anno[:, 3]

    err_center = np.sqrt(np.sum((center - center_GT) ** 2, axis=1))

    index = rect_anno > 0
    idx = np.all(index, axis=1)

    tmp = calc_rect_int(rect_mat[idx, :], rect_anno[idx, :])

    errCoverage = -np.ones(len(idx))
    errCoverage[idx] = tmp
    err_center[~idx] = -1

    return errCoverage, err_center