# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num, ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf):
#     fig, ax = plt.subplots()
#     for i in range(num_tracker):
#         ax.plot(threshold_set, ave_success_rate_plot[i, idx_seq_set, :].mean(axis=0), color=plot_style[i]['color'], linestyle=plot_style[i]['lineStyle'], label=name_tracker_all[i])
#     ax.set_xlabel(x_label_name)
#     ax.set_ylabel(y_label_name)
#     ax.set_title(title_name)
#     ax.legend()
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     fig.savefig(os.path.join(save_fig_path, f"{fig_name}.{save_fig_suf}"))
#     plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import os

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

    fig_path = os.path.join(save_fig_path, f'{fig_name}.{save_fig_suf}')
    plt.savefig(fig_path)
    plt.close()