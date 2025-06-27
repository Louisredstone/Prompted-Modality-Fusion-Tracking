from ..evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/public/dataset/RGBT_Datasets/got10k_lmdb'
    settings.got10k_path = '/home/public/dataset/RGBT_Datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/public/dataset/RGBT_Datasets/itb'
    settings.lasot_extension_subset_path_path = '/home/public/dataset/RGBT_Datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/public/dataset/RGBT_Datasets/lasot_lmdb'
    settings.lasot_path = '/home/public/dataset/RGBT_Datasets/lasot'
    settings.network_path = '/home/zhyzhang/vipt-h/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/public/dataset/RGBT_Datasets/nfs'
    settings.otb_path = '/home/public/dataset/RGBT_Datasets/otb'
    settings.prj_dir = '/home/zhyzhang/vipt-h'
    settings.result_plot_path = '/home/zhyzhang/vipt-h/output/test/result_plots'
    settings.results_path = '/home/zhyzhang/vipt-h/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zhyzhang/vipt-h/output'
    settings.segmentation_path = '/home/zhyzhang/vipt-h/output/test/segmentation_results'
    settings.tc128_path = '/home/public/dataset/RGBT_Datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/public/dataset/RGBT_Datasets/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/public/dataset/RGBT_Datasets/trackingnet'
    settings.uav_path = '/home/public/dataset/RGBT_Datasets/uav'
    settings.vot18_path = '/home/public/dataset/RGBT_Datasets/vot2018'
    settings.vot22_path = '/home/public/dataset/RGBT_Datasets/vot2022'
    settings.vot_path = '/home/public/dataset/RGBT_Datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

