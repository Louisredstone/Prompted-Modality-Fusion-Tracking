from ..utils import TrackerParams
# import os
# from ..evaluation.environment import env_settings
# from ...config.vipt.config import default_config, parse_config_from_file
from ...config import MainConfig

# def parameters(yaml_name: str, epoch=None):
def parameters(CONFIG: MainConfig):
    params = TrackerParams()
    # prj_dir = env_settings().prj_dir
    # save_dir = env_settings().save_dir
    # # update default config from yaml file
    # yaml_file = os.path.join(prj_dir, 'experiments/vipt/%s.yaml' % yaml_name)
    # update_config_from_file(yaml_file)
    params.cfg = CONFIG
    # print("[DEBUG] test config: ", cfg)

    # template and search region
    params.template_factor = CONFIG.TEST.TEMPLATE_FACTOR
    params.template_size = CONFIG.TEST.TEMPLATE_SIZE
    params.search_factor = CONFIG.TEST.SEARCH_FACTOR
    params.search_size = CONFIG.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/vipt/%s/ViPTrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH))
    # params.checkpoint = os.path.join(prj_dir, "./output/checkpoints/train/vipt/deep_rgbt/ViPT_%s.pth"%yaml_name)
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/vipt/%s/ViPTrack_ep0060.pth.tar"%yaml_name)
    params.checkpoint = CONFIG.EVALUATE.CHECKPOINT
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
