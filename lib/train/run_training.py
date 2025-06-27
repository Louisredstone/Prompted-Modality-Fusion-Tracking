import logging
logger = logging.getLogger(__name__)
logger.warning('This file "run_training.py" is deprecated and will be removed in the future.')

import os
import sys
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist

import random
import numpy as np
torch.backends.cudnn.benchmark = False

# from . import _init_paths
# from .admin import settings as ws_settings

from ..config import Config, Required, Optional, Deprecated, Auto, MainConfig

class EnvironmentSettings: # SHITCODE
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        
# def default_settings():
#     from ..config import Config, Required
#     config = Config()
#     config.workspace_dir = Required    # Base directory for saving network checkpoints.
#     config.tensorboard_dir = 'tensorboard/'    # Directory for tensorboard files.
#     config.pretrained_networks = 'pretrained_networks/'
#     # config.lasot_dir = ''
#     # config.got10k_dir = ''
#     # config.trackingnet_dir = ''
#     # config.coco_dir = ''
#     # config.lvis_dir = ''
#     # config.sbd_dir = ''
#     # config.imagenet_dir = ''
#     # config.imagenetdet_dir = ''
#     # config.ecssd_dir = ''
#     # config.hkuis_dir = ''
#     # config.msra10k_dir = ''
#     # config.davis_dir = ''
#     # config.youtubevos_dir = ''
#     return config

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def run_training(script_name, config_path, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
#                  use_lmdb=False, script_name_prv=None, config_name_prv=None, use_wandb=False,
#                  distill=None, script_teacher=None, config_teacher=None):
def run_training(config_path, local_rank=-1):
    """Run the train script.
    args:
        config_name: Name of the yaml file in the "experiments/<script_name>". # FIXME
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    CONFIG = MainConfig.from_file(config_path)
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = CONFIG.GENERAL.CUDNN_BENCHMARK

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if local_rank in [-1, 0]:
        print(f'config_name: {config_name}')

    '''2021.1.5 set seed for different process'''
    base_seed = CONFIG.GENERAL.BASE_SEED
    if local_rank != -1: init_seeds(base_seed + local_rank)
    else: init_seeds(base_seed)

    if CONFIG.DISTILL.ENABLED: raise NotImplementedError()
    
    # from lib.train.train_script import run
    from .train_script import run
    run(CONFIG)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, required=True, help='Name of the train script.')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file.")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, help='the directory to save checkpoints and logs')  # ./output
    parser.add_argument('--seed', type=int, default=0, help='seed for random numbers')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, default=None, help='Name of the train script of previous model.')
    parser.add_argument('--config_prv', type=str, default=None, help="Name of the config file of previous model.")
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, script_name_prv=args.script_prv, config_name_prv=args.config_prv,
                 use_wandb=args.use_wandb,
                 distill=args.distill, script_teacher=args.script_teacher, config_teacher=args.config_teacher)


if __name__ == '__main__':
    main()
