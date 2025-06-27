import logging
logger = logging.getLogger(__name__)
import os
from os.path import join as pjoin
import sys
import argparse
import importlib
import random
import numpy as np
import cv2 as cv
import torch.backends.cudnn
torch.backends.cudnn.benchmark = False
import torch.distributed as dist
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils.box_ops import giou_loss
from .trainers import LTRTrainer
from .base_functions import *
from ..models import Tracker
from .actors import ViPTActor
from ..utils.focal_loss import FocalLoss
from ..config import MainConfig

class EnvironmentSettings: # SHITCODE
    def __init__(self):
        logger.warning('This class "EnvironmentSettings" is deprecated and will be removed in the future.')
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

def init_seeds(seed):
    logger.info(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(config_path, local_rank=-1):
    logger.info("Start training process.")
    logger.info(f"Using config file: {config_path}")
    CONFIG = MainConfig.from_file(config_path)
    CONFIG.GENERAL.LOCAL_RANK = local_rank
    # if local_rank in [-1, 0]:
    #     print("New configuration is shown below.")
    #     for key in cfg.keys():
    #         print(f"{key} configuration:", cfg[key])
    #         print('\n')
    if local_rank == -1:
        print(CONFIG)

    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = CONFIG.GENERAL.CUDNN_BENCHMARK

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    # if local_rank in [-1, 0]:
    #     print(f'config_name: {config_name}')

    '''2021.1.5 set seed for different process'''
    base_seed = CONFIG.GENERAL.BASE_SEED
    if local_rank != -1: init_seeds(base_seed + local_rank)
    else: init_seeds(base_seed)

    if CONFIG.GENERAL.DISTILL.ENABLED: raise NotImplementedError()

    # Record the training log
    log_dir = pjoin(CONFIG.GENERAL.WORK_DIR, 'logs')
    if local_rank == -1: os.makedirs(log_dir, exist_ok=True)

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(CONFIG)

    # Create network
    # if settings.script_name == "vipt":
    #     net = build_ostrack_with_prompt(cfg)
    # else:
    #     raise ValueError("illegal script name")
    # if cfg.TRAIN.PROMPT.TYPE in ["promft_deep", "promft"]:
    #     net = ProMFTDeep(cfg)
    # elif cfg.TRAIN.PROMPT.TYPE in ["promft_shallow"]:
    #     net = ProMFTShallow(cfg)
    # elif cfg.TRAIN.PROMPT.TYPE in ["promft_naive"]:
    #     net = ProMFTNaive.build(cfg)
    # elif cfg.TRAIN.PROMPT.TYPE in ["promft_differential"]:
    #     net = ProMFTDifferential(cfg)
    # elif cfg.TRAIN.PROMPT.TYPE in ["vipt"]:
    #     raise NotImplementedError("vipt is not supported yet")
    # else:
    #     raise ValueError(f"illegal prompt type {cfg.TRAIN.PROMPT.TYPE}")
    net = Tracker.build(CONFIG)

    # wrap networks to distributed one
    net.cuda()
    if local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
        CONFIG.GENERAL.DEVICE = torch.device("cuda:%d" % local_rank)
    else:
        CONFIG.GENERAL.DEVICE = torch.device("cuda:0")
    # CONFIG.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    # CONFIG.distill = getattr(cfg.TRAIN, "DISTILL", False)
    # CONFIG.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    # if settings.script_name == "vipt":
    #     # here cls loss and cls weight are not use
    #     focal_loss = FocalLoss()
    #     objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    #     loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
    #     actor = ViPTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    # else:
    #     raise ValueError("illegal script name")
    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': CONFIG.TRAIN.GIOU_WEIGHT, 'l1': CONFIG.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
    actor = ViPTActor(net=net, objective=objective, loss_weight=loss_weight, settings=CONFIG, config=CONFIG) # borrow from vipt

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, CONFIG)
    use_amp = CONFIG.TRAIN.AMP
    # CONFIG.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)
    # CONFIG.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)

    if loader_val is None:
        trainer = LTRTrainer(actor, [loader_train], optimizer, CONFIG, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, CONFIG, lr_scheduler, use_amp=use_amp)
    
    # train process
    trainer.train(CONFIG.TRAIN.EPOCH, load_latest=True, fail_safe=True)
