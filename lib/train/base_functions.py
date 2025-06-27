import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
# from ..data.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
# from ..data.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
# from ..data.dataset import VisEvent, LasHeR, DepthTrack_origin
from ..data.data_resource import DepthTrack, LasHeR, VisEvent
from ..data.data_resource.quicktest import QuickTest
from ..data.utils import sampler, opencv_loader, processing, LTRLoader
from ..data.utils import transforms as tfm
from ..data.data_mixer import RandomDataMixer
from ..utils.misc import is_main_process
from ..config import Config

def update_settings(settings, cfg):
    logger.warning('update_settings is deprecated, please use config.yaml and lib.config.MainConfig instead.')
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer

def createDataset(name, path, use_lmdb, image_loader, lazy_load=False):
    logger.info(f'Creating dataset: {name}...')
    assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val", "COCO17", "VID", "TRACKINGNET",
                    "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train", "LasHeR_val", "VisEvent_train", "VisEvent_test",
                    "QuickTest_train", "QuickTest_val", "QuickTest_test"]
    res = name.split('_')
    if len(res) == 1: name, split = name, 'all'
    elif len(res) == 2: name, split = res
    else: raise ValueError(f"Invalid dataset name {name}")
    if name.startswith('DepthTrack'): result = DepthTrack(path, split=split)
    elif name.startswith('LasHeR'): result = LasHeR(path, split=split)
    elif name.startswith('VisEvent'): result = VisEvent(path, split=split)
    elif name.startswith('QuickTest'): result = QuickTest(path, split=split)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
    # if not lazy_load:
    #     logger.debug(f'Force loading sequences of dataset {name}...')
    #     result.force_load_sequences()
    #     logger.debug(f'Force loading sequences of dataset {name} done.')
    logger.debug(f'Dataset {name} created.')
    return result
    # if name == "DepthTrack_train":
    #     return DepthTrack_origin(path, dtype='rgbcolormap', split='train')
    # elif name == "DepthTrack_val":
    #     return DepthTrack_origin(path, dtype='rgbcolormap', split='val')
    # elif name == "LasHeR_all":
    #     return LasHeR(path, dtype='rgbrgb', split='all')
    # elif name == "LasHeR_train":
    #     return LasHeR(path, dtype='rgbrgb', split='train')
    # elif name == "LasHeR_val":
    #     return LasHeR(path, dtype='rgbrgb', split='val')
    # elif name == "VisEvent_train":
    #     return VisEvent(path, dtype='rgbrgb', split='train')
    # elif name == "VisEvent_test":
    #     return VisEvent(path, dtype='rgbrgb', split='test')
    # elif name == "VOT-RGBD2022":
    #     return None # FIXME
    # Following is rgb dataset
    # elif name == "LASOT":
    #     if use_lmdb:
    #         print("Building lasot dataset from lmdb")
    #         return Lasot_lmdb(path, split='train', image_loader=image_loader)
    #     else:
    #         return Lasot(path, split='train', image_loader=image_loader)
    # elif name == "GOT10K_vottrain":
    #     if use_lmdb:
    #         print("Building got10k from lmdb")
    #         return Got10k_lmdb(path, split='vottrain', image_loader=image_loader)
    #     else:
    #         return Got10k(path, split='vottrain', image_loader=image_loader)
    # elif name == "GOT10K_train_full":
    #     if use_lmdb:
    #         print("Building got10k_train_full from lmdb")
    #         return Got10k_lmdb(path, split='train_full', image_loader=image_loader)
    #     else:
    #         return Got10k(path, split='train_full', image_loader=image_loader)
    # elif name == "GOT10K_votval":
    #     if use_lmdb:
    #         print("Building got10k from lmdb")
    #         return Got10k_lmdb(path, split='votval', image_loader=image_loader)
    #     else:
    #         return Got10k(path, split='votval', image_loader=image_loader)
    # elif name == "GOT10K_official_val":
    #     if use_lmdb:
    #         raise ValueError("Not implement")
    #     else:
    #         return Got10k(path, split=None, image_loader=image_loader)
    # elif name == "COCO17":
    #     if use_lmdb:
    #         print("Building COCO2017 from lmdb")
    #         return MSCOCOSeq_lmdb(path, version="2017", image_loader=image_loader)
    #     else:
    #         return MSCOCOSeq(path, version="2017", image_loader=image_loader)
    # elif name == "VID":
    #     if use_lmdb:
    #         print("Building VID from lmdb")
    #         return ImagenetVID_lmdb(path, image_loader=image_loader)
    #     else:
    #         return ImagenetVID(path, image_loader=image_loader)
    # elif name == "TRACKINGNET":
    #     if use_lmdb:
    #         print("Building TrackingNet from lmdb")
    #         return TrackingNet_lmdb(path, image_loader=image_loader)
    #     else:
    #         # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
    #         return TrackingNet(path, image_loader=image_loader)
    # else:
    #     raise ValueError("Unsupported dataset")

def build_dataloaders(CONFIG):
    logger.info('Building dataloaders...')
    local_rank = CONFIG.GENERAL.LOCAL_RANK
    
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    # transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
    #                                 tfm.RandomHorizontalFlip(probability=0.5))
    def vanilla_transform(image, bbox, *argv, **kwargs): return image, bbox # TODO: FIXME
    transform_joint = vanilla_transform

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=CONFIG.DATA.MEAN, std=CONFIG.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=CONFIG.DATA.MEAN, std=CONFIG.DATA.STD))

    # The tracking pairs processing module
    output_sz = Config(template = CONFIG.DATA.TEMPLATE.SIZE,
                       search = CONFIG.DATA.SEARCH.SIZE)
    search_area_factor = Config(template = CONFIG.DATA.TEMPLATE.FACTOR,
                                search = CONFIG.DATA.SEARCH.FACTOR)
    center_jitter_factor = Config(template = CONFIG.DATA.TEMPLATE.CENTER_JITTER,
                                  search = CONFIG.DATA.SEARCH.CENTER_JITTER)
    scale_jitter_factor = Config(template = CONFIG.DATA.TEMPLATE.SCALE_JITTER,
                                 search = CONFIG.DATA.SEARCH.SCALE_JITTER)

    data_processing_train = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=center_jitter_factor,
                                                       scale_jitter_factor=scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                    #    settings=settings
                                                       )

    data_processing_val = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=center_jitter_factor,
                                                     scale_jitter_factor=scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                    #  settings=settings
                                                     )

    # Train sampler and loader
    # dataset_train = sampler.TrackingSampler(
    #     data_resources=[createDataset(d.NAME, d.PATH, CONFIG.GENERAL.USE_LMDB, opencv_loader) 
    #               for d in CONFIG.DATA.TRAIN.DATASETS],
    #     ratios=[d.RATIO for d in CONFIG.DATA.TRAIN.DATASETS],
    #     samples_per_epoch=CONFIG.DATA.TRAIN.SAMPLE_PER_EPOCH,
    #     max_gap=CONFIG.DATA.MAX_SAMPLE_INTERVAL, 
    #     num_search_frames=CONFIG.DATA.SEARCH.NUMBER,
    #     num_template_frames=CONFIG.DATA.TEMPLATE.NUMBER, 
    #     processing=data_processing_train,
    #     frame_sample_mode=CONFIG.DATA.SAMPLER_MODE, 
    #     train_cls=CONFIG.TRAIN.TRAIN_CLS
    # )
    dataset_train = RandomDataMixer(
        data_resources=[createDataset(d.NAME, d.PATH, CONFIG.GENERAL.USE_LMDB, opencv_loader) 
                  for d in CONFIG.DATA.TRAIN.DATASETS],
        ratios=[d.RATIO for d in CONFIG.DATA.TRAIN.DATASETS],
        samples_per_epoch=CONFIG.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=CONFIG.DATA.MAX_SAMPLE_INTERVAL, 
        num_search_frames=CONFIG.DATA.SEARCH.NUMBER,
        num_template_frames=CONFIG.DATA.TEMPLATE.NUMBER, 
        processing=data_processing_train,
        frame_sample_mode=CONFIG.DATA.SAMPLER_MODE, 
        # train_cls=CONFIG.TRAIN.TRAIN_CLS
    )

    train_sampler = DistributedSampler(dataset_train) if local_rank != -1 else None
    shuffle = True if local_rank == -1 else False

    loader_train = LTRLoader('train', 
                             dataset_train, 
                             training=True, 
                             batch_size=CONFIG.TRAIN.BATCH_SIZE, 
                             shuffle=shuffle,
                             num_workers=0 if CONFIG.GENERAL.DEBUG else CONFIG.TRAIN.NUM_WORKER, 
                             drop_last=True, 
                             stack_dim=1, 
                             sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if len(CONFIG.DATA.VAL.DATASETS) == 0:
        loader_val = None
    else:
        # dataset_val = sampler.TrackingSampler(
        #     data_resources=[createDataset(d.NAME, d.PATH, CONFIG.GENERAL.USE_LMDB, opencv_loader)
        #               for d in CONFIG.DATA.VAL.DATASETS],
        #     ratios=[d.RATIO for d in CONFIG.DATA.VAL.DATASETS],
        #     samples_per_epoch=CONFIG.DATA.VAL.SAMPLE_PER_EPOCH,
        #     max_gap=CONFIG.DATA.MAX_SAMPLE_INTERVAL, 
        #     num_search_frames=CONFIG.DATA.SEARCH.NUMBER,
        #     num_template_frames=CONFIG.DATA.TEMPLATE.NUMBER, 
        #     processing=data_processing_val,
        #     frame_sample_mode=CONFIG.DATA.SAMPLER_MODE, 
        #     train_cls=CONFIG.TRAIN.TRAIN_CLS)
        dataset_val = RandomDataMixer(
            data_resources=[createDataset(d.NAME, d.PATH, CONFIG.GENERAL.USE_LMDB, opencv_loader)
                      for d in CONFIG.DATA.VAL.DATASETS],
            ratios=[d.RATIO for d in CONFIG.DATA.VAL.DATASETS],
            samples_per_epoch=CONFIG.DATA.VAL.SAMPLE_PER_EPOCH,
            max_gap=CONFIG.DATA.MAX_SAMPLE_INTERVAL, 
            num_search_frames=CONFIG.DATA.SEARCH.NUMBER,
            num_template_frames=CONFIG.DATA.TEMPLATE.NUMBER, 
            processing=data_processing_val,
            frame_sample_mode=CONFIG.DATA.SAMPLER_MODE, 
            # train_cls=CONFIG.TRAIN.TRAIN_CLS
            )
        val_sampler = DistributedSampler(dataset_val) if local_rank != -1 else None
        loader_val = LTRLoader('val', 
                               dataset_val, 
                               training=False, 
                               batch_size=CONFIG.TRAIN.BATCH_SIZE,
                               num_workers=0 if CONFIG.GENERAL.DEBUG else CONFIG.TRAIN.NUM_WORKER, 
                               drop_last=True, 
                               stack_dim=1, 
                               sampler=val_sampler,
                               epoch_interval=CONFIG.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, CONFIG):
    prompt_type = getattr(CONFIG.TRAIN.PROMPT, "TYPE", "")
    if 'vipt' in prompt_type or 'promft' in prompt_type: # SHITCODE: 尽量不要用字符串逻辑, 而应尽量使用含类名/方法名的逻辑, 以便阅读和维护.
        # print("Only training prompt parameters. They are: ")
        param_dicts = [ # SHITCODE!!!! 耦合严重, 应当使用类/方法
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]}
        ]
        for n, p in net.named_parameters():
            if "prompt" not in n:
                p.requires_grad = False
            # else:
            #     print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": CONFIG.TRAIN.LR * CONFIG.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if CONFIG.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=CONFIG.TRAIN.LR,
                                      weight_decay=CONFIG.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if CONFIG.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG.TRAIN.LR_DROP_EPOCH)
    elif CONFIG.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=CONFIG.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=CONFIG.TRAIN.SCHEDULER.GAMMA)
    else:
        # lr_scheduler = None
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
