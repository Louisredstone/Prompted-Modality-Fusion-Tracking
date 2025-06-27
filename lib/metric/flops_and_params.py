import logging
logger = logging.getLogger(__name__)

import torch
from fvcore.nn import FlopCountAnalysis
import numpy as np

from ..models.tracker import Tracker
from ..test import ViPT_Test_Wrapper
from ..data.utils.processing_utils import sample_target

def calculate_flops_and_params(CONFIG):
    logger.info(f'Start calculating flops')
    # work_dir = CONFIG.GENERAL.WORK_DIR
    
    wrapper = ViPT_Test_Wrapper(CONFIG)
    
    # image = np.ones([6, 640, 360])
    image = np.ones([6, 360, 640])
    
    wrapper.initialize(image, [270, 130, 100, 100])

    x_patch_arr, resize_factor, x_amask_arr = sample_target(image, wrapper.state, wrapper.search_factor,
                                                                output_sz=wrapper.search_size)  # (x1, y1, w, h)
    search = wrapper.preprocessor.process(x_patch_arr)
    
    flops = FlopCountAnalysis(wrapper.network, (wrapper.z_tensor, search, wrapper.box_mask_z)).total()
    
    total_params = sum(p.numel() for p in wrapper.network.parameters())
    trainable_params = sum(p.numel() for p in wrapper.network.parameters() if p.requires_grad)
    # backbone_params = sum(p.numel() for p in wrapper.network.backbone.parameters() if p.requires_grad)
    # prompt_params = total_params - backbone_params
    prompt_params = sum(p.numel() for n,p in wrapper.network.backbone.named_parameters() if 'prompt' in n)
    backbone_params = total_params - prompt_params
    return dict(
        flops=flops, 
        total_params=total_params, 
        trainable_params=trainable_params, 
        backbone_params=backbone_params, 
        prompt_params=prompt_params
    )