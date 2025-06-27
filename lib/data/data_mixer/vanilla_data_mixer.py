import logging
logger = logging.getLogger(__name__)

import torch


class VanillaDataMixer(torch.utils.data.Dataset):
    def __init__(self):
        logger.error("VanillaDataMixer is not implemented yet.")
        raise NotImplementedError("VanillaDataMixer is not implemented yet.")