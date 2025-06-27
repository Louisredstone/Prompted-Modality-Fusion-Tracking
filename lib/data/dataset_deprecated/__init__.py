from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .imagenetvid import ImagenetVID
from .coco import MSCOCO
from .coco_seq import MSCOCOSeq
from .got10k_lmdb import Got10k_lmdb
from .lasot_lmdb import Lasot_lmdb
from .imagenetvid_lmdb import ImagenetVID_lmdb
from .coco_seq_lmdb import MSCOCOSeq_lmdb
from .tracking_net_lmdb import TrackingNet_lmdb
# RGBT dataloader
from .lasher import LasHeR
# RGBD dataloader
from .depthtrack import DepthTrack_origin
# Event dataloader
from .visevent import VisEvent

import logging
logger = logging.getLogger(__name__)
logger.warning('This module "dataset" (dataset_deprecated) is deprecated and will be removed in future releases. Please use data_resource and data_mixer instead.')