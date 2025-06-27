import logging
logger = logging.getLogger(__name__)

from .base_image_sequence import BaseImageSequence
from .rgb_aux_image_sequence import RGB_AUX_ImageSequence
from .base_data_resource import BaseDataResource
from .base_img_seq_data_resource import BaseImageSequenceDataResource
from .track_result_data_resource import TrackResultDataResource
from .lasher import LasHeR
from .depthtrack import DepthTrack
from .visevent import VisEvent
from .rgbt234 import RGBT234
from .gtot import GTOT
from .vot import VOT_RGBD2022

def build_data_resource(name, root, split=None) -> BaseImageSequenceDataResource:
    if name == 'result':
        return TrackResultDataResource(root)
    if name.endswith('_train'):
        name = name[:-6]
        split = 'train'
    elif name.endswith('_val'):
        name = name[:-4]
        split = 'val'
    elif name.endswith('_test'):
        name = name[:-5]
        split = 'test'
    if name.lower() == 'lasher': return LasHeR(root, split)
    elif name.lower() == 'depthtrack': return DepthTrack(root, split)
    elif name.lower() == 'visevent': return VisEvent(root, split)
    elif name.lower() == 'rgbt234': return RGBT234(root, split)
    elif name.lower() == 'gtot': return GTOT(root, split)
    elif name.lower() == 'vot-rgbd2022': return VOT_RGBD2022(root, split)
    else:
        raise NotImplementedError(f'Data resource {name} not implemented.')
    
__all__ = ['BaseImageSequence', 'RGB_AUX_ImageSequence', 'BaseDataResource', 'BaseImageSequenceDataResource', 'TrackResultDataResource', 
           'LasHeR', 'DepthTrack', 'VisEvent', 'RGBT234', 'GTOT', 'VOT_RGBD2022',
           'build_data_resource']