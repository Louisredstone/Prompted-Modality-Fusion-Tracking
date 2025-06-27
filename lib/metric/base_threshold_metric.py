import logging
logger = logging.getLogger(__name__)

from abc import ABC

# 未实装

class BaseThresholdMetric(ABC):
    def __init__(self):
        pass
    
    def __call__(self, pred, gt):
        pass
    
