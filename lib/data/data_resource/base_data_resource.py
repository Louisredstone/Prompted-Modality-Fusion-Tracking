import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod


class BaseDataResource(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()