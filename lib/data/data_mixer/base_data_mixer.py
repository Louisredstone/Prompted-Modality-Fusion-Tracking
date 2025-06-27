import logging
logger = logging.getLogger(__name__)

import torch.utils.data

from abc import ABC, abstractmethod

from ..data_resource import BaseDataResource


def no_processing(data):
    return data

class BaseDataMixer(torch.utils.data.Dataset, ABC):
    """ BaseDataMixer is a abstract class responsible for sampling frames from training sequences to form batches. 
    
    We have this class due to our need to use torch.nn.DistributedSampler, which requires a Dataset to be passed. In the mean time, we design some DataResource classes to manage the data. Therefore, the calling stack is as follows:
    DataResource -> DataMixer (torch Dataset) -> DistributedSampler -> DataLoader -> Trainer.
    
    We don't use torch Dataset to manage the data, because wrapping a Dataset (who manages the data) with another Dataset (who manages the sampling and will be passed to DistributedSampler) is not elegant.
    """

    def __init__(self, data_resources: list[BaseDataResource]):
        """
        args:
            data_resources - List of data_resources to be used for training
            ratio - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.data_resources: list[BaseDataResource] = data_resources

    @abstractmethod
    def __len__(self):
        return self.samples_per_epoch

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()