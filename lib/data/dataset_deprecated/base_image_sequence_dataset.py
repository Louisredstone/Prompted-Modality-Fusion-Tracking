import logging
logger = logging.getLogger(__name__)
import torch.utils.data
# 2021.1.5 use jpeg4py_loader_w_failsafe as default
from ..utils.image_loader import jpeg4py_loader_w_failsafe

logger.warning('base_image_dataset.py will be deprecated in the future. Use lib.data.data_resource and lib.data.data_mixer instead.')

class BaseImageSequenceDataset(torch.utils.data.Dataset):
    """
    Base class for video datasets
    'Image Sequence Dataset' means this dataset contains a bunch of image sequences.
    We don't call it 'Video Dataset' because the visual information is usually stored
        as multiple image files, rather than a single video file.
    Generally, we regard the atom object of an image dataset as one sequence, while
        frames are its elements.
    Usually, an image sequence dataset looks like:
        - root
            - sequence1
                - frame1.jpg
                - frame2.jpg
                -...
            - sequence2
                - frame1.jpg
                - frame2.jpg
                -...
            -...
    Some image sequence datasets may have annotations for each sequence or frames.
    Some image sequence datasets may have multiple modals, i.e. RGB, TIR, Depth, Events, etc.
    """

    def __init__(self, name, root, image_loader=jpeg4py_loader_w_failsafe):
        """
        args:
            root - The root path to the dataset
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.sequence_list = [] # current implementation: [name], TODO: [Sequence]
        # self.class_list = []

    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.get_num_sequences()

    def __getitem__(self, sequence_name):
        raise NotImplementedError("Check get_frames() instead.") # TODO

    def is_image_sequence_dataset(self):
        """ Returns whether the dataset is a video dataset or an image dataset

        returns:
            bool - True if a video dataset
        """
        return True

    def is_synthetic_video_dataset(self):
        """ Returns whether the dataset contains real videos or synthetic

        returns:
            bool - True if a video dataset
        """
        return False

    def get_name(self):
        """ Name of the dataset

        returns:
            string - Name of the dataset
        """
        raise NotImplementedError()

    def get_num_sequences(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.sequence_list)

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    # def get_num_classes(self):
    #     return len(self.class_list)

    # def get_class_list(self):
    #     return self.class_list

    def get_sequences_in_class(self, class_name):
        raise NotImplementedError()

    def has_segmentation_info(self):
        return False

    def get_sequence_info(self, seq_id):
        """ Returns information about a particular sequences,

        args:
            seq_id - index of the sequence

        returns:
            Dict
            """
        raise NotImplementedError()

    def get_frames(self, seq_id, frame_ids, anno=None):
        """ Get a set of frames from a particular sequence

        args:
            seq_id      - index of sequence
            frame_ids   - a list of frame numbers
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            list - List of frames corresponding to frame_ids
            list - List of dicts for each frame
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        raise NotImplementedError()