import logging
logger = logging.getLogger(__name__)
import torch.utils.data
from ..utils.image_loader import jpeg4py_loader

logger.warning('base_image_dataset.py will be deprecated in the future. Use lib.data.data_resource and lib.data.data_mixer instead.')

class BaseImageDataset(torch.utils.data.Dataset):
    """
    Base class for image datasets
    'Image Dataset' means this dataset contains a bunch of images and their annotations.
    Generally, we regard the atom object of an image dataset as one image.
    Usually, an image dataset looks like:
        - root
            - class1
                - img1.jpg
                - img2.jpg
                -...
            - class2
                - img1.jpg
                - img2.jpg
                -...
            -...
    Some image datasets have sub-classes, like:
        - root
            - class1
                - subclass1.1
                    - img1.jpg
                    - img2.jpg
                    -...
                - subclass1.2
                    - img1.jpg
                    - img2.jpg
                    -...
            - class2
                - subclass2.1
                    - img1.jpg
                    - img2.jpg
                    -...
    """

    def __init__(self, name, root, image_loader=jpeg4py_loader):
        """
        args:
            root - The root path to the dataset
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.image_list = []     # Contains the list of sequences.
        self.class_list = []

    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.get_num_images()

    def __getitem__(self, index):
        """ Not to be used! Check get_frames() instead.
        """
        return None

    def get_name(self):
        """ Name of the dataset

        returns:
            string - Name of the dataset
        """
        raise NotImplementedError

    def get_num_images(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.image_list)

    def has_class_info(self):
        return False

    def get_class_name(self, image_id):
        return None

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_images_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False

    def get_image_info(self, seq_id):
        """ Returns information about a particular image,

        args:
            seq_id - index of the image

        returns:
            Dict
            """
        raise NotImplementedError

    def get_image(self, image_id, anno=None):
        """ Get a image

        args:
            image_id      - index of image
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            image -
            anno -
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        raise NotImplementedError

