import logging
logger = logging.getLogger(__name__)

from abc import abstractmethod
from collections.abc import Iterable
from typing import Union

from .base_data_resource import BaseDataResource
from .base_image_sequence import BaseImageSequence
from .rgb_aux_image_sequence import RGB_AUX_ImageSequence


class BaseImageSequenceDataResource(BaseDataResource):
    """
    Base class for video datasets
    'IMAGE SEQUENCE data resource' means this dataset contains a bunch of image sequences.
    We don't call it 'Video Dataset' because the visual information is usually stored
        as multiple image files, rather than a single video file.
    'Image sequence DATA RESOURCE' means this class does not inherite
        from torch.utils.data.Dataset, but only provides a common interface
        for accessing image sequence data. If you need to use it with torch.utils.data.Dataset,
        use RandomDataMix or VanillaDataMix as wrappers.
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
    
    def __init__(self, name, root, split: Union[str, None, list[str]]=None):
        """
        args:
            root - The root path to the dataset
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        self.name: str = name
        self.root: str = root
        self.splits: set[str]
        if split is None or split == 'all':
            self.splits = self.valid_splits
        elif isinstance(split, str):
            assert split in self.valid_splits, f"Invalid split: {split}"
            self.splits = set((split,))
        elif isinstance(split, Iterable):
            assert all(s in self.valid_splits for s in split), f"Invalid splits: {split}"
            self.splits = set(split)
        else:
            raise ValueError(f"Invalid split: {split}")
        if len(self.splits) == 0: raise ValueError("No valid split is specified.")
        if len(self.splits) < len(self.valid_splits):
            self.name += f"_{'+'.join(sorted(self.splits))}"
        
        self.sequences: list[BaseImageSequence] = []
        self.init_sequences()
        self.sequences_dict: dict[str, BaseImageSequence] = {seq.name: seq for seq in self.sequences}
    
    @property
    def valid_splits(self) -> set[str]:
        return set(['train', 'test'])
    
    @abstractmethod
    def init_sequences(self) -> None:
        """
        Initialize self.sequences.
        """
        raise NotImplementedError()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx) -> BaseImageSequence|list[BaseImageSequence]:
        if isinstance(idx, slice):
            return [self.sequences[i] for i in range(len(self))[idx]]
        elif isinstance(idx, (list, tuple)):
            return [self.sequences[i] for i in idx]
        elif isinstance(idx, str):
            return self.sequences_dict[idx]
        else:
            return self.sequences[idx]

    def getSeqByName(self, name: str, raise_error: bool = False) -> BaseImageSequence:
        if name not in self.sequences_dict:
            if raise_error:
                raise ValueError(f"Sequence {name} not found in {self.name}.")
            else:
                return None
        return self.sequences_dict[name]

    def force_load_sequences(self) -> None:
        """
        Force load all sequences.
        """
        for sequence in self.sequences:
            sequence: BaseImageSequence
            sequence.ensure_load_frames_to_cache()
            if isinstance(sequence, RGB_AUX_ImageSequence):
                sequence.ensure_load_auxiliary_images()