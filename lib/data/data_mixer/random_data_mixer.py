import logging
logger = logging.getLogger(__name__)

import random
import torch.utils.data

from collections import OrderedDict

from ...utils import TensorDict
from ..data_resource import BaseDataResource, BaseImageSequenceDataResource, BaseImageSequence
from .base_data_mixer import BaseDataMixer


def no_processing(data):
    return data

# class RandomDataMixer(torch.utils.data.Dataset):
class RandomDataMixer(BaseDataMixer):
    """ RandomDataMixer is a class responsible for sampling frames from training sequences to form batches. 
    
    See BaseDataMixer for more details.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, data_resources: list[BaseDataResource], 
                       ratios, 
                       samples_per_epoch, 
                       max_gap,
                       num_search_frames, 
                       num_template_frames=1, 
                       processing=no_processing, 
                       frame_sample_mode='causal',
                       classification: bool=False, 
                       pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
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
        self.classification: bool = classification  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if ratios is None:
            ratios = [len(d) for d in self.data_resources]

        # Normalize
        r_total = sum(ratios)
        self.ratio = [x / r_total for x in ratios]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, 
                                  num_ids=1, 
                                  min_id=None, 
                                  max_id=None,
                                  allow_blocked=False, 
                                  force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_blocked:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)
    
    # def _sample_visible_ids(self, visible, 
    #                               num_ids=1, 
    #                               min_id=None, 
    #                               max_id=None,
    #                               allow_blocked=False, 
    #                               force_invisible=False):
    #     """ Samples num_ids frames between min_id and max_id for which target is visible

    #     args:
    #         visible - 1d Tensor indicating whether target is visible for each frame
    #         num_ids - number of frames to be samples
    #         min_id - Minimum allowed frame number
    #         max_id - Maximum allowed frame number

    #     returns:
    #         list - List of sampled frame numbers. None if not sufficient visible frames could be found.
    #     """
    #     if num_ids == 0:
    #         return []
    #     if min_id is None or min_id < 0:
    #         min_id = 0
    #     if max_id is None or max_id > len(visible):
    #         max_id = len(visible)
    #     # get valid ids
    #     if force_invisible:
    #         valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
    #     else:
    #         if allow_blocked:
    #             valid_ids = [i for i in range(min_id, max_id)]
    #         else:
    #             valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    #     # No visible ids
    #     if len(valid_ids) == 0:
    #         return None

    #     return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.classification:
            return self.getitem_classification()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        logger.debug('Getting item (randomly, without index)')
        # Select a dataset
        data_resource: BaseDataResource = random.choices(self.data_resources, self.ratio)[0]

        # is_image_sequence_dataset = data_resource.is_image_sequence_dataset()
        is_image_sequence_dataset = isinstance(data_resource, BaseImageSequenceDataResource)

        # sample a sequence from the given dataset
        # seq_id, visible, seq_info_dict = self.sample_seq_from_seqs(data_resource, is_image_sequence_dataset) # FIXME
        seq_id, seq = self.sample_seq_from_seqs(data_resource, is_image_sequence_dataset)
        # visible = seq['visible']
        visible = seq.unblocked

        if is_image_sequence_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5

            elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
            elif self.frame_sample_mode == "stark":
                template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq.valid)
            else:
                raise ValueError("Illegal frame sample mode")
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames
        data = TensorDict({'template_images': seq[template_frame_ids].frames,
                           'template_anno': seq[template_frame_ids].bboxes_ltwh,
                           'search_images': seq[search_frame_ids].frames,
                           'search_anno': seq[search_frame_ids].bboxes_ltwh,
                           'dataset': data_resource.name,
                           'test_class': None})
        # make data augmentation
        logger.debug('Processing data')
        data = self.processing(data) # FIXME: Better processing function
        return data

    def getitem_classification(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.data_resources, self.ratio)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_seqs(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_seqs(self, data_resource: BaseImageSequenceDataResource, is_image_sequences: bool):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        random_ids = list(range(len(data_resource)))
        random.shuffle(random_ids)
        for seq_id in random_ids:
        # while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, len(data_resource) - 1)
            seq: BaseImageSequence = data_resource[seq_id]

            # Sample frames
            # seq_info_dict = data_resource.get_sequence_info(seq_id)
            # visible = seq_info_dict['visible']
            unblocked = torch.Tensor(seq.unblocked)

            enough_visible_frames = unblocked.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(unblocked) >= 20

            enough_visible_frames = enough_visible_frames or not is_image_sequences
            
            if enough_visible_frames: break
        if not enough_visible_frames:
            raise ValueError("Not enough visible frames found in the dataset")
        return seq_id, seq

    def get_one_search(self):
        # Select a dataset
        data_resource = random.choices(self.data_resources, self.ratio)[0]

        # is_video_dataset = data_resource.is_video_sequence()
        is_video_dataset = not isinstance(data_resource, BaseImageSequenceDataResource)
        # sample a sequence
        seq_id, seq = self.sample_seq_from_seqs(data_resource, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq.valid, num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(seq.unblocked, num_ids=1, allow_blocked=True)
        else:
            search_frame_ids = [1]

        return seq.frames[search_frame_ids],\
               dict(bbox=seq.bboxes_ltwh[search_frame_ids], valid=seq.valid[search_frame_ids], visible=seq.unblocked[search_frame_ids]),\
               OrderedDict({'object_class_name': None,
                            'motion_class': None,
                            'major_class': None,
                            'root_class': None,
                            'motion_adverb': None})
               # TODO: 'visible' -> 'unblocked'

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_blocked=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids
