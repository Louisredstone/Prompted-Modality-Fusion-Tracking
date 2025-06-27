import logging
logger = logging.getLogger(__name__)

import time
import os
import cv2
import torch
import pandas as pd
import multiprocessing
import numpy as np

from numpy import ndarray
from numpy import concatenate as cat

from ..data.data_resource import RGB_AUX_ImageSequence, BaseImageSequenceDataResource
from .tracker.vipt import ViPT_Test_Wrapper

def test_infer_on_sequence(sequence: RGB_AUX_ImageSequence, dataset_name, config, free_gpus, worker_id=None):
    # sequence means subtask of evaluate.
    # seq_name = seq_path.split('/')[-1]
    logger.info(f'Start processing sequence: {sequence.name}')
    # if 'VTUAV' in dataset_name:
    #     seq_txt = seq_name.split('/')[1]
    # else:
    #     seq_txt = seq_name
    save_folder = f'{config.GENERAL.WORK_DIR}/results/{dataset_name}/'
    save_path = f'{save_folder}/{sequence.name}.csv'
    os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(save_path):
        logger.warning(f'Result file already exists: {save_path}. Skip.')
        return
    n_gpus = len(free_gpus)
    
    if worker_id is None: # parallel mode but worker_id not specified
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        local_rank = free_gpus[worker_id % n_gpus]
        time.sleep(0.5 * worker_id)
        torch.cuda.set_device(local_rank)

    # params = rgbt_prompt_params.parameters(config)
    wrapper = ViPT_Test_Wrapper(config)  # "GTOT" # dataset_name

    logger.info(f'[{worker_id}] Processing sequence: {sequence.name}')
    tic, toc = cv2.getTickCount(), 0
    result = pd.DataFrame(columns=['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'bbox_right', 'bbox_bottom'])
    
    def find_first_true_index(arr)-> int: return int(np.argmax(arr))
    first_valid_index = find_first_true_index(sequence.valid)
    
    # for i, item in enumerate(sequence):
    for i, item in enumerate(sequence):
        tic = cv2.getTickCount()
        # rgb_image, aux_image = frame['image', 'aux_image']
        frame: ndarray[(6, 'H', 'W')] = item.frames[0] # Note: item.frames: ndarray[(1, 6, H, W)] here.
        # rgb_image, aux_image = frame[:3, :, :], frame[3:, :, :]
        if i < first_valid_index:
            result.loc[i] = [np.nan]*6
        elif i == first_valid_index:
            # bbox = frame['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']
            bbox_ltwh = item.bboxes_ltwh
            bbox_ltrb = item.bboxes_ltrb
            # wrapper.initialize(np.concatenate((rgb_image, aux_image), axis=2), bbox.tolist())  # xywh
            wrapper.initialize(frame, bbox_ltwh.tolist())  # ltwh
            # QUESTION: ltwh or xywh? (xywh: centerX, centerY, width, height)
            # result.iloc[0] = bbox + [bbox[0]+bbox[2], bbox[1]+bbox[3]]
            result.loc[i] = cat((bbox_ltwh, bbox_ltrb[2:4]), axis=0).tolist()
        else: # i > first_valid_index
            # bbox_pred, confidence = wrapper.track(np.concatenate((rgb_image, aux_image), axis=2))  # xywh
            bbox_pred, confidence = wrapper.track(frame)  # ltwh
            result.loc[i] = bbox_pred + [bbox_pred[0]+bbox_pred[2], bbox_pred[1]+bbox_pred[3]]
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    
    result.to_csv(save_path, index=False)
    logger.success(f'[{worker_id}] {sequence.name} processed in {toc:.3f}s, fps: {len(sequence) / toc}')