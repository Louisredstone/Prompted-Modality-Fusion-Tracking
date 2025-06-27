# This script is used to generate videos for qualitative comparison among different methods.
# These result videos can help to understand the differences among different methods, and to pick out nice samples for paper.
# requirements: dataset, results of different methods
# output: 
#   - work_dirs/qualitative-compare/<dataset_name>/<seq_name>.mp4
#   - work_dirs/qualitative-compare/<dataset_name>/<seq_name>/<frame_id>.png
# every frame will be like:
# /-----------------------------------\
# | #42                               |
# |           |------|                |
# |         |-|----| |                |
# |         | |    | |                |
# |         | |    | |                |
# |         | |----|-|                |
# |         |      |                  |
# |         |------|                  |
# |                                   |
# \-----------------------------------/

import logging
logger = logging.getLogger(__name__)
import os
import sys
import cv2
import numpy as np
import webcolors
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from os.path import join as pjoin
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
add_path('.')

palette = {
    'groundtruth': 'red',
    'ProMFT': 'yellow',
    # 'ProMFT-Naive': 'blue',
    # 'ProMFT-Shallow': 'green',
    'ViPT': 'green'
}

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate qualitative comparison videos for different methods.')
    parser.add_argument('--force', action='store_true', help='Whether to overwrite existing images and videos.')
    return parser.parse_args()

args = parse_args()
force = args.force

# Step 1: load datasets (DataResource, mainly gt)

from lib.data.data_resource import LasHeR, DepthTrack, VisEvent
logger.info('Loading datasets...')
lasher = LasHeR('datasets/LasHeR', 'test')
depthtrack = DepthTrack('datasets/depthtrack', 'test')
visevent = VisEvent('datasets/VisEvent', 'test')
all_data_resources = {'lasher': lasher, 'depthtrack': depthtrack, 'visevent': visevent}

# Step 2: load results of different methods as DataResource
from lib.data.data_resource.track_result_data_resource import ProMFT_TrackResultDataResource, ViPT_TrackResultDataResource
from lib.data.data_resource import BaseImageSequence

lasher_results_dir = 'work_dirs/results-on-datasets/LasHeR_test'
depthtrack_results_dir = 'work_dirs/results-on-datasets/DepthTrack_test'
visevent_results_dir = 'work_dirs/results-on-datasets/VisEvent_test'
all_results_dirs = {'lasher': lasher_results_dir, 'depthtrack': depthtrack_results_dir, 'visevent': visevent_results_dir}
all_results = {'lasher': {}, 'depthtrack': {}, 'visevent': {}}
for dataset_name, data_resource in all_data_resources.items():
    logger.info(f'Loading results for {dataset_name}...')
    results_dir_of_dataset = all_results_dirs[dataset_name]
    for dir_ in os.listdir(results_dir_of_dataset):
        if dir_ not in palette: continue
        if dir_.startswith('ProMFT'):
            all_results[dataset_name][dir_] = ProMFT_TrackResultDataResource(pjoin(results_dir_of_dataset, dir_))
        elif dir_.startswith('ViPT'):
            all_results[dataset_name][dir_] = ViPT_TrackResultDataResource(pjoin(results_dir_of_dataset, dir_))
        else:
            logger.warning(f'Unknown method {dir_}, skipped.')

# Step 3: generate best collections

qualitative_compare_dir = 'work_dirs/qualitative-compare'
os.makedirs(qualitative_compare_dir, exist_ok=True)
lasher_qc_dir = pjoin(qualitative_compare_dir, 'LasHeR')
depthtrack_qc_dir = pjoin(qualitative_compare_dir, 'DepthTrack')
visevent_qc_dir = pjoin(qualitative_compare_dir, 'VisEvent')
all_qc_dirs = {'lasher': lasher_qc_dir, 'depthtrack': depthtrack_qc_dir, 'visevent': visevent_qc_dir}

TOP = 800

warnings = []

from numpy.linalg import norm
for dataset_name, data_resource in all_data_resources.items():
    logger.info(f'Generating best collection for dataset {dataset_name}...')
    center_error_analysis = []
    qc_dir_of_dataset = all_qc_dirs[dataset_name]
    dataset_results = all_results[dataset_name] # {method_name: method_results}
    for seq_name, seq in data_resource.sequences_dict.items():
        seq_name: str
        seq: BaseImageSequence
        logger.info(f'Processing sequence {seq_name}...')
        center_error_df = pd.DataFrame(columns=dataset_results.keys(), index=list(range(len(seq))), data=np.full((len(seq), len(dataset_results)), np.nan))
        try:
            for method_name, method_results in dataset_results.items():
                if seq_name not in method_results.sequences_dict:
                    logger.warning(f'Sequence {seq_name} not found in method {method_name} results, skipped.')
                    warnings.append(f'Sequence {seq_name} not found in method {method_name} results.')
                    continue
                method_seq = method_results.sequences_dict[seq_name]
                for frame_id in range(len(seq)):
                    frame_id: int
                    bbox_ltrb_gt = seq.bboxes_ltrb[frame_id]
                    if np.nan in bbox_ltrb_gt: continue # skip invalid gt
                    center_gt = (bbox_ltrb_gt[[2, 3]] + bbox_ltrb_gt[[0, 1]]) / 2
                    bbox_ltrb_pred = method_seq.bboxes_ltrb[frame_id]
                    if np.nan in bbox_ltrb_pred: continue # skip invalid pred
                    center_pred = (bbox_ltrb_pred[[2, 3]] + bbox_ltrb_pred[[0, 1]]) / 2
                    center_error_df.loc[frame_id, method_name] = norm(center_gt - center_pred)
            for frame_id in range(len(seq)):
                center_errors = center_error_df.loc[frame_id]
                if not np.isnan(center_errors).any():
                    center_error_analysis.append(dict(seq_name=seq_name, frame_id=frame_id, center_errors=center_errors .to_dict()))
        except Exception as e:
            logger.warning(f'Error when processing sequence {seq_name}, skipped. Error: {e}')
            warnings.append(f'Error when processing sequence {seq_name}. Error: {e}')
            continue
    # center_error_analysis = sorted(center_error_analysis, key=lambda x: max(x['center_errors'].values()))
    center_error_analysis = [item for item in center_error_analysis if 'ProMFT' in item['center_errors'] and 'ViPT' in item['center_errors']]
    center_error_analysis = sorted(center_error_analysis, key=lambda item: item['center_errors']['ViPT']-item['center_errors']['ProMFT'], reverse=True)

    logger.info(f"Annotating frames for dataset {dataset_name}...")
    for i, item in enumerate(center_error_analysis[:TOP]):
        seq_name = item['seq_name']
        seq = data_resource.sequences_dict[seq_name]
        frame_id = item['frame_id']
        logger.info(f'Annotating sequence {seq_name} frame {frame_id}...')
        dir_ = pjoin(qc_dir_of_dataset, 'annotated-frames', seq_name)
        os.makedirs(dir_, exist_ok=True)
        
        rgb_image_path = f'{dir_}/#{frame_id:04d}_rgb.png'
        aux_image_path = f'{dir_}/#{frame_id:04d}_aux.png'
        combined_image_path = f'{dir_}/#{frame_id:04d}.png'
        if os.path.exists(rgb_image_path) and os.path.exists(aux_image_path) and not force:
            continue
        frame = seq.load_frame(frame_id) # 6 x H x W
        rgb_image_chw, aux_image_chw = frame[:3], frame[3:]
        rgb_image_hwc = rgb_image_chw.transpose(1, 2, 0).copy()
        aux_image_hwc = aux_image_chw.transpose(1, 2, 0).copy()
        rgb_image_hwc_bgr = cv2.cvtColor(rgb_image_hwc, cv2.COLOR_RGB2BGR)
        aux_image_hwc_bgr = cv2.cvtColor(aux_image_hwc, cv2.COLOR_RGB2BGR)
        if dataset_name == 'depthtrack': #拉高对比度
            aux_image_hwc_bgr = cv2.convertScaleAbs(aux_image_hwc_bgr, alpha=2, beta=0)
        bbox_ltrb_gt = seq.bboxes_ltrb[frame_id]
        bbox_ltrb_gt = [int(p) for p in bbox_ltrb_gt]
        # bboxes_ltrb = {'groundtruth': bbox_ltrb_gt}
        gt_color = webcolors.name_to_rgb(palette['groundtruth'])
        gt_color = (gt_color[2], gt_color[1], gt_color[0])
        cv2.rectangle(rgb_image_hwc_bgr, (bbox_ltrb_gt[0], bbox_ltrb_gt[1]), (bbox_ltrb_gt[2], bbox_ltrb_gt[3]), gt_color, 2)
        cv2.rectangle(aux_image_hwc_bgr, (bbox_ltrb_gt[0], bbox_ltrb_gt[1]), (bbox_ltrb_gt[2], bbox_ltrb_gt[3]), gt_color, 2)
        bboxes_ltrb = {}
        for method_name, method_result in dataset_results.items():
            logger.info(f'Processing method {method_name}...')
            bbox_ltrb_ = method_result.sequences_dict[seq_name].bboxes_ltrb[frame_id]
            bbox_ltrb_ = [int(p) for p in bbox_ltrb_]
            bboxes_ltrb[method_name] = bbox_ltrb_
            color = webcolors.name_to_rgb(palette[method_name])
            color = (color[2], color[1], color[0])
            cv2.rectangle(rgb_image_hwc_bgr, (bbox_ltrb_[0], bbox_ltrb_[1]), (bbox_ltrb_[2], bbox_ltrb_[3]), color, 2)
            cv2.rectangle(aux_image_hwc_bgr, (bbox_ltrb_[0], bbox_ltrb_[1]), (bbox_ltrb_[2], bbox_ltrb_[3]), color, 2)
        # cv2.putText(
        #     rgb_image_hwc_bgr,
        #     f'#{frame_id:04d}',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_COMPLEX,  # 字体类型（必需）
        #     fontScale=1,
        #     color=(255, 255, 255),     # 白色（BGR 格式）
        #     thickness=1
        # )
        # cv2.putText(
        #     aux_image_hwc_bgr,
        #     f'#{frame_id:04d}',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_COMPLEX,  # 字体类型（必需）
        #     fontScale=1,
        #     color=(255, 255, 255),     # 白色（BGR 格式）
        #     thickness=1
        # )
        cv2.imwrite(rgb_image_path, rgb_image_hwc_bgr)
        cv2.imwrite(aux_image_path, aux_image_hwc_bgr)
        # combined_image_hwc = np.concatenate([rgb_image_hwc, aux_image_hwc], axis=1)
        combined_image_hwc = np.concatenate([rgb_image_hwc_bgr, aux_image_hwc_bgr], axis=0)
        
        cv2.imwrite(combined_image_path, combined_image_hwc)

        # soft link
        best_collection_dir = pjoin(qc_dir_of_dataset, f'best-collection')
        os.makedirs(best_collection_dir, exist_ok=True)
        target_path = pjoin(best_collection_dir, f'{i:04d}-{dataset_name}-{seq_name}-#{frame_id:04d}.png')
        if os.path.islink(target_path): os.remove(target_path)
        import shutil
        shutil.copy(combined_image_path, target_path)
        # linkpath =  pjoin(best_collection_dir, f'{i:04d}.png')
        # if os.path.exists(linkpath): 
        #     logger.warning(f'Link {linkpath} already exists as file, skipped.')
        #     continue
        # if os.path.islink(linkpath): os.remove(linkpath)
        # os.symlink(combined_image_path, linkpath)

print(f'Warnings: '+"\n".join(warnings))
































# Deprecated:

# # Step 3: generate frame images and videos for each sequence

# qualitative_compare_dir = 'work_dirs/qualitative-compare'
# os.makedirs(qualitative_compare_dir, exist_ok=True)
# lasher_qc_dir = pjoin(qualitative_compare_dir, 'LasHeR')
# depthtrack_qc_dir = pjoin(qualitative_compare_dir, 'DepthTrack')
# visevent_qc_dir = pjoin(qualitative_compare_dir, 'VisEvent')
# all_qc_dirs = {'lasher': lasher_qc_dir, 'depthtrack': depthtrack_qc_dir, 'visevent': visevent_qc_dir}

# for dataset_name, data_resource in all_data_resources.items():
#     logger.info(f'Processing dataset {dataset_name}...')
#     qc_dir_of_dataset = all_qc_dirs[dataset_name]
#     method_results_of_dataset = all_method_results[dataset_name]
#     for seq_name, seq in data_resource.sequences_dict.items():
#         seq_name: str
#         seq: BaseImageSequence
#         video_path = pjoin(qc_dir_of_dataset, f'{seq_name}.mp4')
#         logger.info(f'Processing sequence {seq_name}...')
#         dir_ = pjoin(qc_dir_of_dataset, seq_name)
#         os.makedirs(dir_, exist_ok=True)
#         video_frames = []
#         for frame_id in range(len(seq)):
#             frame_id: int
#             rgb_image_path = qc_dir_of_dataset+'/'+seq_name+f'#{frame_id:04d}_rgb.png'
#             aux_image_path = qc_dir_of_dataset+'/'+seq_name+f'#{frame_id:04d}_aux.png'
#             if os.path.exists(rgb_image_path) and os.path.exists(aux_image_path) and not force:
#                 continue
#             frame = seq.load_frame(frame_id) # 6 x H x W
#             rgb_image_chw, aux_image_chw = frame[:3], frame[3:]
#             rgb_image_hwc = rgb_image_chw.transpose(1, 2, 0).copy()
#             aux_image_hwc = aux_image_chw.transpose(1, 2, 0).copy()
#             rgb_image_hwc_bgr = cv2.cvtColor(rgb_image_hwc, cv2.COLOR_RGB2BGR)
#             aux_image_hwc_bgr = cv2.cvtColor(aux_image_hwc, cv2.COLOR_RGB2BGR)
#             bboxes_ltrb = {'gt': seq.bboxes_ltrb[frame_id]}
#             for method_name, method_result in method_results_of_dataset.items():
#                 logger.info(f'Processing method {method_name}...')
#                 # bbox_ltrb_ = method_result.sequences_dict[seq_name].bboxes_ltrb[frame_id]
#                 bbox_ltrb_ = method_result.sequences_dict[seq_name].bboxes_ltrb[frame_id]
#                 bbox_ltrb_ = [int(p) for p in bbox_ltrb_]
#                 bboxes_ltrb[method_name] = bbox_ltrb_
#                 # bboxes_ltrb[method_name] = method_result.sequences_dict[seq_name].bboxes_ltrb[frame_id]
#             # for method_name, bbox_ltrb_ in bboxes_ltrb.items():
#                 color = webcolors.name_to_rgb(palette[method_name])
#                 color = (color[2], color[1], color[0])
#                 cv2.rectangle(rgb_image_hwc_bgr, (bbox_ltrb_[0], bbox_ltrb_[1]), (bbox_ltrb_[2], bbox_ltrb_[3]), color, 2)
#                 cv2.rectangle(aux_image_hwc_bgr, (bbox_ltrb_[0], bbox_ltrb_[1]), (bbox_ltrb_[2], bbox_ltrb_[3]), color, 2)
#             # cv2.addText(rgb_image_hwc_bgr, f'#{frame_id:04d}', (10, 20), nameFont="Arial", color=(255, 255, 255))
#             # cv2.addText(aux_image_hwc_bgr, f'#{frame_id:04d}', (10, 20), nameFont="Arial", color=(255, 255, 255))
#             cv2.putText(
#                 rgb_image_hwc_bgr,
#                 f'#{frame_id:04d}',
#                 (10, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型（必需）
#                 fontScale=0.5,
#                 color=(255, 255, 255),     # 白色（BGR 格式）
#                 thickness=1
#             )
#             cv2.putText(
#                 aux_image_hwc_bgr,
#                 f'#{frame_id:04d}',
#                 (10, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型（必需）
#                 fontScale=0.5,
#                 color=(255, 255, 255),     # 白色（BGR 格式）
#                 thickness=1
#             )
#             cv2.imwrite(rgb_image_path, rgb_image_hwc_bgr)
#             cv2.imwrite(aux_image_path, aux_image_hwc_bgr)
#             # combined_image_hwc = np.concatenate([rgb_image_hwc, aux_image_hwc], axis=1)
#             combined_image_hwc = np.concatenate([rgb_image_hwc_bgr, aux_image_hwc_bgr], axis=0)
#             cv2.imwrite(qc_dir_of_dataset+'/'+seq_name+f'#{frame_id:04d}.png', combined_image_hwc)
#             video_frames.append(combined_image_hwc)
#         video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (combined_image_hwc.shape[1], combined_image_hwc.shape[0]))
#         for frame in video_frames:
#             video_writer.write(frame)
#         video_writer.release()

# # Step 4: find frames with greatest difference among different methods

# from numpy.linalg import norm
# for dataset_name, data_resource in all_data_resources.items():
#     logger.info(f'Generating best collection for dataset {dataset_name}...')
#     center_error_analysis = []
#     qc_dir_of_dataset = all_qc_dirs[dataset_name]
#     for seq_name, seq in data_resource.sequences_dict.items():
#         seq_name: str
#         seq: BaseImageSequence
#         for frame_id in range(len(seq)):
#             frame_id: int
#             bbox_ltrb_gt = seq.bboxes_ltrb[frame_id]
#             center_gt = (bbox_ltrb_gt[[2, 3]] + bbox_ltrb_gt[[0, 1]]) / 2
#             center_errors = {}
#             for method_name, method_results_of_dataset in all_method_results[dataset_name].items():
#                 bbox_ltrb_pred = method_results_of_dataset.sequences_dict[seq_name].bboxes_ltrb[frame_id]
#                 center_pred = (bbox_ltrb_pred[[2, 3]] + bbox_ltrb_pred[[0, 1]]) / 2
#                 center_errors[method_name] = norm(center_gt - center_pred)
#             center_error_analysis.append(dict(seq_name=seq_name, frame_id=frame_id, center_errors=center_errors))
#     # center_error_analysis = sorted(center_error_analysis, key=lambda x: max(x['center_errors'].values()))
#     center_error_analysis = sorted(center_error_analysis, key=lambda x: x['center_errors']['ViPT']-x['center_errors']['ProMFT'])
#     best_collection_dir = pjoin(qc_dir_of_dataset, f'best-collection')
#     for i, item in enumerate(center_error_analysis):
#         seq_name = item['seq_name']
#         frame_id = item['frame_id']
#         combined_image_path = pjoin(all_qc_dirs[dataset_name], seq_name, f'{frame_id:04d}.png')
#         # soft link
#         os.symlink(combined_image_path, pjoin(best_collection_dir, f'{i:04d}.png'))