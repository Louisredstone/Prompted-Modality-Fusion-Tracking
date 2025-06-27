# author: zhyzhang

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from collections import Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Analysis')
    parser.add_argument('datasetType', type=str, help='dataset type (e.g. DepthTrack, GTOT, LasHeR, LaSOT, RGBT210, RGBT234, VisEvent, etc.)')
    parser.add_argument('data_dir', type=str, help='path to dataset')
    return parser.parse_args()

def get_split_sequences(dataset_type, data_dir):
    split_subdir_dict = dict(
        DepthTrack=dict(train='train', test='test'),
        GTOT=dict(all='.'),
        LasHeR=dict(train='TrainingSet/trainingset', test='TestingSet/testingset'),
        LaSOT=dict(train='train', test='test'),
        RGBT210=dict(all='.'),
        RGBT234=dict(all='.'),
        VisEvent=dict(train='train_subset', test='test_subset')
    )
    if dataset_type not in split_subdir_dict:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    split_subdirs = split_subdir_dict[dataset_type]
    from os.path import join
    if dataset_type == 'LaSOT':
        return {key: [join(data_dir, split_subdir, subdir, subsubdir) for subdir in next(os.walk(os.path.join(data_dir, split_subdir)))[1] for subsubdir in next(os.walk(join(data_dir, subdir)))[1]] for key, split_subdir in split_subdirs.items()}
    else: 
        return {key: [join(data_dir, split_subdir, subdir) for subdir in next(os.walk(join(data_dir, split_subdir)))[1]] for key, split_subdir in split_subdirs.items()}

def get_modal_dirnames(dataset_type):
    return dict(
        DepthTrack=dict(rgb='color', aux='depth'),
        GTOT=dict(rgb='i', aux='v'),
        LasHeR=dict(rgb='visible', aux='infrared'),
        RGBT210=dict(rgb='visible', aux='infrared'),
        RGBT234=dict(rgb='visible', aux='infrared'),
        VisEvent=dict(rgb='vis_imgs', aux='event_imgs')
    )[dataset_type]

def main():
    args = parse_args()
    dataset_type = args.datasetType
    data_dir = args.data_dir
    split_sequences = get_split_sequences(dataset_type, data_dir)
    modal_dirnames = get_modal_dirnames(dataset_type)
    rgb_dirname, aux_dirname = modal_dirnames['rgb'], modal_dirnames['aux']
    n_frames_counter = Counter()
    rgb_wh_counter = Counter()
    aux_wh_counter = Counter()
    for split, seqs in split_sequences.items():
        for seq in tqdm(seqs):
            rgb_dir = os.path.join(seq, rgb_dirname)
            aux_dir = os.path.join(seq, aux_dirname)
            n_frames = len(os.listdir(rgb_dir))
            if not len(os.listdir(aux_dir)) == n_frames:
                print(f'Warning: {seq} has different number of frames in {rgb_dir} and {aux_dir}')
            n_frames_counter[n_frames] += 1
            for rgb_img_path in os.listdir(rgb_dir):
                if not os.path.splitext(rgb_img_path)[1] in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                try:
                    rgb_img = Image.open(os.path.join(rgb_dir, rgb_img_path))
                except UnidentifiedImageError:
                    print(f'Warning: {os.path.join(rgb_dir, rgb_img_path)} is not a valid image')
                    continue
                rgb_wh_counter[rgb_img.size] += 1
            for aux_img_path in os.listdir(aux_dir):
                if not os.path.splitext(aux_img_path)[1] in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                aux_img = Image.open(os.path.join(aux_dir, aux_img_path))
                aux_wh_counter[aux_img.size] += 1
    print(f'Dataset type: {dataset_type}')
    print(f'Data directory: {data_dir}')
    print(f'Split sequences: {", ".join([f"{split}: {len(seqs)}" for split, seqs in split_sequences.items()])}')
    print(f'Number of frames: {n_frames_counter}')
    print(f'\tmin: {min(n_frames_counter.keys())}, max: {max(n_frames_counter.keys())}')
    print(f'RGB image sizes: {rgb_wh_counter}')
    print(f'\tmin: {min(rgb_wh_counter.keys(), key=lambda x: x[0]*x[1])}, max: {max(rgb_wh_counter.keys(), key=lambda x: x[0]*x[1])}')
    print(f'Auxiliary image sizes: {aux_wh_counter}')
    print(f'\tmin: {min(aux_wh_counter.keys(), key=lambda x: x[0]*x[1])}, max: {max(aux_wh_counter.keys(), key=lambda x: x[0]*x[1])}')
    
if __name__ == '__main__':
    main()