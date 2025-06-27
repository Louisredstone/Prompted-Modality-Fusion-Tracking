import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Check VOT and Depthtrack results')
    parser.add_argument('--vot-dir', type=str, required=True, help='path to VOT results')
    parser.add_argument('--depthtrack-dir', type=str, required=True, help='path to Depthtrack results')
    
    args = parser.parse_args()
    return args

def get_sequences_from_dir(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

def main():
    args = parse_args()
    vot_dir = args.vot_dir
    depthtrack_dir = args.depthtrack_dir
    
    depthtrack_train_seqs = get_sequences_from_dir(os.path.join(depthtrack_dir, 'train'))
    depthtrack_test_seqs = get_sequences_from_dir(os.path.join(depthtrack_dir, 'test'))
    vot_seqs = get_sequences_from_dir(vot_dir)
    
    depthtrack_train_collection = set(tuple(seq.split('_')[:2]) for seq in depthtrack_train_seqs)
    depthtrack_test_collection = set(tuple(seq.split('_')[:2]) for seq in depthtrack_test_seqs)
    vot_collection = set(tuple(seq.split('_')[:2]) for seq in vot_seqs)
    
    res = vot_collection.intersection(depthtrack_train_collection)
    print('VOT and Depthtrack train collection intersection:', res)
    
    res = vot_collection.intersection(depthtrack_test_collection)
    print('VOT and Depthtrack test collection intersection:', res)
    
    res = depthtrack_test_collection - vot_collection
    print('Depthtrack test collection - VOT collection:', res)
    
    res = vot_collection - depthtrack_test_collection
    print('VOT collection - Depthtrack test collection:', res)
    
if __name__ == '__main__':
    main()