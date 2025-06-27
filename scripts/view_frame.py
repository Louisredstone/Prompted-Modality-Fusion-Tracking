import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='View frame from a dataset')
    parser.add_argument('dataset_type', type=str, help='type of the dataset')
    parser.add_argument('seq_name', type=str, help='name of the sequence to view')
    parser.add_argument('frame_idx', type=int, help='index of the frame to view')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_type = args.dataset_type.lower()
    seq_name = args.seq_name
    frame_idx = args.frame_idx
    assert dataset_type in ['lasher', 'depthtrack', 'visevent'], 'Invalid dataset type'