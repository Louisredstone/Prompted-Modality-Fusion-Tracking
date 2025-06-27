def config_sequence(type, path_anno):
    import os

    # dataset_name = {
    #     'test_set': 'testing_set.txt',
    #     'extension_test_set': 'extension_testing_set.txt',
    #     'all': 'all_dataset.txt'
    # }.get(type)

    # if not dataset_name:
    #     raise ValueError("Error in evaluation dataset type! Either 'testing_set', 'extension_test_set', or 'all'.")

    # if not os.path.exists(dataset_name):
    #     raise FileNotFoundError(f"{dataset_name} is not found!")

    # with open(dataset_name, 'r') as fid:
    #     sequences = [line.strip() for line in fid if line.strip()]

    print('[DEBUG] Using DepthTrack_test dataset for evaluation.')

    txt_files = os.listdir(path_anno)
    sequences = [os.path.splitext(txt_file)[0] for txt_file in txt_files if txt_file.endswith('.txt')]

    return sequences