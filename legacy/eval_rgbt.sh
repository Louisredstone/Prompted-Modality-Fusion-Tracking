# test lasher
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name deep_rgbt

# test rgbt234
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name RGBT234 --yaml_name deep_rgbt

CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name GTOT --yaml_name shaw_rgbt