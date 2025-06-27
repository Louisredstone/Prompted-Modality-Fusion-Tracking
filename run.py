import os
from os.path import join as pjoin
import sys
import time
import argparse
import random
import torch
import datetime
import torch.distributed as dist
import multiprocessing
import yaml
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
add_path('.')

from lib.config import MainConfig, Auto
from lib.utils.cuda_info import get_free_gpus
from lib.utils import is_subpath
from lib.log import setup_logging
from lib.data.data_resource import build_data_resource
from lib.test import test_infer_on_sequence
from lib.metric.toolkit import calculate_metrics
from lib.metric.flops_and_params import calculate_flops_and_params
from lib.metric.analyzer import Analyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Launch process.')
    parser.add_argument('config', type=str, help='yaml configure file path')
    parser.add_argument('--only-train', '--train', action='store_true', help='only train the model')
    parser.add_argument('--only-test', '--test', '--eval', action='store_true', help='only test the model')
    parser.add_argument('--mode', type=str, choices=[
            "single", "parallel", "dist", # master modes
            "train-subprocess" # for train-subprocess of torch.distributed.launch
        ], default="parallel",
                        help="run tasks on single gpu or multiple gpus (parallel)")
    parser.add_argument('--title', type=str, default=None, help='title of task (determining work_dir)')
    parser.add_argument('--cuda', type=str, default='auto', help='alias of CUDA_VISIBLE_DEVICES. If set to "auto", will automatically select free GPU(s).')
    # parser.add_argument('--local-rank', type=int, default=-1, help='local rank of the current process. (necessary for torch.distributed.launch)')
    parser.add_argument('--work-dir', type=str, default=None, help='working directory (empty for auto)')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--checkpoint', '--ckpt', type=str, default=None, help='checkpoint file path (empty for auto)')
    parser.add_argument('--metrics', nargs='+', default=None, help='metrics to test')
    parser.add_argument('--max-test-threads', type=int, default=16, help='max number of threads for parallel testing. (only functional in parallel mode)')
    parser.add_argument('--skip-test-inference', action='store_true', help='skip inferencing while testing the model, only calculate metrics with pre-computed results.')

    args = parser.parse_args()

    return args

def raise_no_free_gpu_error():
    raise ValueError("No free GPU available. All busy. CUDA_VISIBLE_DEVICES: " + (os.getenv('CUDA_VISIBLE_DEVICES') if os.getenv('CUDA_VISIBLE_DEVICES') is not None else "<empty>"))

def main():
    # Get local_rank
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    is_main_process = (local_rank == -1)
    
    
    # Check Arguments
    args = parse_args()
    if args.only_train and args.only_test:
        raise ValueError("Cannot use --only-train and --only-test at the same time.")
    if not args.only_test and args.checkpoint is not None:
        raise ValueError("Cannot use --checkpoint without --only-test. Test process will use the checkpoint of last epoch of training, if training is not skipped.")
    if args.only_train and args.metrics is not None:
        raise ValueError("Cannot use --metrics with --only-train.")
    if args.only_train and args.skip_test_inference:
        raise ValueError("Cannot use --skip-test-inference with --only-train.")
    
    if os.getenv('CUDA_VISIBLE_DEVICES') and args.cuda != 'auto':
        raise ValueError("Cannot use --cuda=<int> and env var CUDA_VISIBLE_DEVICES at the same time. Consider using --cuda=auto instead.")
    if args.mode == "train-subprocess" and is_main_process:
        raise ValueError("Cannot use --mode=train-subprocess and env var LOCAL_RANK==-1 at the same time.")
    if args.mode == "train-subprocess" and args.cuda != 'auto':
        raise ValueError("Cannot use --mode=single, env var LOCAL_RANK!=-1, and --cuda=<int> at the same time, because this indicates re-assigning CUDA devices in train-subprocess of torchrun.")
    if args.mode == "parallel" and not is_main_process:
        raise ValueError("Cannot use --mode=parallel and LOCAL_RANK!=-1 at the same time.")
    if args.mode == "dist":
        raise NotImplementedError("Distributed training is not implemented yet.")
    
    
    # Load config from file
    main_config = MainConfig.from_file(args.config)
    
    
    # Override config with command line args
    ## debug
    if args.debug: main_config.GENERAL.DEBUG = True
    if is_main_process and main_config.GENERAL.DEBUG:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
    ## title
    if args.title is not None: main_config.GENERAL.TITLE = args.title
    main_config.GENERAL.TITLE = main_config.GENERAL.TITLE.replace(' ', '_')
    
    ## work_dir
    if is_main_process: # is main process
        if args.work_dir is not None:
            # if work_dir is specified, use it as is
            main_config.GENERAL.WORK_DIR = os.path.expanduser(args.work_dir)
        elif main_config.GENERAL.WORK_DIR == Auto:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            auto_work_dir = f"work_dirs/{timestamp}-{main_config.GENERAL.TITLE}"
            main_config.GENERAL.WORK_DIR = auto_work_dir
        else:
            main_config.GENERAL.WORK_DIR = os.path.expanduser(main_config.GENERAL.WORK_DIR)
    work_dir = main_config.GENERAL.WORK_DIR
    assert isinstance(work_dir, str), "work_dir should be a string here."
    
    ## checkpoint
    if args.checkpoint is not None:
        main_config.TEST.CHECKPOINT = os.path.expanduser(args.checkpoint)
    
    ## metrics
    if args.metrics is not None:
        main_config.TEST.METRICS = args.metrics
    
    
    # About work_dir
    if os.path.exists(work_dir) and is_subpath(args.config, work_dir):
        # run test under finished training work_dir, or resume training
        is_inplace = True
    else:
        is_inplace = False
        if is_main_process: os.makedirs(work_dir, exist_ok=True)
        else: raise ValueError("Cannot create work_dir on non-main process.")
    
    
    # Setup logging
    log_filepath = pjoin(work_dir, ('main' if is_main_process else f"rank_{local_rank}") + '.log')
    logger = setup_logging(log_filepath)
    
    
    # Save overriden config to work_dir
    if is_main_process:
        config_path_in_work_dir = pjoin(work_dir, 'config.yaml')
        main_config.save_as_yaml(config_path_in_work_dir)
    else: # is subprocess
        assert is_subpath(args.config, work_dir), f"Config file {args.config} is not under work_dir {work_dir}."
        config_path_in_work_dir = args.config
        
    # Configure CUDA
    
    if args.mode == "single":
        chosen_gpu: int
        if args.cuda == 'auto': # automatically select an available GPU
            free_gpus: list[int] = get_free_gpus() # CUDA_VISIBLE_DEVICES has been considered.
            if len(free_gpus) == 0: raise_no_free_gpu_error()
            chosen_gpu = free_gpus[0]
        else: # select an available GPU specified by cuda
            # Need to mention that, in this case, CUDA_VISIBLE_DEVICES must be empty,
            # because previously we've claimed that CUDA_VISIBLE_DEVICES and --cuda cannot be used at the same time.
            cuda_visible_devices: list[int] = [int(s.strip()) for s in args.cuda.split(',')]
            if len(cuda_visible_devices) == 0:
                raise SyntaxError("--cuda should not be empty.")
            os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in cuda_visible_devices)
            free_gpus = get_free_gpus()
            if len(free_gpus) == 0: raise_no_free_gpu_error()
            chosen_gpu = free_gpus[0]
        logger.info(f"Using GPU {chosen_gpu} for training.")
        torch.cuda.set_device(chosen_gpu)
    elif args.mode == "train-subprocess":
        free_gpus = get_free_gpus()
        if local_rank not in free_gpus:
            warnings.warn(f"GPU {local_rank} is not free, but is specified by env var LOCAL_RANK. This may cause errors.")
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    elif args.mode == "parallel":
        nproc_per_node: int
        if args.cuda == 'auto':
            # re-assign CUDA visible devices to the intersection of CUDA_VISIBLE_DEVICES and all free GPUs
            CUDA_VISIBLE_FREE_DEVICES = get_free_gpus(IGNORE_CUDA_VISIBLE_DEVICES=False)
            os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in CUDA_VISIBLE_FREE_DEVICES)
            nproc_per_node = len(CUDA_VISIBLE_FREE_DEVICES)
        else:
            # Need to mention that, in this case, CUDA_VISIBLE_DEVICES must be empty,
            # because previously we've claimed that CUDA_VISIBLE_DEVICES and --cuda cannot be used at the same time.
            CUDA_PREFERRED_DEVICES = [s.strip() for s in args.cuda.split(',')]
            if len(CUDA_PREFERRED_DEVICES) == 0: raise ValueError("--cuda should not be empty.")
            CUDA_PREFERRED_FREE_DEVICES = list(set(CUDA_PREFERRED_DEVICES) & set(get_free_gpus(IGNORE_CUDA_VISIBLE_DEVICES=False)))
            os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in CUDA_PREFERRED_FREE_DEVICES)
            nproc_per_node = len(CUDA_PREFERRED_FREE_DEVICES)


    # Record pid
    pid = os.getpid()
    logger.info(f"Process ID: {pid}")
    
    # Launch tasks
    train: bool = not args.only_test
    test: bool = not args.only_train
    
    
    # Record basic information
    if is_main_process:
        logger.info(f"Working directory: {work_dir}\n"
                    f"Config file: {args.config}\n"
                    f"Mode: {args.mode}\n"
                    f"CUDA (arg): {args.cuda}\n"
                    f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}\n"
                    )
        
    
    if train: # Launch training
        train_start = time.time()
        if args.mode in ["single", "train-subprocess"]:
            if args.mode == "single":
                logger.info(f"Running training on single GPU.")
            from lib.train.train_script import run
            run(config_path_in_work_dir, local_rank=local_rank)
            if args.mode == "train-subprocess": return # exit train-subprocess
        elif args.mode == "parallel":
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(random.randint(10000, 50000))
            os.environ["WORLD_SIZE"] = str(nproc_per_node)
            from torch.distributed.run import main as torchrun
            logger.info(f"Running training on multiple GPUs. Launching train-subprocess with command: `torchrun --nproc_per_node {nproc_per_node} --master_port {random.randint(10000, 50000)} {__file__} {config_path_in_work_dir} --mode train-subprocess --work-dir {work_dir}`")
            torchrun([f"--nproc_per_node={nproc_per_node}",
                    f"--master_port={random.randint(10000, 50000)}",
                    f"{__file__}",
                    f"{config_path_in_work_dir}",
                    f"--only-train",
                    f"--mode=train-subprocess",
                    f'--work-dir="{work_dir}"'])
        else:
            raise ValueError("mode should be 'single' or 'parallel' or 'train-subprocess'.")
        if is_main_process:
            logger.success(f"Training totally costs {time.time()-train_start} seconds!")
            logger.success(f"Training finished. Check {work_dir} for logs and saved models.")
            with open(pjoin(work_dir, 'FINISHED'), 'w') as f: pass
    
    if test: # Launch test
        if main_config.TEST.CHECKPOINT == Auto:
            if train or is_inplace:
                chosen_ckpt = sorted(os.listdir(os.path.join(work_dir, 'checkpoints')))[-1]
                main_config.TEST.CHECKPOINT = os.path.join(work_dir, 'checkpoints', chosen_ckpt)
            else:
                raise ValueError("Cannot auto-detect checkpoint path. You may have moved the work_dir. Please specify checkpoint path in config or command line, or try resolving CONFIG.GENERAL.WORK_DIR to the correct path.")
        test_datasets = dict()
        for dataset_info in main_config.DATA.TEST.DATASETS:
            if dataset_info.NAME not in main_config.DATA.TRAIN.DATASETS:
                test_datasets[dataset_info.NAME] = dataset_info
            else:
                logger.warning(f"Duplicated dataset {dataset_info.NAME} detected in config. Only the first occurrence will be used.")
        if len(test_datasets) == 0:
            raise ValueError("No test dataset specified in config.")
        for dataset_info in test_datasets.values():
            data_resource = build_data_resource(dataset_info.NAME, dataset_info.PATH)
            
            test_start = time.time()
            
            if not args.skip_test_inference: # infer, get results.
                if args.mode == "parallel":
                    free_gpus: list[int] = get_free_gpus()
                    logger.info(f"Running inference on {dataset_info.NAME}...")
                    multiprocessing.set_start_method('spawn', force=True)
                    with multiprocessing.Pool(processes=args.max_test_threads) as pool:
                        pool.starmap(test_infer_on_sequence, [(sequence, dataset_info.NAME, main_config, free_gpus) for sequence in data_resource])
                else: # single
                    for sequence in data_resource:
                        test_infer_on_sequence(sequence, dataset_info.NAME, main_config, free_gpus, worker_id=-1)

        # calculate metrics
        metrics_of_datasets = {}
        for dataset_info in test_datasets.values():
            logger.info(f"Calculating metrics on {dataset_info.NAME}...")
            data_resource = build_data_resource(dataset_info.NAME, dataset_info.PATH)
            track_result = build_data_resource('result', f'{work_dir}/results/{dataset_info.NAME}')
            analyzer = Analyzer(tracker_name='ProMFT', golden=data_resource, track_result=track_result, work_dir=work_dir)
            metrics_of_datasets[dataset_info.NAME] = analyzer.metrics
            logger.info(f"Test results on {dataset_info.NAME}:\n{yaml.dump(analyzer.metrics)}")
        flops_and_params = calculate_flops_and_params(main_config)
        metrics_of_datasets['general'] = flops_and_params
        metrics_path = f'{work_dir}/metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics_of_datasets, f)
        logger.info(f"Test totally costs {time.time()-test_start} seconds!")
    
if __name__ == "__main__":
    main()
