import os
import subprocess
import csv
import sys
import io

def get_gpu_usage(IGNORE_CUDA_VISIBLE_DEVICES=True) -> list[list[str]]:
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
    
    command = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']
    # Note: the result of nvidia-smi is not affected by CUDA_VISIBLE_DEVICES environment variable.
    result = subprocess.run(command, stdout=subprocess.PIPE)
    info_table: str = result.stdout.decode('utf-8')
    # output example: 
    # 0, 100MiB / 11178MiB, 31% Utilization
    # 1, 100MiB / 11178MiB, 31% Utilization
    
    # 解析 CSV 输出
    reader = csv.reader(io.StringIO(info_table))
    all_gpu_usage = list(reader)
    
    # 如果 CUDA_VISIBLE_DEVICES 未设置，返回所有 GPU 的信息
    if not IGNORE_CUDA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is None:
        return all_gpu_usage
    
    # 将 CUDA_VISIBLE_DEVICES 转换为整数列表
    visible_gpu_indices = [int(index) for index in CUDA_VISIBLE_DEVICES.split(',')]
    
    # 过滤出可见 GPU 的信息
    visible_gpu_usage = []
    for index, usage in enumerate(all_gpu_usage):
        id_ = int(usage[0])
        if id_ in visible_gpu_indices:
            local_rank = visible_gpu_indices.index(id_)
            usage[0] = str(local_rank)
            visible_gpu_usage.append(usage)
    
    return visible_gpu_usage

def is_gpu_in_use(gpu_info):
    index, memory_used, memory_total, gpu_util = gpu_info
    memory_used = int(memory_used.split()[0])
    memory_total = int(memory_total.split()[0])
    gpu_util = int(gpu_util.split()[0])
    
    memory_usage_percentage = (memory_used / memory_total) * 100
    
    # 判断标准
    if memory_usage_percentage > 10 or gpu_util > 10:
        return True
    return False

def get_free_gpus(gpu_usage=None, IGNORE_CUDA_VISIBLE_DEVICES=True) -> list[int]:
    # Mention: This function returns a list of free GPU local_rank indices.
    # In other words, when CUDA_VISIBLE_DEVICES is set, it will only return the free GPUs that are visible to the current process.
    if gpu_usage is None:
        gpu_usage = get_gpu_usage(IGNORE_CUDA_VISIBLE_DEVICES=IGNORE_CUDA_VISIBLE_DEVICES)
    free_gpus = []
    for gpu in gpu_usage:
        if not is_gpu_in_use(gpu):
            free_gpus.append(int(gpu[0]))
    
    return free_gpus