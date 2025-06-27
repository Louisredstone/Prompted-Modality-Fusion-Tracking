# In this file, we provide a function to set up logging for the project.
# All submodules will be logged to the same file and console.
# (After setup,) each submodule has a logger named after its module name.
# Therefore, you are recommended to run `logger = logging.getLogger(__name__)` in each submodule to get its own logger.

import logging
import os
import pkgutil
import importlib

# DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50
VERBOSE = 5
logging.addLevelName(VERBOSE, 'VERBOSE')
def verbose(self: logging.Logger, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kws)
logging.Logger.verbose = verbose
logging.VERBOSE = VERBOSE

SUCCESS = 25
logging.addLevelName(SUCCESS, 'SUCCESS')
def success(self: logging.Logger, message, *args, **kws):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kws)
logging.Logger.success = success
logging.SUCCESS = SUCCESS

def get_log_level(level: str) -> int:
    return getattr(logging, level.upper())

def setup_logging(log_filepath):
    top_level_module = __name__.split('.')[0] # 'lib'
    main_logger = logging.getLogger(top_level_module)
    if main_logger.hasHandlers():
        raise ValueError('Logging already set up')
    # main_logger.setLevel(log_level)
    main_logger.setLevel(logging.VERBOSE)
    
    if not os.path.exists(os.path.dirname(log_filepath)):
        os.makedirs(os.path.dirname(log_filepath))
    
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s]\n[%(levelname)s] %(message)s'))
    
    verbose_filepath = log_filepath.replace('.log', '.verbose.log')
    verbose_handler = logging.FileHandler(verbose_filepath)
    verbose_handler.setLevel(logging.VERBOSE)
    verbose_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s]\n[%(levelname)s] %(message)s'))
    
    local_rank = int(os.getenv('LOCAL_RANK', '-1'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    if local_rank == -1:
        console_handler.setFormatter(logging.Formatter('[%(asctime)s] [main] [%(levelname)s] %(message)s'))
    else:
        console_handler.setFormatter(logging.Formatter(f'[%(asctime)s] [rank {local_rank}] [%(levelname)s] %(message)s'))
    
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)
    main_logger.addHandler(verbose_handler)

    def configure_submodules(package_name):
        package = importlib.import_module(package_name)
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f'{package_name}.{module_name}'
            logger = logging.getLogger(full_module_name)
            if logger.hasHandlers(): continue
            logger.propagate = True
            if is_pkg:
                configure_submodules(full_module_name)
    
    configure_submodules(top_level_module)
    
    dist_logger = logging.getLogger('torch.distributed')
    dist_logger.setLevel(logging.DEBUG)
    
    return main_logger

def add_memory_monitor(pid, logger, interval=10):
    import psutil
    import time
    import threading
    def mem_log(pid, logger, interval):
        while True:
            main_process = psutil.Process(pid)
            total_rss = 0
            total_vms = 0
            for child in main_process.children(recursive=True):
                try:
                    info = child.memory_info()
                    total_rss += info.rss
                    total_vms += info.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            logger.info(f"Memory usage: {total_rss / 1024 ** 2:.2f} MB ({total_vms / 1024 ** 2:.2f} MB virtual)")
            time.sleep(interval)
    thread = threading.Thread(target=mem_log, args=(pid, logger, interval,))
    thread.daemon = True # thread dies with the program
    thread.start()