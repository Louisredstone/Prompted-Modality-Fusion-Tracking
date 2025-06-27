import logging
logger = logging.getLogger(__name__)
import os
import datetime
from collections import OrderedDict
from ..trainers import BaseTrainer
from ..admin import AverageMeter, StatValue
from ..admin import TensorboardWriter
import torch
import time
import numpy as np  
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from matplotlib import pyplot as plt
from ...utils.misc import get_world_size
from ...config import MainConfig

DEBUG = True

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, config: MainConfig, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            config - Training configuration
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, config, lr_scheduler)

        # self._set_default_config()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if config.GENERAL.LOCAL_RANK in [-1, 0]:
            # tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path) # SHITCODE
            tensorboard_writer_dir = os.path.join(self.config.GENERAL.WORK_DIR, 'tensorboard')
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if config.GENERAL.USE_WANDB:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * config.TRAIN.BATCH_SIZE)  # * interval

        self.move_data_to_gpu = config.GENERAL.MOVE_DATA_TO_GPU
        self.config = config
        self.use_amp = use_amp
        if use_amp: self.scaler = GradScaler()

    # def _set_default_config(self):
    #     # Dict of all default values
    #     default = {'print_interval': 10,
    #                'print_stats': None,
    #                'description': ''}

    #     for param, default_value in default.items():
    #         if getattr(self.config, param, None) is None:
    #             setattr(self.config, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        logger.info('Cycling dataset')
        logger.info(f'Starting epoch {self.epoch} for {loader.name}')

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        '''add fix rgb pretrained net bn, only used in box_head'''
        if self.config.TRAIN.FIX_BN:
            self.actor.fix_bns()

        self._init_timing()

        for i, data in enumerate(loader, 1):
            logger.debug('Data loaded from loader.')
            # if DEBUG:
            #     one, n_images, channel, height, width = data['template_images'].shape
            #     rows = int(np.ceil(np.sqrt(n_images)))
            #     cols = int(np.ceil(n_images / rows))
            #     fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
            #     axes = axes.flatten()
                # for idx, img in enumerate(data['template_images'][0]):
                #     rgb = (img[:3,:,:].numpy() * np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis] + np.array([0.486, 0.457, 0.407])[:, np.newaxis, np.newaxis]).transpose(1,2,0)
                #     # aux = (img[3:,:,:].numpy() * np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis] + np.array([0.486, 0.457, 0.407])[:, np.newaxis, np.newaxis]).transpose(1,2,0)
                #     axes[idx].imshow(rgb)
                #     # axes[idx].axis('off')
                # plt.show()
                # input('pause...')
                # for idx, img in enumerate(data['search_images'][0]):
                #     rgb = (img[:3,:,:].numpy() * np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis] + np.array([0.486, 0.457, 0.407])[:, np.newaxis, np.newaxis]).transpose(1,2,0)
                #     # aux = (img[3:,:,:].numpy() * np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis] + np.array([0.486, 0.457, 0.407])[:, np.newaxis, np.newaxis]).transpose(1,2,0)
                #     axes[idx].imshow(rgb)
                #     # axes[idx].axis('off')
                # plt.show()
                # input('pause...')
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['config'] = self.config
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.config.TRAIN.GRAD_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.config.TRAIN.GRAD_CLIP_NORM)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print_str = f'[local_rank: {self.config.GENERAL.LOCAL_RANK}] '
        print_str += f'Epoch Time: {datetime.timedelta(seconds=epoch_time)}, '
        print_str += f'Avg Data Time: {self.avg_date_time / self.num_frames * batch_size:.5f}, '
        print_str += f'Avg GPU Trans Time: {self.avg_gpu_trans_time / self.num_frames * batch_size:.5f}, '
        print_str += f'Avg Forward Time: {self.avg_forward_time / self.num_frames * batch_size:.5f}'
        print(print_str)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.config.GENERAL.LOCAL_RANK in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.config.TRAIN.PRINT_INTERVAL == 0 or i == loader.__len__():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_str = f'[{timestamp}] [local_rank: {self.config.GENERAL.LOCAL_RANK}]\n'
            # print_str += '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += f'Epoch: {self.epoch}/?, Batch: {i}/{loader.__len__()}, LR: {self.lr_scheduler.get_last_lr()[0]}, FPS: {average_fps:.1f} ({batch_fps:.1f})'
            print_str += f'DataTime: {self.avg_date_time / self.num_frames * batch_size:.3f} ({self.avg_gpu_trans_time / self.num_frames * batch_size:.3f}), '
            print_str += f'ForwardTime: {self.avg_forward_time / self.num_frames * batch_size:.3f}, '
            print_str += f'TotalTime: {(current_time - self.start_time) / self.num_frames * batch_size:.3f}\n'
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            # print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            # print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.config.GENERAL.PRINT_STATS is None or name in self.config.GENERAL.PRINT_STATS):
                    if hasattr(val, 'avg'):
                        print_str += f'{name}: {val.avg:.5f}, '
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            # print(print_str[:-5])
            print(print_str)
            # log_str = print_str[:-5] + '\n'
            log_str = print_str + '\n'
            with open(self.config.GENERAL.WORK_DIR + '/train_log', 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.config.GENERAL.TITLE, self.config.GENERAL.DESCRIPTION)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
