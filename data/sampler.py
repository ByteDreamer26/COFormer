import os
import torch
import math
import numpy as np
from data.dataset import BlendableDataset
from utils.utils_data import RLTaskInput
from typing import List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def my_collate_fn(data_list: Union[List[Tuple[RLTaskInput, Tuple]], List[tuple]]):
    """ This is a collate function that try to concatenate all tasks of the same type in a batch 
        so that it will be more efficient than multiple forwards """
    task_list = []
    batch_data_info, batch_raw_obs = None, None
    if isinstance(data_list[0], tuple):
        batch_data_info, batch_raw_obs = [], []
        for task in data_list:
            batch_data_info.append(task[1])
            task_list.append(task[0][0])
            batch_raw_obs.append(task[0][1])
    else:
        task_list = data_list

    task_merged = RLTaskInput.merge_into_one(task_list)
    prefix_mask = task_merged.prefix_mask
    task_merged.prefix_mask = None          # prefix_mask中各样本对应timestep不同，无法合并单独处理
    task_merged.apply(torch.cat, dim=0)     # 各成员 tensor 尺寸均为 (data_num, seq_len)
    task_merged.prefix_mask = prefix_mask   # len=data_num
    return task_merged, batch_data_info, batch_raw_obs

def build_dataloader(args, dataset:BlendableDataset, is_eval:bool=False, current_batch:int=0, seed:int=42):
    """ Buld DDP dataloader given an input dataset. """
    if dataset is None:
        return None
    
    batch_size_per_gpu = args.eval_batch_size_per_gpu if is_eval else args.batch_size_per_gpu

    # The DistributedSampler automatically blocks the data and sends it to each Gpu, which can avoid data overlap
    sampler = MyDistributedSampler(
        dataset=dataset,
        num_replicas=int(os.environ.get("WORLD_SIZE", default='1')),
        rank=int(os.environ.get("LOCAL_RANK", default='0')),
        seed=seed,
        init_batch=current_batch, 
        batch_size_per_gpu=batch_size_per_gpu,
        batch_num_per_iter=args.eval_batch_num if is_eval else args.batch_num,
        drop_last=False
    )

    # build DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        pin_memory=True,                    # pin data in memory, which enable DMA transfer tube between CPU and GPU
        shuffle=False,                      # Must be False, as the DistributedSampler will handle the shuffling
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=my_collate_fn
    )

class MyDistributedSampler(DistributedSampler):
    def __init__(
        self, dataset: Dataset, num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False,
        init_batch: int = 0, batch_size_per_gpu: int = 64, batch_num_per_iter: int = 64
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size_per_gpu * num_replicas

        # when self.__iter__ called, a iterator with size of macro_batch_size returned
        self.macro_batch_size = batch_num_per_iter * self.batch_size     
        self.macro_batch_num_per_epoch = len(dataset) // (self.macro_batch_size) if self.drop_last else math.ceil(len(dataset) / (self.macro_batch_size))
        self.epoch_total_samples = self.macro_batch_num_per_epoch * self.macro_batch_size
        self.batch_num_per_epoch = int(self.epoch_total_samples / self.batch_size)

        # get current epoch and data offset, which is useful for resuming training
        self.current_epoch = init_batch // self.batch_num_per_epoch
        self.current_data_offset = self.batch_size * int(init_batch % self.batch_num_per_epoch)

        # generate the sample permutation order of current epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed + self.current_epoch)
            self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()
        else:
            self.sample_idxs = list(range(self.epoch_total_samples))
    
    def update_sample_idxs(self, new_epoch:bool=True):
        # re-generate the sample permutation when the current epoch is finished 
        if self.current_data_offset + self.macro_batch_size > self.epoch_total_samples:
            self.current_data_offset = 0
            self.current_epoch = self.current_epoch + 1 if new_epoch else self.current_epoch
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.current_epoch)
                self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()

    def __iter__(self):
        # When init with a different macro_batch_size from snapshot, sometimes need to update the sample permutation
        # this func should only work at the first __iter__() called after training resumed
        self.update_sample_idxs(new_epoch=False)
        
        # clip a segment from the permutation with size of macro_batch_size
        indices = self.sample_idxs[self.current_data_offset : self.current_data_offset + self.macro_batch_size]
        indices = (np.array(indices) % len(self.dataset)).tolist()  # prevent index out of range
        assert len(indices) == self.macro_batch_size
        self.current_data_offset += self.macro_batch_size

        # re-generate the sample permutation when the current epoch is finished 
        self.update_sample_idxs(new_epoch=True)

        # subsample
        indices = indices[self.rank : self.macro_batch_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.macro_batch_num_per_epoch

    def set_batch(self, batch:int, macro_batch_now:int, is_train:bool) -> None:
        if is_train:
            # 训练集，确保每个 batch 取数据无重叠
            assert self.current_data_offset == self.batch_size * int(batch % self.batch_num_per_epoch)
        else:
            # 验证集，每 n 个训练 epoch 执行一次，每次重新随机取数据
            g = torch.Generator()
            g.manual_seed(self.seed + macro_batch_now)
            self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()
            self.current_data_offset = 0

'''
def build_dataloader(
    args, dataset:BlendableDataset, 
    epoch_total_samples:int=None, is_eval=False, seed=0, current_epoch=0
):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None

    epoch_total_samples = len(dataset) if epoch_total_samples is None else epoch_total_samples
    batch_size = args.eval_batch_size if is_eval else args.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=MyDistributedSampler(
            dataset=dataset,
            epoch_total_samples=epoch_total_samples,
            seed=seed,
            current_epoch=current_epoch
        ),    
        collate_fn=my_collate_fn
    )

class MyDistributedSampler(DistributedSampler):
    def __init__(
            self, 
            dataset: Dataset,                   # 采样目标数据集
            num_replicas: Optional[int] = None, # DDP 并行的进程数
            rank: Optional[int] = None,         # 当前进程 rank
            shuffle: bool = True,               # 是否要打乱顺序（用这个打乱的话需要每个epoch入口处设置本类的epoch变量值）
            seed: int = 0,                      # 用于打乱的随机种子
            drop_last: bool = False,            # 若无法在各个进程间均匀分割数据，是否丢弃尾部无法均分的部分
            epoch_total_samples: int = 0,       # 一个 epoch 的总样本量
            current_epoch: int = 0              # 起始 epoch，从 snapshot 或 ckpt 得到
        ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.epoch = current_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch_total_samples = epoch_total_samples
        self.dataset_total_samples = len(dataset)

        # 计算一个 epoch 需要返回的索引总量 epoch_total_samples
        if self.drop_last and self.epoch_total_samples % self.num_replicas != 0: 
            # 如果 drop_last 且无法整除，则每个进程分到的样本数向下取整 
            self.num_samples = int(self.epoch_total_samples / self.num_replicas)
        else:
            # 如果不 drop_last（需填充重复元素补满）或可以整除，则每个进程分到的样本数向上取整
            self.num_samples = math.ceil(self.epoch_total_samples / self.num_replicas)             
        self.epoch_total_samples = self.num_samples * self.num_replicas

        # 计算遍历一次数据集的总数据量 total_samples
        if self.epoch_total_samples > self.dataset_total_samples:
            print(f'Note that some samples will be repeated in one epoch: there are {self.dataset_total_samples} samples in dataset and we want {self.epoch_total_samples}')
            self.total_samples = self.epoch_total_samples
        else:
            last_data_num = self.dataset_total_samples % self.epoch_total_samples
            self.total_samples = self.dataset_total_samples + (self.epoch_total_samples - last_data_num)    # 令 self.total_samples 可以整除 self.epoch_total_samples
        assert self.total_samples % self.epoch_total_samples == 0

        #self.current_data_offset = self.epoch_total_samples * current_epoch
        self.current_data_offset = self.epoch_total_samples * int(current_epoch % (self.total_samples / self.epoch_total_samples))
        self.init_current_data_offset = self.current_data_offset

        # 初始样本排序
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed + current_epoch)
            self.sample_idxs = torch.randperm(self.total_samples, generator=g).tolist()
        else:
            self.sample_idxs = list(range(self.total_samples))
        
    def __iter__(self):
        # 从排列中依次截取长度 epoch_total_samples 的片段
        indices = self.sample_idxs[self.current_data_offset : self.current_data_offset + self.epoch_total_samples]
        assert len(indices) == self.epoch_total_samples
        self.current_data_offset += self.epoch_total_samples

        # 已经完全遍历一次，按 shuffle 标记重新生成排序
        if self.current_data_offset >= self.total_samples:
            self.current_data_offset = 0
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                self.sample_idxs = torch.randperm(self.total_samples, generator=g).tolist()
        
        # subsample
        indices = indices[self.rank : self.epoch_total_samples : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def reset(self):
        self.current_data_offset = self.init_current_data_offset

    def set_epoch(self, epoch: int, is_train: bool) -> None:
        self.epoch = epoch
        if is_train:
            # 训练集，确保每个 epoch 取数据无重叠
            assert self.current_data_offset == self.epoch_total_samples * int(epoch % (self.total_samples / self.epoch_total_samples))
        else:
            # 验证集，每 n 个训练 epoch 执行一次，每次重新随机取数据
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.sample_idxs = torch.randperm(self.total_samples, generator=g).tolist()
            self.current_data_offset = 0
'''