import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.BaseEnv_COP import Logger_COP
from environment.used.Env_bp_v1 import BP_V1, DDP_BP_V1
from data.raw.bp import MAX_CAPACITY, MAX_ITEM_VOLUME

class BP_V2(BP_V1):
    def __init__(self, render_mode="rgb_array", scale=20, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_BP_V2'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'capacity_left': spaces.Box(low=0, high=MAX_CAPACITY[self.item_num], shape=(1,), dtype=np.int32),
            'visited': spaces.MultiDiscrete([2]*self.item_num),
            'item_volumes': spaces.Box(low=0, high=MAX_ITEM_VOLUME[self.item_num], shape=(self.item_num,), dtype=np.int32),
            'item_values': spaces.Box(low=0, high=int(1.5*MAX_ITEM_VOLUME[self.item_num]), shape=(self.item_num,), dtype=np.int32),
        })

    def _get_observation(self):
        item_volumes = self.item_volumes.copy()
        item_values = self.item_values.copy()
        visited = self.visited.copy()
        item_volumes[visited==1] = 0
        item_values[visited==1] = 0
        
        return {
            'capacity_left': np.array([self.capacity_left,], dtype=np.int32),   # (1,)
            'visited': visited.astype(np.int32),                                # (item_num, )
            'item_volumes': item_volumes,                                       # (item_num, )
            'item_values': item_values,                                         # (item_num, )
        }

class BP_logger_V2(Logger_COP):
    def __init__(self, env_name='Env_01BP', dataset_name='01BP'):
        super().__init__(env_name, dataset_name)

    def log_episode(self, desc, is_eval, episode, epoch_num=0, episode_num=0, time_used=0, seed=0):
        phase = 'eval/log' if is_eval else 'train' 
        local_rank = os.getenv('LOCAL_RANK')
        log_floder_path = f'{base_path}/visualize/{phase}/{self.env_name}/{self.dataset_name}/seed-{seed}'
        log_path = f'{log_floder_path}/[GPU{local_rank}] {desc}.txt' if local_rank is not None else \
                     f'{log_floder_path}/{desc}.txt'

        # 初次 log 时创建 log 文件
        create_file_if_not_exist(log_path)

        '''
        # 追加 log 信息
        with open(log_path, 'a') as file:
            file.write('-'*15+f' epoch-{epoch_num}; episode-{episode_num}; time-{round(time_used, 2)}'+'-'*15+'\n')
            obss = episode['observations']
            capacities = obss['capacity_left']
            item_volumes = obss['item_volumes']
            item_values = obss['item_values']
            acts = episode['actions']
            rewards = episode['rewards']
            act_value_space = episode['act_value_space']
            for t in range(len(rewards)):
                act = acts[t].item()
                file.write(f'capacity_left:\t{capacities[t]}\n')
                file.write(f'item_volumes:\t{[round(volum, 2) for volum in item_volumes[t]]}\n')
                file.write(f'item_values:\t{[round(value, 2) for value in item_values[t]]}\n')
                file.write(f'action_space:\t{act_value_space[t][0]}\n')
                file.write(f'take action:\t{act}\n')
                file.write(f'get value:  \t{item_values[t][act]:.2f}\n')
                file.write(f'get volume: \t{item_volumes[t][act]:.2f}\n')
                file.write(f'get reward: \t{rewards[t]}\n\n')
        '''

class DDP_BP_V2(DDP_BP_V1):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.batch_size = batch_size
        self.name = 'Env_BP_V2'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'capacity_left': spaces.Box(low=0, high=MAX_CAPACITY[self.item_num], shape=(batch_size,), dtype=np.int32),
            'visited': spaces.MultiDiscrete([[2]*self.item_num for _ in range(batch_size)]),
            'item_volumes': spaces.Box(low=0, high=MAX_ITEM_VOLUME[self.item_num], shape=(batch_size, self.item_num,), dtype=np.int32),
            'item_values': spaces.Box(low=0, high=int(1.5*MAX_ITEM_VOLUME[self.item_num]), shape=(batch_size, self.item_num,), dtype=np.int32),
        })

    def _get_observation(self):
        capacity_left = self.capacity_left.copy()
        item_volumes = self.item_volumes.copy()
        item_values = self.item_values.copy()
        visited = self.visited.copy()
        item_volumes[visited==1] = 0
        item_values[visited==1] = 0
        
        return {
            'capacity_left': capacity_left.astype(np.int32),    # (batch_size, )
            'visited': visited.astype(np.int32),                # (batch_size, item_num, )
            'item_volumes': item_volumes,                       # (batch_size, item_num)
            'item_values': item_values,                         # (batch_size, item_num)
        }