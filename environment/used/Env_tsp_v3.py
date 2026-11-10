import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.BaseEnv_COP import Logger_COP
from environment.used.Env_tsp_v2 import TSP_V2
from environment.used.Env_tsp_v2 import TSP_V2, DDP_TSP_V2

class TSP_V3(TSP_V2):
    @classmethod
    def get_range_offset(cls):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5},
            'prefix':{'position':-0.5}
        }
        return cls.adj_range_offset

    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_TSP_V3'

        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([2]*self.num_nodes), 
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
        })

    def _get_observation(self):
        current_position = self.position[self.current_index].copy().astype(np.float32)
        if self.adj_range:
            current_position += self.adj_range_offset['observation']['current_position']

        return {
            'visited': self.visited.copy().astype(np.int32),
            'current_position': current_position
        }

    def get_prefix(self):
        position = self.position.copy().flatten().astype(np.float32)
        if self.adj_range:
            position += self.adj_range_offset['prefix']['position']
            
        return {
            'position': position    # (node_num*2, )
        }
    
    def get_prefix_mask(self):
        return {
            'position': np.repeat(self.visited, 2, axis=-1).astype(bool)    # (node_num*2, )
        }

class TSP_logger_V3(Logger_COP):
    def __init__(self, env_name='Env_TSP', dataset_name='TSP'):
        super().__init__(env_name, dataset_name)

    def log_episode(self, desc, is_eval, episode, epoch_num=0, episode_num=0, time_used=0, seed=0):
        phase = 'eval/log' if is_eval else 'train' 
        log_floder_path = f'{base_path}/visualize/{phase}/{self.env_name}/{self.dataset_name}/seed-{seed}'
        log_path = f'{log_floder_path}/[GPU{self.local_rank}] {desc}.txt' if self.local_rank is not None else \
                     f'{log_floder_path}/{desc}.txt'
        
        # 初次 log 时创建 log 文件
        create_file_if_not_exist(log_path)

        # 追加 log 信息
        with open(log_path, 'a') as file:
            file.write('-'*15+f' epoch-{epoch_num}; episode-{episode_num}; time-{round(time_used, 2)}'+'-'*15+'\n')
            acts = episode['actions']
            rewards_AM = episode['rewards']['AM']
            rewards_DB1 = episode['rewards']['DB1']
            assert len(rewards_AM) == len(rewards_DB1)
            obss = episode['observations']
            visiteds = None if 'visited' not in obss else obss['visited']
            current_poss = obss['current_position']   
            act_value_space = episode['act_value_space']         

            position = episode['prefix']['position'].reshape((-1, 2)) 
            file.write(f'city position:\n{position}\n\n')
            for t in range(len(rewards_AM)):           
                if visiteds:
                    file.write(f'visited:         \t{visiteds[t]}\n')
                file.write(f'current location:\t{current_poss[t]}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t].item()}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_TSP_V3(DDP_TSP_V2):
    @classmethod
    def get_range_offset(cls):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5},
            'prefix':{'position':-0.5}
        }
        return cls.adj_range_offset
    
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_TSP_V3'
        
        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([[2]*self.num_nodes for _ in range(batch_size)]),           # (batch_size, node_num)
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),     # (batch_size, 2)
        })
    
    def _get_observation(self):
        current_position = self.position[np.arange(self.batch_size), self.current_index].copy().astype(np.float32)
        if self.adj_range:
            current_position += self.adj_range_offset['observation']['current_position']
        
        return {
            'visited': self.visited.copy().astype(np.int32),        # (batch_size, node_num)
            'current_position': current_position                    # (batch_size, 2)
        }
    
    def get_prefix(self):
        position = self.position.copy().reshape(self.batch_size, -1).astype(np.float32)
        if self.adj_range:
            position += self.adj_range_offset['prefix']['position']

        return {
            'position': position    # (batch_size, node_num*2)
        }

    def get_prefix_mask(self):
        return {
            'position': np.repeat(self.visited, 2, axis=-1).reshape(self.batch_size, -1).astype(bool)   # (batch_size, node_num*2)
        }