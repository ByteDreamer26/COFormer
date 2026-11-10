import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.Env_pctsp_v1 import PCTSP_V1, PCTSP_logger_V1, DDP_PCTSP_V1
from data.raw.pctsp import OFFSET_PRIZE, OFFSET_PENALTY

class PCTSP_V3(PCTSP_V1):
    @classmethod
    def get_range_offset(cls, node_num):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5, 'prize2go':-0.5},
            'prefix':{'pos_depot':-0.5, 'pos_node':-0.5, 'prize':OFFSET_PRIZE[node_num], 'penalty':OFFSET_PENALTY[node_num]}
        }
        return cls.adj_range_offset

    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_PCTSP_V3'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([2]*self.node_num),  
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, dtype=np.float32),
        })

    def _get_observation(self):
        current_position = self.pos[self.current_index].copy().astype(np.float32)
        prize2go = np.array([self.prize2go,]).astype(np.float32)
        if self.adj_range:
            prize2go += self.adj_range_offset['observation']['prize2go']
            current_position += self.adj_range_offset['observation']['current_position']

        return {
            'prize2go': prize2go,
            'current_position': current_position,
            'visited': self.visited.copy().astype(np.int32)
        }

    def get_prefix(self):
        pos_depot = self.pos_depot.copy().astype(np.float32)
        pos_node = self.pos_node.copy().flatten().astype(np.float32)
        prize = self.prize.astype(np.float32)
        penalty = self.penalty.astype(np.float32)
        if self.adj_range:
            pos_depot += self.adj_range_offset['prefix']['pos_depot']
            pos_node += self.adj_range_offset['prefix']['pos_node']
            prize += self.adj_range_offset['prefix']['prize']
            penalty += self.adj_range_offset['prefix']['penalty']

        return {
            'pos_depot': pos_depot,     # (2, )
            'pos_node': pos_node,       # (node_num*2, )
            'prize': prize,             # (node_num, )
            'penalty': penalty,         # (node_num, )
        }

    def get_prefix_mask(self):
        return {
            'pos_depot': np.zeros(2, dtype=bool),                               # (2, )
            'pos_node': np.repeat(self.visited, 2).astype(bool),                # (node_num*2, )
            'prize': self.visited.copy().astype(bool),                          # (node_num, )
            'penalty': self.visited.copy().astype(bool),                        # (node_num, )
        }

class PCTSP_logger_V3(PCTSP_logger_V1):
    def __init__(self, env_name='Env_PCTSP', dataset_name='PCTSP'):
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
            acts = episode['actions']
            rewards_AM = episode['rewards']['AM']
            rewards_DB1 = episode['rewards']['DB1']
            assert len(rewards_AM) == len(rewards_DB1)
            obss = episode['observations']
            act_value_space = episode['act_value_space']    
            prefix_mask = episode['prefix_masks']
            prefix = episode['prefix']
            assert prefix_mask['pos_depot'].sum() == 0

            file.write('-'*15+f' epoch-{epoch_num}; episode-{episode_num}; time-{round(time_used, 2)}'+'-'*15+'\n')
            file.write(f'pos_depot: \t{prefix["pos_depot"]}\n\n')
            for t in range(len(rewards_AM)):
                current_location = obss['current_position'][t]
                prize2go = obss['prize2go'][t].item()
                visited = obss['visited'][t]
                assert prize2go >= 0

                masked_prize = prefix['prize'].copy()
                masked_penalty = prefix['penalty'].copy()
                masked_prize[prefix_mask['prize'][t]] = 0
                masked_penalty[prefix_mask['penalty'][t]] = 0

                node_info = np.hstack((
                    #np.arange(len(prize), dtype=np.int32)[:,None],
                    prefix["pos_node"].reshape(-1,2), 
                    masked_prize[:,None], 
                    masked_penalty[:,None],
                    visited[:,None]
                ))
                #node_info = np.round(node_info, 3)
                prize = prefix['prize']
                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t]} (to node {acts[t]-1})\n')
                file.write(f'prize to go:     \t{prize2go}\n')
                file.write(f'get prize:       \t{prize[acts[t]-1]}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_PCTSP_V3(DDP_PCTSP_V1):
    @classmethod
    def get_range_offset(cls, node_num):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5, 'prize2go':-0.5},
            'prefix':{'pos_depot':-0.5, 'pos_node':-0.5, 'prize':OFFSET_PRIZE[node_num], 'penalty':OFFSET_PENALTY[node_num]}
        }
        return cls.adj_range_offset

    def __init__(self, render_mode="rgb_array", scale=10, batch_size=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_PCTSP_V3'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([[2]*self.node_num for _ in range(batch_size)]),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, shape=(batch_size, ), dtype=np.float32),
        })
       
    def _get_observation(self):
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy().astype(np.float32)
        prize2go = self.prize2go.copy().astype(np.float32)
        if self.adj_range:
            current_position += self.adj_range_offset['observation']['current_position']
            prize2go += self.adj_range_offset['observation']['prize2go']

        obs = {
            'prize2go': prize2go,                                   # (batch_size, )
            'current_position': current_position,                   # (batch_size, 2)
            'visited': self.visited.copy().astype(np.int32)         # (batch_size, node_num)
        }
        return obs

    def get_prefix(self):
        pos_depot = self.pos_depot.copy().astype(np.float32)
        pos_node = self.pos_node.reshape(self.batch_size, self.node_num*2).astype(np.float32)
        prize = self.prize.astype(np.float32)
        penalty = self.penalty.astype(np.float32)
        if self.adj_range:
            pos_depot += self.adj_range_offset['prefix']['pos_depot']
            pos_node += self.adj_range_offset['prefix']['pos_node']
            prize += self.adj_range_offset['prefix']['prize']
            penalty += self.adj_range_offset['prefix']['penalty']

        return {
            'pos_depot': pos_depot,                     # (batch_size, 2)
            'pos_node': pos_node,                       # (batch_size, node_num*2)
            'prize': prize,                             # (batch_size, node_num)
            'penalty': penalty,                         # (batch_size, node_num)
        }

    def get_prefix_mask(self):
        return {
            'pos_depot': np.zeros((self.batch_size, 2), dtype=bool),                                    # (batch_size, 2)
            'pos_node': np.repeat(self.visited, 2, axis=-1).reshape(self.batch_size, -1).astype(bool),  # (batch_size, node_num*2)
            'prize': self.visited.copy().astype(bool),                                                  # (batch_size, node_num)
            'penalty': self.visited.copy().astype(bool),                                                # (batch_size, node_num)
        }