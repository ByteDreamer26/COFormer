import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.Env_op_v2 import OP_V2, DDP_OP_V2, OP_logger_V2
from data.raw.op import OFFSET_LENGTH

class OP_V4(OP_V2):
    @classmethod
    def get_range_offset(cls, node_num):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5, 'prize':-0.5, 'length': OFFSET_LENGTH[node_num]},
            'prefix':{'pos_depot':-0.5, 'pos_node':-0.5, 'prize':-0.5}
        }
        return cls.adj_range_offset
    
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_OP_V4'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([2]*self.node_num),  
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'prize': spaces.Box(low=0, high=1, dtype=np.float32),
            'length': spaces.Box(low=0, high=4, dtype=np.float32),
        })

    def _get_observation(self):
        prize = 0 if self.model_answer == [] else self.prize[self.model_answer[-1]-1]
        current_position = self.pos[self.current_index].copy().astype(np.float32)
        length_left = self.length_left
        if self.adj_range:
            prize += self.adj_range_offset['observation']['prize']
            current_position += self.adj_range_offset['observation']['current_position']
            length_left += self.adj_range_offset['observation']['length']

        obs = {
            'visited': self.visited.copy().astype(np.int32),        # (node_num, )
            'prize': np.array([prize, ]).astype(np.float32),        # (1, )                
            'length': np.array([length_left,]).astype(np.float32),  # (1, )
            'current_position': current_position,                   # (2, )
        }
        return obs

    def get_prefix(self):
        pos_depot = self.pos_depot.copy().astype(np.float32)
        pos_node = self.pos_node.copy().flatten().astype(np.float32)
        prize = self.prize.astype(np.float32)
        if self.adj_range:
            pos_depot += self.adj_range_offset['prefix']['pos_depot']
            pos_node += self.adj_range_offset['prefix']['pos_node']
            prize += self.adj_range_offset['prefix']['prize']

        return {
            'pos_depot': pos_depot,             # (2, )
            'pos_node': pos_node,               # (node_num*2, )
            'prize': prize,                     # (node_num, )
        }

    def get_prefix_mask(self):
        return {
            'pos_depot': np.zeros(2, dtype=bool),                               # (2, )
            'pos_node': np.repeat(self.visited, 2).astype(bool),                # (node_num*2, )
            'prize': self.visited.copy().astype(bool),                          # (node_num, )
        }

class OP_logger_V4(OP_logger_V2):
    def __init__(self, env_name='Env_OP', dataset_name='OP'):
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
            acts = episode['actions']
            rewards_AM = episode['rewards']['AM']
            rewards_DB1 = episode['rewards']['DB1']
            obss = episode['observations']
            act_value_space = episode['act_value_space']    

            file.write('-'*15+f' epoch-{epoch_num}; episode-{episode_num}; time-{round(time_used, 2)}'+'-'*15+'\n')
            file.write(f'pos_depot: \t{obss["pos_depot"][0]}\n\n')
            for t in range(len(rewards_AM)):
                prize = obss['prize'][t]
                current_location = obss['current_position'][t]
                length_left = obss['length'][t]
                node_info = obss['node_info'][t].reshape((-1, 3))
                node_info = np.hstack((node_info, prize[:,None]))
                assert prize[acts[t]-1].item() != 0

                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'length left:     \t{length_left}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t]} (node{acts[t]-1})\n')
                file.write(f'get prize:       \t{prize[acts[t]-1]}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')
        '''
        
class DDP_OP_V4(DDP_OP_V2):
    @classmethod
    def get_range_offset(cls, node_num):
        cls.adj_range_offset = {
            'observation':{'current_position':-0.5, 'prize':-0.5, 'length': OFFSET_LENGTH[node_num]},
            'prefix':{'pos_depot':-0.5, 'pos_node':-0.5, 'prize':-0.5}
        }
        return cls.adj_range_offset
    
    def __init__(self, render_mode="rgb_array", scale=10, batch_size=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_OP_V4'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([[2]*self.node_num for _ in range(batch_size)]),  
            'prize': spaces.Box(low=0, high=1, shape=(batch_size,), dtype=np.float32),
            'length': spaces.Box(low=0, high=4, shape=(batch_size, ), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
        })

    def _get_observation(self):
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy().astype(np.float32)
        prize = np.array([0 if answer == [] else self.prize[i][answer[-1]-1] for i, answer in enumerate(self.model_answer)]).astype(np.float32)
        length_left = self.length_left.copy().astype(np.float32)
        if self.adj_range:
            prize += self.adj_range_offset['observation']['prize']
            current_position += self.adj_range_offset['observation']['current_position']
            length_left += self.adj_range_offset['observation']['length']

        obs = {
            'visited': self.visited.copy().astype(np.int32),    # (batch_size, node_num)
            'prize': prize,                                     # (batch_size, )
            'length': length_left,                              # (batch_size, )
            'current_position': current_position                # (batch_size, 2)
        }
        return obs
    
    def get_prefix(self):
        pos_depot = self.pos_depot.copy().astype(np.float32)
        pos_node = self.pos_node.reshape(self.batch_size, self.node_num*2).astype(np.float32)
        prize = self.prize.astype(np.float32)
        if self.adj_range:
            pos_depot += self.adj_range_offset['prefix']['pos_depot']
            pos_node += self.adj_range_offset['prefix']['pos_node']
            prize += self.adj_range_offset['prefix']['prize']

        return {
            'pos_depot': pos_depot,     # (batch_size, 2)
            'pos_node': pos_node,       # (batch_size, node_num*2)
            'prize': prize,             # (batch_size, node_num)
        }

    def get_prefix_mask(self):
        return {
            'pos_depot': np.zeros((self.batch_size, 2), dtype=bool),                                    # (batch_size, 2)
            'pos_node': np.repeat(self.visited, 2, axis=-1).reshape(self.batch_size, -1).astype(bool),  # (batch_size, node_num*2)
            'prize': self.visited.copy().astype(bool),                                                  # (batch_size, node_num)
        }