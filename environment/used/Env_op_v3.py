# 相比 op_v1 调整了动作空间取值范围
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.Env_op_v2 import OP_V2, DDP_OP_V2, OP_logger_V2

class OP_V3(OP_V2):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_OP_V3'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(3*self.node_num,), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'prize': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'length': spaces.Box(low=0, high=4, dtype=np.float32),
        })

    def _get_observation(self):
        pos_node = self.pos_node.copy()
        visited = self.visited.copy()
        node_info = np.hstack((pos_node, visited[:,None]))
        current_position = self.pos[self.current_index].copy()
        prize = self.prize.copy()
        prize[self.visited==1] = 0
                    
        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),
            'node_info': node_info.flatten().astype(np.float32),
            'prize': prize.astype(np.float32),
            'length': np.array([self.length_left,]).astype(np.float32),
            'current_position': current_position.astype(np.float32),
        }
        return obs

class OP_logger_V3(OP_logger_V2):
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

class DDP_OP_V3(DDP_OP_V2):
    def __init__(self, render_mode="rgb_array", scale=10, batch_size=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_OP_V3'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(batch_size, 3*self.node_num), dtype=np.float32),
            'prize': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num,), dtype=np.float32),
            'length': spaces.Box(low=0, high=4, shape=(batch_size, ), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
        })
       
    def _get_observation(self):
        pos_node = self.pos_node.copy()                                     # (batch_size, node_num, 2)
        visited = self.visited.copy()                                       # (batch_size, node_num, )
        node_info = np.concatenate((pos_node, visited[:,:,None]), axis=-1)  # (batch_size, node_num, 3)
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy()
        prize = self.prize.copy()
        prize[self.visited==1] = 0
        
        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),      # (batch_size, 2)
            'node_info': node_info.reshape(self.batch_size, self.node_num*3).astype(np.float32),    # (batch_size, 3*node_num)
            'prize': prize.astype(np.float32),                          # (batch_size, )
            'length': self.length_left.copy().astype(np.float32),       # (batch_size, )
            'current_position': current_position.astype(np.float32),    # (batch_size, 2)
        }
        return obs