import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.BaseEnv_COP import Logger_COP
from environment.used.Env_cvrp_v1 import CVRP_V1, DDP_CVRP_V1
from data.raw.cvrp import CAPACITIES

class CVRP_V2(CVRP_V1):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_CVRP_V2'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(3*self.node_num,), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'demand': spaces.MultiDiscrete([10]*self.node_num),     # 应该是 [1,9], 但是 MultiDiscrete 只能从 0 开始，此处设置为 [0,9]
            'capacity': spaces.MultiDiscrete([50,]),
        })

    def _get_observation(self):
        pos_node = self.pos_node.copy()
        visited = self.visited.copy()
        demand = self.demand.copy()
        current_position = self.pos_depot.copy() if self.current_index == 0 else self.pos_node[self.current_index-1].copy()
        node_info = np.hstack((pos_node, visited[:,None]))

        demand[visited==1] = 0
        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),
            'node_info': node_info.flatten().astype(np.float32),
            'demand': demand.astype(np.int32),
            'capacity': np.array([self.capacity_left,], dtype=np.int32),
            'current_position': current_position.copy().astype(np.float32),
        }
        return obs

class CVRP_logger_V2(Logger_COP):
    def __init__(self, env_name='Env_CVRP', dataset_name='CVRP'):
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
                demand = obss['demand'][t]
                current_location = obss['current_position'][t]
                capacity_left = obss['capacity'][t]
                node_info = obss['node_info'][t].reshape((-1, 3))
                node_info = np.hstack((node_info, demand[:,None]))
                
                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'capacity left:   \t{capacity_left}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t].item()} (node{acts[t].item()-1})\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_CVRP_V2(DDP_CVRP_V1):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_CVRP_V2'
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(batch_size, 3*self.node_num), dtype=np.float32),
            'demand': spaces.MultiDiscrete([[10]*self.node_num for _ in range(batch_size)]), # 应该是 [1,9], 但是 MultiDiscrete 只能从 0 开始，此处设置为 [0,9]
            'capacity': spaces.MultiDiscrete([[CAPACITIES[self.node_num]] for _ in range(batch_size)]),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
        })

    def _get_observation(self):
        current_position = self.pos[np.arange(self.batch_size), self.current_index, :].copy()   # (batch_size, 2)  
        demand = self.demand.copy()
        capacity_left = self.capacity_left.copy()
        pos_node = self.pos_node.copy()
        visited = self.visited.copy()
        node_info = np.concatenate((pos_node, visited[:,:,None]), axis=-1)
        demand[visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),                              # (batch_size, 2)
            'node_info': node_info.reshape(self.batch_size, self.node_num*3).astype(np.float32),# (batch_size, node_num * 3)
            'demand': demand.astype(np.int32),                                                  # (batch_size, node_num)
            'capacity': capacity_left.astype(np.int32),                                         # (batch_size, )
            'current_position': current_position.astype(np.float32),                            # (batch_size, 2)
        }
        return obs