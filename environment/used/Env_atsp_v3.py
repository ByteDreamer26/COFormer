# For GATO
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

class ATSP_V3(TSP_V2):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_ATSP_V3'
        
        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([2]*self.num_nodes), 
            'current_embedding': spaces.Box(low=0, high=1, shape=(16, ), dtype=np.float32),
            'current_embedding': spaces.Box(low=0, high=1, shape=(16*self.num_nodes, ), dtype=np.float32),
        })

    def _recover(self, problem_info, problem_obj):
        '''还原到评估问题的初始状态'''
        if problem_obj is None:
            self.problem_best_obj = self.problem_random_obj = None
        else:
            self.problem_best_obj, self.problem_random_obj = problem_obj
        
        prefix, problem, answer = problem_info
        assert isinstance(problem['node_embedding'], np.ndarray)
        self.position = problem['position'].copy().reshape((self.num_nodes, 2)) 
        self.node_embedding = problem['node_embedding'].copy().reshape((self.num_nodes, 16)) 
        #distance, real_answer = TSP_lkh(self.position)
        return answer
    
    def _get_observation(self):
        mask = np.repeat(self.visited, 16, axis=-1).astype(bool)
        node_embedding = self.node_embedding.copy()
        node_embedding[mask==1] = 0
        return {
            'visited': self.visited.copy().astype(np.int32),
            'current_embedding': self.node_embedding[self.current_index].copy().astype(np.float32),
            'node_embedding': node_embedding.flatten().astype(np.float32)   # (node_num*16, )
        }

class ATSP_logger_V3(Logger_COP):
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
            node_embedding = episode['prefix']['node_embedding'].reshape((-1, 16)) 
            file.write(f'city position:\n{position}\n\n')
            for t in range(len(rewards_AM)):           
                if visiteds:
                    file.write(f'visited:         \t{visiteds[t]}\n')
                file.write(f'current location:\t{current_poss[t]}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t].item()}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_ATSP_V3(DDP_TSP_V2):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_ATSP_V3'
        self.node_embedding = np.zeros((batch_size, self.num_nodes, 16), dtype=np.float32)  # (batch_size, node_num, 16)
        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'visited': spaces.MultiDiscrete([[2]*self.num_nodes for _ in range(batch_size)]),                       # (batch_size, node_num)
            'current_embedding': spaces.Box(low=0, high=1, shape=(batch_size, 16), dtype=np.float32),               # (batch_size, 2)
            'node_embedding': spaces.Box(low=0, high=1, shape=(batch_size, 16*self.num_nodes), dtype=np.float32),   # (batch_size, node_num*16)
        })

    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) <= self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch或最大问题数量更新问题
        assert isinstance(problem_list[0]['position'], np.ndarray)
        new_position = []
        new_node_embedding = []
        for problem in problem_list:
            position = problem['position'].reshape((-1, 2))
            node_embedding = problem['node_embedding'].reshape((-1, 16))
            new_position.append(position)
            new_node_embedding.append(node_embedding)
        new_position = np.array(new_position)
        new_node_embedding = np.array(new_node_embedding)
        self.position[problem_idx_list] = new_position      # (batch_size, node_num, 2)
        self.node_embedding[problem_idx_list] = new_node_embedding      # (batch_size, node_num, 16)
        self.visited[:,0] = True
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
    
        # TSP 问题的解从 idx=0 开始回到 idx=0，解格式中包含起点的 0，不含终点的 0
        return answer_list  
    
    def _get_observation(self):
        return {
            'visited': self.visited.copy().astype(np.int32),                                                            # (batch_size, node_num)
            'current_embedding': self.node_embedding[np.arange(self.batch_size), self.current_index].copy().astype(np.float32),# (batch_size, 16)
            'node_embedding': self.node_embedding.copy().reshape(self.batch_size, -1).astype(np.float32) # (batch_size, node_num*16)
        }
