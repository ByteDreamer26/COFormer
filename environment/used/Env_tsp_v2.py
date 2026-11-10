import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist
from environment.used.BaseEnv_COP import Logger_COP
from environment.used.Env_tsp_v1 import TSP_V1, DDP_TSP_V1

class TSP_V2(TSP_V1):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_TSP_V2'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=1, shape=(2*self.num_nodes, ), dtype=np.float32),
            'visited': spaces.MultiDiscrete([2]*self.num_nodes), 
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
        })
    
    def _recover(self, problem_info, problem_obj):
        '''还原到评估问题的初始状态'''
        if problem_obj is None:
            self.problem_best_obj = self.problem_random_obj = None
        else:
            self.problem_best_obj, self.problem_random_obj = problem_obj
        
        prefix, problem, answer = problem_info
        assert isinstance(problem['position'], np.ndarray)
        self.position = problem['position'].copy().reshape((self.num_nodes, 2)) 
        
        #distance, real_answer = TSP_lkh(self.position)
        return answer
    
    def _get_observation(self):
        position = self.position.copy()
        position[self.visited==1] = 0

        return {
            'position': position.flatten().astype(np.float32),     # (2*num_nodes)
            'visited': self.visited.copy().astype(np.int32),
            'current_position': self.position[self.current_index].copy().astype(np.float32)
        }

class TSP_logger_V2(Logger_COP):
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
            positions = obss['position']
            visiteds = obss['visited']
            current_poss = obss['current_position']   
            act_value_space = episode['act_value_space']         

            for t in range(len(rewards_AM)):
                citys = positions[t].reshape((-1, 2))
                location = current_poss[t]
                file.write(f'city position:\n{citys}\n')
                file.write(f'current location:  \t{location}\n')
                file.write(f'visited:           \t{visiteds[t]}\n')
                file.write(f'action_space:      \t{act_value_space[t][0]}\n')
                file.write(f'take action:       \t{acts[t].item()}\n')
                file.write(f'get reward:        \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_TSP_V2(DDP_TSP_V1):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_TSP_V2'
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=1, shape=(batch_size, 2*self.num_nodes), dtype=np.float32),  # (batch_size, node_num*2)
            'visited': spaces.MultiDiscrete([[2]*self.num_nodes for _ in range(batch_size)]),               # (batch_size, node_num)
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),         # (batch_size, 2)
        })
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) <= self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch或最大问题数量更新问题
        assert isinstance(problem_list[0]['position'], np.ndarray)
        new_position = []
        for problem in problem_list:
            position = problem['position'].reshape((-1, 2))
            new_position.append(position)
        new_position = np.array(new_position)
        self.position[problem_idx_list] = new_position      # (batch_size, node_num, 2)
        self.visited[:,0] = True
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
    
        # TSP 问题的解从 idx=0 开始回到 idx=0，解格式中包含起点的 0，不含终点的 0
        return answer_list  

    def _get_observation(self):
        position = self.position.copy() # （batch_size, node_num, 2）
        position[self.visited] = 0

        return {
            'position': position.reshape(self.batch_size, self.num_nodes*2).astype(np.float32),                         # (batch_size, node_num*2)
            'visited': self.visited.copy().astype(np.int32),                                                            # (batch_size, node_num)
            'current_position': self.position[np.arange(self.batch_size), self.current_index].copy().astype(np.float32) # (batch_size, 2)
        }