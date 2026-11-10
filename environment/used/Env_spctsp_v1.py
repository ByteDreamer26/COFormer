import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD
from utils.utils_slover import calc_pctsp_total, calc_pctsp_cost
from environment.used.BaseEnv_COP import Env_COP, Logger_COP
from data.raw.spctsp import MAX_LENGTHS
from environment.used.Env_spctsp_v2 import SPCTSP_V2, DDP_SPCTSP_V2

class SPCTSP_V1(SPCTSP_V2):
    ### stoc_prize 是真实奖励real_prize事先不确定，det_prize 是预期奖励expect_prize事先确定
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode, scale, adj_range)
        self.name = 'Env_SPCTSP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(2*self.node_num,), dtype=np.float32),
            'visited': spaces.MultiDiscrete([2]*self.node_num), 
            'det_prize': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'penalty': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, dtype=np.float32),
        })
    
    def _get_observation(self):
        pos_node = self.pos_node.copy()
        visited = self.visited.copy()
        current_position = self.pos[self.current_index].copy()
        stoc_prize = self.stoc_prize.copy()
        det_prize = self.det_prize.copy()
        penalty = self.penalty.copy()
        stoc_prize[self.visited==1] = 0
        det_prize[self.visited==1] = 0
        penalty[self.visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),          # (2, )
            'pos_node': pos_node.flatten().astype(np.float32),              # (node_num*2, )
            'visited': visited.astype(np.int32),                            # (node_num, )
            'det_prize': det_prize.astype(np.float32),                      # (node_num, )
            'penalty': penalty.astype(np.float32),                          # (node_num, )
            'prize2go': np.array([self.prize2go,]).astype(np.float32),      # (1, )
            'current_position': current_position.copy().astype(np.float32), # (2, )
        }
        return obs

class DDP_SPCTSP_V1(DDP_SPCTSP_V2):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode, scale, batch_size, adj_range)
        self.name = 'Env_SPCTSP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(batch_size, 2*self.node_num,), dtype=np.float32),
            'visited': spaces.MultiDiscrete([[2]*self.node_num for _ in range(batch_size)]),               # (batch_size, node_num)
            'det_prize': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num), dtype=np.float32),
            'penalty': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, shape=(batch_size, ), dtype=np.float32),
        })

    def _get_observation(self):
        pos_node = self.pos_node.copy()                                     # (batch_size, node_num, 2)
        visited = self.visited.copy()                                       # (batch_size, node_num, )
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy()
        stoc_prize = self.stoc_prize.copy()
        det_prize = self.det_prize.copy()
        penalty = self.penalty.copy()
        stoc_prize[self.visited==1] = 0
        det_prize[self.visited==1] = 0
        penalty[self.visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),                              # (batch_size, 2)
            'pos_node': pos_node.reshape(self.batch_size, self.node_num*2).astype(np.float32),  # (batch_size, 2*node_num)
            'visited': visited.astype(np.int32),                                                # (batch_size, node_num)
            'det_prize': det_prize.astype(np.float32),                                          # (batch_size, node_num)
            'penalty': penalty.astype(np.float32),                                              # (batch_size, node_num)
            'prize2go': self.prize2go.copy().astype(np.float32),                                # (batch_size, )
            'current_position': current_position.copy().astype(np.float32),                     # (batch_size, 2)
        }
        return obs

    def _get_info(self):
        return {'obj': np.array([calc_pctsp_cost(self.pos[i], self.penalty[i], self.stoc_prize[i], answer[:-1]) for i, answer in enumerate(self.model_answer)])}