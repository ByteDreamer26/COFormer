import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

import gym
import numpy as np
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Union
from utils.utils import create_file_if_not_exist
from utils.utils_data import DataProblem, RawData

class Env_COP(gym.Env, metaclass=ABCMeta):
    metadata = {
        "render_modes": ["rgb_array",],     # 支持的渲染模式
        "render_fps": 500,                  # 渲染帧率
    }                 

    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        self.render_mode = render_mode

        self.scale = scale                      # 问题规模
        self.adj_range = adj_range              # 调整数值范围关于 0 点对称
        self.observation_space = None           # 观测空间
        self.action_space = None                # 动作空间
        self.action_value_space = []            # 动作各维度对应的 token 取值范围

        self.real_answer = []                   # 经典求解器给出的解
        self.model_answer = []                  # 被测模型给出的解

        self.rng = np.random.RandomState(None)  # 随机数生成器

    @abstractmethod
    def _is_terminated(self) -> bool:
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        obs = self._get_observation()   # 利用当前观测判断是否正常结束
        return False

    @abstractmethod
    def _pred_qulity(self) -> float:
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        assert len(self.real_answer) == len(self.model_answer)
        model_value = 0.5
        real_value = 1
        qulity = 1 - (real_value-model_value)/real_value
        return qulity

    @abstractmethod
    def _recover(self, problem_info:tuple) -> List[int]:
        '''还原到评估问题的初始状态'''
        prefix, problem, real_answer = problem_info
        return real_answer

    @abstractmethod
    def is_same_episode(self, acts1:np.array, acts2:np.array) -> bool:
        ''' 判断两条轨迹是否完全相同, 由于是确定性环境, 仅比较两条act序列即可 '''
        return False

    @abstractmethod
    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None) -> List:
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space 
            注意action_value_space是一个列表的列表, 其中每个元素代表action一个维度的取值范围
        '''
        self.action_value_space = []
        return self.action_value_space

    @abstractmethod
    def step(self, action:Union[np.ndarray, int, float]) -> tuple:
        ''' 环境处理环境转移逻辑, 要生成 terminated、truncated、reward 信号 
            注意若出现 truncated==True, reward 设置为 COP_FAILED_RWD 返回
        '''
        terminated = truncated = False
        reward = {'AM':0, 'DB1':0}
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    @abstractmethod
    def reset(self, seed:int=None, problem_info:DataProblem=None, options=None) -> tuple:
        ''' 重置环境, 若提供了 problem_info 则调用 _recover 设置环境准备求解目标问题; 否则调用 _gen_quesition 生成随机问题
            注意 seed 值仅在初始化环境时提供
        '''
        super().reset(seed=seed)
        return self._get_observation(), self._get_info()

    @abstractmethod
    def _get_observation(self) -> dict:
        return {}

    @abstractmethod
    def _get_info(self) -> dict:
        return {}

    def get_prefix(self) -> dict:
        return {}
    
    def get_prefix_mask(self) -> dict:
        return None

class Logger_COP(ABC):
    def __init__(self, env_name='example', dataset_name='example'):
        self.env_name = env_name
        self.dataset_name = dataset_name
        self.local_rank = os.getenv('LOCAL_RANK')

    def log_data(self, logged_data, seed=0, is_train=True) -> None:
        ''' 记录train/eval过程中样本在数据集中的索引 '''
        data_type = 'train' if is_train else 'eval'
        log_floder_path = f'{base_path}/visualize/{data_type}/{self.env_name}/{self.dataset_name}/seed-{seed}' if is_train else \
                        f'{base_path}/visualize/{data_type}/log/{self.env_name}/{self.dataset_name}/seed-{seed}'
        log_path = f'{log_floder_path}/[GPU{self.local_rank}] {data_type}_data.txt' if self.local_rank is not None else \
                     f'{log_floder_path}/{data_type}_data.txt'
        
        # 初次 log 时创建 log 文件
        create_file_if_not_exist(log_path)
        
        # 追加 log 信息
        with open(log_path, 'a') as file:
            file.write(str(logged_data[self.dataset_name]))
            file.write('\n')

    @abstractmethod
    def log_episode(self, desc='example', is_eval=False, episode=None, epoch_num=0, episode_num=0, seed=0) -> None:
        ''' 记录策略评估过程中的 rollout 结果 '''
        phase = 'eval/log' if is_eval else 'train' 
        log_floder_path = f'{base_path}/visualize/{phase}/{self.env_name}/{self.dataset_name}/seed-{seed}'
        log_path = f'{log_floder_path}/[GPU{self.local_rank}] {desc}.txt' if self.local_rank is not None else \
                     f'{log_floder_path}/{desc}.txt'
        
        # 初次 log 时创建 log 文件
        create_file_if_not_exist(log_path)

        # 追加 log 信息，将 rollout 轨迹 episode 写入 log 文件中
        # ...