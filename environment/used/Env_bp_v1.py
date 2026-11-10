import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD
from environment.used.BaseEnv_COP import Env_COP, Logger_COP
from utils.utils_slover import calc_bp_total
from data.raw.bp import MAX_CAPACITY

class BP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale=20, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.name = 'Env_BP_V1'
        self.item_num = scale

        # 观测空间、动作空间、可行动作取值范围
        self.observation_space = spaces.Dict({
            'capacity_left': spaces.Box(low=0, high=MAX_CAPACITY[self.item_num], shape=(1,), dtype=np.int32),
            'visited': spaces.MultiDiscrete([2]*self.item_num), 
        })
        self.action_space = spaces.Discrete(self.item_num, start=0)     # [0, 1,...,item_num-1]
        self.action_value_space = [list(range(0, self.item_num)),]      # token 取值范围和 action 相同 

        # 初始化状态
        self.capacity_left = np.zeros(1, dtype=np.int32)
        self.item_volumes = np.zeros(self.item_num, dtype=np.int32)
        self.item_values = np.zeros(self.item_num, dtype=np.int32)
        self.visited = np.zeros(self.item_num, dtype=np.int32)
        self.real_answer = []
        self.model_answer = []
        self.problem_best_obj = self.problem_random_obj = None

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        action_value_space = self.get_action_value_space(hard_action_constraint=True)[0]
        return action_value_space.size == 0 # 无可选 item 则正常结束轨迹

    def _pred_qulity(self):
        rnd_obj = self.problem_random_obj
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else self.item_values[self.real_answer].sum()
        model_obj = self.item_values[self.model_answer].sum()
        qulity_AM = 1 - abs((model_obj - best_obj)/best_obj)
        if best_obj == rnd_obj:
            qulity_DB1 = 1 if model_obj >= best_obj else 0
        else:
            qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1, acts2):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        # 对两个ndarray进行元素去重，并排序
        unique_arr1, counts_arr1 = np.unique(acts1, return_counts=True)
        unique_arr2, counts_arr2 = np.unique(acts2, return_counts=True)
        
        # 去重后的 ndarray 相同且相同元素个数相同
        if np.array_equal(unique_arr1, unique_arr2) and np.array_equal(counts_arr1, counts_arr2):   
            return True
        return False

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            value_space = np.where((self.visited==0) & (self.item_volumes <= self.capacity_left))[0]
            self.action_value_space = [value_space.astype(np.int32), ]
        else:
            self.action_value_space = [np.arange(self.item_num, dtype=np.int32), ]
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj):
        '''还原到评估问题的初始状态'''
        if problem_obj is None:
            self.problem_best_obj = self.problem_random_obj = None
        else:
            self.problem_best_obj, self.problem_random_obj = problem_obj

        prefix, problem, answer = problem_info
        assert isinstance(problem['capacity_left'], np.ndarray)
        self.capacity_left = problem['capacity_left'].copy()
        self.item_volumes = problem['item_volumes'].copy()
        self.item_values = problem['item_values'].copy()
        
        #max_value, selection = knapsack_dc(self.capacity_left, self.item_volumes, self.item_values)
        return answer

    def reset(self, seed=None, options:dict=None):
        super().reset(seed=seed)
        if seed is not None:
            return self._get_observation(), self._get_info()
        
        # 初始化为预生成的评估问题并求解
        real_answer = self._recover(problem_info=options['problem_info'], problem_obj=options['problem_obj'])
        
        # 初始化状态
        self.visited = np.zeros(self.item_num, dtype=np.int32)
        self.model_answer = []
        self.real_answer = real_answer

        return self._get_observation(), self._get_info()

    def step(self, action):
        terminated = truncated = False
        reward = {'AM': 0, 'DB1': 0}

        selected_item = int(action)
        if self.visited[selected_item] == 1:
            truncated = True                                    # 如果自动设置了可行 action 范围，这种情况不应该发生
            reward['AM'] = reward['DB1'] = COP_FAILED_RWD
        else:
            self.capacity_left -= self.item_volumes[selected_item]
            if self.capacity_left >= 0:
                # 有剩余空间足够装入新物品
                self.model_answer.append(selected_item)                
                self.visited[selected_item] = 1

                # 由环境判断是否终止
                if self._is_terminated():
                    terminated = True
                    reward = self._pred_qulity()
            else:
                # 剩余空间无法装入新物品
                truncated = True
                reward['AM'] = reward['DB1'] = COP_FAILED_RWD   # 如果自动设置了可行 action 范围，这种情况不应该发生

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {'obj': calc_bp_total(self.item_values, self.model_answer)}

    def _get_observation(self):
        return {
            'capacity_left': self.capacity_left.astype(np.int32),           # (1,)
            'visited': self.visited.copy().astype(np.int32),                # (item_num, )
        }

    def get_prefix(self):
        return {
            'item_volumes': self.item_volumes.copy().astype(np.int32),      # (item_num, )
            'item_values': self.item_values.copy().astype(np.int32),        # (item_num, )
        }
    
    def get_prefix_mask(self):
        return {
            'item_volumes': self.visited.copy().astype(bool),               # (item_num, )
            'item_values': self.visited.copy().astype(bool),                # (item_num, )
        }

class BP_logger_V1(Logger_COP):
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

class DDP_BP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.item_num = scale
        self.batch_size = batch_size
        self.name = 'Env_BP_V1'

        # 观测空间、动作空间、动作取值范围
        self.observation_space = spaces.Dict({
            'capacity_left': spaces.Box(low=0, high=MAX_CAPACITY[self.item_num], shape=(batch_size,), dtype=np.int32),
            'visited': spaces.MultiDiscrete([[2]*self.item_num for _ in range(batch_size)]),
        })
        self.action_space = spaces.MultiDiscrete([self.item_num] * batch_size)
        action_value_space = [np.arange(self.item_num, dtype=np.int32) for _ in range(batch_size)]
        self.action_value_space = [action_value_space,]         

        # 初始化状态
        self.capacity_left = np.zeros(batch_size, dtype=np.int32)
        self.item_volumes = np.zeros((batch_size, self.item_num), dtype=np.int32)
        self.item_values = np.zeros((batch_size, self.item_num), dtype=np.int32)
        self.visited = np.zeros((batch_size, self.item_num), dtype=np.int32)
        self.real_answer = [[] for _ in range(batch_size)]              # 相同规模的 bp 问题解向量长度可能不同，不宜使用np.ndarray形式存储
        self.model_answer = [[] for _ in range(batch_size)]
        
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        action_value_spaces = self.get_action_value_space(hard_action_constraint=True)[0]
        terminated = np.array([len(space) == 0 for space in action_value_spaces], dtype=bool)   # 无可选 item 则正常结束轨迹
        return terminated

    def _pred_qulity(self, item_values, model_answer, problem_best_obj, problem_random_obj):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        rnd_obj = problem_random_obj
        best_obj = problem_best_obj
        if best_obj == 0:
            print()
        model_obj = item_values[model_answer].sum()
        qulity_AM = 1 - abs((model_obj - best_obj)/best_obj)
        if best_obj == rnd_obj:
            qulity_DB1 = 1 if model_obj >= best_obj else 0
        else:
            qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)            
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1:np.array, acts2:np.array):
        pass

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:            
            action_value_space = []
            for i in range(self.batch_size):
                value_space = np.where((self.visited[i]==0) & (self.item_volumes[i] <= self.capacity_left[i]))[0]
                action_value_space.append(value_space.astype(np.int32))
        else:
            action_value_space = [np.arange(self.item_num, dtype=np.int32) for _ in range(self.batch_size)]
        
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert isinstance(problem_list[0]['capacity_left'], np.ndarray)
        new_item_values, new_item_volumes, new_capacity = [], [], []
        for problem in problem_list:
            new_capacity.append(problem['capacity_left'])
            new_item_values.append(problem['item_values'])
            new_item_volumes.append(problem['item_volumes'])
        new_capacity = np.array(new_capacity)
        new_item_values = np.vstack(new_item_values)
        new_item_volumes = np.vstack(new_item_volumes)

        problem_idx_list = problem_idx_list[:len(answer_list)]  # 最后可能会出现剩余数据量不足已完成数据量的情况
        self.capacity_left[problem_idx_list] = new_capacity     # (batch_size, )
        self.item_volumes[problem_idx_list] = new_item_volumes  # (batch_size, item_num)
        self.item_values[problem_idx_list] = new_item_values    # (batch_size, item_num)
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
        return answer_list

    def reset(self, seed=None, options=None):
        # 现在不使用并行环境生成问题，不应提供 seed 参数，保留此分支以向前兼容
        super().reset(seed=seed)
        if seed is not None:
            return self._get_observation(), self._get_info()
        
        # 初始化为预生成的评估问题并求解
        problem_info = options['problem_info']
        problem_idx = options['problem_idx']
        problem_obj = None if 'problem_obj' not in options else options['problem_obj']
        problem_real_answer = self._recover(problem_info, problem_obj, problem_idx)

        # 初始化状态
        self.visited[problem_idx] = np.zeros(self.item_num, dtype=np.int32)     # (batch_size, node_num)
        for idx, answer in zip(problem_idx, problem_real_answer):
            self.real_answer[idx] = answer
            self.model_answer[idx] = []
            
        return self._get_observation(), self._get_info()

    def step(self, action:np.ndarray):
        terminated = np.zeros(self.batch_size, dtype=bool)
        truncated = np.zeros(self.batch_size, dtype=bool)
        reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }
        problem_idx = np.arange(self.batch_size)
        selected_item = action.astype(np.int32)

        # 拿取物品减少剩余容量
        self.capacity_left -= self.item_volumes[problem_idx, selected_item]

        # 考察拿取的物品是否已经拿取过，如果自动设置了可行 action 范围，这种情况不应该发生
        for i in range(self.batch_size):
            if self.visited[i][selected_item[i]] == 1:
                truncated[i] = True        # 如果自动设置了可行 action 范围，这种情况不应该发生
                reward['AM'][i] = reward['DB1'][i] = COP_FAILED_RWD

        # 考察正常结束情况
        self.visited[problem_idx, selected_item] = 1
        terminated = self._is_terminated()
        for i in range(self.batch_size):
            if not truncated[i]:
                if selected_item[i] in self.model_answer[i]:    # 重复放置，这时 DDP_wrapper 中 onging_problem_idx 会自动过滤掉这些问题
                    reward['AM'][i] = reward['DB1'][i] = COP_FAILED_RWD
                else:
                    self.model_answer[i].append(selected_item[i])
                    if terminated[i]: 
                        qulity = self._pred_qulity(self.item_values[i], self.model_answer[i], self.problem_best_obj[i], self.problem_random_obj[i])                
                        reward['AM'][i] = qulity['AM']
                        reward['DB1'][i] = qulity['DB1']

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        return {
            'capacity_left': self.capacity_left.copy().astype(np.int32),    # (batch_size, )
            'visited': self.visited.copy().astype(np.int32)                 # (batch_size, item_num)
        }

    def get_prefix(self):
        return {
            'item_volumes': self.item_volumes.copy().astype(np.int32),      # (batch_size, item_num)
            'item_values': self.item_values.copy().astype(np.int32),        # (batch_size, item_num)
        }
    
    def get_prefix_mask(self):
        return {
            'item_volumes': self.visited.copy().astype(bool),               # (batch_size, item_num)
            'item_values': self.visited.copy().astype(bool),                # (batch_size, item_num)
        }

    def _get_info(self):
        return {'obj': np.array([calc_bp_total(self.item_values[i], answer) for i, answer in enumerate(self.model_answer)])}
        
    def render(self):   
        pass
