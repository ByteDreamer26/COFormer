import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils_slover import TSP_lkh, calc_tsp_distance
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD
from environment.used.BaseEnv_COP import Env_COP, Logger_COP

class TSP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.num_nodes = scale
        self.name = 'Env_TSP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=1, shape=(2*self.num_nodes, ), dtype=np.float32),
            'visited': spaces.MultiDiscrete([2]*self.num_nodes),  
            'first_index': spaces.MultiDiscrete([self.num_nodes]),       # [0, 1,..., num_nodes-1]
            'current_index': spaces.MultiDiscrete([self.num_nodes]),     # [0, 1,..., num_nodes-1]
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(self.num_nodes)

        # 动作各维度对应的 token 取值范围
        # tsp 环境中动作 [0, num_nodes-1] 表示移动去对应的城市
        self.action_value_space = [list(range(0, self.num_nodes)),]      # 由于 action 都是自然数，token 取值范围和 action 相同 

        # 初始化状态
        self.position = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.visited = np.zeros(self.num_nodes, dtype=np.int32)
        self.first_index = 0                     
        self.current_index = 0  # [0, node_num-1] 代表各个城市
        self.real_answer = []
        self.model_answer = []
        self.problem_best_obj = self.problem_random_obj = None

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return self.visited.all()    # 全部结点访问后只能回到起点

    def _pred_qulity(self):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        rnd_obj = self.problem_random_obj
        model_obj = calc_tsp_distance(self.position, self.model_answer)
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else calc_tsp_distance(self.position, self.real_answer)

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj) if self.problem_best_obj is not None else 0
        #qulity_AM = 1 if qulity_AM < 1 and abs(qulity_AM-1) < 1e-4 else qulity_AM
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1:np.array, acts2:np.array):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            action_value_space = np.where(self.visited==0)[0]
        else:
            action_value_space = np.array(range(self.num_nodes))
        
        action_value_space = action_value_space.astype(np.int32)
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj):
        '''还原到评估问题的初始状态'''
        if problem_obj is None:
            self.problem_best_obj = self.problem_random_obj = None
        else:
            self.problem_best_obj, self.problem_random_obj = problem_obj
            
        prefix, problem, answer = problem_info
        assert isinstance(problem['position'], np.ndarray)    
        self.position = problem['position'].copy().reshape((self.num_nodes, 2)) 
        self.first_index = problem['first_index'].item()          
        self.current_index = problem['current_index'].item() 
        return answer

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
            return self._get_observation(), self._get_info()
        
        # 初始化为预生成的评估问题
        assert 'problem_info' in options
        self.real_answer = self._recover(problem_info=options['problem_info'], problem_obj=options['problem_obj'])

        # 初始化状态
        self.first_index = 0                     
        self.current_index = 0
        self.model_answer = [0,]
        self.visited = np.array([1]+[0]*(self.num_nodes-1), dtype=np.int32)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        terminated = truncated = False
        reward = {'AM': 0, 'DB1': 0}

        selected_city = int(action)
        if self.visited[selected_city] == 1:
            truncated = True        # 如果自动设置了可行 action 范围，这种情况不应该发生
            reward['AM'] = reward['DB1'] = COP_FAILED_RWD
        else:
            self.visited[selected_city] = 1
            self.current_index = selected_city
            self.model_answer.append(selected_city)

            if self._is_terminated():
                terminated = True
                reward = self._pred_qulity()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        return {
            'position': self.position.copy().flatten(),     # (2*num_nodes)
            'visited': self.visited.copy().astype(np.int32),
            'first_index': np.array([self.first_index,], dtype=np.int32),
            'current_index': np.array([self.current_index, ], dtype=np.int32)
        }

    def _get_info(self):
        return {'obj': calc_tsp_distance(self.position, self.model_answer)}
    
    def get_prefix_mask(self):
        return None

    def render(self):   
        pass

class TSP_logger(Logger_COP):
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
            obss = episode['observations']
            positions = obss['position']
            visiteds = obss['visited']
            first_idxs = obss['first_index']
            current_idxs = obss['current_index']   
            act_value_space = episode['act_value_space']         
            assert (first_idxs==0).all()

            file.write(f'city position:\n{positions[0].reshape((-1, 2))}\n\n')                
            for t in range(len(rewards_AM)):
                file.write(f'visited:    \t{visiteds[t]}\n')
                file.write(f'current idx:\t{current_idxs[t]}\n')
                file.write(f'action_space:\t{act_value_space[t][0]}\n')
                file.write(f'take action:\t{acts[t].item()}\n')
                file.write(f'get reward: \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_TSP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, rnd_obj:float=0, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.num_nodes = scale
        self.batch_size = batch_size
        self.rnd_obj = rnd_obj
        self.name = 'Env_TSP_V1'
        self.timestep = 0
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=1, shape=(batch_size, 2*self.num_nodes), dtype=np.float32),
            'first_index': spaces.MultiDiscrete([[self.num_nodes, ] for _ in range(batch_size)]),    # [0, 1,..., num_nodes-1]
            'current_index': spaces.MultiDiscrete([[self.num_nodes, ] for _ in range(batch_size)]),  # [0, 1,..., num_nodes-1]
            'visited': spaces.MultiDiscrete([[2]*self.num_nodes for _ in range(batch_size)]), 
        })

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.num_nodes] * batch_size)

        # 动作各维度对应的 token 取值范围
        # tsp 环境中动作 [0, num_nodes-1] 表示移动去对应的城市
        action_value_space = [np.arange(self.num_nodes, dtype=np.int32) for _ in range(batch_size)]     # 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space,]                                                 # 动作只有一个维度

        # 初始化状态
        self.position = np.zeros((batch_size, self.num_nodes, 2), dtype=np.float32)
        self.visited = np.zeros((batch_size, self.num_nodes), dtype=bool)
        self.first_index = np.zeros(batch_size, dtype=np.int32)
        self.current_index = np.zeros(batch_size, dtype=np.int32)                       # [0, node_num-1] 代表各个城市
        self.real_answer = np.zeros((batch_size, self.num_nodes), dtype=np.int32)       # TSP 问题的解从 idx=0 开始回到 idx=0，解格式中包含起点的 0，不含终点的 0
        self.model_answer = np.zeros((batch_size, self.num_nodes), dtype=np.int32)
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)

    def _get_observation(self):
        return {
            'position': self.position.copy().reshape(self.batch_size, self.num_nodes*2).astype(np.float32), # (batch_size, node_num*2)
            'visited': self.visited.copy().astype(np.int32),                                                # (batch_size, node_num, )
            'first_index': self.first_index.astype(np.int32),                                               # (batch_size, )
            'current_index': self.current_index.astype(np.int32),                                           # (batch_size, )
        }

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return self.visited.all(axis=1)     # (batch_size, ) 全部结点访问后只能回到起点

    def _pred_qulity(self):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        rnd_obj = self.problem_random_obj
        model_obj = calc_tsp_distance(self.position, self.model_answer)    # (problem_batch_size_per_gpu, )
        best_obj = self.problem_best_obj
        #best_obj = calc_tsp_distance(self.position, self.real_answer)      # (problem_batch_size_per_gpu, )
        #assert np.array_equal(best_obj, self.problem_best_obj)

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1:np.array, acts2:np.array):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        assert hard_action_constraint == True
        if hard_action_constraint:
            idx, city = np.where(self.visited==0)
            action_value_space = [city[idx==i].astype(np.int32) for i in range(self.batch_size)]
        else:
            action_value_space = [np.arange(self.num_nodes, dtype=np.int32) for _ in range(self.batch_size)]
        
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space

    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) <= self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch更新问题
        assert isinstance(problem_list[0]['position'], np.ndarray)
        new_position = [problem['position'] for problem in problem_list]
        new_position = np.array(new_position)
        self.position[problem_idx_list] = new_position          # (batch_size, node_num, 2)
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
        
        # TSP 问题的解从 idx=0 开始回到 idx=0，解格式中包含起点的 0，不含终点的 0
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

        # 初始化状态 (TSP一定是成批更新，以下flag可以批量处理)
        self.first_index = np.zeros(self.batch_size, dtype=np.int32)
        self.current_index = np.zeros(self.batch_size, dtype=np.int32)          # [0, node_num-1] 代表各个城市
        self.real_answer = np.vstack(problem_real_answer).astype(np.int32)      # (batch_size, node_num), TSP 问题的解从 idx=0 开始回到 idx=0，解格式中包含起点的 0，不含终点的 0
        self.model_answer = np.zeros((self.batch_size, self.num_nodes), dtype=np.int32)
        self.visited = np.zeros((self.batch_size, self.num_nodes), dtype=bool)
        self.visited[:,0] = True    # 起止城市固定为第一个，reset时即标记0索引城市标记为已经访问过
        self.timestep = 0           # 起止城市固定为第一个，reset时看作0时刻

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.timestep += 1
        assert 1 <= self.timestep < self.num_nodes
        terminated = np.zeros(self.batch_size, dtype=bool)
        truncated = np.zeros(self.batch_size, dtype=bool)
        reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }
        selected_node = action.astype(np.int32)

        # 考察去向的站点是否已经访问过，如果自动设置了可行 action 范围，这种情况不应该发生
        # 由于该环境要求 hard_action_constraint==True，故总有truncated==False
        problem_idx = np.arange(self.batch_size)
        selected_visited = self.visited[problem_idx, selected_node]
        selected_visited_idx = np.where(selected_visited)[0]
        assert selected_visited_idx.size == 0   
        #truncated[selected_visited_idx] = True
        #reward[selected_visited_idx] = COP_FAILED_RWD

        # 转移到新城市并配送货
        self.current_index = selected_node
        self.visited[problem_idx, selected_node] = True

        # 更新模型解
        self.model_answer[:, self.timestep] = action    # (batch_size, timestep)

        # 考察正常结束的情况
        terminated = self._is_terminated()              # (batch_size, )

        # 所有问题同步结束，计算质量
        if terminated.sum() != 0:
            assert terminated.sum() == self.batch_size
            reward = self._pred_qulity()                # (batch_size, )
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {'obj': calc_tsp_distance(self.position, self.model_answer)}
        
    def get_prefix_mask(self):
        return None
    
    def render(self):   
        pass
