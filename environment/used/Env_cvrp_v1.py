import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils_slover import CVRP_lkh, calc_vrp_distance
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD
from environment.used.BaseEnv_COP import Env_COP, Logger_COP
from data.raw.cvrp import CAPACITIES

class CVRP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.node_num = scale
        self.name = 'Env_CVRP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(2*self.node_num,), dtype=np.float32),
            'demand': spaces.MultiDiscrete([10]*self.node_num),         # 应该是 [1,9], 但是 MultiDiscrete 只能从 0 开始，此处设置为 [0,9]
            'capacity': spaces.MultiDiscrete([50,]),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'visited': spaces.MultiDiscrete([2]*self.node_num),  
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(self.node_num+1)

        # 动作各维度对应的 token 取值范围
        # cvrp 环境中动作 0 表示移动去仓库；动作 [1, node_num] 表示移动去 [0, node_num-1] 站点
        self.action_value_space = [list(range(0, self.node_num+1)),]    # 由于 action 都是自然数，token 取值范围和 action 相同 

        # 初始化状态
        self.pos_depot = np.random.uniform(0, 1, (2, )).astype(np.float32)
        self.pos_node = np.random.uniform(0, 1, (self.node_num, 2)).astype(np.float32)
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node))   # (node_num+1, 2)
        self.demand = np.ones(self.node_num, dtype=np.int32)
        self.capacity_left = CAPACITIES[self.node_num]
        
        self.current_index = 0  # 0 代表 depot, [1, node_num] 代表各个站点
        self.visited = np.ones(self.node_num, dtype=np.int32)
        self.real_answer = []
        self.model_answer = []
        self.problem_best_obj = 0
        self.problem_random_obj = 0

    def _get_observation(self):
        pos_node = self.pos_node.copy()
        demand = self.demand.copy()
        pos_node[self.visited==1] = 0
        demand[self.visited==1] = 0

        current_position = self.pos_depot.copy() if self.current_index == 0 else \
                            self.pos_node[self.current_index-1].copy()
        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),
            'pos_node': pos_node.flatten().astype(np.float32),
            'demand': demand.astype(np.int32),
            #'capacity': int(self.capacity_left),
            'capacity': np.array([self.capacity_left,], dtype=np.int32),
            'current_position': current_position.copy().astype(np.float32),
            'visited': self.visited.copy().astype(np.int32)
        }
        return obs

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return self.visited.all()    # 全部结点访问后只能回到起点

    def _pred_qulity(self):
        rnd_obj = self.problem_random_obj
        model_obj = calc_vrp_distance(self.pos, self.model_answer)
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else calc_vrp_distance(self.pos, self.real_answer)
        #assert abs(best_obj - self.problem_best_obj) < 1e-4
        assert best_obj <= CAPACITIES[self.node_num]
        assert model_obj <= CAPACITIES[self.node_num]

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj-best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1, acts2):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            obs = self._get_observation()
            demand = self.demand
            capacity_left = obs['capacity']
            visited = self.visited
            unvisited = np.where(visited==0)[0]

            # 如果节点已经访问过或需求超出现有容量，禁止访问
            action_value_space = np.array(
                [idx for idx in unvisited if capacity_left >= demand[idx]]
            )
            action_value_space += 1

            # 如果当前已经在depot，禁止连续访问
            if self.current_index != 0:
                action_value_space = np.insert(action_value_space, 0, 0)
        else:
            action_value_space = np.array(range(self.node_num + 1))

        action_value_space = action_value_space.astype(np.int32)
        assert action_value_space.size > 0
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj):
        '''还原到评估问题的初始状态'''
        if problem_obj is None:
            self.problem_best_obj = self.problem_random_obj = None
        else:
            self.problem_best_obj, self.problem_random_obj = problem_obj

        prefix, problem, answer = problem_info
        assert isinstance(problem['pos_node'], np.ndarray)
        self.pos_depot = problem['pos_depot'].copy().reshape((2,))
        self.pos_node = problem['pos_node'].copy().reshape((self.node_num, 2))
        self.demand = problem['demand'].copy()                  
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node)) 
        return answer

    def reset(self, seed=None, options:dict=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
            return self._get_observation(), self._get_info()
        
        # 初始化为预生成的评估问题并求解
        assert 'problem_info' in options
        self.real_answer = self._recover(problem_info=options['problem_info'], problem_obj=options['problem_obj'])
        self.capacity_left = CAPACITIES[self.node_num]
        self.current_index = 0  # 0 代表 depot
        self.visited = np.zeros(self.node_num, dtype=np.int32)
        self.model_answer = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        terminated = truncated = False
        reward = {'AM':0, 'DB1':0}
        selected_node = int(action)

        self.current_index = selected_node
        if selected_node > 0:
            self.model_answer.append(selected_node)
            if self.visited[selected_node-1] == 1:
                truncated = True            # 如果自动设置了可行 action 范围，这种情况不应该发生
                reward['AM'] = reward['DB1'] = COP_FAILED_RWD
            else:
                self.visited[selected_node-1] = 1                
                self.capacity_left -= self.demand[selected_node-1]

                if self.capacity_left < 0:
                    truncated = True        # 如果自动设置了可行 action 范围，这种情况不应该发生
                    reward['AM'] = reward['DB1'] = COP_FAILED_RWD

                if self._is_terminated():
                    terminated = True
                    reward = self._pred_qulity()
        else:
            self.model_answer.append(selected_node)
            self.capacity_left = CAPACITIES[self.node_num]

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {'obj': calc_vrp_distance(self.pos, self.model_answer)}

    def render(self):   
        pass

class CVRP_logger_V1(Logger_COP):
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
                node = obss['pos_node'][t].reshape((-1, 2))
                demand = obss['demand'][t]
                node_info = np.hstack((node, demand[:,None]))
                current_location = obss['current_position'][t]
                visited = obss['visited'][t]
                capacity_left = obss['capacity'][t]

                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'capacity left:   \t{capacity_left}\n')
                file.write(f'visited:         \t{visited}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t].item()} (node{acts[t].item()-1})\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_CVRP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.node_num = scale
        self.batch_size = batch_size
        self.name = 'Env_CVRP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(batch_size, 2*self.node_num), dtype=np.float32),
            'demand': spaces.MultiDiscrete([[10]*self.node_num for _ in range(batch_size)]), # 应该是 [1,9], 但是 MultiDiscrete 只能从 0 开始，此处设置为 [0,9]
            'capacity': spaces.MultiDiscrete([[CAPACITIES[self.node_num]] for _ in range(batch_size)]),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'visited': spaces.MultiDiscrete([[2]*self.node_num for _ in range(batch_size)]),  
        })

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.node_num + 1] * batch_size)

        # 动作各维度对应的 token 取值范围
        # cvrp 环境中动作 0 表示移动去仓库；动作 [1, node_num] 表示移动去 [0, node_num-1] 站点
        #self.action_value_space = [list(range(0, node_num+1)),]    
        action_value_space = [np.arange(self.node_num+1, dtype=np.int32) for _ in range(batch_size)]# 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space, ]                                            # 动作只有一个维度

        # 初始化状态 
        self.pos_depot = np.random.uniform(0, 1, (batch_size, 2)).astype(np.float32)                # (batch_size, 2)
        self.pos_node = np.random.uniform(0, 1, (batch_size, self.node_num, 2)).astype(np.float32)  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)              # (batch_size, 1 + node_num, 2)
        self.demand = np.ones((batch_size, self.node_num), dtype=np.int32)                          # (batch_size, node_num)
        self.capacity_left = np.array([CAPACITIES[self.node_num]] * batch_size)                     # (batch_size, )
        self.current_index = np.zeros(batch_size, dtype=np.int32)                                   # (batch_size, ) 0 代表 depot, [1, node_num] 代表各个站点
        self.visited = np.zeros((batch_size, self.node_num), dtype=np.int32)                        # (batch_size, node_num)
        self.real_answer = [[] for _ in range(batch_size)]  # 相同规模的cvrp问题，解向量长度可能不同，不宜使用np.ndarray形式存储
        self.model_answer = [[] for _ in range(batch_size)]
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)

    def _get_observation(self):
        current_position = self.pos[np.arange(self.batch_size), self.current_index, :].copy()   # (batch_size, 2)  
        pos_node = self.pos_node.copy()
        demand = self.demand.copy()
        capacity_left = self.capacity_left.copy()
        pos_node[self.visited==1] = 0
        demand[self.visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),                              # (batch_size, 2)
            'pos_node': pos_node.reshape(self.batch_size, self.node_num*2).astype(np.float32),  # (batch_size, node_num * 2)
            'demand': demand.astype(np.int32),                                                  # (batch_size, node_num)
            'capacity': capacity_left.astype(np.int32),                                         # (batch_size, )
            'current_position': current_position.astype(np.float32),                            # (batch_size, 2)
            'visited': self.visited.copy().astype(np.int32)                                     # (batch_size, node_num)
        }
        return obs

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return self.visited.all(axis=1)     # (batch_size, ) 全部结点访问后只能回到起点

    def _pred_qulity(self, pos, real_answer, model_answer, problem_best_obj, problem_random_obj):
        '''  
        pos:  (node_num+1, 2)
        real_answer: List
        model_answer: List
        '''
        rnd_obj = problem_random_obj
        model_obj = calc_vrp_distance(pos, model_answer)
        best_obj = problem_best_obj
        #best_obj = calc_vrp_distance(pos, real_answer)
        #assert abs(best_obj - problem_best_obj) < 1e-4

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj - best_obj) / best_obj if best_obj > 0 else -1e3
        qulity_DB1 = (model_obj - rnd_obj) / (best_obj - rnd_obj) if best_obj - rnd_obj != 0 else -1e3
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1, acts2):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            obs = self._get_observation()
            demand = self.demand                            # (batch_size, node_num)
            capacity_left = obs['capacity']                 # (batch_size, )
            visited = self.visited                          # (batch_size, node_num)

            capacity = capacity_left[:, np.newaxis]         # (batch_size, 1)
            viable = (visited == 0) & (demand <= capacity)  # (batch_size, node_num)
            
            problem_action_value_space = []
            problem_idxs, viable_node_idxs = np.where(viable)
            for i in range(self.batch_size):
                # 如果当前不在depot，总是允许访问depot
                value_space = [0, ] if self.current_index[i] != 0 else []
                # 未访问过且需求 <= 现有容量的节点，允许访问
                value_space.extend(viable_node_idxs[problem_idxs==i]+1)
                
                assert len(value_space) >= 0
                problem_action_value_space.append(np.array(value_space, dtype=np.int32))
        else:
            problem_action_value_space = [np.arange(self.node_num+1, dtype=np.int32) for _ in range(self.batch_size)]

        self.action_value_space = [problem_action_value_space, ]    # 动作只有一个维度
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert isinstance(problem_list[0]['pos_node'], np.ndarray)
        new_pos_depot, new_pos_node, new_demand = [], [], []
        for problem in problem_list:
            new_pos_depot.append(problem['pos_depot'])
            new_pos_node.append(problem['pos_node'].reshape((1, self.node_num, 2)))
            new_demand.append(problem['demand'])
        new_demand = np.vstack(new_demand)
        new_pos_depot = np.vstack(new_pos_depot)
        new_pos_node = np.vstack(new_pos_node)
        
        problem_idx_list = problem_idx_list[:len(answer_list)]  # 最后可能会出现剩余数据量不足已完成数据量的情况
        self.demand[problem_idx_list] = new_demand                                      # (batch_size, node_num)
        self.pos_depot[problem_idx_list] = new_pos_depot                                # (batch_size, 2)
        self.pos_node[problem_idx_list] = new_pos_node                                  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)  # (batch_size, 1 + node_num, 2)                                        
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
        return answer_list

    def reset(self, seed=None, options:dict=None):
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
        self.capacity_left[problem_idx] = CAPACITIES[self.node_num]                 # (batch_size, )
        self.current_index[problem_idx] = 0                                         # (batch_size, ) 0 代表 depot, [1, node_num] 代表各个站点
        self.visited[problem_idx] = np.zeros(self.node_num, dtype=np.int32)         # (batch_size, node_num)
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
        selected_node = action.astype(np.int32)

        to_node = selected_node != 0
        to_depot = selected_node == 0
        problem_idx = np.arange(self.batch_size)

        # 考察去向的站点是否已经访问过，如果自动设置了可行 action 范围，这种情况不应该发生
        selected_visited = self.visited[problem_idx, selected_node-1]
        selected_visited_idx = np.where((selected_visited==1) & to_node)[0] 
        truncated[selected_visited_idx] = True
        reward['AM'][selected_visited_idx] = reward['DB1'][selected_visited_idx] = COP_FAILED_RWD

        # 转移到新城市并配送货
        self.current_index = selected_node
        self.visited[problem_idx[to_node], (selected_node-1)[to_node]] = 1
        self.capacity_left[to_node] -= self.demand[problem_idx, selected_node-1][to_node]
        
        # 考察车辆容量是否无法满足需求，如果自动设置了可行 action 范围，这种情况不应该发生
        capacity_insufficient_idx = np.where((self.capacity_left<0) & to_node)[0]
        truncated[capacity_insufficient_idx] = True
        reward['AM'][capacity_insufficient_idx] = reward['DB1'][capacity_insufficient_idx] = COP_FAILED_RWD

        # 考察正常结束的情况
        terminated = self._is_terminated()
        
        # 转移到仓库的情况
        self.capacity_left[to_depot] = CAPACITIES[self.node_num]

        # 更新模型解
        for i in range(self.batch_size):
            self.model_answer[i].append(action[i])
            if terminated[i]:
                qulity = self._pred_qulity(self.pos[i], self.real_answer[i], self.model_answer[i], self.problem_best_obj[i], self.problem_random_obj[i])
                reward['AM'][i] = qulity['AM']
                reward['DB1'][i] = qulity['DB1']
                
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {'obj': np.array([calc_vrp_distance(self.pos[i], answer) for i, answer in enumerate(self.model_answer)])}