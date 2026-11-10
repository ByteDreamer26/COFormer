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

class SPCTSP_V2(Env_COP):
    ### stoc_prize 是真实奖励real_prize事先不确定，det_prize 是预期奖励expect_prize事先确定
    def __init__(self, render_mode="rgb_array", scale:int=10, adj_range:bool=False):
        super().__init__(render_mode)
        self.name = 'Env_SPCTSP_V2'
        self.node_num = scale
        self.adj_range = adj_range
        
        # 定义动作空间
        self.action_space = spaces.Discrete(self.node_num+1)

        # 动作各维度对应的 token 取值范围
        # cvrp 环境中动作 0 表示移动去仓库；动作 [1, num_nodes] 表示移动去 [0, num_nodes-1] 站点
        self.action_value_space = [list(range(0, self.node_num + 1)),]                  # 由于 action 都是自然数，token 取值范围和 action 相同 

        # 初始化状态
        self.pos_depot = np.random.uniform(0, 1, (2, )).astype(np.float32)              # (2, )
        self.pos_node = np.random.uniform(0, 1, (self.node_num, 2)).astype(np.float32)  # (node_num, 2)
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node))                   # (node_num+1, 2)
        self.det_prize = np.zeros(self.node_num, dtype=np.float32)
        self.stoc_prize = np.zeros(self.node_num, dtype=np.float32)
        self.penalty = np.zeros(self.node_num, dtype=np.float32)
        
        self.prize2go = 1
        self.current_index = 0      # 0 代表 depot, [1, node_num] 代表各个站点
        self.visited = np.zeros(self.node_num, dtype=np.int32)
        self.real_answer = []
        self.model_answer = []
        self.problem_best_obj = 0
        self.problem_random_obj = 0

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(3*self.node_num,), dtype=np.float32),
            'det_prize': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'penalty': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, dtype=np.float32),
        })

    def _is_terminated(self, select):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return select == 0
    
    def _pred_qulity(self):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        #assert len(self.real_answer) == len(self.model_answer) # pctsp 问题中解长度不一定相等
        assert self.real_answer[-1] ==  self.model_answer[-1] == 0
        assert calc_pctsp_total(self.stoc_prize, self.real_answer[:-1]) >= 1
        assert calc_pctsp_total(self.stoc_prize, self.model_answer[:-1]) >= 1

        rnd_obj = self.problem_random_obj
        model_obj = calc_pctsp_cost(self.pos, self.penalty, self.stoc_prize, self.model_answer[:-1])
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else calc_pctsp_cost(self.pos, self.penalty, self.stoc_prize, self.real_answer[:-1])
        #assert abs(best_obj - self.problem_best_obj) < 1e-4
        assert best_obj is not None and model_obj is not None
        
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity
    
    def is_same_episode(self, acts1:np.array, acts2:np.array):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)
    
    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            visited = self.visited
            unvisited = np.where(visited==0)[0]
            action_value_space = unvisited + 1
            if not (unvisited.size > 0 and self.prize2go > 0):
                # 如果还有站点没有访问，且当前累计奖励尚未达到要求，禁止访问仓库
                action_value_space = np.insert(action_value_space, 0, 0)
        else:
            action_value_space = np.array(range(self.node_num + 1))
        
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
        assert isinstance(problem['pos_node'], np.ndarray)
        self.pos_depot = problem['pos_depot'].copy().reshape((2,))
        self.pos_node = problem['pos_node'].copy().reshape((self.node_num, 2))
        self.det_prize = problem['det_prize'].copy()    # (node_num, )   #按照预期奖励det_prize计算
        self.stoc_prize = problem['stoc_prize'].copy()  # (node_num, )   #按照真实奖励stoc_prize计算
        self.penalty = problem['penalty'].copy()                  
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node))               # (node_num+1, 2)
        return answer
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
            return self._get_observation(), self._get_info()
        
        # 初始化为预生成的评估问题并求解
        assert 'problem_info' in options
        real_answer = self._recover(problem_info=options['problem_info'], problem_obj=options['problem_obj'])

        # 初始化状态          
        self.prize2go = 1        
        self.current_index = 0
        self.stoc_visited = np.zeros(self.node_num, dtype=np.int32)
        self.det_visited = np.zeros(self.node_num, dtype=np.int32)
        self.model_answer = []
        self.real_answer = real_answer
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        terminated = truncated = False
        reward = {'AM': 0, 'DB1': 0}

        selected_city = int(action)
        if self._is_terminated(selected_city):
            assert selected_city == 0   # pctsp 轨迹的终止是由agent自己控制的            
            self.model_answer.append(selected_city)
            terminated = True
            reward = self._pred_qulity()
        else:
            assert selected_city > 0
            if self.visited[selected_city-1] == 1:
                truncated = True        # 如果自动设置了可行 action 范围，这种情况不应该发生
                reward['AM'] = reward['DB1'] = COP_FAILED_RWD
            else:
                self.current_index = selected_city
                self.model_answer.append(selected_city)
                self.prize2go -= self.stoc_prize[selected_city-1]
                self.prize2go = max(0, self.prize2go)
                self.visited[selected_city-1] = 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        pos_node = self.pos_node.copy()
        visited = self.visited.copy()
        node_info = np.hstack((pos_node, visited[:,None]))
        current_position = self.pos[self.current_index].copy()
        stoc_prize = self.stoc_prize.copy()
        det_prize = self.det_prize.copy()
        penalty = self.penalty.copy()
        stoc_prize[self.visited==1] = 0
        det_prize[self.visited==1] = 0
        penalty[self.visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),          # (2, )
            'node_info': node_info.flatten().astype(np.float32),            # (node_num*3, )
            'det_prize': det_prize.astype(np.float32),                      # (node_num, )
            'penalty': penalty.astype(np.float32),                          # (node_num, )
            'prize2go': np.array([self.prize2go,]).astype(np.float32),      # (1, )
            'current_position': current_position.copy().astype(np.float32), # (2, )
        }
        return obs

    
    def _get_info(self):
        return {'obj': calc_pctsp_cost(self.pos, self.penalty, self.stoc_prize, self.model_answer[:-1])}
        
    def render(self):   
        pass

class SPCTSP_logger_V2(Logger_COP):
    def __init__(self, env_name='Env_SPCTSP', dataset_name='SPCTSP'):
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
            acts = episode['actions']
            rewards_AM = episode['rewards']['AM']
            rewards_DB1 = episode['rewards']['DB1']
            assert len(rewards_AM) == len(rewards_DB1)
            obss = episode['observations']
            act_value_space = episode['act_value_space']    
            prefix_mask = episode['prefix_masks']
            prefix = episode['prefix']
            assert prefix_mask['pos_depot'].sum() == 0

            file.write('-'*15+f' epoch-{epoch_num}; episode-{episode_num}; time-{round(time_used, 2)}'+'-'*15+'\n')
            file.write(f'pos_depot: \t{prefix["pos_depot"]}\n\n')
            for t in range(len(rewards_AM)):
                current_location = obss['current_position'][t]
                prize2go = obss['stoc_prize2go'][t].item()
                visited = obss['visited'][t]
                assert prize2go >= 0

                masked_prize = prefix['det_prize'].copy()
                masked_penalty = prefix['penalty'].copy()
                masked_prize[prefix_mask['det_prize'][t]] = 0
                masked_penalty[prefix_mask['penalty'][t]] = 0

                node_info = np.hstack((
                    #np.arange(len(prize), dtype=np.int32)[:,None],
                    prefix["pos_node"].reshape(-1,2), 
                    masked_prize[:,None], 
                    masked_penalty[:,None],
                    visited[:,None]
                ))
                #node_info = np.round(node_info, 3)
                det_prize = prefix['det_prize']
                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t]} (to node {acts[t]-1})\n')
                file.write(f'stoc_prize (real) to go:     \t{prize2go}\n')
                file.write(f'get det prize (expected):       \t{det_prize[acts[t]-1]}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_SPCTSP_V2(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=10, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode)
        self.node_num = scale
        self.adj_range = adj_range
        self.batch_size = batch_size
        self.name = 'Env_SPCTSP_V2'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'node_info': spaces.Box(low=0, high=1, shape=(batch_size, 3*self.node_num,), dtype=np.float32),
            'det_prize': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num), dtype=np.float32),
            'penalty': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'prize2go': spaces.Box(low=0, high=1, shape=(batch_size, ), dtype=np.float32),
        })

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.node_num+1] * batch_size)

        # 动作各维度对应的 token 取值范围
        # pctsp 环境中动作 0 表示移动去仓库；动作 [1, num_nodes] 表示移动去 [0, num_nodes-1] 站点
        action_value_space = [np.arange(self.node_num+1, dtype=np.int32) for _ in range(batch_size)]# 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space, ]     

        # 初始化状态
        self.pos_depot = np.random.uniform(0, 1, (batch_size, 2)).astype(np.float32)                # (batch_size, 2)
        self.pos_node = np.random.uniform(0, 1, (batch_size, self.node_num, 2)).astype(np.float32)  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)     
        self.det_prize = np.zeros((batch_size, self.node_num), dtype=np.float32)
        self.stoc_prize = np.zeros((batch_size, self.node_num), dtype=np.float32)
        self.penalty = np.zeros((batch_size, self.node_num), dtype=np.float32)
        self.prize2go = np.ones(batch_size, dtype=np.float32)
        self.current_index = np.zeros(batch_size, dtype=np.int32)               # 0 代表 depot, [1, node_num] 代表各个站点
        self.visited = np.zeros((batch_size, self.node_num), dtype=np.int32)
        self.real_answer = [[] for _ in range(batch_size)]                      # 相同规模的pctsp问题，解向量长度可能不同，不宜使用np.ndarray形式存储
        self.model_answer = [[] for _ in range(batch_size)]
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)

    def _is_terminated(self, actions:np.ndarray):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return actions == 0 # (batch_size, ) 执行动作0返回仓库时结束轨迹

    def _pred_qulity(self, pos, stoc_prize, penalty, real_answer, model_answer, problem_best_obj, problem_random_obj):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        #assert len(self.real_answer) == len(self.model_answer) # pctsp 问题中解长度不一定相等
        #assert real_answer[-1] == model_answer[-1] == 0
        #assert calc_pctsp_total(prize, real_answer[:-1]) >= 1
        #assert calc_pctsp_total(prize, model_answer[:-1]) >= 1
        rnd_obj = problem_random_obj
        model_obj = calc_pctsp_cost(pos, penalty, stoc_prize, model_answer[:-1])
        best_obj = problem_best_obj
        #best_obj = calc_pctsp_cost(pos, penalty, prize, real_answer[:-1])
        #assert abs(best_obj - problem_best_obj) < 1e-4
        assert best_obj is not None and model_obj is not None
        
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity

    def is_same_episode(self, acts1:np.array, acts2:np.array):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if hard_action_constraint:
            visited = self.visited            # (batch_size, node_num)
            unvisited_idx, unvisited = np.where(visited==0)
            
            action_value_space = []
            for i in range(self.batch_size):
                space = unvisited[unvisited_idx==i]
                space += 1
                if not (space.size > 0 and self.prize2go[i] > 0):
                # 如果还有站点没有访问，且当前累计奖励尚未达到要求，禁止访问仓库
                    space = np.concatenate(([0], space))
                action_value_space.append(space)
        else:
            action_value_space = [np.arange(self.node_num+1, dtype=np.int32) for _ in range(self.batch_size)]
        
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert isinstance(problem_list[0]['pos_node'], np.ndarray)
        new_pos_depot, new_pos_node, new_stoc_prize, new_det_prize, new_penalty = [], [], [], [], []
        for problem in problem_list:
            new_pos_depot.append(problem['pos_depot'])
            new_pos_node.append(problem['pos_node'].reshape((1, self.node_num, 2)))
            new_stoc_prize.append(problem['stoc_prize'])
            new_det_prize.append(problem['det_prize'])
            new_penalty.append(problem['penalty'])
        new_stoc_prize = np.vstack(new_stoc_prize)
        new_det_prize = np.vstack(new_det_prize)
        new_penalty = np.vstack(new_penalty)
        new_pos_depot = np.vstack(new_pos_depot)
        new_pos_node = np.vstack(new_pos_node)

        problem_idx_list = problem_idx_list[:len(answer_list)]  # 最后可能会出现剩余数据量不足已完成数据量的情况
        self.stoc_prize[problem_idx_list] = new_stoc_prize                                        # (batch_size, node_num)
        self.det_prize[problem_idx_list] = new_det_prize
        self.penalty[problem_idx_list] = new_penalty                                    # (batch_size, node_num)
        self.pos_depot[problem_idx_list] = new_pos_depot                                # (batch_size, 2)
        self.pos_node[problem_idx_list] = new_pos_node                                  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)  # (batch_size, 1 + node_num, 2)                                                      
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
        self.prize2go[problem_idx] = 1                                      # (batch_size, )
        self.current_index[problem_idx] = 0                                 # (batch_size, )
        self.visited[problem_idx] = np.zeros(self.node_num, dtype=np.int32) # (batch_size, node_num)
        for idx, answer in zip(problem_idx, problem_real_answer):
            self.real_answer[idx] = answer
            self.model_answer[idx] = []
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        terminated = np.zeros(self.batch_size, dtype=bool)
        truncated = np.zeros(self.batch_size, dtype=bool)
        reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }
        selected_node = action.astype(np.int32)
        problem_idx = np.arange(self.batch_size)

        # 转移到新城市并配获取奖励
        self.prize2go -= self.stoc_prize[problem_idx, selected_node-1]
        self.prize2go[self.prize2go < 0] = 0

        # 考察正常结束的情况，pctsp 轨迹的终止是由agent自己控制的（执行action=0则终止）   
        terminated = self._is_terminated(selected_node)
        for i in range(self.batch_size):
            if terminated[i]: 
                self.model_answer[i].append(selected_node[i])
                qulity = self._pred_qulity(self.pos[i], self.stoc_prize[i], self.penalty[i], self.real_answer[i], self.model_answer[i], self.problem_best_obj[i], self.problem_random_obj[i])
                reward['AM'][i] = qulity['AM']
                reward['DB1'][i] = qulity['DB1']
            else:
                if self.visited[i][selected_node[i]-1] == 1:
                    truncated[i] = True        # 如果自动设置了可行 action 范围，这种情况不应该发生
                    reward['AM'][i] = reward['DB1'][i] = COP_FAILED_RWD
                else:
                    self.current_index[i] = selected_node[i]
                    self.model_answer[i].append(selected_node[i])
                    self.visited[i][selected_node[i]-1] = 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        pos_node = self.pos_node.copy()                                     # (batch_size, node_num, 2)
        visited = self.visited.copy()                                       # (batch_size, node_num, )
        node_info = np.concatenate((pos_node, visited[:,:,None]), axis=-1)  # (batch_size, node_num, 3)
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy()
        stoc_prize = self.stoc_prize.copy()
        det_prize = self.det_prize.copy()
        penalty = self.penalty.copy()
        stoc_prize[self.visited==1] = 0
        det_prize[self.visited==1] = 0
        penalty[self.visited==1] = 0

        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),                              # (batch_size, 2)
            'node_info': node_info.reshape(self.batch_size, self.node_num*3).astype(np.float32),# (batch_size, 3*node_num)
            'det_prize': det_prize.astype(np.float32),                                          # (batch_size, node_num)
            'penalty': penalty.astype(np.float32),                                              # (batch_size, node_num)
            'prize2go': self.prize2go.copy().astype(np.float32),                                # (batch_size, )
            'current_position': current_position.copy().astype(np.float32),                     # (batch_size, 2)
        }
        return obs

    def _get_info(self):
        return {'obj': np.array([calc_pctsp_cost(self.pos[i], self.penalty[i], self.stoc_prize[i], answer[:-1]) for i, answer in enumerate(self.model_answer)])}