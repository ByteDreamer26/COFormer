import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from utils.utils_slover import calc_op_distance, calc_op_total
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD
from environment.used.BaseEnv_COP import Env_COP, Logger_COP
from data.raw.op import MAX_LENGTHS

class OP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale=10, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.node_num = scale
        self.name = 'Env_OP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(2*self.node_num,), dtype=np.float32),
            'prize': spaces.Box(low=0, high=1, shape=(self.node_num,), dtype=np.float32),
            'length': spaces.Box(low=0, high=4, dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
            'visited': spaces.MultiDiscrete([2]*self.node_num),  
        })

        # 定义动作空间
        self.action_space = spaces.Discrete(self.node_num)

        # 动作各维度对应的 token 取值范围
        # 动作 [0, num_nodes-1] 表示移动去对应的站点
        self.action_value_space = [list(range(0, self.node_num)),]   # 由于 action 都是自然数，token 取值范围和 action 相同 

        # 初始化状态
        self.pos_depot = np.random.uniform(0, 1, (2, )).astype(np.float32)              # (2,)
        self.pos_node = np.random.uniform(0, 1, (self.node_num, 2)).astype(np.float32)  # (node_num, 2)
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node))                   # (node_num+1, 2)
        self.prize = np.zeros(self.node_num, dtype=np.float32)                          # (node_num, )
        self.length_left = MAX_LENGTHS[self.node_num]
        
        self.current_index = -1     # -1 代表 depot, [0, node_num-1] 代表各个结点
        self.visited = np.zeros(self.node_num, dtype=np.int32)                          # (node_num, )
        self.distance_node = np.zeros((self.node_num, self.node_num), dtype=np.float32) # (node_num, node_num)
        self.distance_depot = np.zeros(self.node_num, dtype=np.float32)                 # (node_num, )
        self.real_answer = []
        self.model_answer = []
        self.problem_best_obj = 0
        self.problem_random_obj = 0

    def _set_distance(self):
        for i in range(self.node_num):
            self.distance_depot[i] = np.linalg.norm(self.pos_node[i] - self.pos_depot)
            for j in range(self.node_num):
                self.distance_node[i,j] = np.linalg.norm(self.pos_node[i] - self.pos_node[j])

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        action_value_space = self.get_action_value_space(hard_action_constraint=True)[0]
        return action_value_space.size == 0 # 无可行node节点则正常结束轨迹

    def _pred_qulity(self):
        model_answer, real_answer = np.array(self.model_answer), np.array(self.real_answer) # 取值均在 [0,19]

        # 判断 model answer 合法性
        assert calc_op_distance(self.pos, real_answer+1) <= MAX_LENGTHS[self.node_num] + 1e-5, "Tour exceeds max_length!"
        assert model_answer.min() >= 0 and model_answer.max() <= self.node_num - 1
        assert real_answer.min() >= 0 and real_answer.max() <= self.node_num - 1
        
        # 计算 model_answer 质量
        rnd_obj = self.problem_random_obj
        model_obj = calc_op_total(self.prize, model_answer)
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else calc_op_total(self.prize, real_answer)
        #assert abs(best_obj - self.problem_best_obj) < 1e-4
        
        qulity_AM = 1 - (best_obj - model_obj)/best_obj
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
            length_left = obs['length']
            visited = obs['visited']
            unvisited = np.where(visited==0)[0]

            if self.current_index == -1:
                # 当前在仓库（一定是初始状态），只考虑长度限制
                assert np.allclose(obs['current_position'], self.pos_depot)
                assert length_left == MAX_LENGTHS[self.node_num]
                action_value_space = np.array([
                    idx for idx in range(self.node_num) \
                    if length_left >= self.distance_depot[idx] * 2   
                ])
            else:
                # 旅途当中，节点如果已经访问过，或访问后无法在长度限制内返回仓库，则禁止访问
                assert np.allclose(obs['current_position'], self.pos_node[self.current_index])
                action_value_space = np.array([
                    idx for idx in unvisited \
                    if length_left >= self.distance_node[self.current_index, idx] + \
                                    self.distance_depot[idx]   
                ])
        else:
            action_value_space = np.array(range(self.node_num))
        
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
        self.prize = problem['prize'].copy()                  
        self.pos = np.vstack((self.pos_depot[None,:], self.pos_node))               # (node_num+1, 2)
        self._set_distance()
        return answer

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
            return self._get_observation(), self._get_info()
        
        self.length_left = MAX_LENGTHS[self.node_num]
        self.current_index = -1     # -1 代表 depot
        self.visited = np.zeros(self.node_num, dtype=np.int32)
        self.model_answer = []
        
        # 初始化为预生成的评估问题，判断合法性
        assert 'problem_info' in options
        real_answer = self._recover(problem_info=options['problem_info'], problem_obj=options['problem_obj'])
        assert len(np.unique(real_answer)) == len(real_answer), "Tour cannot contain duplicates"
        assert calc_op_distance(self.pos, np.array(real_answer)+1) <= MAX_LENGTHS[self.node_num] + 1e-5, "Tour exceeds max_length!"
        self.real_answer = real_answer

        return self._get_observation(), self._get_info()

    def step(self, action):
        terminated = truncated = False
        reward = {'AM': 0, 'DB1': 0}
        selected_node = int(action)

        self.model_answer.append(selected_node)
        if self.visited[selected_node] == 1:
            # 目标节点已经被访问过，轨迹失败（如果自动设置了可行 action 范围，这种情况不应该发生）
            truncated = True            
            reward['AM'] = reward['DB1'] = COP_FAILED_RWD
        else:
            # 转移去目标节点
            if self.current_index == -1:
                self.length_left -= self.distance_depot[selected_node]
            else:             
                self.length_left -= self.distance_node[self.current_index, selected_node]
            self.current_index = selected_node
            self.visited[selected_node] = 1

            # 若剩余距离不足以返回仓库，轨迹失败（如果自动设置了可行 action 范围，这种情况不应该发生）
            if self.length_left < self.distance_depot[selected_node]:
                truncated = True       
                reward['AM'] = reward['DB1'] = COP_FAILED_RWD
            
            # 检查轨迹是否正常结束
            if self._is_terminated():
                terminated = True
                reward = self._pred_qulity()
    
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        pos_node = self.pos_node.copy()
        prize = self.prize.copy()
        pos_node[self.visited==1] = 0
        prize[self.visited==1] = 0

        current_position = self.pos_depot.copy() if self.current_index == -1 else \
                            self.pos_node[self.current_index].copy()
        obs = {
            'pos_depot': self.pos_depot.copy().astype(np.float32),
            'pos_node': pos_node.flatten().astype(np.float32),
            'prize': prize.astype(np.float32),
            'length': np.array([self.length_left,]).astype(np.float32),
            'current_position': current_position.copy().astype(np.float32),
            'visited': self.visited.copy().astype(np.int32)
        }
        return obs

    def _get_info(self):
        #return {'obs': self._get_observation()}
        return {'obj': calc_op_total(self.prize, self.model_answer)}
    
    def render(self):   
        pass

class OP_logger_V1(Logger_COP):
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
                node = obss['pos_node'][t].reshape((-1, 2))
                prize = obss['prize'][t]
                node_info = np.hstack((node, prize[:,None]))
                current_location = obss['current_position'][t]
                visited = obss['visited'][t]
                length_left = obss['length'][t]
                assert prize[acts[t]].item() != 0

                file.write(f'node info:\n{node_info}\n')
                file.write(f'current location:\t{current_location}\n')
                file.write(f'length left:     \t{length_left}\n')
                file.write(f'visited:         \t{visited}\n')
                file.write(f'action_space:    \t{act_value_space[t][0]}\n')
                file.write(f'take action:     \t{acts[t]}\n')
                file.write(f'get prize:       \t{prize[acts[t]]}\n')
                file.write(f'get reward:      \tAM:{rewards_AM[t]}; DB1:{rewards_DB1[t]}\n\n')

class DDP_OP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale=10, batch_size=32, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.node_num = scale
        self.batch_size = batch_size
        self.name = 'Env_OP_V1'

        # 定义观测空间
        self.observation_space = spaces.Dict({
            'pos_depot': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'pos_node': spaces.Box(low=0, high=1, shape=(batch_size, 2*self.node_num), dtype=np.float32),
            'prize': spaces.Box(low=0, high=1, shape=(batch_size, self.node_num,), dtype=np.float32),
            'length': spaces.Box(low=0, high=4, shape=(batch_size, ), dtype=np.float32),
            'current_position': spaces.Box(low=0, high=1, shape=(batch_size, 2), dtype=np.float32),
            'visited': spaces.MultiDiscrete([[2]*self.node_num for _ in range(batch_size)]),  
        })

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.node_num] * batch_size)

        # 动作各维度对应的 token 取值范围
        # op_v1 环境中动作 [0, num_nodes-1] 表示移动去对应的站点
        action_value_space = [np.arange(self.node_num, dtype=np.int32) for _ in range(batch_size)]  # 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space,]                                             # 动作只有一个维度

        # 初始化状态
        self.pos_depot = np.random.uniform(0, 1, (batch_size, 2)).astype(np.float32)                # (batch_size, 2)
        self.pos_node = np.random.uniform(0, 1, (batch_size, self.node_num, 2)).astype(np.float32)  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)              # (batch_size, 1 + node_num, 2)
        self.prize = np.zeros((batch_size, self.node_num), dtype=np.float32)                        # (batch_size, node_num)
        self.length_left = np.array([MAX_LENGTHS[self.node_num]] * batch_size)                      # (batch_size, )
        self.visited = np.zeros((batch_size, self.node_num), dtype=np.int32)                        # (batch_size, node_num)
        self.distance = np.zeros((batch_size, self.node_num+1, self.node_num+1), dtype=np.float32)  # (batch_size, node_num+1, node_num+1)
        
        # 类内使用0表示仓库，[1, num_nodes] 表示移动去对应的站点以提高效率
        # 类外使用-1表示仓库，[0, num_nodes-1] 表示移动去对应的站点，和OP_V1兼容
        self.current_index = np.zeros(batch_size, dtype=np.int32)   # (batch_size, ), 取值[0, num_nodes], 0 代表 depot, [1, node_num] 代表各个结点
        self.real_answer = [[] for _ in range(batch_size)]          # 相同规模的cvrp问题，解向量长度可能不同，不宜使用np.ndarray形式存储
        self.model_answer = [[] for _ in range(batch_size)]
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)

    def _set_distance(self):
        self.distance = np.sqrt(np.sum((self.pos[:,:,np.newaxis] - self.pos[:,np.newaxis]) ** 2, axis=-1))  # (batch_size, node_num+1, node_num+1)

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        terminated = np.zeros(self.batch_size, dtype=bool)
        action_value_space = self.get_action_value_space(hard_action_constraint=True)[0]
        for i, space in enumerate(action_value_space):
            terminated[i] = space.size == 0 # 无可行node节点则正常结束轨迹
        return terminated 

    def _pred_qulity(self, prize, real_answer, model_answer, problem_best_obj, problem_random_obj):
        '''  
        prize:  (node_num, )
        real_answer:    List, 取值[0,19]
        model_answer:   List, 取值[0,19]
        '''
        # 判断 model answer 合法性
        model_answer, real_answer = np.array(model_answer), np.array(real_answer)
        #assert model_answer.min() >= 0 and model_answer.max() <= self.node_num - 1
        #assert real_answer.min() >= 0 and real_answer.max() <= self.node_num - 1
        #assert calc_op_distance(pos, real_answer+1) <= MAX_LENGTHS[self.node_num] + 1e-5
        #assert len(np.unique(model_answer)) == len(model_answer), "Tour cannot contain duplicates"
        
        # 计算 model_answer 质量
        rnd_obj = problem_random_obj
        model_obj = calc_op_total(prize, model_answer)
        best_obj = problem_best_obj
        #best_obj = calc_op_total(prize, real_answer)
        #assert abs(best_obj - problem_best_obj) < 1e-4

        qulity_AM = 1 - (best_obj - model_obj)/best_obj
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
            length_left = obs['length']             # (batch_size, )
            visited = obs['visited']                # (batch_size, node_num)

            at_depot = self.current_index == 0          # (batch_size, )
            at_node = self.current_index != 0           # (batch_size, )
            at_node_idx = self.current_index[at_node]   # (at_node_num, )
            problem_idx = np.arange(at_node_idx.size)

            distance_to_depot = np.zeros((self.batch_size, self.node_num), dtype=np.float32)
            # 当前在仓库（一定是初始状态），只考虑长度限制
            distance_to_depot[at_depot] = self.distance[at_depot][:, 0, 1:] * 2
            # 旅途当中，节点如果已经访问过，或访问后无法在长度限制内返回仓库，则禁止访问
            distance_to_depot[at_node] = (self.distance[at_node][problem_idx, at_node_idx] + self.distance[at_node][problem_idx, 0])[:,1:]
            # 找出满足长度要求的站点
            problem_idx, action_value_spaces = np.where(distance_to_depot < length_left[:,None])

            action_value_space = []
            for i in range(self.batch_size):
                space = action_value_spaces[problem_idx==i]
                action_value_space.append(np.array([a for a in space if visited[i][a] == 0], dtype=np.int32))     
        else:
            action_value_space = [np.arange(self.node_num, dtype=np.int32) for _ in range(self.batch_size)]
        
        self.action_value_space = [action_value_space, ]     
        return self.action_value_space

    def _gen_question(self):
        pass

    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert isinstance(problem_list[0]['pos_node'], np.ndarray)
        new_pos_depot, new_pos_node, new_prize = [], [], []
        for problem in problem_list:
            new_pos_depot.append(problem['pos_depot'])
            new_pos_node.append(problem['pos_node'].reshape((1, self.node_num, 2)))
            new_prize.append(problem['prize'])
        new_prize = np.vstack(new_prize)
        new_pos_depot = np.vstack(new_pos_depot)
        new_pos_node = np.vstack(new_pos_node)

        problem_idx_list = problem_idx_list[:len(answer_list)]  # 最后可能会出现剩余数据量不足已完成数据量的情况
        self.prize[problem_idx_list] = new_prize                                        # (batch_size, node_num)
        self.pos_depot[problem_idx_list] = new_pos_depot                                # (batch_size, 2)
        self.pos_node[problem_idx_list] = new_pos_node                                  # (batch_size, node_num, 2)
        self.pos = np.concatenate((self.pos_depot[:, None, :], self.pos_node), axis=1)  # (batch_size, 1 + node_num, 2)                                                      
        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
    
        self._set_distance()
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
        self.length_left[problem_idx] = MAX_LENGTHS[self.node_num]          # (batch_size, )
        self.current_index[problem_idx] = 0                                 # (batch_size, ) 0 代表 depot, [1, node_num] 代表各个站点
        self.visited[problem_idx] = np.zeros(self.node_num, dtype=np.int32) # (batch_size, node_num)
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
        selected_node = action.astype(np.int32) # (batch_size, ) 取值 [0,19]

        # 目标节点已经被访问过，轨迹失败（如果自动设置了可行 action 范围，这种情况不应该发生）
        problem_idx = np.arange(self.batch_size)
        selected_visited = self.visited[problem_idx, selected_node]
        selected_visited_idx = np.where(selected_visited)[0]
        truncated[selected_visited_idx] = True
        reward['AM'][selected_visited_idx] = reward['DB1'][selected_visited_idx] = COP_FAILED_RWD

        # 转移去目标节点
        self.length_left -= self.distance[problem_idx, self.current_index, selected_node + 1]
        self.current_index = selected_node + 1
        self.visited[problem_idx, selected_node] = 1

        # 若剩余距离不足以返回仓库，轨迹失败（如果自动设置了可行 action 范围，这种情况不应该发生）
        len_insufficient_idx = np.where(self.length_left < self.distance[problem_idx, self.current_index, 0])[0]
        truncated[len_insufficient_idx] = True
        reward['AM'][len_insufficient_idx] = reward['DB1'][len_insufficient_idx] = COP_FAILED_RWD

        # 考察正常结束的情况
        terminated = self._is_terminated()

        # 检查轨迹是否正常结束
        for i in range(self.batch_size):
            self.model_answer[i].append(action[i])
            if terminated[i]:
                qulity = self._pred_qulity(self.prize[i], self.real_answer[i], self.model_answer[i], self.problem_best_obj[i], self.problem_random_obj[i])
                reward['AM'][i] = qulity['AM']
                reward['DB1'][i] = qulity['DB1']
                    
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        pos_node = self.pos[:, 1:].copy()
        pos_depot = self.pos[:, 0].copy()
        prize = self.prize.copy()
        length = self.length_left.copy()  
        current_position = self.pos[np.arange(self.batch_size), self.current_index].copy()
        pos_node[self.visited==1] = 0
        prize[self.visited==1] = 0
        
        obs = {
            'pos_depot': pos_depot.astype(np.float32),                                          # (batch_size, 2)
            'pos_node': pos_node.reshape(self.batch_size, self.node_num*2).astype(np.float32),  # (batch_size, 2*node_num)
            'prize': prize.astype(np.float32),                          # (batch_size, )
            'length': length.astype(np.float32),                        # (batch_size, )
            'current_position': current_position.astype(np.float32),    # (batch_size, 2)
            'visited': self.visited.copy().astype(np.int32)             # (batch_size, node_num)
        }
        return obs

    def _get_info(self):
        return {'obj': np.array([calc_op_total(self.prize[i], answer) for i, answer in enumerate(self.model_answer)])}

    def render(self):   
        pass