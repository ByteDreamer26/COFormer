# 弃用
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from environment.used.BaseEnv_COP import Env_COP
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD

class DDP_MIS_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=20, batch_size:int=32, adj_range:bool=False):
        super().__init__(render_mode)
        self.adj_range = adj_range
        self.name = 'Env_MIS_V1'
        self.node_num = scale
        self.batch_size = batch_size
 
        self.node_rows = np.zeros((batch_size, self.node_num, 20), dtype=np.float32)    # (batch_size, node_num, 20)
        self.observation_space = spaces.Dict({
            'current_state': spaces.Box(low=0, high=2, shape=(batch_size, 20), dtype=np.int32),     # (batch_size, 20)
        })


        # 动作各维度对应的 token 取值范围
        # tsp 环境中动作 [0, num_nodes-1] 表示移动去对应的城市
        action_value_space = [np.arange(self.node_num, dtype=np.int32) for _ in range(batch_size)]  # 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space,]                                             # 动作只有一个维度

        # 初始化状态
        self.timestep = 0
        self.adj_mat = np.zeros((batch_size, self.node_num, self.node_num), dtype=np.float32)
        self.current_state = np.zeros((batch_size, self.node_num+1), dtype=np.int32)                 # 每个点有-1, 0, 1三种状态, 0代表未决定，-1代表不选，1代表选; 最后留一个维度代表跳过      
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)


    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) <= self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch或最大问题数量更新问题
        #assert isinstance(problem_list[0]['adj_mat'], np.ndarray)
        new_adj_mat = []
        for problem in problem_list:
            adj_mat = problem['adj_mat'].reshape((self.node_num, self.node_num))
            new_adj_mat.append(adj_mat)
        new_adj_mat = np.array(new_adj_mat)
        self.adj_mat[problem_idx_list] = new_adj_mat     # (batch_size, node_num, node_num)

        if problem_obj is not None:
            self.problem_best_obj[problem_idx_list] = problem_obj[0]
            self.problem_random_obj[problem_idx_list] = problem_obj[1]
    
        return answer_list  
    
    def _get_observation(self):
        return {                                                               
            'current_state': self.current_state[:,:-1].copy().astype(np.int32)  # (batch_size, node_num)
        }
    
    def get_prefix(self):
        return {
            'adj_mat': self.adj_mat.copy().reshape(self.batch_size, -1).astype(np.int32)        # (batch_size, node_num*node_num)
        }

    def get_prefix_mask(self):
        index_mask = self.current_state[:,:-1] !=0 # (batch_size, node_num)
        
        batch_idx, mask_idx = np.where(index_mask)
        prefix_mask = np.zeros((self.batch_size, self.node_num, self.node_num), dtype=bool)
        prefix_mask[batch_idx, mask_idx, :] = True
        prefix_mask[batch_idx, :, mask_idx] = True
        return {
            'adj_mat': prefix_mask.reshape(self.batch_size, -1).astype(bool)    #(batch_size, node_num, node_num) -> (batch_size, node_num*20)
        }

    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return (self.current_state!=0).all(axis=1)     # (batch_size, ) 全部结点访问后只能回到起点
    
    def _pred_qulity(self, problem_current_state, problem_best_obj, problem_random_obj):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        assert (problem_current_state[:-1]!=0).all()
        rnd_obj = problem_random_obj
        model_obj = (problem_current_state[:-1]==1).sum()    # (problem_batch_size, )
        best_obj = problem_best_obj
        #best_obj = calc_tsp_distance(self.position, self.real_answer)      # (problem_batch_size, )
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

        idx, node = np.where(self.current_state[:,:-1]==0)  #正常取值范围最大为0 -> node_num-1
        action_value_space = []
        for i in range(self.batch_size): 
            if (idx==i).any():  #对于第i个问题，还有为0的位置
                action_value_space.append(node[idx==i].astype(np.int32))
            else:
                action_value_space.append(np.array([self.node_num,], dtype=np.int32))  #跳过，等待其他环境

        self.action_value_space = [action_value_space, ]     
        return self.action_value_space

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
        self.timestep = 0

        self.current_state = np.zeros((self.batch_size, self.node_num+1), dtype=np.int32)                 # 每个点有-1, 0, 1三种状态, 0代表未决定，-1代表不选，1代表选; 最后留一个维度代表跳过

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.timestep += 1
        assert 1 <= self.timestep < self.node_num
        terminated = np.zeros(self.batch_size, dtype=bool)
        truncated = np.zeros(self.batch_size, dtype=bool)
        reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }
        selected_node = action.astype(np.int32)  # (batch_size, )

        # 考察去向的站点是否已经访问过，如果自动设置了可行 action 范围，这种情况不应该发生
        # 由于该环境要求 hard_action_constraint==True，故总有truncated==False
        problem_idx = np.arange(self.batch_size)
        selected_visited = self.current_state[problem_idx, selected_node] == 0
        assert selected_visited[selected_node!=20].all()   # 20是跳过的标记

        # 排除那些有连接的点
        expanded = np.concatenate([self.adj_mat, np.zeros((self.batch_size, self.node_num, 1), dtype=np.int32)], axis=2) # (batch_size, node_num, node_num+1)
        expanded = np.concatenate([expanded, np.zeros((self.batch_size, 1, self.node_num+1), dtype=np.int32)], axis=1)  # (batch_size, node_num+1, node_num+1)
        whether_connected = expanded[problem_idx, selected_node, :] # (batch_size, node_num + 1)
        self.current_state[whether_connected==1] = 2  # (batch_size, node_num + 1) 那些相连接的点标记为2


        # 将新选取的节点标记为1
        self.current_state[problem_idx, selected_node] = 1 #有可能也标记最后一个维度，但是不会被使用

        # 考察正常结束的情况
        terminated = self._is_terminated()              # (batch_size, )

        # 所有问题同步结束，计算质量
        #if terminated.all():
        #    reward = self._pred_qulity()                # (batch_size, )
        
        for i in range(self.batch_size):
            if terminated[i]: 
                qulity = self._pred_qulity(self.current_state[i], self.problem_best_obj[i], self.problem_random_obj[i])
                reward['AM'][i] = qulity['AM']
                reward['DB1'][i] = qulity['DB1']
 
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {'obj': (self.current_state[:,:-1]==1).sum(axis = 1)}