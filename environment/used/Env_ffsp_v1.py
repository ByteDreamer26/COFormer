import os
import sys
matnet_base_path = os.path.abspath('/data1/xxx/MatNet/FFSP/FFSP_MatNet')
sys.path.append(matnet_base_path)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
from dataclasses import dataclass
import itertools
from environment.used.BaseEnv_COP import Env_COP
from utils.utils import create_file_if_not_exist, COP_FAILED_RWD

#from matnet_test import eval_main, env_params, tester_params

@dataclass
class Reset_State:
    problems_list: list
    # len(problems_list) = stage_cnt
    # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
    # float type


@dataclass
class Step_State:
    BATCH_IDX: np.array
    POMO_IDX: np.array
    # shape: (batch, pomo)
    #--------------------------------------
    step_cnt: int = 0
    stage_idx: np.array = None
    # shape: (batch, pomo)
    stage_machine_idx: np.array = None
    # shape: (batch, pomo)
    job_ninf_mask: np.array = None
    # shape: (batch, pomo, job+1)
    finished: np.array = None
    # shape: (batch, pomo)
    machine_query: np.array = None
    prefix_mask: np.array = None

class DDP_FFSP_V1(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=20, batch_size:int=48, adj_range:bool=False):
        super().__init__(render_mode)
        self.batch_size = batch_size
        self.name = 'Env_FFSP_V1'
        self.timestep = 0
        
        # 问题有关参数
        self.stage_cnt = 3
        self.machine_cnt_list = [4,4,4]
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = scale
        self.process_time_params = {
            'time_low': 2,
            'time_high': 10,
        },
        self.pomo_size = 24  # assuming 4 machines at each stage! 4*3*2*1
        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)


        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'machine_query': spaces.Box(low=0, high=1, shape=(batch_size, self.job_cnt+1), dtype=np.float32), 
        })

        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([self.job_cnt+1] * batch_size)

        # 动作各维度对应的 token 取值范围
        # tsp 环境中动作 [0, num_nodes-1] 表示移动去对应的城市
        action_value_space = [np.arange(self.job_cnt+1, dtype=np.int32) for _ in range(batch_size)]  # 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space,]                                         # 动作只有一个维度
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)


        # Const @Load_Problem
        ####################################
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)

        self.job_durations = None
        # shape: (batch, job+1, total_machine)
        # last job means NO_JOB ==> duration = 0

        # Dynamic
        ####################################
        self.time_idx = None
        # shape: (batch, pomo)
        self.sub_time_idx = None  # 0 ~ total_machine_cnt-1
        # shape: (batch, pomo)
        self.machine_idx = None  # must update according to sub_time_idx
        # shape: (batch, pomo)

        self.schedule = None
        # shape: (batch, pomo, machine, job+1)
        # records start time of each job at each machine
        self.machine_wait_step = None
        # shape: (batch, pomo, machine)
        # How many time steps each machine needs to run, before it become available for a new job
        self.job_location = None
        # shape: (batch, pomo, job+1)
        # index of stage each job can be processed at. if stage_cnt, it means the job is finished (when job_wait_step=0)
        self.job_wait_step = None
        # shape: (batch, pomo, job+1)
        # how many time steps job needs to wait, before it is completed and ready to start at job_location
        self.finished = None  # is scheduling done?
        # shape: (batch, pomo)

        # STEP-State
        ####################################
        self.step_state = None
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) == self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch或最大问题数量更新问题
        assert isinstance(problem_list[0]['durations'], np.ndarray)
        
        self.BATCH_IDX = np.arange(self.batch_size)[:, None]  # shape: (batch_size, 1)
        self.BATCH_IDX = np.broadcast_to(self.BATCH_IDX, (self.batch_size, self.pomo_size))  # 扩展到 (batch_size, pomo_size)

        self.POMO_IDX = np.arange(self.pomo_size)[None, :]  # shape: (1, pomo_size)
        self.POMO_IDX = np.broadcast_to(self.POMO_IDX, (self.batch_size, self.pomo_size))

        new_durations = []
        for problem in problem_list:
            durations = problem['durations']
            new_durations.append(durations)
        new_durations = np.array(new_durations)

        self.job_durations = np.empty(shape=(self.batch_size, self.job_cnt+1, self.total_machine_cnt), dtype=np.int64)
        self.job_durations[problem_idx_list, :, :] = new_durations      # (batch_size, job_cnt+1, 3*machine_cnt)
        assert (self.job_durations>=0).all() # 仅最后一个job可以为0
        assert (self.job_durations<=10).all()

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
        problem_real_answer = self._recover(problem_info, problem_obj, problem_idx)  #载入问题

        # 初始化状态

        self.time_idx = np.zeros((self.batch_size, self.pomo_size), dtype=np.int64)      ###真实的时间time
        # shape: (batch, pomo)
        self.sub_time_idx = np.zeros((self.batch_size, self.pomo_size), dtype=np.int64)  ###用来表示在pomo设定下，现在处在第几个状态，取值为0-11
        # shape: (batch, pomo)
        self.machine_idx = self.sm_indexer.get_machine_index(self.POMO_IDX, self.sub_time_idx)     ### 按照pomo的顺序，获取当前的machine的index
        # shape: (batch, pomo)   pomo = 24 = permutation(4) 把4个机器按照不同顺序排列的所有可能

        # self.schedule: shape: (batch, pomo, machine, job+1)
        self.schedule = np.full(shape=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt + 1),
                                fill_value=-999999, dtype=np.int64)

        # self.machine_wait_step: shape: (batch, pomo, machine)
        self.machine_wait_step = np.zeros(shape=(self.batch_size, self.pomo_size, self.total_machine_cnt), dtype=np.int64)

        # self.job_location: shape: (batch, pomo, job+1)
        self.job_location = np.zeros(shape=(self.batch_size, self.pomo_size, self.job_cnt + 1), dtype=np.int64)

        # self.job_wait_step: shape: (batch, pomo, job+1)
        self.job_wait_step = np.zeros(shape=(self.batch_size, self.pomo_size, self.job_cnt + 1), dtype=np.int64)

        # self.finished: shape: (batch, pomo)
        self.finished = np.full(shape=(self.batch_size, self.pomo_size), fill_value=False, dtype=bool)
        # shape: (batch, pomo)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        ### pre_step
        self._update_step_state()
        self.step_state.step_cnt = 0
        reward = None
        done = False

        return self._get_observation(), self._get_info()
    
    def _update_step_state(self):
        self.step_state.step_cnt += 1

        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)  ###储存stage值, 意味着当前时刻对于当前机器只决策这个stage的任务
        # shape: (batch, pomo)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.POMO_IDX, self.sub_time_idx)  ###储存machine编号的真值
        # shape: (batch, pomo)

        job_loc = self.job_location[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)
        job_wait_t = self.job_wait_step[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)

        job_in_stage = job_loc == self.step_state.stage_idx[:, :, None]
        # shape: (batch, pomo, job)
        job_not_waiting = (job_wait_t == 0)
        # shape: (batch, pomo, job)
        job_available = job_in_stage & job_not_waiting              ### job匹配所处于的stage，且有可以马上开始的job
        # shape: (batch, pomo, job)

        job_in_previous_stages = (job_loc < self.step_state.stage_idx[:, :, None]).any(axis=2)
        # shape: (batch, pomo)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(axis=2)
        # shape: (batch, pomo)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished  ### 允许进行跳过操作的条件 ==（还在上一个stage的任务 或 当前正在进行处理的子任务 或 已经完成的任务）
        # shape: (batch, pomo)

       # self.step_state.job_ninf_mask: shape: (batch, pomo, job+1)
        self.step_state.job_ninf_mask = np.full(shape=(self.batch_size, self.pomo_size, self.job_cnt + 1),
                                                fill_value=float('-inf'))

        # job_enable: 拼接 job_available 和 wait_allowed
        job_enable = np.concatenate((job_available, wait_allowed[:, :, np.newaxis]), axis=2)
        # shape: (batch, pomo, job+1)
        self.step_state.job_ninf_mask[job_enable] = 0
        # shape: (batch, pomo, job+1)

        self.step_state.finished = self.finished
        # shape: (batch, pomo)

        machine_idx = self.sm_indexer.get_machine_index(self.POMO_IDX, self.sub_time_idx)  ###在3个stage下的全局machine编号
        # shape: (batch, pomo)
        self.step_state.machine_query = self.job_durations[self.BATCH_IDX, : , machine_idx] ### 获取当前machine，当前stage的原始obs
        # shape: (batch, pomo, job+1)
        prefix_mask = np.ones(shape=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=bool)
        # shape: (batch, pomo, job+1)
        prefix_mask[job_enable] = False
        self.step_state.prefix_mask = prefix_mask
    
    def _pred_qulity(self):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        rnd_obj = self.problem_random_obj
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else eval_main(self.job_durations[:, :-1, :])[1]

        job_durations_perm = np.transpose(self.job_durations, (0, 2, 1))  ### 转置，方便计算
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, None, :, :]  ### 记录了每个机器，在每个job上的结束时间
        # shape: (batch, pomo, machine, job+1)

        end_time_max = end_schedule[:, :, :, :self.job_cnt].max(axis=3)
        # shape: (batch, pomo, machine)
        end_time_max = end_time_max.max(axis=2)
        # shape: (batch, pomo)

        model_obj = end_time_max.min(axis=1)    # (problem_batch_size, )

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity
    
    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        job_enable = (self.step_state.job_ninf_mask == 0).reshape(-1, self.job_cnt+1) #(batch* pomo, job+1)
        idx, job = np.where(job_enable)
        action_value_space = [job[idx==i].astype(np.int32) for i in range(self.batch_size*self.pomo_size)]  #length = batch_size*pomo_size

        self.action_value_space = [action_value_space, ]     
        return self.action_value_space
    
    def step(self, job_idx):
        job_idx = job_idx.reshape(self.batch_size, self.pomo_size)  ## (batch, pomo)
        # job_idx.shape: (batch, pomo)

        ### 检查外部给入的动作是否合法
        assert np.sum(self.step_state.job_ninf_mask[self.BATCH_IDX, self.POMO_IDX, job_idx]) == 0

        self.schedule[self.BATCH_IDX, self.POMO_IDX, self.machine_idx, job_idx] = self.time_idx ### 记录每个job在每个machine上的开始时间
        # shape: (batch, pomo, machine, job+1)
        job_length = self.job_durations[self.BATCH_IDX, job_idx, self.machine_idx]
        # shape: (batch, pomo)     获取各个任务在当前stage机器上的工作时间
        self.machine_wait_step[self.BATCH_IDX, self.POMO_IDX, self.machine_idx] = job_length
        # shape: (batch, pomo, machine)     按照机器，更新各个机器对于当前任务剩余的工作时间
        self.job_location[self.BATCH_IDX, self.POMO_IDX, job_idx] += 1
        # shape: (batch, pomo, job+1)       按照任务，更新各任务目前进行到第几个stage （正在进行）
        self.job_wait_step[self.BATCH_IDX, self.POMO_IDX, job_idx] = job_length
        # shape: (batch, pomo, job+1)       按照任务，更新各个job对于当前子任务剩余的工作时间
        self.finished = (self.job_location[:, :, :self.job_cnt] == self.stage_cnt).all(axis=2)
        # shape: (batch, pomo)              

        ####################################
        done = self.finished.all()

        if done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._move_to_next_machine()
            self._update_step_state()

        if done:
            reward = self._pred_qulity()  
            # shape: (batch, pomo)
        else:
            reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }

        terminated = self.finished
        truncated = np.zeros(self.batch_size, dtype=bool)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_to_next_machine(self):

        # b_idx: shape: (batch*pomo,) == (not_ready_cnt,)
        b_idx = self.BATCH_IDX.flatten()

        # p_idx: shape: (batch*pomo,) == (not_ready_cnt,)
        p_idx = self.POMO_IDX.flatten()

        # ready: shape: (batch*pomo,) == (not_ready_cnt,)
        ready = self.finished.flatten()

        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        p_idx = p_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        cnt = 0
        while ~ready.all():
            cnt+=1
            # print('cnt in state loop:', cnt)

            new_sub_time_idx = self.sub_time_idx[b_idx, p_idx] + 1       ### 这里不是真的time，是sub_step, 遍历各个stage各个机器哪个可以继续选；需要往前推演状态，直到machine和job双双ready的状态
            # shape: (not_ready_cnt,)   
            step_time_required = new_sub_time_idx == self.total_machine_cnt  ### 所有的状态都不满足，意味着当前时刻，没有找到合适的机器/stage的对应，需要推进真实时间
            # shape: (not_ready_cnt,)
            self.time_idx[b_idx, p_idx] += step_time_required.astype(np.int64)     ### 每个job在每个machine上的开始时间，进行一次更新
            new_sub_time_idx[step_time_required] = 0                     ### 取值范围在0-11之间
            self.sub_time_idx[b_idx, p_idx] = new_sub_time_idx           


            machine_wait_steps = self.machine_wait_step[b_idx, p_idx, :]  
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.machine_wait_step[b_idx, p_idx, :] = machine_wait_steps  ### 

            new_machine_idx = self.sm_indexer.get_machine_index(p_idx, new_sub_time_idx)  ### pomo的作用在这里体现：提取不同machine顺序排列下的case，实现多样性，取值为0-11不重复                             
            self.machine_idx[b_idx, p_idx] = new_machine_idx 
            machine_ready = self.machine_wait_step[b_idx, p_idx, new_machine_idx] == 0  ### 等待时间为0意味着这个machine可以给新任务了
            # shape: (not_ready_cnt,)
            
            job_wait_steps = self.job_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.job_wait_step[b_idx, p_idx, :] = job_wait_steps  ###

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)                   
            # shape: (not_ready_cnt,)
            job_ready_1 = (self.job_location[b_idx, p_idx, :self.job_cnt] == new_stage_idx[:, None])   ### new_sub_time_idx 轮转，直到回到现有的job的stage
            # shape: (not_ready_cnt, job)
            job_ready_2 = (self.job_wait_step[b_idx, p_idx, :self.job_cnt] == 0)
            # shape: (not_ready_cnt, job)
            job_ready = (job_ready_1 & job_ready_2).any(axis=1)     ### 条件1；2同满足，说明这个job可以开始了；判断现在有没有任务可以开始
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready     ###  有机器和新任务派遣，说明这个case可以继续变化
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            p_idx = p_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
    
    
    def _get_info(self):
        if self.time_idx is None:
            return {'obj': np.zeros(self.batch_size, dtype=np.float32)}
        else:
            return {'obj': np.max(self.time_idx, axis=1)}  # (batch_size, pomo) -> (batch_size, )

    def get_prefix(self):
        return {
            'durations': self.job_durations.copy().repeat(24, axis=0).reshape(self.batch_size*24, -1).astype(np.int32) # (batch_size*24, job_cnt+1  * total_machine_cnt)
        }
    
    def _get_observation(self):
        if self.step_state is None:
            return None
        else:
            return {
                'machine_query': self.step_state.machine_query.copy().reshape(self.batch_size*24, self.job_cnt+1).astype(np.int32) # (batch_size*24, job_cnt+1)
            }
    
    
    def get_prefix_mask(self):
        return {
            'durations': self.step_state.prefix_mask.repeat(self.total_machine_cnt, axis=-1).reshape(self.batch_size*24, -1).astype(bool) # (batch_size*24, job_cnt+1  * total_machine_cnt)
        }


    def _is_terminated(self):
        ''' 由环境判断当前是否 terminated, 注意本方法不处理 truncated 信号 '''
        return (self.job_location[:, :, :self.job_cnt] == self.stage_cnt).all(axis=2)     # (batch_size, ) 全部结点访问后只能回到起点

    def is_same_episode(self, acts1:np.array, acts2:np.array):
        '''判断两条轨迹是否完全相同，由于是确定性环境，仅比较两条act序列是否相同即可'''
        return np.array_equal(acts1, acts2)

class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        assert env.machine_cnt_list == [4, 4, 4]
        assert env.pomo_size == 24

        machine_SUBindex_0 = np.array(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_1 = np.array(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_2 = np.array(list(itertools.permutations([0, 1, 2, 3])))
        self.machine_SUBindex_table = np.concatenate((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2), axis=1)
        # machine_SUBindex_table.shape: (pomo, total_machine)
        '''
        tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [0, 1, 3, 2, 0, 1, 3, 2, 0, 1, 3, 2],
        [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
        [0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1],
        [0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2],
        [0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1],
        [1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3],
        [1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2],
        [1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3],
        [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0],
        [1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2],
        [1, 3, 2, 0, 1, 3, 2, 0, 1, 3, 2, 0],
        [2, 0, 1, 3, 2, 0, 1, 3, 2, 0, 1, 3],
        [2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1],
        [2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3],
        [2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0],
        [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        [2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0],
        [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        [3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1],
        [3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2],
        [3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0],
        [3, 2, 0, 1, 3, 2, 0, 1, 3, 2, 0, 1],
        [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]])
        '''
        starting_SUBindex = [0, 4, 8]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        self.machine_table = np.concatenate((machine_order_0, machine_order_1, machine_order_2), axis=1)
        # machine_table.shape: (pomo, total_machine)
        '''
        tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
        [ 0,  1,  3,  2,  4,  5,  7,  6,  8,  9, 11, 10],
        [ 0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11],
        [ 0,  2,  3,  1,  4,  6,  7,  5,  8, 10, 11,  9],
        [ 0,  3,  1,  2,  4,  7,  5,  6,  8, 11,  9, 10],
        [ 0,  3,  2,  1,  4,  7,  6,  5,  8, 11, 10,  9],
        [ 1,  0,  2,  3,  5,  4,  6,  7,  9,  8, 10, 11],
        [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10],
        [ 1,  2,  0,  3,  5,  6,  4,  7,  9, 10,  8, 11],
        [ 1,  2,  3,  0,  5,  6,  7,  4,  9, 10, 11,  8],
        [ 1,  3,  0,  2,  5,  7,  4,  6,  9, 11,  8, 10],
        [ 1,  3,  2,  0,  5,  7,  6,  4,  9, 11, 10,  8],
        [ 2,  0,  1,  3,  6,  4,  5,  7, 10,  8,  9, 11],
        [ 2,  0,  3,  1,  6,  4,  7,  5, 10,  8, 11,  9],
        [ 2,  1,  0,  3,  6,  5,  4,  7, 10,  9,  8, 11],
        [ 2,  1,  3,  0,  6,  5,  7,  4, 10,  9, 11,  8],
        [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9],
        [ 2,  3,  1,  0,  6,  7,  5,  4, 10, 11,  9,  8],
        [ 3,  0,  1,  2,  7,  4,  5,  6, 11,  8,  9, 10],
        [ 3,  0,  2,  1,  7,  4,  6,  5, 11,  8, 10,  9],
        [ 3,  1,  0,  2,  7,  5,  4,  6, 11,  9,  8, 10],
        [ 3,  1,  2,  0,  7,  5,  6,  4, 11,  9, 10,  8],
        [ 3,  2,  0,  1,  7,  6,  4,  5, 11, 10,  8,  9],
        [ 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8]])
        '''
        # assert env.pomo_size == 1
        # self.machine_SUBindex_table = torch.tensor([[0,1,2,3,0,1,2,3,0,1,2,3]])
        # self.machine_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11]])

        self.stage_table = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, POMO_IDX, sub_time_idx):
        # POMO_IDX.shape: (batch, pomo)
        # sub_time_idx.shape: (batch, pomo)
        return self.machine_table[POMO_IDX, sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]
