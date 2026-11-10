import os
import sys
matnet_base_path = os.path.abspath('/data1/xxx/MatNet/FFSP/FFSP_MatNet')
sys.path.append(matnet_base_path)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from gym import spaces
import numpy as np
import torch
from dataclasses import dataclass
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

    ### 新加的部分
    action_mask: torch.Tensor = None
    prefix_mask: torch.Tensor = None # true被屏蔽掉，false可选
    machine_wait_time: torch.Tensor = None
    job_location: torch.Tensor = None
    time_idx: torch.Tensor = None
    prefix_mask_machines: torch.Tensor = None #对于全部机器的mask
    machine_idx : torch.Tensor = None

class DDP_FFSP_V2(Env_COP):
    def __init__(self, render_mode="rgb_array", scale:int=20, batch_size:int=48, adj_range:bool=False):
        super().__init__(render_mode)
        self.batch_size = batch_size
        self.name = 'Env_FFSP_V2'
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
        self.act_num = self.total_machine_cnt


        # 观测空间中去掉 position 字段
        self.observation_space = spaces.Dict({
            'time_idx': spaces.Box(low=0, high=70, shape=(batch_size, 1), dtype=np.int32), 
            'machine_wait_time': spaces.Box(low=0, high=10, shape=(batch_size, self.total_machine_cnt), dtype=np.int32),
        })

        # 定义动作空间
        # self.action_space = spaces.MultiDiscrete([self.job_cnt+1] * batch_size)

        # 动作各维度对应的 token 取值范围
        # tsp 环境中动作 [0, num_nodes-1] 表示移动去对应的城市
        action_value_space = [np.arange(self.job_cnt+1, dtype=np.int32) for _ in range(batch_size)]  # 由于 action 都是自然数，token 取值范围和 action 相同 
        self.action_value_space = [action_value_space for _ in range(self.act_num)]        # action num == 总共的机器数量
        self.problem_best_obj = np.zeros(batch_size, dtype=np.float32)
        self.problem_random_obj = np.zeros(batch_size, dtype=np.float32)


        # Const @Load_Problem
        ####################################
        self.BATCH_IDX = None

        self.job_durations = None
        # shape: (batch, job+1, total_machine)
        # last job means NO_JOB ==> duration = 0

        # Dynamic
        ####################################
        self.time_idx = None
        # shape: (batch,)
        self.sub_time_idx = None  # 0 ~ total_machine_cnt-1
        # shape: (batch,)
        self.machine_idx = None  # must update according to sub_time_idx
        # shape: (batch,)

        self.schedule = None
        # shape: (batch, machine, job+1)
        # records start time of each job at each machine
        self.machine_wait_step = None
        # shape: (batch, machine)
        # How many time steps each machine needs to run, before it become available for a new job
        self.job_location = None
        # shape: (batch, job+1)
        # index of stage each job can be processed at. if stage_cnt, it means the job is finished (when job_wait_step=0)
        self.job_wait_step = None
        # shape: (batch, job+1)
        # how many time steps job needs to wait, before it is completed and ready to start at job_location
        self.finished = None  # is scheduling done?
        # shape: (batch,)

        # STEP-State
        ####################################
        self.step_state = None
    
    def _recover(self, problem_info, problem_obj, problem_idx_list):
        '''还原到评估问题的初始状态'''
        prefix_list, problem_list, answer_list = problem_info
        assert len(problem_idx_list) == len(answer_list) == self.batch_size     # 相同规模的tsp问题解张量尺寸一致，总是按完整batch或最大问题数量更新问题
        assert isinstance(problem_list[0]['durations'], np.ndarray)
        
        self.BATCH_IDX = np.arange(self.batch_size) # shape: (batch_size,)
        self.MACHINE_IDX = np.arange(self.total_machine_cnt) # shape: (total_machine_cnt,)

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

        self.time_idx = np.zeros((self.batch_size,), dtype=np.int64)      ###真实的时间time
        # shape: (batch,)
        self.sub_time_idx = np.zeros((self.batch_size,), dtype=np.int64)  ###用来表示在pomo设定下，现在处在第几个状态，取值为0-11
        # shape: (batch,)
        self.machine_idx = self.sm_indexer.get_machine_index(self.sub_time_idx)     ### 按照pomo的顺序，获取当前的machine的index
        # shape: (batch,) 

        # self.schedule: shape: (batch, machine, job+1)
        self.schedule = np.full(shape=(self.batch_size, self.total_machine_cnt, self.job_cnt + 1),
                                fill_value=-999999, dtype=np.int64)

        # self.machine_wait_step: shape: (batch, machine)
        self.machine_wait_step = np.zeros(shape=(self.batch_size, self.total_machine_cnt), dtype=np.int64)

        # self.job_location: shape: (batch, job+1)
        self.job_location = np.zeros(shape=(self.batch_size, self.job_cnt + 1), dtype=np.int64)

        # self.job_wait_step: shape: (batch, job+1)
        self.job_wait_step = np.zeros(shape=(self.batch_size, self.job_cnt + 1), dtype=np.int64)

        # self.finished: shape: (batch,)
        self.finished = np.full(shape=(self.batch_size), fill_value=False, dtype=bool)
        # shape: (batch,)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX)

        ### pre_step
        self._update_step_state()
        self.step_state.step_cnt = 0
        reward = None
        done = False

        return self._get_observation(), self._get_info()
    
    def _update_step_state(self):
        self.step_state.step_cnt += 1

        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)  ###储存stage值, 意味着当前时刻对于当前机器只决策这个stage的任务
        # shape: (batch,)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.sub_time_idx)  ###储存machine编号的真值
        # shape: (batch,)

        job_loc = self.job_location[:, :self.job_cnt]
        # shape: (batch, job)
        job_wait_t = self.job_wait_step[:, :self.job_cnt]
        # shape: (batch, job)

        job_in_stage = job_loc == self.step_state.stage_idx[:, None]
        # shape: (batch, job)
        job_not_waiting = (job_wait_t == 0)
        # shape: (batch, job)
        job_available = job_in_stage & job_not_waiting              ### job匹配所处于的stage，且有可以马上开始的job
        # shape: (batch, job)

        job_in_previous_stages = (job_loc < self.step_state.stage_idx[:, None]).any(axis=1)
        # shape: (batch,)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(axis=1)
        # shape: (batch,)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.finished  ### 允许进行跳过操作的条件 ==（还在上一个stage的任务 或 当前正在进行处理的子任务 或 已经完成的任务）
        # shape: (batch,)

       # self.step_state.job_ninf_mask: shape: (batch, job+1)
        self.step_state.job_ninf_mask = np.full(shape=(self.batch_size, self.job_cnt + 1),
                                                fill_value=float('-inf'))

        # job_enable: 拼接 job_available 和 wait_allowed
        job_enable = np.concatenate((job_available, wait_allowed[:, np.newaxis]), axis=1)
        # shape: (batc, job+1)
        self.step_state.job_ninf_mask[job_enable] = 0
        # shape: (batch, job+1)

        self.step_state.finished = self.finished
        # shape: (batch,)

        machine_idx = self.sm_indexer.get_machine_index(self.sub_time_idx)  ###在3个stage下的全局machine编号
        # shape: (batch,)

        prefix_mask = np.ones(shape=(self.batch_size, self.job_cnt+1), dtype=bool)
        # shape: (batch, job+1)
        prefix_mask[job_enable] = False
        self.step_state.prefix_mask = prefix_mask

        self.step_state.machine_wait_time = self.machine_wait_step.copy()
        self.step_state.job_location = self.job_location.copy()
        self.step_state.time_idx = self.time_idx.copy()

        ### 对于当前所有machine的mask
        # shape: (batch, pomo, machine, job+1)
        prefix_mask_machines = np.ones((self.batch_size, self.total_machine_cnt, self.job_cnt+1), dtype=bool) # (batch, act_num, job+1)
        job_in_stage_machines = job_loc[:, None, :] == self.sm_indexer.stage_table[None, :, None] #(batch, act_num, job)
        job_available_machines = job_in_stage_machines & job_not_waiting[:,None,:] #(batch, act_num, job)
        job_in_previous_stages_machines = (job_loc[:, None, :] < self.sm_indexer.stage_table[None, :, None]).any(axis=2) #(batch, act_num)
        job_in_after_stages_machines = (job_loc[:, None, :] > self.sm_indexer.stage_table[None, :, None]).all(axis=2) #(batch, act_num)
        job_waiting_in_stage_machines = (job_in_stage_machines & (job_wait_t[:,None,:] > 0)).any(axis=2) #(batch, act_num)
        wait_allowed_machines = job_in_previous_stages_machines + job_in_after_stages_machines + job_waiting_in_stage_machines + self.finished[:,None] #(batch, act_num)
        job_enable_machines = np.concatenate((job_available_machines, wait_allowed_machines[:, :, None]), axis=2) #(batch, act_num, job+1)
        prefix_mask_machines[job_enable_machines] = False # (batch, act_num, job+1)
        self.step_state.action_mask = prefix_mask_machines.copy() # (batch, act_num, job+1)  当前是对于action的mask，还需要处理到对prefix token的mask
        
        #制作prefix mask放到专门函数去


    
    def _pred_qulity(self):
        ''' 基于 real_answer 和 model_answer 计算模型给出解的质量，取值应当在 [0, 1] '''
        rnd_obj = self.problem_random_obj
        best_obj = self.problem_best_obj if self.problem_best_obj is not None else eval_main(self.job_durations[:, :-1, :])[1]

        job_durations_perm = np.transpose(self.job_durations, (0, 2, 1))  ### 转置，方便计算
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, :, :]  ### 记录了每个机器，在每个job上的结束时间
        # shape: (batch, machine, job+1)

        end_time_max = end_schedule[:, :, :self.job_cnt].max(axis=2)
        # shape: (batch, machine)
        end_time_max = end_time_max.max(axis=1)
        # shape: (batch,)

        model_obj = end_time_max

        #assert best_obj <= model_obj # LKH 算法不一定能给出最优解
        qulity_AM = 1 - (model_obj - best_obj)/best_obj
        qulity_DB1 = (model_obj - rnd_obj)/(best_obj - rnd_obj)
        qulity = {'AM':qulity_AM, 'DB1':qulity_DB1}
        return qulity
    
    def _get_action_value_space_at_timestep(self):
        job_enable = (self.step_state.action_mask == False) # (batch, act_num, job+1)
        action_value_space = []
        for i in range(self.act_num):
            mac_job_enable = job_enable[:, i, :]  # shape: (batch, job+1)
            idx, job = np.where(mac_job_enable)
            dim_action_value_space = [job[idx==j].astype(np.int32) for j in range(self.batch_size)]  # length = batch_size
            action_value_space.append(dim_action_value_space)    
        return action_value_space   

    def get_action_value_space(self, hard_action_constraint=False, generated_actions:np.ndarray=None):
        ''' 根据当前状态和约束条件生成可行动作范围 action_value_space '''
        if generated_actions.size == 0:
            # generated_actions: ()
            self.action_value_space = self._get_action_value_space_at_timestep()    # act_num * [batch_size * [available job idx]]
            for space in self.action_value_space:
                assert len(space) > 0
        else:
            # generated_actions: (batch_size, generated_act_dim_num)
            for p in range(self.batch_size):
                # 获取问题 p 已经生成的动作（不含20）
                generated_dim_act_for_p = generated_actions[p]
                generated_act = [v for v in generated_dim_act_for_p if v != 20]

                # 对于问题 p 的下一个动作维度，把已经生成过的动作排除
                next_dim = len(generated_dim_act_for_p)
                next_dim_action_value_space = self.action_value_space[next_dim][p]
                next_dim_action_value_space = next_dim_action_value_space[~np.isin(next_dim_action_value_space, generated_act)]
                assert len(next_dim_action_value_space) > 0
                self.action_value_space[next_dim][p] = next_dim_action_value_space

        return self.action_value_space  #act_num * [ batch_size * [available job idx]]
    
    def step(self, job_idx):
        # job_idx.shape: (batch, act_num)

        ### 检查外部给入的动作是否合法
        for i, job_idx_p in enumerate(job_idx):
            assert np.sum(self.step_state.action_mask[i, self.MACHINE_IDX, job_idx_p]) == 0
            temp = [v for v in job_idx_p if v != 20]
            assert len(set(temp)) == len(temp)
        
        self.schedule[self.BATCH_IDX[:, None], self.MACHINE_IDX[None, :], job_idx] = self.time_idx[:, None] ### 记录每个job在每个machine上的开始时间
        # shape: (batch, machine, job+1)
        job_length = self.job_durations[self.BATCH_IDX[:, None], job_idx, self.MACHINE_IDX[None, :]]
        # shape: (batch, machine)     获取各个任务在当前stage机器上的工作时间
        self.machine_wait_step[self.BATCH_IDX[:,None], self.MACHINE_IDX[None,:]] += job_length
        # shape: (batch, machine)     按照机器，更新各个机器对于当前任务剩余的工作时间
        for i in range(self.act_num):
            self.job_location[self.BATCH_IDX, job_idx[:, i]] += 1
            # shape: (batch, job+1)       依次遍历本轮被选的任务，更新各任务目前进行到第几个stage （正在进行）
            self.job_wait_step[self.BATCH_IDX, job_idx[:, i]] = job_length[:, i]
            # shape: (batch, job+1)       依次遍历本轮被选的任务，更新各任务需要新等待的时间 （正在进行）

        self.finished = (self.job_location[:, :self.job_cnt] == self.stage_cnt).all(axis=1)  #全部开始第三阶段即结束
        # shape: (batch)              

        ####################################
        done = self.finished.all()

        if done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._move_to_next_machine()
            self._update_step_state()

        if done:
            reward = self._pred_qulity()  
            # shape: (batch,)
        else:
            reward = {
            'AM': np.zeros(self.batch_size, dtype=np.float32), 
            'DB1': np.zeros(self.batch_size, dtype=np.float32),       
        }

        terminated = self.finished
        truncated = np.zeros(self.batch_size, dtype=bool)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_to_next_machine(self):

        # b_idx: shape: (batch,) == (not_ready_cnt,)
        b_idx = self.BATCH_IDX.flatten()
        
        has_step_time = np.zeros(self.batch_size, dtype=bool)
        # p_idx: shape: (batch,) == (not_ready_cnt,)

        # ready: shape: (batch,) == (not_ready_cnt,)
        ready = self.finished.flatten()

        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)

        cnt = 0
        while ~ready.all():
            cnt+=1
            print('cnt in state loop:', cnt)

            new_sub_time_idx = self.sub_time_idx[b_idx] + 1       ###  遍历各个stage的机器，哪个可以继续选；需要往前推演状态，直到machine和job双双ready的状态
            # shape: (not_ready_cnt,)   
            step_time_required = new_sub_time_idx == self.total_machine_cnt  ### 所有的状态都不满足，意味着当前时刻，没有找到合适的机器/stage的对应，需要推进真实时间
            # shape: (not_ready_cnt,)
            self.time_idx[b_idx] += step_time_required.astype(np.int64)     ### 每个job在每个machine上的开始时间，进行一次更新
            new_sub_time_idx[step_time_required] = 0                     ### 取值范围在0-11之间
            self.sub_time_idx[b_idx] = new_sub_time_idx   

            has_step_time[b_idx] = has_step_time[b_idx] | step_time_required  ###是否已经time idx有变化       


            machine_wait_steps = self.machine_wait_step[b_idx, :]  
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.machine_wait_step[b_idx, :] = machine_wait_steps  ### 

            new_machine_idx = self.sm_indexer.get_machine_index(new_sub_time_idx)  ### pomo的作用在这里体现：提取不同machine顺序排列下的case，实现多样性，取值为0-11不重复                             
            self.machine_idx[b_idx] = new_machine_idx 
            machine_ready = self.machine_wait_step[b_idx, new_machine_idx] == 0  ### 等待时间为0意味着这个machine可以给新任务了
            # shape: (not_ready_cnt,)
            
            job_wait_steps = self.job_wait_step[b_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.job_wait_step[b_idx, :] = job_wait_steps  ###

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)                   
            # shape: (not_ready_cnt,)
            job_ready_1 = (self.job_location[b_idx, :self.job_cnt] == new_stage_idx[:, None])   ### new_sub_time_idx 轮转，直到回到现有的job的stage
            # shape: (not_ready_cnt, job)
            job_ready_2 = (self.job_wait_step[b_idx, :self.job_cnt] == 0)
            # shape: (not_ready_cnt, job)
            job_ready = (job_ready_1 & job_ready_2).any(axis=1)     ### 条件1；2同满足，说明这个job可以开始了；判断现在有没有任务可以开始
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready & has_step_time[b_idx]     ###  有机器和新任务派遣，时间有所增加，可以往后生成新的一组action
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
    
    
    def _get_info(self):
        if self.time_idx is None:
            return {'obj': np.zeros(self.batch_size, dtype=np.float32)}
        else:
            return {'obj': self.time_idx}  # (batch_size) -> (batch_size, )

    def get_prefix(self):
        if self.job_durations.shape[1] == self.job_cnt+1 and self.job_durations.shape[2] == self.total_machine_cnt:
            durations = self.job_durations.copy()  # (batch, job+1, total_machine)
        elif self.job_durations.shape[1] == self.total_machine_cnt and self.job_durations.shape[2] == self.job_cnt+1:
            durations = self.job_durations.copy().transpose(0, 2, 1)  # (batch, job+1, total_machine)
        else:
            raise ValueError('job_durations shape error')
        return {
            'durations': durations.reshape(self.batch_size, -1).astype(np.int32) # (batch_size, job_cnt+1  * total_machine_cnt)
        }
    
    def _get_observation(self):
        if self.step_state is None:
            return None
        else:
            return {
                'machine_wait_time': self.step_state.machine_wait_time.copy().astype(np.int32), # (batch_size, machine)
                'time_idx': self.step_state.time_idx.copy().astype(np.int32), # (batch_size, 1)
            }
    
    def get_prefix_mask(self):
        prefix_mask_temp = np.ones(shape=(self.batch_size, self.act_num, self.total_machine_cnt, self.job_cnt + 1), dtype=bool) # (batch, act_num, machine, job+1)
        prefix_mask_temp[:, 0:4, 0:4, :] = False
        prefix_mask_temp[:, 4:8, 4:8, :] = False
        prefix_mask_temp[:, 8:12, 8:12, :] = False
        prefix_mask = self.step_state.action_mask[:, :, None, :] | prefix_mask_temp  # (batch, act_num, machine, job+1)
        prefix_mask = prefix_mask.transpose(0, 1, 3, 2)  # (batch, act_num, job+1, machine)
        return {
            'durations': prefix_mask.reshape(self.batch_size, self.act_num, -1).astype(bool) # (batch, job_cnt+1  * total_machine_cnt)
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

        machine_SUBindex_0 = np.array([0, 1, 2, 3])
        machine_SUBindex_1 = np.array([0, 1, 2, 3])
        machine_SUBindex_2 = np.array([0, 1, 2, 3])
        self.machine_SUBindex_table = np.concatenate((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2), axis=0)
        # machine_SUBindex_table.shape: (total_machine)
        '''
        tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        '''
        starting_SUBindex = [0, 4, 8]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        self.machine_table = np.concatenate((machine_order_0, machine_order_1, machine_order_2), axis=0)
        # machine_table.shape: (total_machine)
        '''
        tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        '''
        # assert env.pomo_size == 1
        # self.machine_SUBindex_table = torch.tensor([[0,1,2,3,0,1,2,3,0,1,2,3]])
        # self.machine_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11]])

        self.stage_table = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, sub_time_idx):
        # sub_time_idx.shape: (batch,)
        return self.machine_table[sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, sub_time_idx):
        return self.machine_SUBindex_table[sub_time_idx]
