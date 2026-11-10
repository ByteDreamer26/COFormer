import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

import numpy as np
import time
from typing import Union
from tqdm import tqdm
from utils.utils import set_seed
from utils.utils_data import split_dataproblem, get_best_obj_value
from environment.used.BaseEnv_COP import DataProblem
from environment.wrapper import LMPromptEnv
from environment.DDP_wrapper import DDP_LMPromptEnv

class RawProblemLoader():
    def __init__(self, problem_dataset:DataProblem) -> None:
        super().__init__()
        self.problem_dataset = problem_dataset
        self.len = len(problem_dataset.answer_list)
        self.cnt = 0
        
    def __len__(self):
        return self.len
    
    def reset(self):
        self.cnt = 0

    def get_problem(self, num=1):
        prefix_list, problem_list, answer_list = [], [], []
        if self.cnt >= self.len:
            return ([], [], []), np.array([])
        
        cnt_to = min(self.cnt+num, self.len)
        prefix_list = self.problem_dataset.prefix_list[self.cnt: cnt_to] if self.problem_dataset.prefix_list is not None else None
        problem_list = self.problem_dataset.problem_list[self.cnt: cnt_to]
        answer_list = self.problem_dataset.answer_list[self.cnt: cnt_to]
        
        problem_idx_array = np.arange(self.cnt, cnt_to)
        self.cnt = cnt_to
        problem_info_ddp = (prefix_list, problem_list, answer_list)
        problem_info_list = [(None, problem, answer) for problem, answer in zip(problem_list, answer_list)] if self.problem_dataset.prefix_list is None else \
                            [(prefix, problem, answer) for prefix, problem, answer in zip(prefix_list, problem_list, answer_list)]
        return problem_info_list, problem_info_ddp, problem_idx_array

def get_obj_value(problems:DataProblem, env_basic:LMPromptEnv, disable_tqdm=False, is_rl=False):
    rng = np.random.default_rng(42)
    raw_problem_loader = RawProblemLoader(problems)
    problem_dataset_size = len(raw_problem_loader)
    raw_problem_loader.reset()

    # init obj value
    if env_basic.env_name.startswith('Env_FFSP'):
        best_obj_array = np.ones(problem_dataset_size, dtype=np.float32) * 27.31    # (problem_dataset_size, ) init random obj value of all problems
    else:
        best_obj_array = get_best_obj_value(env_basic.env_name, problems)           # (problem_dataset_size, ) best obj value of all problems
    random_obj_array = np.zeros(problem_dataset_size, dtype=np.float32)             # (problem_dataset_size, ) init random obj value of all problems

    # 初始化第一个问题
    start_time = time.time()
    problem_info_list, _, _ = raw_problem_loader.get_problem(num=1)
    env_basic.reset(options={
        'problem_info':problem_info_list[0],
        'problem_obj':(best_obj_array[0], 0),
    })
    
    # 开始交互
    slove_cnt = 0
    obj_list = []
    with tqdm(total=problem_dataset_size, desc=f'Rollout random policy on {env_basic.env_name}', disable=disable_tqdm) as pbar:
        while True:
            # 随机生成动作并进行环境转移
            action_space = env_basic.env.get_action_value_space(hard_action_constraint=True)[0]
            act = rng.choice(action_space)
            _, _, terminated, truncated, info = env_basic.step(act)
            assert not truncated

            # 处理轨迹结束情况
            if terminated:
                obj_list.append(info['obj'])
                slove_cnt += 1
                
                # 更新进度条
                pbar_info = {
                    'obj':  f'{0 if len(obj_list) == 0 else np.mean(obj_list):.2f}',
                    'time': f'{0 if len(obj_list) == 0 else (time.time() - start_time)/len(obj_list):.4f}',
                }
                pbar.set_postfix(pbar_info)
                pbar.update(1)
                
                if slove_cnt >= problem_dataset_size:
                    break
                
                # 加载新问题
                problem_info_list, _, problem_idx_array = raw_problem_loader.get_problem(num=1)
                assert problem_idx_array.item() == slove_cnt     
                env_basic.reset(options={
                    'problem_info':problem_info_list[0],
                    'problem_obj':(best_obj_array[slove_cnt], 0),
                })                

    random_obj_array = np.array(obj_list)
    assert len(obj_list) == problem_dataset_size
    assert (random_obj_array!=0).sum() == problem_dataset_size

    if not is_rl:
        # 启发式求解器不一定得到最优解，要求以99%的概率保证best obj优于rnd obj
        if env_basic.env_name.startswith('Env_ATSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_TSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_BP'):
            assert (best_obj_array >= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_PCTSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_SPCTSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_CVRP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_OP'):
            assert (best_obj_array >= random_obj_array).mean() > 0.99
        elif env_basic.env_name.startswith('Env_OBP3D'):
            pass
        elif env_basic.env_name.startswith('Env_FFSP'):
            pass
        elif env_basic.env_name.startswith('Env_MIS'):
            pass
        else:
            raise NotImplementedError
    return best_obj_array, random_obj_array

def get_obj_value_ddp(problems:DataProblem, env_ddp:DDP_LMPromptEnv, disable_tqdm=False, is_rl=False):
    rng = np.random.default_rng(42)
    batch_size = env_ddp.env.batch_size
    act_dim = env_ddp.dataset.act_dim

    # 构造 RawProblemLoader
    raw_problem_loader = RawProblemLoader(problems)
    problem_dataset_size = len(raw_problem_loader)
    raw_problem_loader.reset()

    # init obj value
    if is_rl:
        best_obj_array = np.ones(problem_dataset_size, dtype=np.float32)        # (problem_dataset_size, ) init random obj value of all problems
    else:
        if env_ddp.env_name.startswith('Env_FFSP'):
            best_obj_array = np.ones(problem_dataset_size, dtype=np.float32) * 27.31 # (problem_dataset_size, ) init random obj value of all problems
        else:
            best_obj_array = get_best_obj_value(env_ddp.env_name, problems)         # (problem_dataset_size, ) best obj value of all problems
    random_obj_array = np.zeros(problem_dataset_size, dtype=np.float32)         # (problem_dataset_size, ) init random obj value of all problems

    # 开始评估 rnd obj value，初始化第一批问题
    start_time = time.time()
    _, problem_info_ddp, problem_idx_array = raw_problem_loader.get_problem(num=batch_size)
    problem_num = len(problem_info_ddp[2])
    _, ddp_info = env_ddp.reset(options={
        'problem_info':problem_info_ddp, 
        'problem_idx':list(range(problem_num)),
        'problem_obj':(best_obj_array[problem_idx_array], np.zeros(problem_num, np.float32)),   # 只需要info['obj']不需要reward，problem obj不重要
    })
    
    # 开始交互
    slove_cnt = 0
    done_idx_list = []  # 维护当前timestep时已经结束的问题索引，仅用于并行环境 & 串行环境一致性检查
    with tqdm(total=problem_dataset_size, desc=f'Rollout random policy on {env_ddp.env_name}', disable=disable_tqdm) as pbar:
        while True:
            # 随机生成一批动作
            batch_action = []
            for d in range(act_dim):
                generated_actions = np.array(batch_action).T
                ddp_action_space = env_ddp.env.get_action_value_space(True, generated_actions)[d]       # [batch_size * [available job idx]]
                actions = [0 if len(ddp_action_space[i])==0 else rng.choice(ddp_action_space[i]) for i in range(len(ddp_action_space))]
                batch_action.append(actions)

            # 批量环境转移
            batch_action = np.array(batch_action).T.squeeze()   # (batch_size, act_dim) or (batch_size, ) when act_dim==1
            batch_action = np.expand_dims(batch_action, axis=0) if batch_action.ndim == 0 else batch_action
            _, _, ddp_terminated, ddp_truncated, ddp_info = env_ddp.step(batch_action)
            assert ddp_truncated.sum() == 0
            
            # 获取终止轨迹的 obj value
            done_idx = []   # 环境转移后终止的batch内问题索引
            done_idx.extend(np.where(ddp_terminated)[0].tolist())
            done_idx.extend(np.where(ddp_truncated)[0].tolist())
            if len(done_idx) != 0:
                done_problem_idx = problem_idx_array[done_idx]                  # 环境转移后终止的全局问题索引
                assert random_obj_array[done_problem_idx].sum() == 0            # 检查是否有重复终止的问题
                random_obj_array[done_problem_idx] = ddp_info['obj'][done_idx]  # 保存 random policy obj value
                done_idx_list.extend(done_idx)

            # 更新进度条
            info = {
                'obj':  f'{0 if (random_obj_array!=0).sum() == 0 else random_obj_array[random_obj_array!=0].mean():.2f}',
                'time': f'{0 if (random_obj_array!=0).sum() == 0 else (time.time() - start_time)/(random_obj_array!=0).sum():.4f}',
            }
            pbar.set_postfix(info)
            pbar.update(len(done_idx))

            # 等待整个 batch problem 评估完再批量更新问题
            if len(done_idx_list) == problem_num:
                slove_cnt += problem_num
                if slove_cnt >= problem_dataset_size:
                    break
                
                # 加载一批新问题，数量为batch_size，直到问题数量不足为止
                done_idx_list = []
                _, problem_info_ddp, problem_idx_array = raw_problem_loader.get_problem(num=batch_size)            
                problem_num = len(problem_info_ddp[2])
                if problem_num > 0:
                    _, ddp_info = env_ddp.reset(options={
                        'problem_info':problem_info_ddp, 
                        'problem_idx':list(range(problem_num)),
                        'problem_obj':(best_obj_array[problem_idx_array], np.zeros(problem_num, np.float32)),   # 只需要info['obj']不需要reward，problem obj不重要
                    })

    assert (random_obj_array!=0).sum() == problem_dataset_size
    if not is_rl:
        # 启发式求解器不一定得到最优解，要求以99%的概率保证best obj优于rnd obj
        if env_ddp.env_name.startswith('Env_ATSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_TSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_BP'):
            assert (best_obj_array >= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_PCTSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_SPCTSP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_CVRP'):
            assert (best_obj_array <= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_OP'):
            assert (best_obj_array >= random_obj_array).mean() > 0.99
        elif env_ddp.env_name.startswith('Env_OBP3D'):
            pass
        elif env_ddp.env_name.startswith('Env_FFSP'):
            pass
        elif env_ddp.env_name.startswith('Env_MIS'):
            pass
        else:
            raise NotImplementedError
    return best_obj_array, random_obj_array
    
class ProblemLoader():
    def __init__(self, problem_dataset:DataProblem, env:Union[DDP_LMPromptEnv, LMPromptEnv], is_rl=False) -> None:
        super().__init__()
        self.rank = int(os.environ.get("RANK", default='0'))
        self.world_size = int(os.environ.get("WORLD_SIZE", default='1'))

        self.problem_dataset = problem_dataset
        self.problem_num = len(problem_dataset.answer_list)
        self.problem_num_per_rank = int(self.problem_num/self.world_size)
        self.cnt = self.rank * self.problem_num_per_rank

        assert self.problem_num % self.world_size == 0
        assert self.problem_dataset.problem_list is not None
        assert self.problem_dataset.answer_list is not None
        
        self.best_obj_array = np.zeros(self.problem_num, dtype=np.float32)
        self.random_obj_array = np.zeros(self.problem_num, dtype=np.float32)
        
        from_idx = self.rank*self.problem_num_per_rank
        to_idx = (self.rank+1)*self.problem_num_per_rank
        problmes = split_dataproblem(problem_dataset, from_idx, to_idx)
        if isinstance(env, DDP_LMPromptEnv):
            disable_tqdm = not env.env_name.startswith('Env_FFSP')
            self.best_obj_array[from_idx:to_idx], self.random_obj_array[from_idx:to_idx] = get_obj_value_ddp(problmes, env, disable_tqdm=disable_tqdm, is_rl=is_rl)
        elif isinstance(env, LMPromptEnv):
            self.best_obj_array[from_idx:to_idx], self.random_obj_array[from_idx:to_idx] = get_obj_value(problmes, env, disable_tqdm=True, is_rl=is_rl)
        else:
            raise NotImplementedError

    def __len__(self):
        return self.problem_num_per_rank
    
    def reset(self):
        self.cnt = self.rank * self.problem_num_per_rank

    def get_problem(self, num=1):
        end_idx = min(self.cnt+num, (self.rank+1)*self.problem_num_per_rank)
        prefix_list = None if self.problem_dataset.prefix_list is None else self.problem_dataset.prefix_list[self.cnt: end_idx]
        problem_list = self.problem_dataset.problem_list[self.cnt: end_idx]
        answer_list = self.problem_dataset.answer_list[self.cnt: end_idx]
        best_obj_array = self.best_obj_array[self.cnt: end_idx]
        random_obj_array = self.random_obj_array[self.cnt: end_idx]
        self.cnt = end_idx

        problem_info_ddp = (prefix_list, problem_list, answer_list)
        problem_obj_ddp = (best_obj_array, random_obj_array)
        return problem_info_ddp, problem_obj_ddp