"""Input types for DB1 of different modalities and tasks"""
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pickle
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, Union, List, Dict
from utils.utils_slover import calc_vrp_distance, calc_tsp_distance, calc_op_total, calc_pctsp_cost, knapsack_dp, calc_bp_total

@dataclass
class GatoInputBase:
    # 各种任务的通用属性
    position_id: Optional[torch.tensor]
    attention_mask: Optional[torch.tensor]
    loss_mask: Optional[torch.tensor]
    label: Optional[torch.tensor]

    def get_datasize(self):
        ''' 计算当前对象占用内存大小(单位为GB) '''
        gb = 0
        for e in [self.position_id, self.attention_mask, self.loss_mask, self.label]:
            if e is not None:
                gb += e.element_size() * e.nelement()
        gb /= 1024 * 1024 * 1024
        return gb

    def to(self, **kwargs):
        ''' 将对象中所有不为 None 的张量都移动到指定的设备上 '''
        for k, v in asdict(self).items():
            if v is not None and isinstance(v, torch.Tensor):
                setattr(self, k, v.to(**kwargs))

    def apply(self, fn, *args, **kwargs):
        ''' 对所有不为 None 的张量都应用 fn 方法, 并传入 args 和 **kwargs 参数 '''
        for k, v in asdict(self).items():
            if v is not None:
                setattr(self, k, fn(v, *args, **kwargs))

    def append(self, other):
        ''' 将另一个GatoInputBase对象的所有属性都添加到当前对象的相应属性中 '''
        assert type(self).__name__ == type(other).__name__
        for k, v in asdict(self).items():
            if v is None:
                continue
            other_v = getattr(other, k)
            # assert v.ndim == other_v.ndim == 2, (k, v.shape, other_v.shape)
            new_v = torch.cat([v, other_v], dim=0)
            # new_v = np.concatenate([v, other_v], axis=0)
            setattr(self, k, new_v)

    @staticmethod
    def merge_into_one(data2merge: List["GatoInputBase"]):
        ''' 将多个GatoInputBase对象合并成一个对象 '''
        t = data2merge[0]
        t.apply(lambda x: [x])
        for i in range(1, len(data2merge)):
            for k, v in asdict(data2merge[i]).items():
                if v is not None:
                    t_v = getattr(t, k)
                    t_v.append(v)
            # set(t, k, t_v)
        return t

@dataclass
class RLTaskInput(GatoInputBase):
    tensor_seq: Union[List, torch.tensor]
    seq_len: Union[List, torch.tensor]
    prefix_mask: Union[List, torch.tensor]
    rewards: Union[List, torch.tensor] = None

# --------------------------------- DataProblem ---------------------------------
@dataclass
class DataProblem:
    prefix_list: List = None
    problem_list: List = None
    answer_list: List = None

def split_dataproblem(problem:DataProblem, from_idx:int, to_idx:int):
    assert problem.answer_list is not None
    return DataProblem(
        prefix_list = None if problem.prefix_list is None else problem.prefix_list[from_idx:to_idx],
        problem_list = problem.problem_list[from_idx:to_idx],
        answer_list = problem.answer_list[from_idx:to_idx]
    )

def rnd_sub_dataproblem(problem:DataProblem, num:int, rng:None):
    assert problem.answer_list is not None and rng is not None
    indices = rng.choice(len(problem.answer_list), num, replace=num>len(problem.answer_list))
    return DataProblem(
        prefix_list = None if problem.prefix_list is None else [problem.prefix_list[i] for i in indices],
        problem_list = [problem.problem_list[i] for i in indices],
        answer_list = [problem.answer_list[i] for i in indices],
    )

def load_data(data_name:str, data_file_name:str, num_problem:int=0, num_episode:int=0, data_base_path:str=base_path, data_source:str=None):
    ''' 加载已生成的 .pkl 轨迹或问题数据文件 '''
    cop_type = data_file_name.split('_')[0]
    data_dir = f'{data_base_path}/data/used/{data_name}/{cop_type}' if data_source is None else \
                f'{data_base_path}/data/used/{data_name}/{cop_type}_{data_source}'
    with open(f'{data_dir}/{data_file_name}.pkl', 'rb') as f:
        data = pickle.load(f)  
    
    if data_file_name.endswith('problem'):
        return data if num_problem == 0 else split_dataproblem(data, 0, num_problem)
    return data if num_episode == 0 else data[:num_episode]

def get_best_obj_value(env_name, problems:DataProblem):
    obj_list = []
    if env_name in ['Env_BP_V1', 'Env_BP_V2']:
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            #value, _ = knapsack_dp(problem['capacity_left'].item(), problem['item_volumes'], problem['item_values'])
            value = calc_bp_total(problem['item_values'], answer)
            obj_list.append(value)
    elif env_name in ['Env_ATSP_V1', 'Env_ATSP_V2', 'Env_ATSP_V3', 'Env_TSP_V1', 'Env_TSP_V2', 'Env_TSP_V3', 'Env_TSP_V4']:
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            position = problem['position'].reshape((-1,2))
            obj_list.append(calc_tsp_distance(position, answer))
    elif env_name.startswith('Env_CVRP'):
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            pos = np.vstack((problem['pos_depot'][None,:], problem['pos_node'].reshape((-1, 2))))
            obj_list.append(calc_vrp_distance(pos, answer))
    elif env_name == 'Env_OP_V1':
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            obj_list.append(calc_op_total(problem['prize'], answer))
    elif env_name in ['Env_OP_V2', 'Env_OP_V3', 'Env_OP_V4']:
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            obj_list.append(calc_op_total(problem['prize'], np.array(answer)-1))
    elif env_name.startswith('Env_PCTSP'):
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            pos = np.vstack((problem['pos_depot'][None,:], problem['pos_node'].reshape((-1, 2))))
            obj_list.append(calc_pctsp_cost(pos, problem['penalty'], problem['prize'], answer[:-1]))
    elif env_name.startswith('Env_SPCTSP'):
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            pos = np.vstack((problem['pos_depot'][None,:], problem['pos_node'].reshape((-1, 2))))
            obj_list.append(calc_pctsp_cost(pos, problem['penalty'], problem['stoc_prize'], answer[:-1]))
    elif env_name.startswith('Env_FFSP'):
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            obj_list.append(27.31)
    elif env_name.startswith('Env_MIS'):
        for problem, answer in zip(problems.problem_list, problems.answer_list):
            obj_list.append(10.45645)
    else:
        raise NotImplementedError

    assert None not in obj_list
    return np.array(obj_list)

def load_data_and_check_quantity(file_names:List, env_name:str, num_problem:int=0, num_episode:int=0, check_ave_obj:bool=False, data_base_path=base_path, data_source:str=None):
    file_data = {}
    print('-'*60)
    for file_name in file_names:
        data_name = file_name[:-4]
        records = load_data(env_name, data_name, num_problem, num_episode, data_base_path, data_source)  
        if data_name.endswith('problem'):
            if check_ave_obj:
                obj_array = get_best_obj_value(env_name=f'Env_{env_name}', problems=records)
                print(f'{data_name:20}\t{len(records.answer_list)}\t{obj_array.mean()}')
            else:
                print(f'{data_name:20}\t{len(records.answer_list)}')
            
            obss_range = {k: [float('inf'), -float('inf')] for k in records.problem_list[0]}
            for problem in records.problem_list:
                for k, v in problem.items():
                    if v.min() < obss_range[k][0]:
                        obss_range[k][0] = v.min()
                    if v.max() > obss_range[k][1]:
                        obss_range[k][1] = v.max()
            for k, v in obss_range.items():
                print(f'\t{k:18}\tin range {v}')
        else:
            print(f'{data_name:20}\t{len(records)}')
            obss_range = {k: [float('inf'), -float('inf')] for k in records[0]['observations']}
            for data in records:
                for k, v in data['observations'].items():
                    if v.min() < obss_range[k][0]:
                        obss_range[k][0] = v.min()
                    if v.max() > obss_range[k][1]:
                        obss_range[k][1] = v.max()
            for k, v in obss_range.items():
                print(f'\t{k:18}\tin range {v}')
        file_data[data_name] = records
        print()
    print('-'*60)
    return file_data

# --------------------------------- RawData ---------------------------------
@dataclass
class RawData:
    seed_list: List = None
    problem_list: List = None
    answer_list: List = None
    cost_list: List = None

def split_raw_dataset(dataset:RawData, from_idx:int, to_idx:int):
    assert dataset.answer_list is not None
    return RawData(
        seed_list = dataset.seed_list,
        problem_list = dataset.problem_list[from_idx:to_idx],
        answer_list = dataset.answer_list[from_idx:to_idx],
        cost_list = dataset.cost_list[from_idx:to_idx],
    )

def merge_raw_datasets(dataset_list:List[RawData]):
    merged_dataset = RawData(seed_list=[], problem_list=[], answer_list=[], cost_list=[])
    for dataset in dataset_list:
        merged_dataset.seed_list.extend(dataset.seed_list)
        merged_dataset.problem_list.extend(dataset.problem_list)
        merged_dataset.answer_list.extend(dataset.answer_list)
        merged_dataset.cost_list.extend(dataset.cost_list)
    return merged_dataset

def load_raw_dataset(dataset_path):
    ex_data = RawData(seed_list=[], problem_list=[], answer_list=[], cost_list=[])
    if os.path.isfile(dataset_path) and os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            ex_data = pickle.load(f)
            if isinstance(ex_data, dict):
                ex_data = RawData(**ex_data)
    return ex_data

def load_all_raw_dataset(data_base_path, cop_type, scale, data_num:int=None, show_info:bool=False, data_source:str=None):
    ex_datasets=  {}
    for data_type in ['problem', 'train', 'prompt']:
        dataset_path = f'{data_base_path}/data/raw/{cop_type}/{cop_type}{scale}/{cop_type}{scale}_{data_type}.pkl' if data_source is None else \
                        f'{data_base_path}/data/raw/{cop_type}/{cop_type}{scale}_{data_source}/{cop_type}{scale}_{data_type}.pkl'
        dataset = load_raw_dataset(dataset_path)
        if data_num is not None:
            dataset = split_raw_dataset(dataset, 0, data_num)
        ex_datasets[data_type] = dataset

    if show_info:
        print('-'*50)
        for datatype in ['train', 'problem', 'prompt']:
            raw_dataset = ex_datasets[datatype]
            print(f'[{len(raw_dataset.answer_list)}] raw data for [{datatype}] loaded, ave cost [{np.mean(raw_dataset.cost_list)}], [{len(set(raw_dataset.seed_list))}] seeds used')
        print('-'*50)
    
    return ex_datasets

def load_and_merge_all_raw_dataset(data_base_path, cop_type, scale, data_num:int=None):
    ex_datasets = load_all_raw_dataset(data_base_path, cop_type, scale, data_num)
    dataset = merge_raw_datasets(list(ex_datasets.values()))
    return dataset

def merge_to_dataset(from_dataset_path_list, to_dataset_path, backup_path):
    ''' 将多个RawData类型子数据集的数据合并到目标数据集  '''
    # 加载可能存在的目标数据集
    target_data = load_raw_dataset(to_dataset_path)
    
    # 合并所有子数据集
    for sub_dataset_path in from_dataset_path_list:
        sub_dataset = load_raw_dataset(sub_dataset_path)
        assert all(elem not in target_data.seed_list for elem in sub_dataset.seed_list), "检查到重复的随机种子"
        target_data.seed_list.extend(sub_dataset.seed_list)
        target_data.problem_list.extend(sub_dataset.problem_list)
        target_data.answer_list.extend(sub_dataset.answer_list)
        target_data.cost_list.extend(sub_dataset.cost_list)
        os.remove(sub_dataset_path) 
    
    # 保存到目标路径和备份路径
    with open(to_dataset_path, 'wb') as f:
        pickle.dump(target_data, f)
    with open(backup_path, 'wb') as f:
        pickle.dump(target_data, f)

def to_d4rl_format(episodes:List, env_name:str) -> Dict:
    obss, acts, rwds, terminals = [], [], [], []
    prefixes, prefix_masks, use_prefix = [], [], episodes[0]['prefix'] is not None
    for epi in tqdm(episodes, desc=f'[{env_name}] To D4RL format'):
        obss.append(epi['observations'])
        acts.append(epi['actions'])
        rwds.append(epi['rewards'])
        terminals.append(epi['terminals'])
    
        if use_prefix:
            prefixes.append(epi['prefix'])
            prefix_masks.append(epi['prefix_masks'])

    return {
        'prefixes': None if prefixes == [] else prefixes, 
        'prefix_masks': None if prefix_masks == [] else prefix_masks,
        'observations': obss, 
        'actions': acts, 
        'rewards': rwds, 
        'terminals': terminals
    }