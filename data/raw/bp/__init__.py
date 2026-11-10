import pickle
import random
import numpy as np
from tqdm import tqdm
from utils.utils_slover import knapsack_dp
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset

MAX_ITEM_VOLUME = {
    5: 10,
    10: 20,
    20: 20,
    50: 20,
    100: 20,
}

MAX_CAPACITY = {
    5: 10,
    10: 20,
    20: 30,
    50: 75,
    100: 150
}

def raw_data_to_problem(raw_data:RawData, num:int=0):
    num = len(raw_data.answer_list) if num == 0 else min(num, len(raw_data.answer_list))
    problems = DataProblem(problem_list=raw_data.problem_list[:num], answer_list=raw_data.answer_list[:num])
    # with tqdm(total=num, desc=f'Checking problem') as pbar:
    #     for problem, _, value in zip(problems.problem_list, problems.answer_list, raw_data.cost_list[:num]):
    #         real_value, _ = knapsack_dp(problem['capacity_left'].item(), problem['item_volumes'], problem['item_values'])
    #         assert real_value == value
    #         pbar.update()
    return problems

def check_bp_data(data_base_path, scale, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'bp', scale)
    assert len(dataset.problem_list) == len(dataset.answer_list) == len(dataset.cost_list), '数据长度不匹配'
    assert len(set(dataset.seed_list)) == len(dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    values = np.array([problem['item_values'] for problem in dataset.problem_list])         # (data_num, scale)
    volumes = np.array([problem['item_volumes'] for problem in dataset.problem_list])       # (data_num, scale)
    assert values.shape[0] == np.unique(values, axis=0).shape[0] and volumes.shape[0] == np.unique(volumes, axis=0).shape[0], '存在完全相同的问题'

    if show_info:
        print('-'*80+f'\nThere are [{len(dataset.answer_list)}] raw data generated with [{len(dataset.seed_list)}] different random seeds, ave cost = [{np.mean(dataset.cost_list):.3f}]\n'+'-'*80)

    # 随机检查 check_data_num 个数据
    if check_data_num > 0:
        dataset_size = len(dataset.answer_list)
        check_idx = random.sample(range(dataset_size), min(dataset_size, check_data_num))
        for idx in tqdm(check_idx, desc=f"Check data validity"):
            volumes = dataset.problem_list[idx]['item_volumes']
            values = dataset.problem_list[idx]['item_values']
            capacity = dataset.problem_list[idx]['capacity_left']
            data_value, data_answer = dataset.cost_list[idx], dataset.answer_list[idx]
            value, real_answer = knapsack_dp(capacity.item(), volumes, values)        
            assert abs(value - data_value) < 1e-4
            assert real_answer == data_answer

def gen_raw_bp(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock=None):
    # 生成问题，求解并写入 dataset
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象
    with tqdm(total=data_num_per_process, desc=f'Gen 01BP data with seed {seed}', position=tqdm_pos) as pbar:
        for _ in range(data_num_per_process):
            values = rng.randint(2, MAX_ITEM_VOLUME[scale], scale)              # 随机生成物品价值
            volumes = values + 0.5*(values * rng.choice([-1, 1], size=(scale,)))   # 物品体积基于价值生成，总体呈正相关   
            volumes = volumes.astype(np.int32)
            assert volumes.min() > 0        
            capacity = np.array(MAX_CAPACITY[scale], dtype=np.int32)
            value, real_answer = knapsack_dp(capacity.item(), volumes, values)
            dataset.problem_list.append({'item_values': values, 'item_volumes': volumes, 'capacity_left': capacity})
            dataset.answer_list.append(real_answer)
            dataset.cost_list.append(value)
            pbar.update()

    # 保存该进程生成的原始数据
    dataset_path = f'{data_base_path}/data/raw/bp/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def get_rl_bp(rng, batch_size, scale):
    dataproblem = DataProblem(answer_list=[], problem_list=[])
    for i in range(batch_size):    
        values = rng.integers(2, MAX_ITEM_VOLUME[scale], scale)              # 随机生成物品价值
        volumes = values + 0.5*(values * rng.choice([-1, 1], size=(scale,)))   # 物品体积基于价值生成，总体呈正相关   
        volumes = volumes.astype(np.int32)
        assert volumes.min() > 0        
        capacity = np.array(MAX_CAPACITY[scale], dtype=np.int32)

        dataproblem.problem_list.append({'item_values': values, 'item_volumes': volumes, 'capacity_left': capacity})
        dataproblem.answer_list.append([0])

    return dataproblem