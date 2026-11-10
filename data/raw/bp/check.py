import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import random
import numpy as np
from tqdm import tqdm
from utils.utils_slover import knapsack_dp
from utils.utils_data import merge_raw_datasets, load_all_raw_dataset
from data.raw.bp import MAX_CAPACITY

def check_data(data_base_path, scale, load_data_num=None, check_data_num=5000, data_source=None):
    ''' 检查生成的数据是否存在问题 '''
    # 加载所有源数据
    ex_datasets = load_all_raw_dataset(data_base_path, 'bp', scale, load_data_num, data_source=data_source)
    merged_dataset = merge_raw_datasets(list(ex_datasets.values()))
    
    assert len(merged_dataset.problem_list) == len(merged_dataset.answer_list) == len(merged_dataset.cost_list), '数据长度不匹配'
    assert len(set(merged_dataset.seed_list)) == len(merged_dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    volumes = np.array([problem['item_volumes'] for problem in merged_dataset.problem_list])           # (data_num, node_num, 2)
    capacities = np.array([problem['capacity_left'] for problem in merged_dataset.problem_list])       # (data_num, node_num, 2)
    assert volumes.shape[0] == np.unique(volumes, axis=0).shape[0], '存在完全相同的问题'
    assert np.all(capacities == MAX_CAPACITY[scale]), '存在总容量有误的问题'
    assert volumes.min() > 0, '存在0体积物品'

    # 打印数据集信息
    dataset_train, dataset_prompt, dataset_problem = ex_datasets['train'], ex_datasets['prompt'], ex_datasets['problem']
    print('-'*80)
    print(f'There are [{len(merged_dataset.answer_list)}] raw data generated with [{len(merged_dataset.seed_list)}] different random seeds, ave cost = [{np.mean(merged_dataset.cost_list):.3f}]')
    print(f'\tTrain dataset:    [{len(dataset_train.answer_list)}] raw data with [{len(dataset_train.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_train.cost_list):.3f}]')
    print(f'\tProblem dataset:  [{len(dataset_problem.answer_list)}] raw data with [{len(dataset_problem.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_problem.cost_list):.3f}]')
    print(f'\tPrompt dataset:   [{len(dataset_prompt.answer_list)}] raw data with [{len(dataset_prompt.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_prompt.cost_list):.3f}]')
    print('-'*80)

    # 随机检查 check_data_num 个数据
    if check_data_num > 0:
        dataset_size = len(merged_dataset.answer_list)
        check_idx = random.sample(range(dataset_size), min(dataset_size, check_data_num))
        for idx in tqdm(check_idx, desc=f"Check data validity"):
            capacity = merged_dataset.problem_list[idx]['capacity_left']
            volumes = merged_dataset.problem_list[idx]['item_volumes']
            values = merged_dataset.problem_list[idx]['item_values']
            data_answer = merged_dataset.answer_list[idx]
            data_cost = merged_dataset.cost_list[idx]
            cost, answer = knapsack_dp(capacity.item(), volumes, values) 
            assert abs(cost - data_cost) < 1e-5
            assert answer == data_answer

if __name__ == "__main__":
    #data_base_path = base_path
    data_base_path = '/data1/autoco/unico'    
    scale = 5
    source_data = check_data(data_base_path, scale, load_data_num=None)
    


    

    