import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
from utils.utils_slover import PCTSP_ILS
from utils.utils_data import merge_raw_datasets, load_all_raw_dataset

def check_data(data_base_path, scale, load_data_num=None, check_data_num=5000, data_source=None):
    ''' 检查生成的数据是否存在问题 '''
    # 加载所有源数据
    ex_datasets = load_all_raw_dataset(data_base_path, 'spctsp', scale, load_data_num, data_source=data_source)
    merged_dataset = merge_raw_datasets(list(ex_datasets.values()))

    assert len(merged_dataset.problem_list) == len(merged_dataset.answer_list) == len(merged_dataset.cost_list), '数据长度不匹配'
    assert len(set(merged_dataset.seed_list)) == len(merged_dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    positions = np.array([problem['pos_node'] for problem in merged_dataset.problem_list])         # (data_num, node_num, 2)
    assert positions.shape[0] == np.unique(positions, axis=0).shape[0], '存在完全相同的问题'

    # 打印数据集信息
    dataset_train, dataset_prompt, dataset_problem = ex_datasets['train'], ex_datasets['prompt'], ex_datasets['problem']
    print('-'*80)
    print(f'There are [{len(merged_dataset.answer_list)}] raw data generated with [{len(merged_dataset.seed_list)}] different random seeds, ave cost = [{np.mean(merged_dataset.cost_list):.3f}]')
    print(f'\tTrain dataset:    [{len(dataset_train.answer_list)}] raw data with [{len(dataset_train.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_train.cost_list):.3f}]')
    print(f'\tProblem dataset:  [{len(dataset_problem.answer_list)}] raw data with [{len(dataset_problem.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_problem.cost_list):.3f}]')
    print(f'\tPrompt dataset:   [{len(dataset_prompt.answer_list)}] raw data with [{len(dataset_prompt.seed_list)}] different random seeds, ave cost = [{np.mean(dataset_prompt.cost_list):.3f}]')
    print('-'*80)

if __name__ == "__main__":
    #data_base_path = base_path
    data_base_path = '/data3/xxx/unico'    
    scale = 100
    data_source = 'am'
    source_data = check_data(data_base_path, scale, load_data_num=None, data_source=data_source)