import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import pickle
import random
import numpy as np
from tqdm import tqdm
from utils.utils_slover import CVRP_lkh, calc_vrp_distance
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset

CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.
}

COST_LIMIT = {
    10: 10.,
    20: 10.,
    50: 25.,
    100: 40
}

def raw_data_to_problem(raw_data:RawData, num:int=0):
    num = len(raw_data.answer_list) if num == 0 else min(num, len(raw_data.answer_list))
    problems = DataProblem(problem_list=raw_data.problem_list[:num], answer_list=raw_data.answer_list[:num])
    with tqdm(total=num, desc=f'Checking') as pbar:
        for problem, answer, cost in zip(problems.problem_list, problems.answer_list, raw_data.cost_list[:num]):
            position = np.vstack((problem['pos_depot'][None,:], problem['pos_node']))
            assert abs(calc_vrp_distance(position, answer) - cost) < 1e-5
            pbar.update()
    return problems

def check_cvrp_data(data_base_path, scale, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'cvrp', scale)
    assert len(dataset.problem_list) == len(dataset.answer_list) == len(dataset.cost_list), '数据长度不匹配'
    assert len(set(dataset.seed_list)) == len(dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    positions = np.array([problem['pos_node'] for problem in dataset.problem_list])         # (data_num, scale, 2)
    assert positions.shape[0] == np.unique(positions, axis=0).shape[0], '存在完全相同的问题'

    # 打印数据集信息
    if show_info:
        print('-'*80+f'\nThere are [{len(dataset.answer_list)}] raw data generated with [{len(dataset.seed_list)}] different random seeds, ave cost = [{np.mean(dataset.cost_list):.3f}]\n'+'-'*80)

    # 随机检查 check_data_num 个数据
    if check_data_num > 0:
        dataset_size = len(dataset.answer_list)
        check_idx = random.sample(range(dataset_size), min(dataset_size, check_data_num))
        for idx in tqdm(check_idx, desc=f"Check data validity"):
            pos_depot = dataset.problem_list[idx]['pos_depot']
            pos_node = dataset.problem_list[idx]['pos_node']
            demand = dataset.problem_list[idx]['demand']
            capacity_left = CAPACITIES[pos_node.shape[0]]
            data_answer = dataset.answer_list[idx]
            data_cost = dataset.cost_list[idx]
            cost, answer, _ = CVRP_lkh(pos_depot.tolist(), pos_node.tolist(), demand.tolist(), capacity_left)
            assert abs(cost - data_cost) < 1e-5
            assert answer == data_answer

def gen_raw_cvrp(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock=None):
    ''' 数据生成进程的目标方法 '''
    # 生成问题，求解并写入 dataset
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象
    with tqdm(total=data_num_per_process, desc=f'Gen CVRP data with seed {seed}', position=tqdm_pos) as pbar:
        for n in range(data_num_per_process):
            #lock.acquire()  
            real_answer = None
            #while real_answer is None or (cost is not None and cost > COST_LIMIT[scale]) :  # 生成的解有概率不满足约束或错误，这种情况下重新生成
            while real_answer is None:
                pos_depot = rng.uniform(0, 1, size=(2, ))
                pos_node = rng.uniform(0, 1, size=(scale, 2))
                demand = rng.randint(1, 10, size=(scale, ))
                capacity_left = CAPACITIES[scale]
                # CVRP 问题的解从仓库出发，可能多次回到仓库，最终在仓库结束。仓库索引为 0，站点索引从 1 开始
                # 调用 LKH 方法求得的解格式中首尾的仓库都不包含
                cost, real_answer, _ = CVRP_lkh(pos_depot.tolist(), pos_node.tolist(), demand.tolist(), capacity_left)
            #lock.release()
            dataset.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'demand': demand})
            dataset.answer_list.append(real_answer)
            dataset.cost_list.append(cost)
            
            info_cost = 0 if len(dataset.cost_list) == 0 else np.mean(dataset.cost_list)
            info = {
                'cost': f'{info_cost:.2f}',
                'success': f'{len(dataset.cost_list)/(n+1):.2f}'
            }
            pbar.set_postfix(info)
            pbar.update()
            
    # 保存该进程生成的原始数据
    dataset_path = f'{data_base_path}/data/raw/cvrp/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def get_rl_cvrp(rng, batch_size, scale):
    dataproblem = DataProblem(answer_list=[], problem_list=[])
    
    pos_depots = rng.uniform(0, 1, size=(batch_size, 2))
    pos_nodes = rng.uniform(0, 1, size=(batch_size, scale, 2))
    demands = rng.integers(1, 10, size=(batch_size, scale))    
    for pos_depot, pos_node, demand in zip(pos_depots, pos_nodes, demands):
        dataproblem.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'demand': demand})
        dataproblem.answer_list.append([0])
    return dataproblem