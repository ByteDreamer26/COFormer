import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import random
import pickle
import numpy as np
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset
from utils.utils_slover import calc_op_total, OP_compass
from tqdm import tqdm

MAX_LENGTHS = {
    10: 1.5,
    20: 2.,
    50: 3.,
    100: 4.
}

OFFSET_LENGTH = {
    10:-0.75, 
    20:-1.0, 
    50:-1.5, 
    100:-2.0
}

def raw_data_to_problem(raw_data:RawData, num:int=0):
    num = len(raw_data.answer_list) if num == 0 else min(num, len(raw_data.answer_list))
    problems = DataProblem(problem_list=raw_data.problem_list[:num], answer_list=[[a+1 for a in answer] for answer in raw_data.answer_list[:num]])
    with tqdm(total=num, desc=f'Checking problem') as pbar:
        for problem, answer, cost in zip(problems.problem_list, problems.answer_list, raw_data.cost_list[:num]):
            assert abs(-calc_op_total(problem['prize'], np.array(answer)-1) - cost) < 1e-3
            pbar.update()
    return problems

def check_op_data(data_base_path, scale, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'op', scale)
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
            prize = dataset.problem_list[idx]['prize']
            length_left = MAX_LENGTHS[pos_node.shape[0]]
            data_answer = dataset.answer_list[idx]
            data_cost = dataset.cost_list[idx]
            #ost, answer = OP_gurobi(pos_depot.tolist(), pos_node.tolist(), prize.tolist(), length_left, )
            cost, answer = OP_compass(pos_depot.tolist(), pos_node.tolist(), prize.tolist(), length_left, )
            assert abs((cost - data_cost)/data_cost) < 0.1
            #assert answer == data_answer

def gen_raw_op(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock=None, prize_type='dist'):
    ''' 数据生成进程的目标方法 '''
    # 生成问题，求解并写入 dataset
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象
    with tqdm(total=data_num_per_process, desc=f'Gen OP data with seed {seed}', position=tqdm_pos) as pbar:
        for n in range(data_num_per_process):
            pos_depot = rng.uniform(0, 1, size=(2, ))
            pos_node = rng.uniform(0, 1, size=(scale, 2))
            length_left = MAX_LENGTHS[scale]

            distance_node = np.zeros((scale, scale), dtype=np.float32)
            distance_depot = np.zeros(scale, dtype=np.float32)
            for i in range(scale):
                distance_depot[i] = np.linalg.norm(pos_node[i] - pos_depot)
                for j in range(scale):
                    distance_node[i,j] = np.linalg.norm(pos_node[i] - pos_node[j])

            # 三种 prize 类型
            if prize_type == 'const':
                prize = np.ones(scale, dtype=np.float32)
            elif prize_type == 'unif':
                prize = (1 + np.random.randint(0, 100, size=(scale,))) / 100.
            else:  # Based on distance to depot
                assert prize_type == 'dist'
                distance = distance_depot
                prize = (1 + (distance / distance.max() * 99).astype(int)) / 100.

            # OP 问题的解从仓库出发，经过若干站点后在仓库结束
            # 仓库索引为 -1，站点索引从 0 开始
            # 调用 gurobi 方法求得的解格式中首尾的仓库都不包含
            #cost, real_answer = OP_gurobi(pos_depot.tolist(), pos_node.tolist(), prize.tolist(), length_left, )
            cost, real_answer = OP_compass(pos_depot.tolist(), pos_node.tolist(), prize.tolist(), length_left, )
            if cost is not None:
                dataset.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'prize': prize})
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
    dataset_path = f'{data_base_path}/data/raw/op/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def get_rl_op(rng, batch_size, scale, prize_type='dist'):
    dataproblem = DataProblem(answer_list=[], problem_list=[])
    
    pos_depots = rng.uniform(0, 1, size=(batch_size, 2))
    pos_nodes = rng.uniform(0, 1, size=(batch_size, scale, 2))
    for pos_depot, pos_node in zip(pos_depots, pos_nodes):
        distance_depot = np.zeros(scale, dtype=np.float32)
        for i in range(scale):
            distance_depot[i] = np.linalg.norm(pos_node[i] - pos_depot)

        if prize_type == 'const':
            prize = np.ones(scale, dtype=np.float32)
        elif prize_type == 'unif':
            prize = (1 + np.random.randint(0, 100, size=(scale,))) / 100.
        else:  # Based on distance to depot
            assert prize_type == 'dist'
            distance = distance_depot
            prize = (1 + (distance / distance.max() * 99).astype(int)) / 100.

        dataproblem.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'prize': prize})
        dataproblem.answer_list.append([0])
    return dataproblem