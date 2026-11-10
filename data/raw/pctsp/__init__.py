import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import random
import pickle
import numpy as np
from tqdm import tqdm
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset
from utils.utils_slover import calc_pctsp_cost, PCTSP_ILS

MAX_LENGTHS = {
    10: 1.5,
    20: 2.,
    50: 3.,
    100: 4.
}

COST_LIMIT = {
    10: 5.,
    20: 5.,
    50: 8.,
    100: 12.
}

OFFSET_PRIZE = {
    20: -0.1, 
    50:-0.04, 
    100:-0.02
}
OFFSET_PENALTY = {
    20: -0.15, 
    50:-0.09, 
    100:-0.06
}

def raw_data_to_problem(raw_data:RawData, num:int=0):
    num = len(raw_data.answer_list) if num == 0 else min(num, len(raw_data.answer_list))
    problems = DataProblem(problem_list=raw_data.problem_list[:num], answer_list=raw_data.answer_list[:num])
    with tqdm(total=num, desc=f'Checking') as pbar:
        for problem, answer, cost in zip(problems.problem_list, problems.answer_list, raw_data.cost_list[:num]):
            position = np.vstack((problem['pos_depot'][None,:], problem['pos_node']))
            assert abs(calc_pctsp_cost(position, problem['penalty'], problem['prize'], answer[:-1]) - cost) < 1e-5
            pbar.update()
    return problems

def check_pctsp_data(data_base_path, scale, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'pctsp', scale)
    assert len(dataset.problem_list) == len(dataset.answer_list) == len(dataset.cost_list), '数据长度不匹配'
    assert len(set(dataset.seed_list)) == len(dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    positions = np.array([problem['pos_node'] for problem in dataset.problem_list])         # (data_num, node_num, 2)
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
            penalty = dataset.problem_list[idx]['penalty']
            prize = dataset.problem_list[idx]['prize']
            data_answer = dataset.answer_list[idx]
            data_cost = dataset.cost_list[idx]
            cost, answer, _ = PCTSP_ILS(
                pos_depot.tolist(), 
                pos_node.tolist(), 
                penalty.tolist(), 
                prize.tolist(),
            )
            assert abs(cost - data_cost) < 1e-5
            assert answer + [0] == data_answer


def gen_raw_pctsp(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock, penalty_factor=3):
    ''' 数据生成进程的目标方法 '''
    # 生成问题，求解并写入 dataset
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象
    with tqdm(total=data_num_per_process, desc=f'Gen PCTSP data with seed {seed}', position=tqdm_pos) as pbar:
        for n in range(data_num_per_process):
            #lock.acquire()  
            real_answer = None
            cost = float('inf')
            # 为了加速生成，runs参数设为2，求解精度较低
            # 这导致生成的解有概率不满足约束或得到次优解（20城市的情况下根据经验设cost>5）这种情况下重新生成
            while real_answer is None or cost > COST_LIMIT[scale]:  
                pos_depot = rng.uniform(0, 1, size=(2, ))
                pos_node = rng.uniform(0, 1, size=(scale, 2))
                penalty_max = MAX_LENGTHS[scale] * (penalty_factor) / float(scale)
                penalty = np.random.uniform(size=(scale, )) * penalty_max
                prize = np.random.uniform(size=(scale,)) * 4 / float(scale)
                while prize.sum() < 1:
                    prize = np.random.uniform(size=(scale,)) * 4 / float(scale)

                # PCTSP 问题的解从仓库出发，经过若干站点后在仓库结束
                # 仓库索引为 0，站点索引从 1 开始
                # 调用 ILS 方法求得的解格式中首尾的仓库都不包含
                cost, real_answer, _ = PCTSP_ILS(
                    pos_depot.tolist(), 
                    pos_node.tolist(), 
                    penalty.tolist(), 
                    prize.tolist(),
                )
            real_answer += [0]  # answer 格式中包含终止的仓库, 不包含出发的仓库
            #lock.release()

            dataset.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'prize': prize, 'penalty': penalty})
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
    dataset_path = f'{data_base_path}/data/raw/pctsp/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def get_rl_pctsp(rng, batch_size, scale, penalty_factor=3):
    dataproblem = DataProblem(answer_list=[], problem_list=[])
    
    pos_depots = rng.uniform(0, 1, size=(batch_size, 2))
    pos_nodes = rng.uniform(0, 1, size=(batch_size, scale, 2))
    penalty_max = MAX_LENGTHS[scale] * (penalty_factor) / float(scale)
    penalties = rng.uniform(size=(batch_size, scale)) * penalty_max
    prizes = rng.uniform(size=(batch_size, scale)) * 4 / float(scale)
    while np.any(prizes.sum(axis=1)) < 1:
        prizes = rng.uniform(size=(batch_size, scale,)) * 4 / float(scale)

    for pos_depot, pos_node, penalty, prize in zip(pos_depots, pos_nodes, penalties, prizes):
        dataproblem.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'prize': prize, 'penalty': penalty})
        dataproblem.answer_list.append(None)
    return dataproblem