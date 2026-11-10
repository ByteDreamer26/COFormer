import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import random
import pickle
import numpy as np
from tqdm import tqdm
from utils.utils_slover import SPCTSP_REOPT, calc_pctsp_cost
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset

MAX_LENGTHS = {
    10: 1.5,
    20: 2.,
    50: 3.,
    100: 4.
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
    problems = DataProblem(problem_list=[], answer_list=[])
    i = 0
    with tqdm(total=num, desc=f'Checking') as pbar:
        while len(problems.answer_list) < num:
            problem, answer, cost = raw_data.problem_list[i], raw_data.answer_list[i], raw_data.cost_list[i]
            position = np.vstack((problem['pos_depot'][None,:], problem['pos_node']))
            calc_cost = calc_pctsp_cost(position, problem['penalty'], problem['stoc_prize'], answer[:-1])
            if calc_cost is None:
                i += 1
                continue

            assert abs(calc_cost - cost) < 1e-5
            problems.problem_list.append(problem)
            problems.answer_list.append(answer)
            pbar.update()
            i += 1

            if i == len(raw_data.answer_list):
                break

    return problems

def check_spctsp_data(data_base_path, scale, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'spctsp', scale)
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
            det_prize = dataset.problem_list[idx]['det_prize']
            stoc_prize = dataset.problem_list[idx]['stoc_prize']
            data_answer = dataset.answer_list[idx]
            data_cost = dataset.cost_list[idx]
            cost, answer, _ = SPCTSP_REOPT(
                pos_depot.tolist(), 
                pos_node.tolist(), 
                penalty.tolist(), 
                det_prize.tolist(),
                stoc_prize.tolist(),
            )
            
            assert abs(cost - data_cost) < 1e-5
            assert answer + [0] == data_answer

def gen_raw_spctsp(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock, penalty_factor=3):
    ''' 数据生成进程的目标方法 '''
    # 生成问题，求解并写入 dataset
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象
    with tqdm(total=data_num_per_process, desc=f'Gen SPCTSP data with seed {seed}', position=tqdm_pos) as pbar:
        for _ in range(data_num_per_process):
            #lock.acquire()  
            real_answer = None
            cost = float('inf')
            # 为了加速生成，runs参数设为2，求解精度较低
            # 这导致生成的解有概率不满足约束或得到次优解（20城市的情况下根据经验设cost>5）这种情况下重新生成
            while real_answer is None or cost > 8:  
                pos_depot = rng.uniform(0, 1, size=(2, ))
                pos_node = rng.uniform(0, 1, size=(scale, 2))
                penalty_max = MAX_LENGTHS[scale] * (penalty_factor) / float(scale)
                penalty = np.random.uniform(size=(scale, )) * penalty_max
                det_prize = np.random.uniform(size=(scale,)) * 4 / float(scale)
                while det_prize.sum() < 1:
                    det_prize = np.random.uniform(size=(scale,)) * 4 / float(scale)
                stoc_prize = np.random.uniform(size=(scale,)) * 4 / float(scale)
                while stoc_prize.sum() < 1:
                    stoc_prize = np.random.uniform(size=(scale,)) * 4 / float(scale)

                # SPCTSP 问题的解从仓库出发，经过若干站点后在仓库结束
                # 仓库索引为 0，站点索引从 1 开始
                # 调用 ILS 方法求得的解格式中首尾的仓库都不包含
                cost, real_answer, _ = SPCTSP_REOPT(
                    pos_depot.tolist(), 
                    pos_node.tolist(), 
                    penalty.tolist(), 
                    det_prize.tolist(),
                    stoc_prize.tolist(),
                )
            real_answer += [0]  # answer 格式中包含终止的仓库, 不包含出发的仓库
            #lock.release()

            dataset.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'det_prize': det_prize, 'stoc_prize': stoc_prize, 'penalty': penalty})
            dataset.answer_list.append(real_answer)
            dataset.cost_list.append(cost)
            pbar.update()

    # 保存该进程生成的原始数据
    dataset_path = f'{data_base_path}/data/raw/spctsp/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def get_rl_spctsp(rng, batch_size, scale, penalty_factor=3):
    dataproblem = DataProblem(answer_list=[], problem_list=[])

    pos_depots = rng.uniform(0, 1, size=(batch_size, 2))
    pos_nodes = rng.uniform(0, 1, size=(batch_size, scale, 2))
    penalty_max = MAX_LENGTHS[scale] * (penalty_factor) / float(scale)
    penalties = rng.uniform(size=(batch_size, scale)) * penalty_max
    det_prizes = rng.uniform(size=(batch_size, scale)) * 4 / float(scale)
    while np.any(det_prizes.sum(axis=1)) < 1:
        det_prizes = rng.uniform(size=(batch_size, scale)) * 4 / float(scale)

    stoc_prizes = rng.uniform(size=(batch_size, scale)) * 4 / float(scale)
    while np.any(stoc_prizes.sum(axis=1)) < 1:
        stoc_prizes = rng.uniform(size=(batch_size, scale)) * 4 / float(scale)

    for pos_depot, pos_node, penalty, det_prize, stoc_prize in zip(pos_depots, pos_nodes, penalties, det_prizes, stoc_prizes):
        dataproblem.problem_list.append({'pos_depot': pos_depot, 'pos_node': pos_node, 'det_prize': det_prize, 'stoc_prize': stoc_prize, 'penalty': penalty})
        dataproblem.answer_list.append([0])
    return dataproblem