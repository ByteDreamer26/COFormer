import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
import pickle
import random
from tqdm import tqdm
from utils.utils_slover import TSP_lkh
from sklearn.datasets import make_blobs, make_circles, make_moons
from utils.utils_data import DataProblem, RawData, load_and_merge_all_raw_dataset

def generate_tsp_data(seed=42, dataset_size=10000, scale=20, distribution='uniform'):
    rng = np.random.RandomState(seed)  # 创建独立的随机数生成器对象

    if distribution == 'uniform':
        data = rng.uniform(size=(dataset_size, scale, 2))
    elif distribution == 'normal':
        data = rng.normal(loc=0.5, scale=1, size=(dataset_size, scale, 2))
    elif distribution == 'exponential':
        data = rng.exponential(scale=0.5, size=(dataset_size, scale, 2))
    elif distribution == 'gamma':
        data = rng.gamma(shape=2.0, scale=1.0, size=(dataset_size, scale, 2))
    elif distribution == 'beta':
        data = rng.beta(a=2.0, b=8.0, size=(dataset_size, scale, 2))
    elif distribution == 'clusters':
        X, _ = make_blobs(n_samples=dataset_size*scale, n_features=2, centers=2, cluster_std=[2, 2], random_state=seed)
        data = X.reshape((dataset_size, scale, 2))
    elif distribution == 'circles':
        X, _ = make_circles(n_samples=dataset_size*scale, noise=0.1, random_state=seed)
        data = X.reshape((dataset_size, scale, 2))
    elif distribution == 'moons':
        X, _ = make_moons(n_samples=dataset_size*scale, noise=0.1, random_state=seed)
        data = X.reshape((dataset_size, scale, 2))
    else:
        raise ValueError("Invalid distribution type. Supported types are 'uniform', 'normal', 'exponential', 'gamma', 'beta', 'clusters', 'circles', 'moons', and 'grid'.")
        
    # 线性归一化到单位方块 [0, 0] 到 [1, 1] 内
    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def check_tsp_data(data_base_path, prob_size, show_info=False, check_data_num=5000):
    # 加载并合并已生成的 train/problem/prompt 源数据
    dataset = load_and_merge_all_raw_dataset(data_base_path, 'tsp', prob_size)
    assert len(dataset.problem_list) == len(dataset.answer_list) == len(dataset.cost_list), '数据长度不匹配'
    assert len(set(dataset.seed_list)) == len(dataset.seed_list), '存在使用相同随机种子的数据生成进程'
    positions = np.array([problem['position'] for problem in dataset.problem_list])         # (data_num, scale, 2)
    assert positions.shape[0] == np.unique(positions, axis=0).shape[0], '存在完全相同的问题'

    if show_info:
        print('-'*80+f'\nThere are [{len(dataset.answer_list)}] raw data generated with [{len(dataset.seed_list)}] different random seeds, ave cost = [{np.mean(dataset.cost_list):.3f}]\n'+'-'*80)

    # 随机检查 check_data_num 个数据
    if check_data_num > 0:
        dataset_size = len(dataset.answer_list)
        check_idx = random.sample(range(dataset_size), min(dataset_size, check_data_num))
        for idx in tqdm(check_idx, desc=f"Check data validity"):
            data_pos = dataset.problem_list[idx]['position']
            data_answer = dataset.answer_list[idx]
            data_cost = dataset.cost_list[idx]
            cost, answer = TSP_lkh(np.array(data_pos, np.float32)) 
            assert abs(cost - data_cost) < 1e-5
            assert answer == data_answer

def gen_raw_tsp(data_base_path, data_num_per_process, seed, scale, tqdm_pos, lock=None, data_distribution='uniform'):
    # 生成问题，求解并写入 dataset
    process_data = generate_tsp_data(seed, data_num_per_process, scale, data_distribution)
    dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    with tqdm(total=data_num_per_process, desc=f'Gen TSP data with seed {seed}', position=tqdm_pos) as pbar:
        for city_pos in process_data:
            lock.acquire()  
            cost, answer = TSP_lkh(np.array(city_pos, np.float32)) 
            lock.release()
            dataset.problem_list.append({'position': city_pos})
            dataset.answer_list.append(answer)
            dataset.cost_list.append(cost)
            pbar.update()

    # 保存该进程生成的原始数据
    dataset_path = f'{data_base_path}/data/raw/tsp/_temp/num{data_num_per_process}_seed{seed}.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset_path

def gen_rl_tsp(rng, batch_size, scale):
    dataproblem = DataProblem(answer_list=[], problem_list=[])
    problems = rng.uniform(size=(batch_size, scale, 2))
    for problem in problems:
        dataproblem.problem_list.append({'position': problem})
        dataproblem.answer_list.append([0] * (scale-1))
    return dataproblem