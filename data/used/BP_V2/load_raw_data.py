import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import List
from environment.used.Env_bp_v2 import BP_V2
from data.raw.bp import raw_data_to_problem
from utils.utils_data import DataProblem, RawData, load_all_raw_dataset, split_raw_dataset
from utils.utils import create_folder_if_not_exist
from utils.utils_slover import knapsack_dp

def raw_data_to_traj(info):
    i, traj_type, get_problem = info
    
    global ex_datasets
    dataset = ex_datasets[traj_type]
    problem, answer, value = dataset.problem_list[i], dataset.answer_list[i], dataset.cost_list[i] 
    real_value, _ = knapsack_dp(problem['capacity_left'].item(), problem['item_volumes'], problem['item_values'])        
    assert real_value == value

    # 重置环境
    env = BP_V2(scale=problem['item_volumes'].shape[0])
    observation, info = env.reset(options={
        'problem_info': (None, problem, answer),
        'problem_obj':(value, 1),
        'use_default_policy_obj': False
    })
    assert answer == env.real_answer

    # 生成 MDP 轨迹
    # 注意 random obj 未知，生成轨迹中的 DB1 reward 不可靠，因此只保留 AM reward
    obss, acts, rewards = [observation, ], [], []
    for action in answer[:-1]:
        observation, reward, terminated, truncated, info = env.step(action)
        acts.append(action)
        rewards.append(reward['AM'])
        obss.append(observation)
        assert not (terminated or truncated)
    action = answer[-1]
    observation, reward, terminated, truncated, info = env.step(action)
    assert abs(reward['AM'] - 1) <= 1e-4 and abs(reward['DB1'] - 1) <= 1e-4    # 由于直接用 real_answer 作为 model_answer, 无论 DB1 reward 是多少，reward['DB1'] == 1
    assert terminated and not truncated
    acts.append(action)
    rewards.append(1)
    assert len(obss) == len(acts) == len(rewards)

    # 处理成 d4rl 格式保存
    obss_visited = np.vstack([obs['visited'] for obs in obss])
    obss_capacity = np.array([obs['capacity_left'] for obs in obss])
    obss_item_values = np.vstack([obs['item_values'] for obs in obss])
    obss_item_volumes = np.vstack([obs['item_volumes'] for obs in obss])

    episode = {
        'prefix': None,
        'prefix_masks': None,
        'observations': {
            'visited': obss_visited,                        # (time_steps, num_item)
            'capacity_left': obss_capacity,                 # (time_steps, 1)
            'item_values': obss_item_values,                # (time_steps, num_item)
            'item_volumes': obss_item_volumes               # (time_steps, num_item)
        },
        'actions': np.array(acts).astype(np.int32),         # (time_steps, )
        'rewards': np.array(rewards).astype(np.float32),    # (time_steps, )
        'terminals': np.array([False] * (len(rewards)-1) + [True], dtype=bool)  # 'terminals' 字段只是模仿 d4rl 的数据形式，当前没有作用
    }
    if get_problem:
        return episode, problem, answer
    return episode

def raw_data_to_traj_multiprocessing(process_num:int=10, data_num:int=0, data_type:str='train', get_problem:bool=False):
    global ex_datasets
    raw_data = ex_datasets[data_type]

    data_num = len(raw_data.answer_list) if data_num == 0 else min(data_num, len(raw_data.answer_list))
    with mp.Pool(processes=process_num) as pool:
        results = tqdm(
            pool.imap_unordered(raw_data_to_traj, [(i, data_type, get_problem) for i in range(data_num)]),
            total=data_num,
            desc=f'Generating {data_type} episodes'
        )  # 'total' is redundant here but can be useful when the size of the iterable is unobvious
        results = list(results)
    
    if not get_problem:
        results = [epi for epi in results if epi is not None]
        return results, None
    else:
        episodes = []
        problems = DataProblem(problem_list=[], answer_list=[])
        for episode, problem, answer in results:
            if episode is not None and problem is not None and answer is not None:
                episodes.append(episode)
                problems.problem_list.append(problem)
                problems.answer_list.append(answer)
        return episodes, problems

def consistency_check(episodes:List, problem:DataProblem, check_num:int=1000):
    data_num = len(episodes)
    random_indices = np.random.choice(data_num, min(data_num, check_num), replace=False)
    for i in random_indices:
        assert episodes[i]['actions'].tolist() == problem.answer_list[i]

if __name__ == "__main__":
    # 构造环境
    scale = 50
    cop_type = 'bp'
    dataset_name = 'BP_V2'
    data_source = None
    gen_test_episodes = False
    get_trian_problem = True
    worker_num = 25

    data_base_path = base_path
    #data_base_path = '/data3/xxx/toy-gato'

    # 用于保存数据的文件夹
    save_to_dir = f'{data_base_path}/data/used/{dataset_name}/{cop_type}{scale}' if data_source is None else \
                f'{data_base_path}/data/used/{dataset_name}/{cop_type}{scale}_{data_source}'
    create_folder_if_not_exist(save_to_dir)

    # 加载三类原始数
    ex_datasets = load_all_raw_dataset(data_base_path, cop_type, scale, data_num=None, show_info=True, data_source=data_source)
    ex_datasets['train'] = split_raw_dataset(ex_datasets['train'], 0, 200000)
    #raw_data_to_traj((0, 'train', True))

    # 训练中未见的测试问题
    test_problem = raw_data_to_problem(ex_datasets['problem']) 
    with open(f'{save_to_dir}/{cop_type}{scale}_problem.pkl', 'wb') as f:
        pickle.dump(test_problem, f)    
        print(f'test_problems saved')

    # 生成轨迹数据                                         
    prompt_episodes, _ = raw_data_to_traj_multiprocessing(                              
        worker_num, data_type='prompt'
    ) 
    train_episodes, train_problem = raw_data_to_traj_multiprocessing(                   
        worker_num, data_type='train', get_problem=get_trian_problem
    )    
    if gen_test_episodes:                           
        test_episodes, _ = raw_data_to_traj_multiprocessing(                            
            worker_num, data_type='test'
        )    

    # 提示MDP轨迹
    with open(f'{save_to_dir}/{cop_type}{scale}_prompt.pkl', 'wb') as f:
        pickle.dump(prompt_episodes, f)    
        print(f'prompt_traj saved')

    # 训练MDP轨迹
    with open(f'{save_to_dir}/{cop_type}{scale}_train.pkl', 'wb') as f:
        pickle.dump(train_episodes, f)    
        print(f'train_traj saved')

    # （可选）训练数据对应的问题，这些问题上的测试性能应该接近 100%
    if get_trian_problem:
        assert train_problem is not None
        consistency_check(train_episodes, train_problem)
        with open(f'{save_to_dir}/{cop_type}{scale}_train_problem.pkl', 'wb') as f:
            pickle.dump(train_problem, f)    
            print(f'train_problems saved')

    # （可选）训练中未见测试问题对应的最优MDP轨迹，这些数据上的损失反映拟合的泛化质量
    if gen_test_episodes: 
        assert test_episodes is not None
        consistency_check(test_episodes, test_problem)
        with open(f'{save_to_dir}/{cop_type}{scale}_test.pkl', 'wb') as f:
            pickle.dump(test_episodes, f)    
            print(f'test_traj saved')   