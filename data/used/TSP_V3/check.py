import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import glob
import random
from tqdm import tqdm
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_data import load_data_and_check_quantity

# 数据量
ENV_NAME = 'TSP_V3'
cop_type = 'tsp'
scale = 100
data_source = None
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集   
file_dir = os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}') if data_source is None else \
            os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}_{data_source}')
file_list = glob.glob(os.path.join(file_dir, '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True, data_source=data_source)

# 一致性检查
if f'{cop_type}{scale}_train_problem' in file_data and f'{cop_type}{scale}_train' in file_data:
    train_problem = file_data[f'{cop_type}{scale}_train_problem']
    train_episodes = file_data[f'{cop_type}{scale}_train']
    num_problem = len(train_problem.problem_list)
    num_episode = len(train_episodes)

    num_samples = min(num_problem, num_episode, 3000)
    check_idxs = random.sample(range(min(num_problem, num_episode)), num_samples)
    for idx in tqdm(check_idxs, desc=f"Check data validity"):
        assert [0] + train_episodes[idx]['actions'].tolist() == train_problem.answer_list[idx]

# 写数据集 log 文件
create_folder_overwrite_if_exist(f'{base_path}/visualize/dataset/{ENV_NAME}')
for file_name in file_names:
    data_name = file_name[:-4]
    # 可视化评估问题
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            for i in range(num_problem):
                answer = problems.answer_list[i]
                position = problems.problem_list[i]['position'].reshape((-1, 2))
                start_location = position[0]

                file.write('-'*25+f' problem-{i} '+'-'*25+'\n')
                file.write(f'city position:\n{position}\n\n')
                file.write(f'start location:\t{start_location}\n')                    
                file.write(f'solution:      \t{answer}\n\n')          

    # 可视化训练序列
    else:
        episodes = file_data[data_name]
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            for i in range(num_episode):
                epi = episodes[i]
                acts = epi['actions']
                rewards = epi['rewards']
                obss = epi['observations']

                file.write('-'*25+f' episode-{i} '+'-'*25+'\n')
                position = epi['prefix']['position'].reshape((-1, 2)) 
                file.write(f'[fixed prefix] city position:\n{position}\n\n')
                for t in range(len(rewards)):
                    current_location = obss['current_position'][t]
                    visited = obss['visited'][t]
                    file.write(f'visited:         \t{visited}\n')
                    file.write(f'current location:\t{current_location}\n')
                    file.write(f'take action:     \t{acts[t]}\n')
                    file.write(f'get reward:      \t{rewards[t]}\n\n')
            
    print(f'{data_name}.txt saved')