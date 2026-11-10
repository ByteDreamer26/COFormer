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
ENV_NAME = 'BP_V1'
cop_type = 'bp'
scale = 100
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集   
file_list = glob.glob(os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}', '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
#file_names = ['bp20_problem.pkl']

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True)

# 一致性检查
if f'{cop_type}{scale}_train_problem' in file_data and f'{cop_type}{scale}_train' in file_data:
    train_problem = file_data[f'{cop_type}{scale}_train_problem']
    train_episodes = file_data[f'{cop_type}{scale}_train']
    num_problem = len(train_problem.problem_list)
    num_episode = len(train_episodes)

    num_samples = min(num_problem, num_episode, 3000)
    check_idxs = random.sample(range(min(num_problem, num_episode)), num_samples)
    for idx in tqdm(check_idxs, desc=f"Check data validity"):
        assert train_episodes[idx]['actions'].tolist() == train_problem.answer_list[idx]

# 写数据集 log 文件
create_folder_overwrite_if_exist(f'{base_path}/visualize/dataset/{ENV_NAME}')
for file_name in file_names:
    data_name = file_name[:-4]
    # 可视化评估问题
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            #for i, obs in enumerate(problems):
            for i in range(num_problem):
                answer = problems.answer_list[i]
                capacity = problems.problem_list[i]['capacity_left']
                item_values = problems.problem_list[i]['item_values']
                item_volumes = problems.problem_list[i]['item_volumes']

                file.write('-'*25+f' problem-{i} '+'-'*25+'\n')
                file.write(f'capacity:      \t{capacity}\n')
                file.write(f'item values:   \t{item_values}\n')
                file.write(f'item volumes:  \t{item_volumes}\n')                    
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
                for t in range(len(rewards)):
                    visited = obss['visited'][t]
                    capacity = obss['capacity_left'][t]
                    item_values = obss['item_values'][t]
                    item_volumes = obss['item_volumes'][t]
                    file.write(f'item values:   \t{item_values}\n')
                    file.write(f'item volumes:  \t{item_volumes}\n')
                    file.write(f'visited:       \t{visited}\n')
                    file.write(f'capacity left: \t{capacity}\n')
                    file.write(f'take action:   \t{acts[t]}\n')
                    file.write(f'get reward:    \t{rewards[t]}\n\n')
            
    print(f'{data_name}.txt saved')