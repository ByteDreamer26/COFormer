import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import pickle
from tqdm import tqdm
from utils.utils_data import load_data_and_check_quantity, DataProblem

# 数据量
ENV_NAME = 'TSP_V3'
cop_type = 'tsp'
scale = 20

# 加载训练数据
file_names = [f'{cop_type}{scale}_train.pkl', ]
file_data = load_data_and_check_quantity(file_names, ENV_NAME, 0, 0, check_ave_obj=True)
train_episodes = file_data[f'{cop_type}{scale}_train']

# 转换为 train_problem
train_problem = DataProblem(answer_list=[],  problem_list=[])
for epi in tqdm(train_episodes):
    train_problem.answer_list.append([0] + epi['actions'].tolist())
    
    position = epi['prefix']['position'].reshape(-1, 2)
    train_problem.problem_list.append(
        {
            'position': position,
            'current_position': position[0],
        }
    )

# 保存数据
data_base_path = '/data3/xxx/unico'
save_to_dir = f'{data_base_path}/data/used/{ENV_NAME}/{cop_type}{scale}'
with open(f'{save_to_dir}/{cop_type}{scale}_train_problem.pkl', 'wb') as f:
    pickle.dump(train_problem, f)    
    print(f'train_problems saved')