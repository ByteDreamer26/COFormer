import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
from utils.utils_data import load_data_and_check_quantity
import pickle

# 数据量
ENV_NAME = 'ATSP_V1'
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集   
file_list = glob.glob(os.path.join(f'{base_path}/data/used/{ENV_NAME}', '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
file_names = ['atsp20_prompt.pkl',]

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True)

for file_name in file_names:
    data_name = file_name[:-4]
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        problems_node_embedding = np.vstack([p['node_embedding'] for p in problems.problem_list])
        print('problems_node_embedding', problems_node_embedding.max())
        if problems_node_embedding.max() > 4:
            for i in range(5000):
                problems.problem_list[i]['node_embedding'] /= 15

            problems_node_embedding = np.vstack([p['node_embedding'] for p in problems.problem_list])
            print('problems_node_embedding', problems_node_embedding.max())
            #with open(f'{base_path}/data/used/{ENV_NAME}/{data_name}.pkl', 'wb') as f:
            #    pickle.dump(problems, f)    
            #    print(f'{data_name} reranged')

    else:
        episodes = file_data[data_name]
        epi_prefix_node_embedding = np.vstack([epi['prefix']['node_embedding'] for epi in episodes])
        epi_obss_current_embedding = np.vstack([epi['observations']['current_embedding'] for epi in episodes])
        print('epi_prefix_node_embedding', epi_prefix_node_embedding.max())
        print('epi_obss_current_embedding', epi_obss_current_embedding.max())
        assert not (epi_prefix_node_embedding.max().item() > 4) ^ (epi_obss_current_embedding.max().item() > 4)
        if epi_prefix_node_embedding.max() > 4:
            for i in range(len(episodes)):
                episodes[i]['prefix']['node_embedding'] /= 15
                episodes[i]['observations']['current_embedding'] /= 15

            epi_prefix_node_embedding = np.vstack([epi['prefix']['node_embedding'] for epi in episodes])
            epi_obss_current_embedding = np.vstack([epi['observations']['current_embedding'] for epi in episodes])
            print('epi_prefix_node_embedding', epi_prefix_node_embedding.max())
            print('epi_obss_current_embedding', epi_obss_current_embedding.max())
            with open(f'{base_path}/data/used/{ENV_NAME}/{data_name}.pkl', 'wb') as f:
                pickle.dump(episodes, f)    
                print(f'{data_name} reranged')