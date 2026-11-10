import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_data import load_data_and_check_quantity
from utils.utils_slover import calc_op_total, OP_gurobi
from environment.used.Env_op_v1 import MAX_LENGTHS

# 数据量
ENV_NAME = 'OP_V4'
cop_type = 'op'
scale = 50
data_source = None
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集
data_base_path = base_path
file_dir = os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}') if data_source is None else \
            os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}_{data_source}')
file_list = glob.glob(os.path.join(file_dir, '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
#file_names = [name for name in file_names if 'op50_' in name]

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True,  data_base_path=data_base_path, data_source=data_source)

# 写数据集 log 文件
create_folder_overwrite_if_exist(f'{base_path}/visualize/dataset/{ENV_NAME}')
for file_name in file_names:
    data_name = file_name[:-4]
    # 可视化评估问题
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        total_prizes = []
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            #for i, obs in enumerate(problems):
            for i in range(len(problems.answer_list)):
                obs = problems.problem_list[i]
                answer = problems.answer_list[i]

                pos_depot = obs['pos_depot']
                pos_node = obs['pos_node'].reshape((-1, 2))
                prize = obs['prize']
                max_len = MAX_LENGTHS[len(prize)]
                total = calc_op_total(prize, [a-1 for a in answer])
                total_prizes.append(total)

                '''
                distance, real_answer = OP_gurobi(
                    pos_depot.tolist(), 
                    pos_node.tolist(), 
                    prize.tolist(), 
                    max_len
                )
                assert abs(total - calc_op_total(prize, real_answer)) < 1e-4
                '''

                node_info = np.hstack((pos_node, prize[:,None]))
                file.write('-'*25+f' problem-{i} '+'-'*25+'\n')
                file.write(f'depot position:\n {pos_depot}\n\n')    
                file.write(f'node info:\n{node_info}\n\n')    
                file.write(f'max length: \t{max_len}\n')
                file.write(f'solution:   \t{answer}\n')
                file.write(f'total prize:\t{round(total, 2)}\n\n')

        print(f'{data_name}.txt saved, ave prize={np.average(total_prizes)}')    
        
    # 可视化训练序列
    else:
        episodes = file_data[data_name]
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            for i, epi in enumerate(episodes):
                acts = epi['actions']
                rewards = epi['rewards']
                obss = epi['observations']
                prefix_mask = epi['prefix_masks']
                prefix = epi['prefix']
                assert prefix_mask['pos_depot'].sum() == 0
                
                prize = prefix['prize']
                total = calc_op_total(prize, [a-1 for a in acts])
                prize_timestep = []

                file.write('-'*25+f' episode-{i} '+'-'*25+'\n')
                file.write(f'pos_depot: \t{prefix["pos_depot"]}\n\n')
                for t in range(len(rewards)):
                    current_location = obss['current_position'][t]
                    length_left = obss['length'][t]
                    get_prize = obss['prize'][t]
                    visited = obss['visited'][t]

                    masked_prize = prize.copy()
                    masked_prize[prefix_mask['prize'][t]] = 0
                    node_info = np.hstack((
                        prefix["pos_node"].reshape(-1,2), 
                        masked_prize[:,None], 
                        visited[:,None]
                    ))

                    file.write(f'node info:\n{node_info}\n')
                    file.write(f'get prize:       \t{get_prize}\n\n')
                    file.write(f'current location:\t{current_location}\n')
                    file.write(f'length left:     \t{length_left}\n')
                    file.write(f'take action:     \t{acts[t]} (node{acts[t]-1})\n')
                    file.write(f'get reward:      \t{rewards[t]}\n')
                    prize_timestep.append(get_prize)
                
                file.write(f'\nget final prize:   \t{prize[acts[t]-1]}\n\n')       
                prize_timestep.append(prize[acts[t]-1])
                assert abs(sum(prize_timestep) - total) < 1e-4

        print(f'{data_name}.txt saved')