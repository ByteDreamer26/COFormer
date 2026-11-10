import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_data import load_data_and_check_quantity
from utils.utils_slover import calc_pctsp_cost

# 数据量
ENV_NAME = 'PCTSP_V1'
cop_type = 'pctsp'
scale = 100
data_source = 'am_sampling'
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集   
file_dir = os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}') if data_source is None else \
            os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}_{data_source}')
file_list = glob.glob(os.path.join(file_dir, '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
#file_names = ['pctsp20_eval.pkl',]

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True, data_source=data_source)

# 写数据集 log 文件
#np.set_printoptions(suppress=True, floatmode='fixed')
create_folder_overwrite_if_exist(f'{base_path}/visualize/dataset/{ENV_NAME}')
for file_name in file_names:
    data_name = file_name[:-4]
    # 可视化评估问题
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        costs = []
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            #for i, obs in enumerate(problems):
            for i in range(len(problems.answer_list)):
                obs = problems.problem_list[i]
                answer = problems.answer_list[i]

                pos_depot = obs['pos_depot']
                pos_node = obs['pos_node'].reshape((-1, 2))
                prize = obs['prize']
                penalty = obs['penalty']
                prize2go = obs['prize2go'].item()
                current_position = obs['current_position']
                visited = obs['visited']
                assert np.array_equal(current_position, pos_depot)
                assert visited.sum() == 0
                cost = calc_pctsp_cost(pos_depot, pos_node, penalty, prize, answer[:-1])
                assert cost is not None
                costs.append(cost)

                '''
                real_answer = None
                while real_answer is None:
                    _, real_answer, _ = PCTSP_ILS(
                        pos_depot.tolist(), 
                        pos_node.tolist(), 
                        penalty.tolist(), 
                        prize.tolist()
                    )
                real_cost = calc_pctsp_cost(pos_depot, pos_node, penalty, prize, real_answer)
                assert real_cost is not None
                assert abs(cost - real_cost) < 1e-4
                '''

                node_info = np.hstack((pos_node, prize[:,None], penalty[:,None]))
                file.write('-'*25+f' problem-{i} '+'-'*25+'\n')
                file.write(f'depot position:\n {pos_depot}\n\n')    
                file.write(f'node info:\n{node_info}\n\n')    
                file.write(f'prize to go:\t{prize2go}\n')
                file.write(f'solution:   \t{answer}\n')
                file.write(f'total cost: \t{round(cost, 2)}\n\n')

        print(f'{data_name}.txt saved, ave prize={np.average(costs)}')    
        
    # 可视化训练序列
    else:
        episodes = file_data[data_name]
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            for i, epi in enumerate(episodes):
                acts = epi['actions']
                rewards = epi['rewards']
                obss = epi['observations']
                
                file.write('-'*25+f' episode-{i} '+'-'*25+'\n')
                file.write(f'pos_depot: \t{obss["pos_depot"][0]}\n\n')
                for t in range(len(rewards)):
                    node = obss['pos_node'][t].reshape((-1, 2))
                    prize = obss['prize'][t]
                    penalty = obss['penalty'][t]
                    node_info = np.hstack((node, prize[:,None], penalty[:,None]))
                    current_location = obss['current_position'][t]
                    visited = obss['visited'][t]
                    prize2go = obss['prize2go'][t].item()

                    file.write(f'node info:\n{node_info}\n')
                    file.write(f'visited:         \t{visited}\n')
                    file.write(f'current location:\t{current_location}\n')
                    file.write(f'take action:     \t{acts[t]} (to node {acts[t]-1})\n')
                    file.write(f'prize to go:     \t{prize2go}\n')
                    file.write(f'get prize:       \t{prize[acts[t]-1]}\n')
                    file.write(f'get reward:      \t{rewards[t]}\n\n')
                    
        print(f'{data_name}.txt saved')