import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
import pickle
from utils.utils import load_data_and_check_quantity


# 数据量
ENV_NAME = 'OBP3D_V1'
num_problem = 10
num_episode = 0

# 获取指定环境的所有数据集   
file_list = glob.glob(os.path.join(f'{base_path}/data/used/{ENV_NAME}', '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
file_names = ['obp3d_train.pkl', 'obp3d_prompt.pkl']

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True)


train = file_data['obp3d_train']
for i in range(len(train)):
    #train[i]['observations']['current_state'] = train[i]['observations']['current_state'].astype(np.int32)
    train[i]['prefix'] = {'tag': np.array([1]*5, dtype=np.int32)}
    train[i]['prefix_masks'] = {'tag': np.zeros((len(train[i]['actions']), 5), dtype=bool)}
with open(f'{base_path}/data/used/{ENV_NAME}/obp3d_train.pkl', 'wb') as f:
    pickle.dump(train, f)    
    
prompt = file_data['obp3d_prompt']
for i in range(len(prompt)):
    #prompt[i]['observations']['current_state'] = prompt[i]['observations']['current_state'].astype(np.int32)
    prompt[i]['prefix'] = {'tag': np.array([1]*5, dtype=np.int32)}
    prompt[i]['prefix_masks'] = {'tag': np.zeros((len(prompt[i]['actions']), 5), dtype=bool)}
with open(f'{base_path}/data/used/{ENV_NAME}/obp3d_prompt.pkl', 'wb') as f:
    pickle.dump(prompt, f)    

print()
'''
# 写数据集 log 文件
np.set_printoptions(suppress=True, floatmode='fixed')
create_folder_overwrite_if_exist(f'{base_path}/visualize/dataset/{ENV_NAME}')
for file_name in file_names:
    data_name = file_name[:-4]
    # 可视化评估问题
    if data_name.endswith('problem'):
        problems = file_data[data_name]
        distances = []
        with open(f'{base_path}/visualize/dataset/{ENV_NAME}/{data_name}.txt', 'w') as file:
            for i in range(len(problems.answer_list)):
                obs = problems.problem_list[i]
                answer = problems.answer_list[i]

                pos_depot = obs['pos_depot']
                pos_node = obs['pos_node'].reshape((-1, 2))
                demand = obs['demand']
                capacity = CAPACITIES[len(demand)]

                distance = calc_vrp_distance(np.vstack((pos_depot[None,:], pos_node)), answer)
                distances.append(distance)
                
                #real_distance, real_answer, _ = CVRP_lkh(
                #    pos_depot.tolist(), 
                #    pos_node.tolist(), 
                #    demand.tolist(), 
                #    capacity
                #)
                #assert abs(distance - real_distance) < 1e-4
                
                node_info = np.hstack((pos_node, demand[:,None]))
                file.write('-'*25+f' problem-{i} '+'-'*25+'\n')
                file.write(f'depot position:\n {pos_depot}\n\n')    
                file.write(f'node info:\n{node_info}\n\n')    
                file.write(f'capacity:\t{capacity}\n')
                file.write(f'solution:\t{answer}\n\n')

        print(f'{data_name}.txt saved, ave distance={np.average(distances)}')    

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
                demand = prefix['demand']

                file.write('-'*25+f' episode-{i} '+'-'*25+'\n')
                file.write(f'pos_depot: \t{prefix["pos_depot"]}\n\n')
                for t in range(len(rewards)):
                    current_location = obss['current_position'][t]
                    capacity_left = obss['capacity'][t]
                    visited = obss['visited'][t]

                    masked_demand = demand.copy()
                    masked_demand[prefix_mask['demand'][t]] = 0
                    node_info = np.hstack((
                        prefix["pos_node"].reshape(-1,2), 
                        masked_demand[:,None], 
                        visited[:,None]
                    ))

                    file.write(f'node info:\n{node_info}\n')
                    file.write(f'current location:\t{current_location}\n')
                    file.write(f'capacity left:   \t{capacity_left}\n')
                    file.write(f'take action:     \t{acts[t]} (node{acts[t]-1})\n')
                    file.write(f'get reward:      \t{rewards[t]}\n\n')

        print(f'{data_name}.txt saved')
'''