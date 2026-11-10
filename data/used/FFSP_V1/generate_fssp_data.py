import os
import sys
matnet_base_path = os.path.abspath('/data1/xxx/MatNet/FFSP/FFSP_MatNet')
sys.path.append(matnet_base_path)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import pickle
import numpy as np
import torch

from environment.used.BaseEnv_COP import DataProblem, RawData
from matnet_test import eval_main, env_params, tester_params

to_data_dir = '/data1/xxx/gato-revise/toy-gato-data/data/used/FFSP_V1/'

data_num_dict = {
        'train': 240000,
        'problem': 12000,
        'prompt': 24000,
}

stage_cnt = env_params['stage_cnt']
machine_cnt_list = env_params['machine_cnt_list']
process_time_params = env_params['process_time_params']
job_cnt = env_params['job_cnt']
batch_size = 1000
total_machine_cnt = sum(machine_cnt_list)

def get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params):

    time_low = process_time_params['time_low']
    time_high = process_time_params['time_high']

    problems_INT_list = []
    for stage_num in range(stage_cnt):
        machine_cnt = machine_cnt_list[stage_num]
        stage_problems_INT = torch.randint(low=time_low, high=time_high, size=(batch_size, job_cnt, machine_cnt))
        problems_INT_list.append(stage_problems_INT)

    return problems_INT_list





#####################################################
### generate prompt.pkl
total_trajs = []
total_problems = []
for _ in range(data_num_dict['prompt']//(batch_size*24)):
    problems_INT_list = get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params)
    trajs, returns = eval_main(problems_INT_list) # list of dict, 24倍
    problems = torch.cat(problems_INT_list, dim=2).cpu().numpy()  # (batch, job, machine_cnt * 3)
    problems = np.concatenate((problems, np.zeros((batch_size, 1, total_machine_cnt))), axis=1)  # (batch, job+1, machine_cnt * 3)
    problems = list(problems.repeat(24, axis=0)) #  batch *24 * (job+1, machine_cnt * 3)
    total_problems+=problems
    total_trajs += trajs
prompt_data = []
for i in range(data_num_dict['prompt']):
    rewards = np.zeros_like(total_trajs[i]['action'])
    rewards[-1] = 0
    terminals = np.zeros_like(total_trajs[i]['action'], dtype=bool)
    terminals[-1] = True
    prompt_data.append({
        'prefix': {'durations': total_problems[i].reshape(-1).astype(np.int32)},  # (job+1)*machine_cnt*3
        'prefix_masks': {'durations': total_trajs[i]['prefix_mask'].repeat(total_machine_cnt, axis=-1).astype(bool)},  #目前只对不同的job做mask，不在machine层做mask
        'observations': {'machine_query': total_trajs[i]['machine_query'].astype(np.int32)},
        'actions': total_trajs[i]['action'],
        'rewards': rewards,
        'terminals': terminals
    })

#####################################################   
### generate problem.pkl
total_trajs = []
total_problems = []
for _ in range(data_num_dict['problem']//(batch_size)):
    problems_INT_list = get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params)
    trajs, returns = eval_main(problems_INT_list) # list of dict, 24倍
    problems = torch.cat(problems_INT_list, dim=2).cpu().numpy()  # (batch, job, machine_cnt * 3)
    problems = np.concatenate((problems, np.zeros((batch_size, 1, total_machine_cnt))), axis=1)  # (batch, job+1, machine_cnt * 3)
    problems = list(problems.repeat(24, axis=0)) #  batch *24 * (job+1, machine_cnt * 3)
    total_problems+=problems
    total_trajs += trajs
problem_list = []
answer_list = []
for i in range(0, data_num_dict['problem']*24, 24):
    problem_list.append({
        'durations': total_problems[i].astype(np.int32),  # (job+1, machine_cnt*3)
    })
    answer_list.append(list(total_trajs[i]['action']))
problem_data = DataProblem(prefix_list=None, problem_list=problem_list, answer_list=answer_list)



#####################################################
### generate train.pkl

total_trajs = []
total_problems = []
for _ in range(data_num_dict['train']//(batch_size*24)):
    problems_INT_list = get_random_problems(batch_size, stage_cnt, machine_cnt_list, job_cnt, process_time_params)
    trajs, returns = eval_main(problems_INT_list) # list of dict, 24倍
    problems = torch.cat(problems_INT_list, dim=2).cpu().numpy()  # (batch, job, machine_cnt * 3)
    problems = np.concatenate((problems, np.zeros((batch_size, 1, total_machine_cnt))), axis=1)  # (batch, job+1, machine_cnt * 3)
    problems = list(problems.repeat(24, axis=0)) #  batch *24 * (job+1, machine_cnt * 3)
    total_problems+=problems
    total_trajs += trajs
train_data = []
for i in range(data_num_dict['train']):
    rewards = np.zeros_like(total_trajs[i]['action'])
    rewards[-1] = 0
    terminals = np.zeros_like(total_trajs[i]['action'], dtype=bool)
    terminals[-1] = True
    train_data.append({
        'prefix': {'durations': total_problems[i].reshape(-1).astype(np.int32)},  # (job+1)*machine_cnt*3
        'prefix_masks': {'durations': total_trajs[i]['prefix_mask'].repeat(total_machine_cnt, axis=-1).astype(bool)},  #目前只对不同的job做mask，不在machine层做mask
        'observations': {'machine_query': total_trajs[i]['machine_query'].astype(np.int32)},
        'actions': total_trajs[i]['action'],
        'rewards': rewards,
        'terminals': terminals
    })

with open(os.path.join(to_data_dir, f'ffsp{job_cnt}_train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
    print(f'train_data saved')

with open(os.path.join(to_data_dir, f'ffsp{job_cnt}_problem.pkl'), 'wb') as f:
    pickle.dump(problem_data, f)
    print(f'problem saved')

with open(os.path.join(to_data_dir, f'ffsp{job_cnt}_prompt.pkl'), 'wb') as f:
    pickle.dump(prompt_data, f)
    print(f'prompt_data saved')


#####################################################
### generate train_problem.pkl
problem_list = []
answer_list = []
if os.path.exists(os.path.join(to_data_dir, f'ffsp{job_cnt}_train.pkl')):
    print('generating train_problem.pkl from train.pkl')
    with open(os.path.join(to_data_dir, f'ffsp{job_cnt}_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
        assert (train_data[0]['prefix']['durations'] == train_data[23]['prefix']['durations']).all()
        for i in range(0, 500, 24):
            problem_list.append({'durations':train_data[i]['prefix']['durations'].reshape(21, 12)})
            answer_list.append(list(train_data[i]['actions']))
problem_data = DataProblem(prefix_list=None, problem_list=problem_list, answer_list=answer_list)
with open(os.path.join(to_data_dir, f'ffsp{job_cnt}_train_problem.pkl'), 'wb') as f:
    pickle.dump(problem_data, f)
    print(f'train_problem data saved')
exit()