import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)
from environment.used.BaseEnv_COP import DataProblem, RawData

import pickle
import glob
import scipy.sparse
import numpy as np
import time



data_num_dict = {
    'train': 200000,
    'problem': 10000,
}

node_num = 20

############################################## problem.pkl
from_data_file = '/data1/xxx/DIFUSCO/data/test_data/*gpickle'
from_label_dir = '/data1/xxx/DIFUSCO/data/test_annot/'
to_data_dir = '/data1/xxx/gato-revise/toy-gato-data/data/used/MIS_V1/'

t1 = time.time()
objs = []
file_lines = glob.glob(from_data_file)
problem_list = []
answer_list = []
for i in range(data_num_dict['problem']):
    if i % 1000 == 0:
        print(f'processing {i} for problem data')
    with open(file_lines[i], "rb") as f:
        graph = pickle.load(f)
    base_label_file = os.path.basename(file_lines[i]).replace('.gpickle', '_unweighted.result')
    node_label_file = os.path.join(from_label_dir, base_label_file)
    with open(node_label_file, 'r') as f:
        node_labels = [int(_) for _ in f.read().splitlines()]
    node_labels = np.array(node_labels, dtype=np.int64)
    assert node_labels.shape[0] == node_num
    objs.append(node_labels.sum())

    edges = np.array(graph.edges, dtype=np.int64)  # (edge_num, 2)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)  # add reverse edge
    # add self loop
    self_loop = np.arange(node_num).reshape(-1, 1).repeat(2, axis=1) # (node_num, 2)
    edges = np.concatenate([edges, self_loop], axis=0) # ( edges, 2)
    edges = edges.T   # (2, edges)
    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edges[0]), (edges[0], edges[1])),
    )
    adj_mat = adj_mat.toarray() # (node_num, node_num)
    actions = np.where(node_labels == 1)[0]
    actions = np.random.shuffle(actions)
    problem_list.append({'adj_mat':adj_mat})
    answer_list.append(actions)
problem_data = DataProblem(prefix_list=None, problem_list=problem_list, answer_list=answer_list)
with open(os.path.join(to_data_dir, f'mis{node_num}_problem.pkl'), 'wb') as f:
    pickle.dump(problem_data, f)
    print(f'problem saved')

t2 = time.time()
t = t2-t1
print(f"程序执行时间: {t:.4f} 秒")
objs = np.array(objs)
print(f'train objs: {objs.mean()}, {objs.std()}')

exit()





########################################## train.pkl
from_data_file = '/data1/xxx/DIFUSCO/data/train_data/*gpickle'
from_label_dir = '/data1/xxx/DIFUSCO/data/train_annot/'
file_lines = glob.glob(from_data_file)
train_data = []
objs = []
for i in range(data_num_dict['train']):
    if i % 1000 == 0:
        print(f'processing {i} for train data')
    with open(file_lines[i], "rb") as f:
        graph = pickle.load(f)
    base_label_file = os.path.basename(file_lines[i]).replace('.gpickle', '_unweighted.result')
    node_label_file = os.path.join(from_label_dir, base_label_file)
    with open(node_label_file, 'r') as f:
        node_labels = [int(_) for _ in f.read().splitlines()]
    node_labels = np.array(node_labels, dtype=np.int64)
    assert node_labels.shape[0] == node_num
    objs.append(node_labels.sum())

    edges = np.array(graph.edges, dtype=np.int64)  # (edge_num, 2)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)  # add reverse edge
    # add self loop
    self_loop = np.arange(node_num).reshape(-1, 1).repeat(2, axis=1) # (node_num, 2)
    edges = np.concatenate([edges, self_loop], axis=0) # ( edges, 2)
    edges = edges.T   # (2, edges)
    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edges[0]), (edges[0], edges[1])),
    )
    adj_mat = adj_mat.toarray() # (node_num, node_num)
    prefix = adj_mat.copy().reshape(-1).astype(np.int32) # (node_num*node_num,)

    actions = np.where(node_labels == 1)[0]
    np.random.shuffle(actions)
    all_states = []
    current_state = np.zeros(node_num, dtype=np.int32)
    ## 追溯每一步动作
    for j in range(actions.shape[0]):
        whether_connected = adj_mat[actions[j], :]
        current_state[whether_connected == 1] = 2 #相邻的边标为2
        current_state[actions[j]] = 1 #当前节点标为1
        all_states.append(current_state.copy())
    all_states = np.array(all_states, dtype=np.int32) # (actions.shape[0], node_num)

    index_mask = all_states != 0 # (actions.shape[0], node_num)
    batch_idx, mask_idx = np.where(index_mask)
    prefix_mask = np.zeros((actions.shape[0], node_num, node_num), dtype=bool)
    prefix_mask[batch_idx, mask_idx, :] = True
    prefix_mask[batch_idx, :, mask_idx] = True
    prefix_mask = prefix_mask.reshape(actions.shape[0], -1) # (actions.shape[0], node_num*node_num)

    ## 补上第一个动作之前的obs和mask
    all_states = np.concatenate([np.zeros((1, node_num), dtype=np.int32), all_states], axis=0)
    prefix_mask = np.concatenate([np.zeros((1, node_num*node_num), dtype=bool), prefix_mask], axis=0)
    ## 去掉最后一个终止状态
    all_states = all_states[:-1]
    prefix_mask = prefix_mask[:-1]

    rewards = np.zeros_like(actions)
    rewards[-1] = 0
    terminals = np.zeros_like(actions, dtype=bool)
    terminals[-1] = True
    train_data.append({
        'prefix': {'adj_mat': prefix}, # (node_num*node_num,)
        'prefix_masks': {'adj_mat': prefix_mask}, # (actions.shape[0], node_num*node_num)
        'observations': {'current_state': all_states}, # (actions.shape[0], node_num)
        'actions': actions, # (actions.shape[0],)
        'rewards': rewards, # (actions.shape[0],)
        'terminals': terminals # (actions.shape[0],)
    })






objs = np.array(objs)
print(f'train objs: {objs.mean()}, {objs.std()}')

with open(os.path.join(to_data_dir, f'mis{node_num}_train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
    print(f'train_data saved')




