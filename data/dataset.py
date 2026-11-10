import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import tree
import torch
import math
import random
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from utils.utils_data import RLTaskInput
from data.tokenizer import ContinuousScalarTokenizer
from data.adapter import DatasetAdapter

def get_loss_flag_and_position_id_v1(seq_length, prefix_dim, obs_dim, act_dim, prepend_trans_num=0, is_obs_pretrain=False):
    ''' There is a bug about position_id_res for prefix model, but some important ckpt was trained by this func '''
    step_size = obs_dim + act_dim + 1
    prepend_mask_length = (prefix_dim + 1) if prefix_dim is not None else (prepend_trans_num * step_size)
    prefix_dim = prefix_dim or 0

    loss_flag_res = np.zeros(seq_length, dtype=np.int64)
    position_id_res = np.zeros(seq_length, dtype=np.int64)

    # 从 1 开始生成观测的局部位置编码，这样动作的特殊位置编码为 0
    for i in range(prefix_dim, seq_length, step_size):
        position_id_res[i : i+obs_dim+1] = 1 + np.arange(min(obs_dim+1, seq_length-i))
    position_id_res[:prefix_dim] = 1 + np.arange(prefix_dim)
    
    # loss flag, 学习策略时对所有 action token 置 1; 观测预训练时对所有 obs token 置 1
    if is_obs_pretrain:
        for i, idx in enumerate(range(prepend_mask_length, seq_length, step_size)):
            if i > 0:
                loss_flag_res[idx: idx+obs_dim] = 1
    else:
        for i, idx in enumerate(range(prepend_mask_length, seq_length, step_size)):
            loss_flag_res[idx+obs_dim+1 : min(seq_length, idx+step_size)] = 1
        
    return loss_flag_res, position_id_res

def get_loss_flag_and_position_id_v2(seq_length, prefix_dim, obs_dim, act_dim, prepend_trans_num=0, is_obs_pretrain=False):
    ''' Bug free version '''
    step_size = obs_dim + act_dim + 1
    loss_flag_res = np.zeros(seq_length, dtype=np.int64)
    position_id_res = np.zeros(seq_length, dtype=np.int64)
    
    if prefix_dim is not None:
        assert prepend_trans_num == 0
        prepend_mask_length = prefix_dim + 1
        
        # 从 1 开始生成观测的局部位置编码，这样动作的特殊位置编码为 0
        position_id_res[:prepend_mask_length] = 1 + np.arange(prepend_mask_length)
        for i in range(prepend_mask_length, seq_length, step_size):
            position_id_res[i : i+obs_dim+1] = 1 + np.arange(min(obs_dim+1, seq_length-i))
    else:
        prefix_dim = 0
        prepend_mask_length = prepend_trans_num * step_size

        # 从 1 开始生成观测的局部位置编码，这样动作的特殊位置编码为 0
        for i in range(0, seq_length, step_size):
            position_id_res[i : i+obs_dim+1] = 1 + np.arange(min(obs_dim+1, seq_length-i))

    # action flag, 对所有 action token 置 1，其余置 0
    if is_obs_pretrain:
        for i, idx in enumerate(range(prepend_mask_length, seq_length, step_size)):
            if i > 0:
                loss_flag_res[idx: idx+obs_dim] = 1
    else:
        for i, idx in enumerate(range(prepend_mask_length, seq_length, step_size)):
            loss_flag_res[idx+obs_dim+1 : min(seq_length, idx+step_size)] = 1
        
    return loss_flag_res, position_id_res

class RLFullDataset(torch.utils.data.Dataset):
    def __init__(self, args, data:Dict, adapter:DatasetAdapter, dataset_name:str, env_name:str,):
        # 参数整理
        self.args = args
        self.data = data
        self.dataset_name = dataset_name
        self.env_name = env_name
        self.adapter = adapter
        dataset_meta = adapter.get_meta_data()
        self.act_type_spec = dataset_meta['act_type_spec']                              # 指示动作的数据类型
        self.prefix_type_spec = dataset_meta['prefix_type_spec']                        # 指示前缀中各个字段的数据类型
        self.obs_type_spec = dataset_meta['obs_type_spec']                              # 指示各个观测的数据类型
        self.act_dim = dataset_meta['act_dim']                                          # 动作张量维度，代表每个 transition action 对应的 token 个数
        self.obs_dim = dataset_meta['obs_dim']                                          # 各个观测的张量维度之和，代表每个 transition observation 对应的 token 个数
        self.prefix_dim = dataset_meta['prefix_dim']                                    # 前缀各个字段的张量维度之和，代表轨迹前缀对应的 token 个数
        self.obs_dims_for_spec = dataset_meta['obs_dims_for_spec']                      # 指示观测中各个字段的张量维度
        self.prefix_dims_for_spec = dataset_meta['prefix_dims_for_spec']                # 指示前缀中各个字段的张量维度
        
        self.traj_type = args.traj_type                                                 # 'all' or 'complete'
        self.prompt_strategy = args.prompt_strategy[:args.prompt_strategy.find(';')]    # 不使用 end of an episode 作为 prompt 时，设置 prompt 的策略
        self.output_sequence_length = args.n_position                                   # GPT 输出 token 序列长度，和输入相同
        self.num_discrete_values = args.num_discrete_values                             # 离散值对应的 token 个数
        self.num_continous_values = args.num_continuous_bin                             # 连续值对应的 token 个数
        self.act_spliter_token = args.special_tokens['<|>']                             # 分隔符对应的 special token
        assert self.act_spliter_token == self.num_discrete_values + self.num_continous_values
        self.discretizer = ContinuousScalarTokenizer(
            self.args.tokenizer_ver,
            self.num_continous_values,
            self.args.discretize_mu,
            self.args.discretize_M
        )         
        self.real_prepend_trans_num = 0                                                 # 最近返回样本中的 prompt transition 数量，仅用于 check_token_list_format 方法
        self.pad_num = 0                                                                # 最近返回样本中的 prompt token 数量，仅用于 check_token_list_format 方法
        
        # 数据整理
        self.prefixes = data['prefixes'] if self.args.use_prefix else None
        self.prefix_masks = data['prefix_masks'] if self.args.use_prefix and 'prefix_masks' in data else None
        assert (self.prefixes is None) == (self.prefix_masks is None)
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.path_lengths = np.asarray([len(x) for x in self.rewards], dtype=np.int32)
        self.traj_returns = np.asarray([e.sum() for e in self.rewards], dtype=np.float32)

        self.trans_dim = self.obs_dim + self.act_dim + 1
        if self.args.use_prefix:
            MDP_seq_len = self.output_sequence_length - self.prefix_dim
            self.transition_num = math.ceil(MDP_seq_len / self.trans_dim)   
            self.prompt_transition_num = 0
        else:
            MDP_seq_len = self.output_sequence_length
            self.transition_num = math.ceil(MDP_seq_len / self.trans_dim)                       # compute ceil of transition num      
            self.prompt_transition_num = int(self.args.prompt_ratio * self.transition_num)      # prompt transition 的数量
        self.predicted_transition_num = self.transition_num - self.prompt_transition_num        # 预测 transition 的数量

        num_of_path = self.path_lengths.shape[0]
        if self.traj_type == 'all':
            # indices 是观测的局部位置编码，设 3 条轨迹的 path_lenghths = [n0, n1, n2]
            # 则 indices 形如 [0,0,n0], [0,1,n0],..., [0,n0-2,n0], [1,0,n1], [1,1,n1],...,[0,n1-2,n1], [2,0,n2], [2,1,n2],...,[2,n2-2,n2]
            # 每条轨迹 ni 对应的 indices 都是 ni-1 少一个，这代表从这个轨迹片段能生成的样本数量（即任意轨迹 timestep 长度）至少为 2
            num_of_idx = np.sum(self.path_lengths-1)
            sample_idx = np.empty((num_of_idx, 3), dtype=np.int32)
            i_sample = 0
            for i in range(num_of_path):
                for j in range(self.path_lengths[i] - 1):
                    sample_idx[i_sample, 0] = i
                    sample_idx[i_sample, 1] = j
                    sample_idx[i_sample, 2] = min(j + self.transition_num, self.path_lengths[i])
                    i_sample += 1
            self.indices = sample_idx.astype(np.int32)
        elif self.traj_type == 'complete':
            # 只保留完整轨迹，这样后续就不会做zero-padding，提高训练效率（该方案还没有测试过）
            num_of_idx = np.sum(self.path_lengths - self.transition_num + 1)
            sample_idx = np.empty((num_of_idx, 3), dtype=np.int32)
            i_sample = 0
            for i in range(num_of_path):
                for j in range(self.path_lengths[i] - self.transition_num + 1):
                    sample_idx[i_sample, 0] = i
                    sample_idx[i_sample, 1] = j
                    sample_idx[i_sample, 2] = min(j + self.transition_num, self.path_lengths[i])
                    i_sample += 1
            self.indices = sample_idx.astype(np.int32)
        else:
            raise NotImplementedError(f"Unknown traj_type: {self.traj_type}")

        # XXX(DB1): we select prompt according to the length, why don't we select the top 10% rewarded-traj?
        # 按轨迹 return 对轨迹索引降序排序
        self.traj_idx_ret_tuples = sorted(
            [(i, self.traj_returns[i]) for i in range(len(self.path_lengths))],
            key=lambda x: x[1],
            reverse=True,
        )
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.get(idx, with_raw_obs=True)

    def split_dataset(self, weights:str):
        '''根据 weights 将数据集切分为训练集、验证集, 并打乱顺序'''
        def _build_dataset(idx_list):
            data = {    
                'prefixes': None if self.data['prefixes'] is None else [],
                'prefix_masks': None if self.data['prefix_masks'] is None else [],
                'observations': [], 'actions': [], 'rewards': [], 'terminals': []
            }
            for i in idx_list:
                data['observations'].append(self.data['observations'][i])
                data['actions'].append(self.data['actions'][i])
                data['rewards'].append(self.data['rewards'][i])
                data['terminals'].append(self.data['terminals'][i])
            if data['prefixes'] is not None:
                for i in idx_list:
                    data['prefixes'].append(self.data['prefixes'][i])
                    data['prefix_masks'].append(self.data['prefix_masks'][i])

            dataset = RLFullDataset(self.args, data, self.adapter, self.dataset_name, self.env_name)
            return dataset

        sample_num = len(self.data['observations'])
        weights = [int(n) for n in weights.split(',')]
        weights = [n/sum(weights) for n in weights]
        amounts = [int(w*sample_num) for w in weights]
        train_num, val_num = amounts[0], amounts[1]
        assert train_num != 0 and val_num != 0

        # Shuffle indices before building datasets
        indices = list(range(sample_num))
        if self.args.eval_problem_set != 'train_problem':
            random.seed(self.args.seed)
            random.shuffle(indices)

        dataset_train = _build_dataset(indices[:train_num])
        dataset_val = _build_dataset(indices[train_num: train_num+val_num])
        return dataset_train, dataset_val

    def get_prefix_obs_action_by_path_idx(
        self, path_ind: int, start_ind: int = None, end_ind: int = None
    ) -> Tuple[Union[Dict, None], Union[Dict, np.ndarray], np.ndarray]:
        start_ind = 0 if start_ind is None else start_ind
        end_ind = len(self.actions[path_ind]) if end_ind is None else end_ind
        actions = self.actions[path_ind][start_ind:end_ind]
        observations = tree.map_structure(
            lambda x: x[start_ind:end_ind], 
            self.observations[path_ind]
        )
        prefix = None if not self.args.use_prefix else self.prefixes[path_ind]
        prefix_mask =  None if self.prefix_masks is None else tree.map_structure(
            lambda x: x[start_ind:end_ind], 
            self.prefix_masks[path_ind]
        )
        return prefix, prefix_mask, observations, actions

    def prepend_prompt(
        self, path_idx: int, observations: Union[Dict, np.ndarray], actions: np.ndarray
    ) -> Tuple[Union[Dict, np.ndarray], np.ndarray]:
        assert self.args.use_prompt
        real_prepend_trans_num = 0  # prompt 序列的 timestep 长度

        # 以 prompt_prob 概率设置 prompt 序列
        if np.random.random() < self.args.prompt_prob:
            assert path_idx >= 0
            _, _, obs_traj, action_traj = self.get_prefix_obs_action_by_path_idx(path_idx)   # 目标 prompt 所在序列的 obss 和 actions
            path_length = self.path_lengths[path_idx]                           # 目标 prompt 所在序列的 timestep 长度
            
            # prompt 序列有概率来自目标轨迹末尾 XXX(DB1) prompt as goal
            if np.random.random() < self.args.prompt_at_final_transition_prob:        
                # 截取轨迹尾部的至多 prompt_transition_num 个 transition, 如果没有这么多，则有多少截取多少
                trans_obs = tree.map_structure(lambda x: x[-self.prompt_transition_num:], obs_traj)
                trans_act = action_traj[-self.prompt_transition_num :]

            # prompt 序列有概率均匀从目标轨迹中采样 XXX(DB1) prompt as exploration
            else:                                                           
                if self.prompt_strategy == "stochastic_timestep":            
                    # 在目标轨迹中均匀采样 prompt_transition_num 个 transition，不一定要连续
                    # 先在其长度范围内均匀选择 prompt_transition_num 个 timestep，然后从小到大排序，提取对应 transition 作为 prompt
                    random_idx = np.random.choice(path_length, self.prompt_transition_num, replace=False)   
                    random_idx.sort()
                    trans_obs = tree.map_structure(lambda x: x[random_idx], obs_traj)
                    trans_act = action_traj[random_idx]
                elif self.prompt_strategy == "stochastic_subseq":
                    # 在可行范围内均匀采样连续一个起点 timestep 然后取连续 prompt_transition_num 个 transition 组成子轨迹作为 prompt
                    random_start = np.random.choice(max(path_length - self.prompt_transition_num, 1))
                    random_end = random_start + self.prompt_transition_num
                    trans_obs = tree.map_structure(lambda x: x[random_start:random_end], obs_traj)
                    trans_act = action_traj[random_start:random_end]
                else:
                    raise NotImplementedError

            # 提取的 prompt 轨迹包含的 timestep 数量
            real_prepend_trans_num = len(trans_act)     

            # 从原轨迹中截取一部分接在 prompt 轨迹之后，这里先在原轨迹可行范围内均匀采样一个起始 timestep，然后往后截取 predicted_transition_num 长度
            offset_range = max(0, len(actions) - self.predicted_transition_num)          
            offset = np.random.choice(offset_range) if offset_range > 0 else offset_range
            observations = tree.map_structure(                                      
                lambda x: x[offset : offset + self.predicted_transition_num],
                observations,
            )

            # 准备一个全 0 数组，然后填充拼接 prompt 结果得到新轨迹观测 obs_holder
            obs_holder = tree.map_structure(                                 
                lambda x, y: np.zeros((x.shape[0] + y.shape[0],) + x.shape[1:], dtype=x.dtype),
                observations, trans_obs,
            )
            for k, v in obs_holder.items():
                v[:real_prepend_trans_num] = trans_obs[k]
                v[real_prepend_trans_num : real_prepend_trans_num + observations[k].shape[0]] = observations[k]

            # 类似地拼接 prompt 得到新轨迹动作 act_holder
            actions = actions[offset : offset + self.predicted_transition_num]
            act_holder = np.zeros(
                (trans_act.shape[0] + actions.shape[0],) + actions.shape[1:],
                dtype=actions.dtype,
            )
            act_holder[:real_prepend_trans_num] = trans_act
            act_holder[real_prepend_trans_num:] = actions

        # 以 1 - prompt_prob 概率不设置 prompt 序列
        else:
            obs_holder = observations
            act_holder = actions

        return obs_holder, act_holder, real_prepend_trans_num

    def postprocess_to_token(
        self, obs_array: Union[Dict, np.ndarray], act_array: np.ndarray, prefix: Union[Dict, None] = None,
    ) -> Tuple[Tuple["Text", "Tensor"], "Actions"]:  # type: ignore pylance
        # post_process for observations & actions
        obs_array = self.adapter.post_process_obs(obs_array)
        act_array = self.adapter.post_process_act(act_array)
        prefix_array = self.adapter.post_process_prefix(prefix)

        def _postprocess_obs(obs_array, obs_type):
            o_tensor = None
            
            if obs_type == "float":
                obs_array = self.discretizer.discretize(obs_array, is_action=False).numpy()
                o_tensor = obs_array + self.num_discrete_values
            elif obs_type == "int":
                assert obs_array.min() >= 0 and obs_array.max() < self.num_discrete_values
                o_tensor = obs_array
            else:
                raise ValueError(f'obs_type_spce can only be "float" or "int", instead of "{obs_type}"')

            # resize to (traj_len, obs_dim)
            if o_tensor is not None:
                assert o_tensor.ndim in [1, 2]   
                if o_tensor.ndim == 1:              # (traj_len, ) -> (traj_len, 1)
                    o_tensor = o_tensor[:, None]    

            return o_tensor

        def _postprocess_prefix(prefix_array, prefix_type):
            if prefix_array is None:
                return None

            if prefix_type == "float":
                prefix_array = self.discretizer.discretize(prefix_array, is_action=False).numpy()
                p_tensor = prefix_array + self.num_discrete_values
            elif prefix_type == "int":
                assert prefix_array.min() >= 0 and prefix_array.max() < self.num_discrete_values
                p_tensor = prefix_array
            else:
                raise ValueError(f'prefix_type_spce can only be "float" or "int", instead of "{prefix_type}"')

            assert p_tensor.ndim == 1   # (prefix_dim, )
            return p_tensor
    
        # obs & prefix tokenize
        p_tensor = None if not self.args.use_prefix else tree.map_structure(
            _postprocess_prefix,
            None if self.prefix_type_spec is None else {k:prefix_array[k] for k in self.prefix_type_spec},
            self.prefix_type_spec,
        )
        o_tensor = tree.map_structure(
            _postprocess_obs,
            None if self.obs_type_spec is None else {k:obs_array[k] for k in self.obs_type_spec},
            self.obs_type_spec,
        )

        # actions tokenize 
        if self.act_type_spec == 'float':
            act_array = self.discretizer.discretize(act_array, is_action=True).numpy()
            processed_act = act_array + self.num_discrete_values
        elif self.act_type_spec == 'int':
            assert act_array.min() >= 0 and act_array.max() < self.num_discrete_values
            # resize to (traj_len, act_dim)
            assert act_array.ndim in [1, 2]
            if act_array.ndim == 1:            
                act_array = act_array[:, None]  # (traj_len, ) -> (traj_len, 1)   
            processed_act = act_array
        else:
            raise ValueError(f'act_type_spce can only be "float" or "int", instead of "{self.act_type_spec}"')

        return (p_tensor, o_tensor), processed_act

    def get(self, idx: int, with_raw_obs=True):
        ''' 获取数据集中索引为 idx 的样本，包装为 RLTaskInput 返回 '''
        def _truncate_or_pad_to_match_seq_len(arr: np.ndarray, seq_len: int):
            if len(arr) > seq_len:
                return arr[:seq_len], 0
            elif len(arr) < seq_len:
                pad_num = seq_len - len(arr)
                return np.pad(arr, (0, pad_num)), pad_num # by default it will pad constant zero
            else:
                return arr, 0

        idx = idx % len(self.indices)                       # 取模以防目标样本索引超过样本个数
        path_ind, start_ind, end_ind = self.indices[idx]    # 样本所在轨迹索引 & 样本起止位置timestep索引
        path_length = self.path_lengths[path_ind]           # 样本所在轨迹的 timestep 度

        # 加载样本轨迹
        prefix, prefix_mask, observations, actions = self.get_prefix_obs_action_by_path_idx(path_ind, start_ind, end_ind)
        
        # 设置 prompt 序列
        self.real_prepend_trans_num = 0  
        if self.args.use_prompt:
            assert prefix is None
            rand_path_idx = np.random.choice(len(self.path_lengths))            
            observations, actions, prepend_trans_num = self.prepend_prompt(rand_path_idx, observations, actions)
            self.real_prepend_trans_num = prepend_trans_num

        # 对观测和动作进行 tokenize 和后处理
        (p_tensor, o_tensor), act_discrete = self.postprocess_to_token(observations, actions, prefix)

        if p_tensor is not None:
            prefix_discrete = [p_tensor[k] for k in sorted(p_tensor)]
            prefix_discrete = np.concatenate(prefix_discrete)                   # (prefix_dim, )
            prefix_mask = [prefix_mask[k] for k in sorted(p_tensor)]
            prefix_mask = np.hstack(prefix_mask)                                # (traj_len, prefix_dim, )
            prefix_mask = prefix_mask.reshape(-1, prefix_mask.shape[-1])        # (act_num, prefix_dim)
        else:
            prefix_discrete = prefix_mask = None

        if o_tensor is not None:
            obs_discrete = [o_tensor[k] for k in sorted(o_tensor)]              # 按字典的键排序
            obs_discrete = np.concatenate(obs_discrete, axis=1)                 # (traj_len, obs_dim)
            assert obs_discrete.shape[1] == self.obs_dim
        
        # 拼接 token 序列
        joined_discrete = np.concatenate(                                       # (traj_len, obs_dim + 1 + act_dim)
            [
                obs_discrete,                                                   # (traj_len, obs_dim)
                self.act_spliter_token * np.ones((act_discrete.shape[0], 1)),   # (traj_len, 1)
                act_discrete,                                                   # (traj_len, act_dim)
            ], axis=1,
        )
        
        # 拉平成一维变成输入 GPT 模型的形式
        joined_discrete = joined_discrete.flatten().astype(np.int64)        # (traj_token_num, )
        if prefix_discrete is not None:
            joined_discrete = np.concatenate([prefix_discrete.squeeze(), [self.args.special_tokens['<X>']], joined_discrete])

        # 获取 loss_flag 和观测局部位置编码 position_id
        seq_len = len(joined_discrete)
        real_seq_len = seq_len - 1                                          # 进行自回归训练需错一位构造样本和标签，真实长度要-1
        get_loss_flag_and_position_id = get_loss_flag_and_position_id_v1 if self.args.local_pos_ver == 'v1' else get_loss_flag_and_position_id_v2
        loss_flag, position_id = get_loss_flag_and_position_id(             # (traj_token_num, )
            seq_len,
            self.prefix_dim if self.args.use_prefix else None,
            self.obs_dim,
            self.act_dim,
            self.real_prepend_trans_num,
            self.args.is_obs_pretrain
        )
        assert end_ind <= path_length

        # truncate to self.output_length+1
        target_seq_len = self.output_sequence_length + 1
        assert position_id.shape[0] == loss_flag.shape[0] == joined_discrete.shape[0]
        
        position_id, _ = _truncate_or_pad_to_match_seq_len(position_id, target_seq_len)                # (target_seq_len, )
        loss_flag, _ = _truncate_or_pad_to_match_seq_len(loss_flag, target_seq_len)                    # (target_seq_len, )
        joined_discrete, pad_num = _truncate_or_pad_to_match_seq_len(joined_discrete, target_seq_len)  # (target_seq_len, )
        self.pad_num = pad_num if pad_num == 0 else pad_num - 1
        prefix_mask = None if prefix_mask is None else prefix_mask[:(joined_discrete == self.act_spliter_token).sum()]

        # 包装成 RLTaskInput
        attention_mask = np.zeros(self.output_sequence_length, dtype=np.uint8)
        last_spliter = np.where(joined_discrete[:-1] == self.act_spliter_token)[0][-1]
        attention_mask[:last_spliter+1] = 1
        res = RLTaskInput(
            position_id=position_id[:-1],       # 局部位置编码，取前 1024
            attention_mask=attention_mask,      # 最后一个分隔符及之前所有位置为1（和 llama 输入要求一致）
            tensor_seq=joined_discrete[:-1],    # token 序列，取前 1024
            loss_mask=loss_flag[1:],            # loss_mask 指出需要计算 loss 的位置，错一位取后 1024
            label=joined_discrete[1:],          # 计算 loss 时的 ground_truth, 错一位取后 1024 
            seq_len=real_seq_len,               # token序列样本长度，用于 auto seq_len
            prefix_mask=prefix_mask
        )
        res.apply(lambda x: torch.tensor(x))    # 将 res 对象中所有不为 None 的张量都转换成 PyTorch 张量
        res.apply(lambda x: x[None, ...])       # 在每个张量的第一维上增加了一个维度，将其转换成一个 batch。这里 position_id、tensor_seq、loss_mask、label 尺寸都变成 (1, 1024)
                
        if with_raw_obs:
            observations = {k:torch.tensor(v) for k, v in observations.items()}
            return res, observations
        else:
            return res

    def sample_expert_demonstration_for_prompt(
        self, strategy: str, strict_length: bool, sample_peak: bool, path_idx: int=None
    ) -> Dict[str, np.ndarray]:
        assert self.args.use_prompt

        # only first n_trans is needed
        assert strategy == "moving_prompt"
        prompt_length = self.prompt_transition_num if strategy == "fixed_prompt" else \
                        self.transition_num - 2     # -1是为了拼接问题初始观测；-1为了解决计算transition_num时的ceil操作。保证第一个动作预测时 token 序列总长度不超过 seq_length
        
        # 候选prompt轨迹索引集合
        if sample_peak:
            # select demonstration from the top 10%
            stop_idx = int(len(self.traj_idx_ret_tuples) * 0.1)
            candidates_idx = [x[0] for x in self.traj_idx_ret_tuples[:stop_idx]]
        else:
            candidates_idx = np.arange(len(self.path_lengths))

        # 随机选择一条轨迹用作prompt
        if path_idx is not None:
            assert path_idx in candidates_idx
        else:
            path_idx = np.random.choice(candidates_idx)
        _, _, observation_traj, action_traj = self.get_prefix_obs_action_by_path_idx(path_idx)

        if strict_length:
            # splice other expert episodes to ensure sampled demonstration lenghth not less than prompt_length
            current_length = len(action_traj)
            obs_list, act_list = [observation_traj], [action_traj]
            while current_length < prompt_length:
                path_idx = np.random.choice(candidates_idx)
                _, _, observation_traj, action_traj = self.get_prefix_obs_action_by_path_idx(path_idx)
                obs_list.append(observation_traj)
                act_list.append(action_traj)
                current_length += len(action_traj)
            observation_traj = tree.map_structure(lambda *xs: np.concatenate(xs, axis=0), *obs_list)
            action_traj = np.concatenate(act_list, axis=0)

        actions = action_traj[:prompt_length]
        observations = tree.map_structure(lambda x: x[:prompt_length], observation_traj)
        (_, o_tensor), a_tensor = self.postprocess_to_token(observations, actions)

        # append parsed obs
        return {
            "a_tensor": a_tensor,       # (prompt_length, act_dim)
            "o_tensor": o_tensor,       # (prompt_length, obs_dim)
            "prompt_raw_obs": observations
        }

    def check_token_list_format(self, res:RLTaskInput):
        def _check_value_range(type_spec, tokens):
            if type_spec == 'float':
                assert tokens.min() >= self.num_discrete_values and tokens.max() <= self.num_discrete_values + self.num_continous_values - 1
            elif type_spec == 'int':
                assert tokens.min() >= 0 and tokens.max() <= self.num_discrete_values - 1
            else:
                raise False
        
        # 分离各类数据对应的 token
        assert self.pad_num >= 0
        joined_discrete = res.tensor_seq.numpy().squeeze()
        joined_discrete = joined_discrete[:-self.pad_num] if self.pad_num > 0 else joined_discrete

        mask = np.empty(self.output_sequence_length - self.pad_num, dtype='U10')
        if self.args.use_prompt:
            for i in range(0, mask.size, self.obs_dim+1+self.act_dim):
                mask[i : i+self.obs_dim] = 'obs'
                mask[i+self.obs_dim : i+self.obs_dim+1] = 'a_spliter'
                mask[i+self.obs_dim+1 : i+self.obs_dim+1+self.act_dim] = 'act'
            observations = joined_discrete[mask=='obs']
            act_spliters = joined_discrete[mask=='a_spliter']
            actions = joined_discrete[mask=='act']
            prefixes = None
        elif self.args.use_prefix:
            mask[: self.prefix_dim] = 'prefix'
            assert joined_discrete[self.prefix_dim].item() == self.args.special_tokens['<X>']
            for i in range(self.prefix_dim+1, mask.size, self.obs_dim+1+self.act_dim):
                mask[i : i+self.obs_dim] = 'obs'
                mask[i+self.obs_dim : i+self.obs_dim+1] = 'a_spliter'
                mask[i+self.obs_dim+1 : i+self.obs_dim+1+self.act_dim] = 'act'
            observations = joined_discrete[mask=='obs']
            actions = joined_discrete[mask=='act']
            act_spliters = joined_discrete[mask=='a_spliter']
            prefixes = joined_discrete[mask=='prefix']
        else:
            raise Exception('Must use prompt or prefix')

        # 检查动作分隔符
        assert (act_spliters == self.act_spliter_token).all()

        # 检查action token是否在指定取值范围内
        _check_value_range(self.act_type_spec, actions)

        # 检查 observation token 是否在指定取值范围内
        obs_dim_sorted = [self.obs_dims_for_spec[obs_name] for obs_name in sorted(self.obs_type_spec)]
        obs_dim_interval = {obs_name: [sum(obs_dim_sorted[:i]), sum(obs_dim_sorted[:i+1])] for obs_name, i in zip(sorted(self.obs_type_spec), range(len(obs_dim_sorted)))} 
        mask_obs = np.empty(observations.shape[0], dtype='U20')
        for i in range(0, mask_obs.size, self.obs_dim):
            for obs_name, dim_interval in obs_dim_interval.items():
                mask_obs[i+dim_interval[0] : i+dim_interval[1]] = obs_name
        obs_spec = {obs_name: observations[mask_obs==obs_name] for obs_name in self.obs_type_spec.keys()}
        for obs_name, obs_content in obs_spec.items():
            _check_value_range(self.obs_type_spec[obs_name], obs_content)

        # 检查 prefix token 是否在指定取值范围内
        if self.args.use_prefix:
            prefix_dim_sorted = [self.prefix_dims_for_spec[prefix_name] for prefix_name in sorted(self.prefix_type_spec)]
            prefix_dim_interval = {prefix_name: [sum(prefix_dim_sorted[:i]), sum(prefix_dim_sorted[:i+1])] for prefix_name, i in zip(sorted(self.prefix_type_spec), range(len(prefix_dim_sorted)))} 
            mask_prefix = np.empty(prefixes.shape[0], dtype='U20')
            for prefix_name, dim_interval in prefix_dim_interval.items():
                mask_prefix[dim_interval[0] : dim_interval[1]] = prefix_name
            prefix_spec = {prefix_name: prefixes[mask_prefix==prefix_name] for prefix_name in self.prefix_type_spec.keys()}
            for prefix_name, prefix_content in prefix_spec.items():
                _check_value_range(self.prefix_type_spec[prefix_name], prefix_content)

        # 检查 loss_mask 是否和 action token 位置匹配
        # loss_mask 为 1 的位置会计算损失，基于 autoregress 原理，应该比 action token 真实位置提前一位
        # 因为不对 prompt/prefix 部分计算 loss，prompt/prefix token 部分全部为 0
        prepend_token_num = self.real_prepend_trans_num * self.trans_dim if self.args.use_prompt else self.prefix_dim
        action_mask = np.array(mask=='act', dtype=np.int32)[prepend_token_num+1:]
        loss_mask = np.concatenate((np.zeros(prepend_token_num, dtype=np.int32), action_mask))  # (seq_len-1, ) 这里从 res.tensor_seq 计算应有的 loss_mask，最后一个算不到
        res_loss_mask = res.loss_mask.numpy().squeeze().astype(np.int32)
        res_loss_mask = res_loss_mask[:-self.pad_num] if self.pad_num > 0 else res_loss_mask
        assert np.array_equal(loss_mask, res_loss_mask[:-1])                                    # 只检查前 seq_len-1 长度
        
        # 检查 position_id 是否和 observation token 位置匹配
        position_id = np.zeros(self.output_sequence_length-self.pad_num, dtype=np.int64)
        if self.args.use_prefix:
            position_id[:self.prefix_dim+1] = np.arange(self.prefix_dim+1, dtype=np.int64) + 1
            for i in range(self.prefix_dim+1, self.output_sequence_length, self.trans_dim):
                position_id[i : i+self.obs_dim+1] = 1 + np.arange(min(self.obs_dim+1, self.output_sequence_length-self.pad_num-i))
        else:
            for i in range(0, self.output_sequence_length, self.trans_dim):
                position_id[i : i+self.obs_dim+1] = 1 + np.arange(min(self.obs_dim+1, self.output_sequence_length-self.pad_num-i))
        res_position_id = res.position_id.numpy().squeeze()
        res_position_id = res_position_id[:-self.pad_num] if self.pad_num > 0 else res_position_id
        assert np.array_equal(position_id, res_position_id)

class BlendableDataset(torch.utils.data.Dataset):
    """A naive implementation of collection of multiple datasets sampled in a weighted round-robin manner."""
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        weights: List,                          # 不同数据集的采样比例
        batch_size: Optional[int] = None,       # 一个 batch 从所有数据集采样的样本总量
        log_data: bool = False,                 # 记录加载数据所在的数据集和数据集内偏移            
        with_dataset_info: bool = False         # 加载数据时，同时返回 tokenize 前的原始数据 & 数据所在数据集的索引
    ):
        super().__init__()
        self.datasets = datasets
        self.with_dataset_info = with_dataset_info
        self.log_data = log_data
        self.logged_data = {dataset.dataset_name:[] for dataset in datasets} if log_data else None

        batch_size = len(datasets) if batch_size is None else batch_size
        assert batch_size >= len(datasets)
        self.sample_batch_size = batch_size    

        # 采样比例转成概率形式
        weights = torch.tensor(weights, dtype=torch.float32)
        assert (weights > 0).all()
        weights /= weights.sum()    

        # 一个 batch 中各个数据集片段的起始索引
        num_samples_one_batch = (batch_size * weights).round()          # 一个长度为数据集数量的一维 tensor，指出每个batch中需要在各个数据集中采样的数据数量，例如 [2, 2, 2, 2, 2]
        offset_in_batch = num_samples_one_batch.cumsum(0).int().numpy() # [2, 4, 6, 8, 10]
        self.offset_in_batch = np.zeros_like(offset_in_batch)
        self.offset_in_batch[1:] = offset_in_batch[:-1]                 # [0, 2, 4, 6, 8] 每个数据集对应批次内的偏移量
        
        # 一个 batch 中各个数据集片段的长度
        self.num_in_batch = np.array(self.offset_in_batch[1:].tolist() + [batch_size]) - self.offset_in_batch
        
        # 计算总样本量，要求遍历一次该尺寸的索引，可以在依概率采样的情况下保证最大的子数据集所有数据都访问一次，小的子数据集部分数据被重复访问
        dlen = torch.tensor([len(dataset) for dataset in self.datasets])
        self.size = math.ceil((dlen / weights).max())

    def get_logged_data(self):
        assert self.log_data
        logged_data = self.logged_data
        self.logged_data = {dataset.dataset_name:[] for dataset in self.datasets}
        return logged_data

    def __len__(self):
        return self.size

    def __getitem__(self, idx, with_raw_obs=True):
        inner_batch_idx = idx % self.sample_batch_size
        mix_batch_num = idx // self.sample_batch_size

        # 找出 idx 刚好大于的那个 offset_in_batch，这是该样本来自的数据集索引
        dataset_idx = np.argwhere(self.offset_in_batch <= inner_batch_idx).max()  
        dataset = self.datasets[dataset_idx]

        # 将混合数据集的索引映射到某个子数据集中
        data_num_in_batch = self.num_in_batch[dataset_idx]
        data_offset_in_batch = self.offset_in_batch[dataset_idx]
        inner_dataset_offset = mix_batch_num * data_num_in_batch + (inner_batch_idx - data_offset_in_batch)
        inner_dataset_offset = int(inner_dataset_offset % len(dataset))

        # 若需要检查训练使用的数据，在此记录
        if self.log_data:
            self.logged_data[dataset.dataset_name].append(inner_dataset_offset)

        if self.with_dataset_info:
            return dataset.get(inner_dataset_offset, with_raw_obs), (dataset.dataset_name, inner_dataset_offset)
        else:
            return dataset.get(inner_dataset_offset, with_raw_obs)