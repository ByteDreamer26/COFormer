import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import tree
import numpy as np
from data.dataset import RLFullDataset
from data.adapter import DatasetAdapter
from typing import Dict, List, Union, Tuple

class DDP_RLFullDataset(RLFullDataset):
    ''' 本类对象仅用于批量评估时采样 batch prompt 序列 '''
    def __init__(self, args, data:Dict, adapter:DatasetAdapter, dataset_name:str, env_name:str):
        super().__init__(args, data, adapter, dataset_name, env_name)
        
    def split_dataset(self, weights:str):
        # 根据weights将数据集切分为训练集、验证集
        def _build_dataset(start, end):
            data = {    
                'prefixes': None if self.data['prefixes'] is None else self.data['prefixes'][start:end],
                'observations':self.data['observations'][start:end],
                'actions':self.data['actions'][start:end],
                'rewards':self.data['rewards'][start:end],
                'terminals':self.data['terminals'][start:end]
            }
            dataset = DDP_RLFullDataset(self.args, data, self.adapter, self.dataset_name, self.env_name)
            #for i in range(len(dataset)):
            #    dataset.check_token_list_format(dataset.get(i))
            return dataset

        sample_num = len(self.data['observations'])
        weights = [int(n) for n in weights.split(',')]
        weights = [n/sum(weights) for n in weights]
        amounts = [int(w*sample_num) for w in weights]
        train_num, val_num = amounts[0], amounts[1]
        assert train_num != 0 and val_num != 0

        dataset_train = _build_dataset(0, train_num)
        dataset_val = _build_dataset(train_num, train_num+val_num)
        return dataset_train, dataset_val

    def get_prefix_obs_action_by_path_idx(
        self, path_idxs: List[int], start_idxs: List[int] = None, end_idxs: List[int] = None
    ) -> Tuple[List, List, List, List]:
        batch_actions = []
        batch_observations = []
        batch_len = []
        if start_idxs is None and end_idxs is None:
            for idx in path_idxs:
                batch_actions.append(self.actions[idx])
                batch_observations.append(self.observations[idx])
                batch_len.append(len(batch_actions[-1]))
        else:
            for idx, start, end in zip(path_idxs, start_idxs, end_idxs):
                start = 0 if start is None else start
                end = len(self.actions[idx]) if end is None else end
                batch_actions.append(self.actions[idx][start:end])
                batch_observations.append(tree.map_structure(
                    lambda x: x[start:end], 
                    self.observations[idx]
                ))
                batch_len.append(len(batch_actions[-1]))
        
        return batch_observations, batch_actions, batch_len
    
    def postprocess_to_token(
        self, obs_array: Union[Dict, np.ndarray], act_array: np.ndarray, prefix: Union[Dict, None] = None,
    ) -> Tuple[Tuple["Text", "Tensor"], "Actions"]:  # type: ignore pylance
        """Process prefix, observations and actions
        Observation: Suppose two forms, a single ndarray or a dict of ndarray
            if float number, do discretize.
        Action:
            if float number, do discretize
        prefix: A dict of ndarray
            if float number, do discretize.

        return: (prefix=None, tensor=None), act
        """
        # post_process for observations & actions
        obs_array = self.adapter.post_process_obs(obs_array)
        act_array = self.adapter.post_process_act(act_array)
        prefix_array = self.adapter.post_process_prefix(prefix)

        # observations tokenize 
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

            # resize o_tensor to (batch_size, traj_len, obs_item_dim)
            if o_tensor is not None:
                assert o_tensor.ndim == 2 or o_tensor.ndim == 3
                if o_tensor.ndim == 2:              
                    o_tensor = o_tensor[:, :, None]     # (batch_size, traj_len, ) -> (batch_size, traj_len, 1)   
           
            return o_tensor

        def _postprocess_prefix(prefix_array, prefix_type):
            if prefix_type == "float":
                prefix_array = self.discretizer.discretize(prefix_array, is_action=False).numpy()
                p_tensor = prefix_array + self.num_discrete_values
            elif prefix_type == "int":
                assert prefix_array.min() >= 0 and prefix_array.max() < self.num_discrete_values
                p_tensor = prefix_array
            else:
                raise ValueError(f'prefix_type_spce can only be "float" or "int", instead of "{prefix_type}"')

            assert p_tensor.ndim == 2   # (batch_size, prefix_dim, )
            return p_tensor
        
        def _postprocess_act(act_array, act_type):
            if act_type == 'float':
                act_array = self.discretizer.discretize(act_array, is_action=True).numpy()
                a_tensor = act_array + self.num_discrete_values
            elif act_type == 'int':
                assert act_array.min() >= 0 and act_array.max() < self.num_discrete_values            
                a_tensor = act_array
            else:
                raise ValueError(f'act_type_spce can only be "float" or "int", instead of "{self.act_type_spec}"')

            # resize a_tensor to (batch_size, traj_len, act_dim)
            if a_tensor is not None:
                assert a_tensor.ndim == 2 or a_tensor.ndim == 3
                if a_tensor.ndim == 2:              
                    a_tensor = a_tensor[:, :, None]     # (batch_size, traj_len, ) -> (batch_size, traj_len, 1)   
            return a_tensor
        
        # obs & prefix tokenize
        p_tensor = None if prefix_array is None else tree.map_structure(
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
        a_tensor = _postprocess_act(act_array, self.act_type_spec)

        return (p_tensor, o_tensor), a_tensor
    
    def sample_expert_demonstration_for_prompt(
        self, strategy: str, strict_length: bool, sample_peak: bool, batch_size: int=1
    ) -> Dict[str, np.ndarray]:
        """Sample an expert demonstration and encode it as a dict {'actions', 'o_tensor'}
        Returns:
            Dict[str, np.ndarray]: A dict of encoded demonstration.
        """
        # only first n_trans is needed
        prompt_length = self.prompt_transition_num if strategy == "fixed_prompt" else \
                         self.transition_num - 2   # -1是为了拼接问题初始观测；-1为了解决计算transition_num时的ceil操作。保证第一个动作预测时 token 序列总长度不超过 seq_length
        assert prompt_length >= 1
        # 候选prompt轨迹索引集合
        if sample_peak:
            # select demonstration from the top 10%
            stop_idx = int(len(self.traj_idx_ret_tuples) * 0.1)
            candidates_idx = [x[0] for x in self.traj_idx_ret_tuples[:stop_idx]]
        else:
            candidates_idx = np.arange(len(self.path_lengths))

        # 随机选择MDP轨迹用作prompt
        path_idxs = np.random.choice(candidates_idx, size=batch_size)
        observation_traj_list, action_traj_list, len_traj_list = self.get_prefix_obs_action_by_path_idx(path_idxs)
        
        len_traj_array = np.array(len_traj_list)
        if strict_length:
            # splice other expert episodes to ensure sampled demonstration lenghth not less than prompt_length
            # 找出 timestep 长度不足 prompt_length 的提示序列，将其观测、动作序列转换为列表
            short_epi_idx = np.where(len_traj_array < prompt_length)[0]
            for i in short_epi_idx:
                observation_traj_list[i] = [observation_traj_list[i], ]
                action_traj_list[i] = [action_traj_list[i], ]
            
            # 采样新的MDP traj补齐提示序列长度
            while short_epi_idx.size > 0:
                path_idxs = np.random.choice(candidates_idx, size=short_epi_idx.size)
                extend_observation_traj, extend_action_traj, extend_len_traj = self.get_prefix_obs_action_by_path_idx(path_idxs)
                for i, observation_traj, action_traj, len_traj in zip(short_epi_idx, extend_observation_traj, extend_action_traj, extend_len_traj):
                    observation_traj_list[i].append(observation_traj)
                    action_traj_list[i].append(action_traj)
                    len_traj_array[i] += len_traj
                    
                    # 长度已经满足要求，将观测、动作列表拼接在一起还原为轨迹形式
                    if len_traj_array[i] >= prompt_length:
                        action_traj_list[i] = np.concatenate(action_traj_list[i], axis=0)
                        observation_traj_list[i] = tree.map_structure(
                            lambda *xs: np.concatenate(xs, axis=0), 
                            *observation_traj_list[i]
                        )
                short_epi_idx = np.where(len_traj_array < prompt_length)[0]
        
        obs_keys = observation_traj_list[0].keys()
        observations = {k:[] for k in obs_keys}
        actions = []
        for i in range(batch_size): 
            action_traj_list[i] = action_traj_list[i][:prompt_length]
            for k in obs_keys:
                observations[k].append(observation_traj_list[i][k][:prompt_length])
        actions = np.array(action_traj_list)
        observations = {k:np.array(v) for k, v in observations.items()}
        (p_tensor, o_tensor), a_tensor = self.postprocess_to_token(observations, actions)

        # append parsed obs
        return {
            "a_tensor": a_tensor,       # (batch_size, prompt_length, act_dim)
            "o_tensor": o_tensor,       # (batch_size, prompt_length, obs_dim)
            "prompt_raw_obs": observations
        }