import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import gym
import torch
import numpy as np
import tree
import argparse
from data.DDP_dataset import DDP_RLFullDataset
from typing import Union, Dict, Tuple

class DDP_LMPromptEnv(gym.Env):
    """Wrap a gym env to make its output suitable for gato"""
    def __init__(self, env:gym.Env, args:argparse.Namespace, dataset:DDP_RLFullDataset, eval_prompt_strat:str='moving_prompt'):
        self.env = env
        self.args = args
        self.dataset = dataset
        self.eval_prompt_strat = eval_prompt_strat
        
        self.seq_length = args.n_position
        self.num_discrete_values = args.num_discrete_values
        self.total_vocab_size =  args.num_discrete_values + args.num_continuous_bin + len(args.special_tokens)

        self.env_name = self.dataset.env_name
        self.cont_tokenizer = self.dataset.discretizer
        self.act_spliter_token = args.special_tokens['<|>']
        assert self.act_spliter_token == self.dataset.act_spliter_token

        self.batch_size = env.batch_size
        self.onging_problem_idx = np.zeros(self.batch_size, dtype=bool) # True指示batch中该problme尚未求解完成
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            assert options is None
            self.env.reset(seed=seed, options=options)
            return
        
        assert options is not None and 'problem_info' in options and 'problem_idx' in options
        # 将新问题插入到problem_idx指定的位置，将这些位置的onging标志置为真
        problem_num = len(options['problem_info'][2])
        self.onging_problem_idx[options['problem_idx'][:problem_num]] = True
        obs, info = self.env.reset(seed=seed, options=options)
        current_seq = self.build_rl_task_input(raw_obs=obs)
        info['obs'] = obs
        return current_seq, info
    
    def step(self, act):
        obs, reward, terminated, truncated, info = self.env.step(act) #obs中的值: (batch size, obs_dim)或（24 batch_size）; terminated: (batch_size) 或（batch_size,24）
        new_seq = self.build_rl_task_input(raw_obs=obs)
        info['obs'] = obs
        terminated = terminated.reshape(terminated.shape[0], -1)
        truncated = truncated.reshape(truncated.shape[0], -1)
        terminated &= self.onging_problem_idx[:, None]
        truncated &= self.onging_problem_idx[:, None]
        reward['AM'] *= self.onging_problem_idx
        reward['DB1'] *= self.onging_problem_idx
        terminated = np.all(terminated, axis=1)
        truncated = np.all(truncated, axis=1)
        
        # 非SPCTSP问题，DB1性能最大值不超过1
        if not self.env_name.startswith('Env_SPCTSP'):
            temp_db1 = reward['DB1'].copy()
            temp_db1[terminated] = 0
            temp_db1[truncated] = 0
            assert temp_db1.max() <= 1.01
            
        minus = torch.where(torch.any(new_seq<0, axis=1))[0]
        for i in minus:
            assert not self.onging_problem_idx[i]
            new_seq[i][new_seq[i]<0] = 0

        
        self.onging_problem_idx[terminated] = False
        self.onging_problem_idx[truncated] = False
        return new_seq, reward, terminated, truncated, info

    def get_action_mask(self, hard_action_constraint=False, generated_actions=None):    
        if self.dataset.act_type_spec == 'float':
            return None
        assert self.dataset.act_type_spec == 'int', f'act_type_spce can only be "float" or "int", instead of "{self.dataset.act_type_spec}"'

        # 动作各维度真实取值范围
        action_value_space = self.env.get_action_value_space(hard_action_constraint, generated_actions)  # len(action_value_space) == act_dim_num
        
        # post_process
        action_token_masks = []
        for dim_value_space in action_value_space:
            env_token_masks = []
            for i, sub_env_act_space in enumerate(dim_value_space):                         # len(dim_value_space) == batch_size or batch_size*24 for FFSP
                assert isinstance(sub_env_act_space, np.ndarray)
                token_space = self.dataset.adapter.post_process_act(sub_env_act_space)
                #if self.onging_problem_idx[i]:
                assert token_space.size == 0 or (token_space.size > 0 and token_space.min() >= 0 and token_space.max() < self.num_discrete_values)
                mask = np.ones(self.total_vocab_size)
                mask[token_space] = 0                            
                env_token_masks.append(mask)
            action_token_masks.append(env_token_masks)
        return action_token_masks

    def get_prefix(self, with_raw=False):
        # 环境中获取当前正在解决的问题对应的prefix序列并进行tokenize，仅用于评估阶段 (evalute_batch_episode)
        def _postprocess_prefix(prefix_array, prefix_type):
            if prefix_array is None:
                return None

            if prefix_type == "float":
                prefix_array = self.dataset.discretizer.discretize(prefix_array, is_action=False).numpy()
                p_tensor = prefix_array + self.num_discrete_values
            elif prefix_type == "int":
                assert prefix_array.min() >= 0 and prefix_array.max() < self.num_discrete_values
                p_tensor = prefix_array
            else:
                raise ValueError(f'prefix_type_spce can only be "float" or "int", instead of "{prefix_type}"')

            # resize o_tensor to (batch_size, prefix_item_dim)
            assert p_tensor.ndim == 2
            return p_tensor

        # post process
        raw_prefix = self.env.get_prefix()
        assert raw_prefix != {}
        prefix = self.dataset.adapter.post_process_prefix(raw_prefix)
        
        # tokenize
        processed_prefix = tree.map_structure(
            _postprocess_prefix,
            prefix,
            self.dataset.prefix_type_spec,
        )

        # flatten to tensor
        prefix_tensor = [processed_prefix[k] for k in sorted(processed_prefix)]
        prefix_tensor = torch.tensor(np.concatenate(prefix_tensor, axis=-1), dtype=torch.long)
        prefix_tensor = prefix_tensor.squeeze() if len(prefix_tensor) > 1 else prefix_tensor
        
        if with_raw:
            return prefix_tensor, raw_prefix
        return prefix_tensor, None

    def get_prefix_mask(self):
        prefix_mask = self.env.get_prefix_mask()
        if prefix_mask is None:
            return None
        prefix_mask = [prefix_mask[k] for k in sorted(prefix_mask)]
        prefix_mask = np.hstack(prefix_mask)
        if prefix_mask.ndim == 1:
            prefix_mask = prefix_mask[None,:]
        return list(torch.tensor(prefix_mask))

    def get_prompt(
        self, 
        strict_length: bool=True,
        minimal_expert_data: bool=False,  # strict length for potential batch eval.
        with_raw=True
    ):        
        # 从数据集采样prompt序列并进行tokenize，得到obs和act的token序列
        encoded_demo: Dict[str, np.ndarray] = self.dataset.sample_expert_demonstration_for_prompt(
            strategy=self.eval_prompt_strat, 
            strict_length=strict_length,
            sample_peak=(not minimal_expert_data),
            batch_size=self.batch_size*24 if self.env_name.startswith('Env_FFSP') else self.batch_size 
        )
        o_tensor = encoded_demo["o_tensor"]             # (batch_size, prompt_length, obs_item_dim)
        a_tensor = encoded_demo["a_tensor"]             # (batch_size, prompt_length, act_dim=1)
        
        # 组合张量观测序列
        prepend_obs = self.build_rl_task_input(o_tensor=o_tensor)   # (batch_size, prompt_length, obs_dim)
        
        # 处理各个timestep的三类token子序列
        prepend_act = (
            torch.from_numpy(a_tensor)
            .long()
            .to(prepend_obs.device)
            .reshape(prepend_obs.shape[0], prepend_obs.shape[1], -1)
        )                                               # (batch_size, prompt_length, act_dim=1)
        spliter_tokens = torch.full(
            (prepend_obs.shape[0], prepend_obs.shape[1], 1), 
            self.act_spliter_token, 
            dtype=torch.long
        ).to(prepend_obs.device)                        # (batch_size, prompt_length, 1)
        
        # 组合并拉平所有timestep，得到完整token序列
        fixed_prompt = torch.cat([prepend_obs, spliter_tokens, prepend_act], dim=-1)    # (batch_size, prompt_length, obs_dim+1+act_dim)
        prepend_tensor = fixed_prompt.flatten(start_dim=1).long()                       # (batch_size, prompt_length*(obs_dim+1+act_dim), )
        
        if with_raw:
            return prepend_tensor, encoded_demo['prompt_raw_obs']
        return prepend_tensor, None
    
    def build_rl_task_input(
        self,
        raw_obs: Union[Dict, np.ndarray] = None,
        o_tensor: Union[Dict, np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        assert (raw_obs is None) ^ (o_tensor is None)
            1. raw_obs is not None 时，该方法是在 reset 和 step 方法中调用，只对一个 obs 进行编码
            2. o_tensor is not None 时，该方法是在 get_prompt 方法中调用，对整个轨迹的 obs 序列进行编码
        '''
        def _encode_obs(obs_array, obs_type):
            o_tensor = None
            if obs_type == "float":
                obs_array = self.cont_tokenizer.discretize(obs_array, is_action=False).numpy()
                o_tensor = obs_array + self.num_discrete_values
            elif obs_type == "int":
                # ongoing_array = obs_array[self.onging_problem_idx]
                # assert ongoing_array.min() >= 0 and ongoing_array.max() < self.num_discrete_values
                o_tensor = obs_array
            else:
                raise ValueError(f'obs_type_spce can only be "float" or "int", instead of "{obs_type}"')
            
            # resize o_tensor to (batch_size, obs_item_dim)
            # 注意这里仅针对单个观测进行处理，dataset 类中的 encode_obs 则对整个观测序列进行编码，因此要调整成的维度也不同
            if o_tensor is not None:
                assert o_tensor.ndim == 1 or o_tensor.ndim == 2
                if o_tensor.ndim == 1:
                    o_tensor = o_tensor[:, None]
            
            return o_tensor
        
        assert (raw_obs is None) ^ (o_tensor is None)
        if raw_obs is not None:
            obs_array = self.dataset.adapter.post_process_obs(raw_obs)
            o_tensor = tree.map_structure(
                _encode_obs,
                {k: obs_array[k] for k in self.dataset.obs_type_spec},
                self.dataset.obs_type_spec,
            )
        
        # 按把各个 obs item 按 key 的字典序拼接
        res = []
        if o_tensor is not None:
            if isinstance(o_tensor, dict):
                for k in sorted(o_tensor):
                    if o_tensor[k] is not None:
                        res.append(o_tensor[k])
            else:
                res.append(o_tensor)
        
        input_tensor = torch.tensor(np.concatenate(res, axis=-1), dtype=torch.long)   # (batch_size, obs_dim)
        return input_tensor