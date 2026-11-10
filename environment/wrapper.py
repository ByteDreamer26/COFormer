import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import gym
import torch
import numpy as np
import tree
import argparse
from data.dataset import RLFullDataset
from typing import Union, Dict, Tuple

class LMPromptEnv(gym.Env):
    """Wrap a gym env to make its output suitable for gato"""
    def __init__(self, env:gym.Env, args:argparse.Namespace, dataset:RLFullDataset, eval_prompt_strat:str='moving_prompt'):
        self.env = env
        self.args = args
        self.dataset = dataset
        self.eval_prompt_strat = eval_prompt_strat

        self.seq_length = args.n_position
        self.num_discrete_values = args.num_discrete_values
        self.total_vocab_size = args.num_discrete_values + args.num_continuous_bin + len(args.special_tokens)
        
        self.env_name = self.dataset.env_name
        self.cont_tokenizer = self.dataset.discretizer
        self.act_spliter_token = args.special_tokens['<|>']
        assert self.act_spliter_token == self.dataset.act_spliter_token

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            assert options is None
            self.env.reset(seed=seed, options=options)
            return
        
        obs, info = self.env.reset(seed=seed, options=options)
        current_seq = self.build_rl_task_input(raw_obs=obs)
        #return obs, current_seq
        info['obs'] = obs
        return current_seq, info
        
    def step(self, act):
        obs, reward, terminated, truncated, info = self.env.step(act)
        new_seq = self.build_rl_task_input(raw_obs=obs)
        info['obs'] = obs
        return new_seq, reward, terminated, truncated, info

    def get_action_mask(self, hard_action_constraint=False, generated_actions=False):    
        if self.dataset.act_type_spec == 'float':
            return None
        assert self.dataset.act_type_spec == 'int', f'act_type_spce can only be "float" or "int", instead of "{self.dataset.act_type_spec}"'
    
        # post_process 
        action_token_masks = []
        action_value_space = self.env.get_action_value_space(hard_action_constraint)    # 动作各维度真实取值范围
        for dim_value_space in action_value_space:
            dim_value_space = np.array(dim_value_space)
            dim_token_space = self.dataset.adapter.post_process_act(dim_value_space)
            if dim_token_space.size > 0:
                assert dim_token_space.min() >= 0 and dim_token_space.max() < self.num_discrete_values            
            
            mask = np.ones(self.total_vocab_size)
            mask[dim_token_space] = 0
            action_token_masks.append(mask)

        return action_token_masks

    def get_prefix(self, with_raw=False):
        # 环境中获取当前正在解决的问题对应的prefix序列并进行tokenize，仅用于评估阶段 (evalute_one_episode)
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

            assert p_tensor.ndim == 1   # (prefix_dim, )
            return p_tensor

        # get raw prefix from env
        raw_prefix = self.env.get_prefix()
        
        # post process
        assert raw_prefix != {}
        prefix = self.dataset.adapter.post_process_prefix(raw_prefix)
        
        # tokenize
        processed_prefix = tree.map_structure(
            _postprocess_prefix,
            prefix,
            self.dataset.prefix_type_spec,
        )

        # flatten to tensor
        prefix_tensor = []
        for k in sorted(processed_prefix):    
            prefix_tensor.append(processed_prefix[k])
        prefix_tensor = torch.tensor(np.concatenate(prefix_tensor, axis=-1), dtype=torch.long).squeeze()
        
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
        minimal_expert_data: bool=False  # strict length for potential batch eval.
    ):        
        # 从数据集采样prompt序列并进行tokenize，得到观测、动作各部分token序列
        encoded_demo: Dict[str, np.ndarray] = self.dataset.sample_expert_demonstration_for_prompt(
            strategy=self.eval_prompt_strat, 
            strict_length=strict_length,
            sample_peak=(not minimal_expert_data)
        )
        o_tensor = encoded_demo["o_tensor"]             # (prompt_length, obs_item_dim)
        a_tensor = encoded_demo["a_tensor"]             # (prompt_length, act_dim=1)

        # 组合张量观测序列
        prepend_obs = self.build_rl_task_input(o_tensor=o_tensor)   # (prompt_length, obs_dim)

        # 处理各个timestep的三类token子序列
        if prepend_obs.ndim == 1:
            prepend_obs.unsqueeze_(-1)                  # (prompt_length, obs_dim)
        prepend_act = (
            torch.from_numpy(a_tensor)
            .long()
            .to(prepend_obs.device)
            .reshape(len(prepend_obs), -1)
        )                                               # (prompt_length, act_dim)
        spliter_tokens = torch.full(
            (prepend_obs.shape[0], 1), 
            self.act_spliter_token, 
            dtype=torch.long
        ).to(prepend_obs.device)                        # (prompt_length, 1)
        
        # 组合并拉平所有timestep，得到完整token序列
        fixed_prompt = torch.cat([prepend_obs, spliter_tokens, prepend_act], dim=-1)    # (prompt_length, obs_dim+1+act_dim)
        prepend_tensor = fixed_prompt.flatten().long()  # (prompt_length*(obs_dim+1+act_dim), )
        
        return prepend_tensor, encoded_demo['prompt_raw_obs']
    
    def build_rl_task_input(
        self,
        raw_obs: Union[Dict, np.ndarray] = None,
        o_tensor: Union[Dict, np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def _encode_obs(obs_array, obs_type):
            o_tensor = None
            if obs_type == "float":
                obs_array = self.cont_tokenizer.discretize(obs_array, is_action=False).numpy()
                o_tensor = obs_array + self.num_discrete_values
            elif obs_type == "int":
                assert obs_array.min() >= 0 and obs_array.max() < self.num_discrete_values
                o_tensor = obs_array
            else:
                raise ValueError(f'obs_type_spce can only be "float" or "int", instead of "{obs_type}"')
            
            # resize o_tensor to (obs_dim, )
            # 注意这里仅针对单个观测进行处理，dataset 类中的 encode_obs 则对整个观测序列进行编码，因此要调整成的维度也不同
            if o_tensor is not None:
                assert o_tensor.ndim == 0 or o_tensor.ndim == 1
                if o_tensor.ndim == 0:
                    o_tensor = o_tensor[None]

            return o_tensor
    
        if raw_obs is not None:
            obs_array = self.dataset.adapter.post_process_obs(raw_obs)
            o_tensor = tree.map_structure(
                _encode_obs,
                {k: obs_array[k] for k in self.dataset.obs_type_spec},
                self.dataset.obs_type_spec,
            )
        assert o_tensor is not None
        
        res = []
        if o_tensor is not None:
            if isinstance(o_tensor, dict):
                for k in sorted(o_tensor):
                    if o_tensor[k] is not None:
                        res.append(o_tensor[k])
            else:
                res.append(o_tensor)
        
        input_tensor = torch.tensor(np.concatenate(res, axis=-1), dtype=torch.long)
        return input_tensor