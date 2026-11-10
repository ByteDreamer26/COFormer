import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import numpy as np
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers import AutoConfig, LlamaModel
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from utils.utils_data import RLTaskInput
import torch.nn.functional as F

class LlamaBackbone(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.min_dtype = torch.finfo(torch.float32).min
        self.prefix_spliter = config.prefix_spliter
        self.action_spliter = config.action_spliter
        self.local_pos_encoding = nn.Embedding(config.vocab_size, config.hidden_size)       # 局部位置编码
        self.post_init()

    def get_causal_mask(self, full_input_ids, prefix_mask_list, compute_loss, past_seen_tokens=0):
        attention_mask = torch.ones_like(full_input_ids, dtype=torch.int8)
        
        dtype, device = torch.float32, full_input_ids.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = target_length = attention_mask.shape[1]
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > torch.arange(target_length, device=device).reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(attention_mask.shape[0], 1, -1, -1)
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        
        if self.prefix_spliter is not None:
            input_ids_temp = full_input_ids.detach().clone().contiguous()
            for i in range(len(causal_mask)):
                # prefix段使用双向注意力
                prefix_spliter_idx = torch.where(full_input_ids[i]==self.prefix_spliter)[0]
                assert len(prefix_spliter_idx) == 1, "prefix_spliter_idx should be 1, but got {}".format(len(prefix_spliter_idx))
                causal_mask[i][0][:,:prefix_spliter_idx] = 0
                
                # 动态prefix
                prefix_mask_of_timestep = prefix_mask_list[i].squeeze()     
                if prefix_mask_of_timestep.ndim == 1:
                    prefix_mask_of_timestep = prefix_mask_of_timestep[None,:]   # (timestep, prefix_dim)
                
                action_predict_idxs = torch.where(input_ids_temp[i] == self.action_spliter)[0]
                if compute_loss:
                    assert len(action_predict_idxs) == len(prefix_mask_of_timestep)
                    for t, act_predict_idx in enumerate(action_predict_idxs):
                        prefix_mask = prefix_mask_of_timestep[t]
                        causal_mask[i][0][act_predict_idx][:prefix_spliter_idx] += (prefix_mask * min_dtype).to(device)
                else:
                    assert len(prefix_mask_of_timestep) == 1
                    act_predict_idx = action_predict_idxs[-1]
                    prefix_mask = prefix_mask_of_timestep[0]
                    causal_mask[i][0][act_predict_idx][:prefix_spliter_idx] += (prefix_mask * min_dtype).to(device)
        
        causal_mask = causal_mask[:, :, past_seen_tokens:, :]    # (batch_size, 1, q_len, k_len)
        return causal_mask

    def get_causal_mask_quick(self, full_input_ids, prefix_mask_tensor, past_seen_tokens=0):
        attention_mask = torch.ones_like(full_input_ids, dtype=torch.int8)
        device = full_input_ids.device
        
        sequence_length = target_length = attention_mask.shape[1]
        causal_mask_quick = torch.full((sequence_length, target_length), fill_value=self.min_dtype, dtype=torch.float32, device=device)
        if sequence_length != 1:
            causal_mask_quick = torch.triu(causal_mask_quick, diagonal=1)
        causal_mask_quick *= torch.arange(target_length, device=device) > torch.arange(target_length, device=device).reshape(-1, 1)
        causal_mask_quick = causal_mask_quick[None, None, :, :].expand(attention_mask.shape[0], 1, -1, -1)
        causal_mask_quick = causal_mask_quick.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask_quick[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask_quick[..., :mask_length] = causal_mask_quick[..., :mask_length].masked_fill(padding_mask, self.min_dtype)
        
        if self.prefix_spliter is not None:
            # prefix段使用双向注意力
            prefix_spliter_idx = torch.where(full_input_ids==self.prefix_spliter)[1]    # (batch_size, )
            assert prefix_spliter_idx.min() == prefix_spliter_idx.max()
            prefix_spliter_pos = prefix_spliter_idx[0].item()
            causal_mask_quick[:, :, :, :prefix_spliter_pos] = 0
            assert (full_input_ids[:, -1] == self.action_spliter).all()
            causal_mask_quick[:, 0, -1, :prefix_spliter_pos] += (prefix_mask_tensor * self.min_dtype).to(device)

        causal_mask_quick = causal_mask_quick[:, :, past_seen_tokens:, :]    # (batch_size, 1, q_len, k_len)
        return causal_mask_quick

    def forward(
        self,
        full_input_ids: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        prefix_mask_list: List = None,
        local_position_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_loss = True
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        assert input_ids is not None and inputs_embeds is None, "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        assert not (self.gradient_checkpointing and self.training and use_cache), "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # token embedding
        local_position_ids = local_position_ids[:, -input_ids.shape[1]:]    # (batch_size, seq_len)
        local_positional_emb = self.local_pos_encoding(local_position_ids)  # (batch_size, seq_len, n_embed)
        inputs_embeds = self.embed_tokens(input_ids)                        # (batch_size, seq_len, n_embed)
        inputs_embeds = inputs_embeds + local_positional_emb                # (batch_size, seq_len, n_embed)

        # kv cache & position cache (for evaluation)
        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            assert not self.training, '训练阶段不使用 kvcache'
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        # attention mask
        if not self.training and not compute_loss:
            prefix_mask_tensor = torch.vstack(prefix_mask_list)             # (batch_size, prefix_dim)
            causal_mask = self.get_causal_mask_quick(full_input_ids, prefix_mask_tensor, past_seen_tokens)
            #assert (causal_mask == causal_mask_quick).all()
        else:
            causal_mask = self.get_causal_mask(full_input_ids, prefix_mask_list, compute_loss, past_seen_tokens)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,     # past_key_values:DynamicCache 会逐层缓存 key & value 
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)        # (batch_size, seq_len, hidden_size)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class TrajLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.wd_decr_style == "constant"                   # 现在只考虑固定的权重衰减系数
        self.config = config
        self.n_embed = config.n_embed                               # token 嵌入维度
        self.n_position = config.n_position                         # 模型 token 位置数量（模型支持的上下文长度）
        self.n_layer = config.n_layer                               # transformer block 数量
        self.n_q_head = config.n_q_head                             # 注意力头总数
        self.n_kv_head = config.n_kv_head                           # 注意力头分组数，每组内所有头共用 key 和 value，即 GQA 机制
        self.d_head = self.n_embed // self.n_q_head                 # 每个 attention head 将 token 嵌入向量投影到的维度                      
        self.d_inner = (int(self.n_embed * 8/3 / 64) + 1) * 64      # SwiGLU 中间层尺寸，取 8/3 倍再按 64 向上取整（用 8/3 而非 2 倍是因为 SwiGLU 包含三个参数矩阵） 
        self.total_vocab_size = config.num_discrete_values + config.num_continuous_bin + len(config.special_tokens)
        
        # build llama backbone
        action_spliter = config.special_tokens['<|>']
        prefix_spliter = None if not config.use_prefix else config.special_tokens['<X>']
        self.llama_config = AutoConfig.for_model(
            model_type="llama",
            vocab_size = self.total_vocab_size,
            hidden_size = self.n_embed,
            intermediate_size = self.d_inner,
            num_hidden_layers = self.n_layer,
            num_attention_heads = self.n_q_head,
            num_key_value_heads = self.n_kv_head,
            max_position_embeddings = self.n_position,
            rms_norm_eps = self.config.rms_norm_eps,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            tie_word_embeddings=self.config.weight_tying,
            attention_bias=False,
            attention_dropout=self.config.dropattn,
            action_spliter=action_spliter,
            prefix_spliter=prefix_spliter,
        )
        self.model = LlamaBackbone._from_config(self.llama_config, torch_dtype=torch.float32)
        if not self.config.weight_tying:
            self.lm_head = nn.Linear(self.n_embed, self.total_vocab_size, bias=False)
            self.lm_head.weight.data.normal_(mean=0.0, std=self.llama_config.initializer_range)

    def configure_optimizers(self):
        """
        utilize weight decay method to avoid overfitting
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_parameters = get_parameter_names(self, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        decay = set([n for n, p in self.named_parameters() if (n in decay_parameters and p.requires_grad)])
        no_decay = set([n for n, p in self.named_parameters() if (n not in decay_parameters and p.requires_grad)])

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.wd_begin},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.config.lr_begin, 
            betas=(self.config.adam_beta1, self.config.adam_beta2), 
            eps=self.config.adam_eps
        )
        return optimizer
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        current_seq_len = input_ids.shape[1]

        if not kwargs.get("use_cache"):
            model_inputs = {
                "input_ids": input_ids.contiguous(),
                "position_ids": torch.arange(current_seq_len, device=input_ids.device).unsqueeze(0),
                "use_cache": False,
                "past_key_values": None,
            }    
        else:
            if past_key_values is None:
                past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)

            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
                if past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]  # 输入序列中去除已经cache的部分

            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                position_ids = torch.arange(current_seq_len, device=input_ids.device).unsqueeze(0)
                position_ids = position_ids[:, -input_ids.shape[1] :]

            # The `input_ids.contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            model_inputs = {
                "input_ids": input_ids.contiguous(),
                "position_ids": position_ids,
                "use_cache": True,
                "past_key_values": past_key_values,
            }

        return model_inputs
    
    def forward(self, tasks_input:RLTaskInput, compute_loss:bool=True, batch_dataset_name:List=None, model_inputs:dict={}):
        label = tasks_input.label                           # (batch_size, seq_len)
        loss_mask = tasks_input.loss_mask                   # (batch_size, seq_len)
        
        outputs = self.model(
            full_input_ids = tasks_input.tensor_seq,        # (batch_size, seq_len)
            prefix_mask_list = tasks_input.prefix_mask,     # [batch_size x (timestep, prefix_dim)]
            local_position_ids = tasks_input.position_id,   # (batch_size, seq_len)
            inputs_embeds = None,
            return_dict = True,
            output_attentions = False,
            output_hidden_states = False,
            compute_loss = compute_loss,
            **model_inputs,
        )
        hidden_states = outputs.last_hidden_state           # (batch_size, seq_len, n_embed)

        # 最后的线性层将 hidden_states 维度由 n_embed 调整为输出尺寸 total_vocab_size
        if self.config.weight_tying:
            lm_logits = F.linear(hidden_states, self.model.embed_tokens.weight)
            assert lm_logits.shape[:-1] == hidden_states.shape[:-1]
        else:
            lm_logits = self.lm_head(hidden_states)         # (batch_size, klen, total_vocab_size)

        # compute loss when training
        loss_ave = None
        loss_datasets = {dataset_name: [] for dataset_name in self.config.eval_dataset_names}
        if compute_loss:
            assert loss_mask is not None and label is not None
            # Flatten the tokens
            loss_mask = loss_mask.detach().float()                                      # (batch_size, klen)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            act_lm_logits = lm_logits[loss_mask==1]                                     # (act_num, vocab_size)
            act_label = label[loss_mask==1]                                             # (act_num, )
            loss = loss_fct(act_lm_logits.view(-1, self.total_vocab_size), act_label)   # (act_num, )
                    
            # 所有数据集的平均损失
            loss_ave = loss.mean()

            # 分别各个数据集上的损失，它们的加权平均值应当和 loss_ave 相同
            loss = loss.detach()
            loss_num = (loss_mask==1).sum(axis=1)
            loss_end = torch.cumsum(loss_num, dim=0).tolist()
            loss_start = [0] + loss_end[:-1]
            for idx, dataset_name in enumerate(batch_dataset_name):
                start, end = loss_start[idx], loss_end[idx]
                loss_datasets[dataset_name].extend(loss[start:end].tolist())
            
            loss_num = len(loss)
            assert sum([len(v) for v in loss_datasets.values()]) == loss_num
            weighted_loss = {k:0 if len(v) == 0 else len(v)/loss_num*np.mean(v) for k,v in loss_datasets.items()}
            assert abs(loss_ave.item() - np.sum(list(weighted_loss.values()))) < 1e-3, f'unequal loss: {loss_ave.item()} != {np.sum(list(weighted_loss.values()))}'
            loss_datasets = {k:0 if len(v) == 0 else np.mean(v) for k,v in loss_datasets.items()}

        return lm_logits, loss_ave, loss_datasets, outputs

class CriticLlama(TrajLlama):
    def __init__(self, config):
        super().__init__(config)
        # 重写 head，采用 regression head 输出标量，不使用weight tying 机制
        self.lm_head = None
        self.critic_head = nn.Linear(self.n_embed, 1)
        self.critic_head.weight.data.normal_(mean=0.0, std=self.llama_config.initializer_range)
        if self.critic_head.bias is not None:
            self.critic_head.bias.data.zero_()
    
    def forward(self, tasks_input, compute_loss:bool=False, batch_dataset_name:list=None, model_inputs:dict={}):
        loss_mask = tasks_input.loss_mask                   # (batch_size, seq_len)
        rewards = tasks_input.rewards                       # (batch_size, timestep)

        # 调用 backbone 部分，获取 hidden_states
        outputs = self.model(
            full_input_ids = tasks_input.tensor_seq,        # (batch_size, seq_len)
            prefix_mask_list = tasks_input.prefix_mask,     # [batch_size x (timestep, prefix_dim)]
            local_position_ids = tasks_input.position_id,   # (batch_size, seq_len)
            inputs_embeds = None,
            return_dict = True,
            output_attentions = False,
            output_hidden_states = False,
            compute_loss = compute_loss,                    # Critic 不计算分类 loss
            **model_inputs,
        )
        hidden_states = outputs.last_hidden_state           # (batch_size, seq_len, n_embed)

        # 得到序列表示，可以选择取最后一个 token 的 hidden state 或者使用平均池化
        if self.config.critic_feature == 'last':
            feature_hidden = hidden_states
        elif self.config.critic_feature == 'mean':
            raise ValueError(f"Not implemented yet: {self.config.critic_feature}")
        else:
            raise ValueError(f"Invalid critic feature: {self.config.critic_feature}")

        # 经过回归 head 得到价值
        values = self.critic_head(feature_hidden).squeeze()         # (batch_size, seq_len)
        
        # 计算 GAE
        adv_list, state_values, td_targets = [], [], []
        for rwd, value, l_mask in zip(rewards, values, loss_mask):
            rwd = rwd[rwd !=- 1e5]                                  # (timestep,)
            state_value = value[l_mask==1]                          # (timestep, )
            next_states_value = torch.tensor(state_value[1:].tolist() + [0], device=value.device)    
            td_target = rwd + self.config.gamma_gae * next_states_value # (timestep, ) 
            td_delta = td_target - state_value                      # (timestep, ) 
            
            adv, advs = 0, []
            for delta in reversed(td_delta.tolist()):
                adv = delta + self.config.gamma_gae * self.config.lambda_gae * adv
                advs.append(adv)
            adv_list.append(list(reversed(advs)))                   # (timestep, )

            state_values.append(state_value)
            td_targets.extend(td_target.tolist())                   # (timestep, )

        # 计算损失
        loss_ave = None
        loss_datasets = {dataset_name: [] for dataset_name in self.config.eval_dataset_names}
        if compute_loss:
            state_values = torch.hstack(state_values)               # (state_num, )
            td_targets = torch.tensor(td_targets).to(state_values.device)
            loss_fct = nn.MSELoss(reduction="none")            
            loss = loss_fct(state_values, td_targets.detach())      # (state_num, )
                    
            # 所有数据集的平均损失
            loss_ave = loss.mean()

            # # 分别各个数据集上的损失，它们的加权平均值应当和 loss_ave 相同
            # loss = loss.detach()
            # loss_num = (loss_mask==1).sum(axis=1)
            # loss_end = torch.cumsum(loss_num, dim=0).tolist()
            # loss_start = [0] + loss_end[:-1]
            # for idx, dataset_name in enumerate(batch_dataset_name):
            #     start, end = loss_start[idx], loss_end[idx]
            #     loss_datasets[dataset_name].extend(loss[start:end].tolist())
            
            # loss_num = len(loss)
            # assert sum([len(v) for v in loss_datasets.values()]) == loss_num
            # weighted_loss = {k:0 if len(v) == 0 else len(v)/loss_num*np.mean(v) for k,v in loss_datasets.items()}
            # assert abs(loss_ave.item() - np.sum(list(weighted_loss.values()))) < 1e-3, f'unequal loss: {loss_ave.item()} != {np.sum(list(weighted_loss.values()))}'
            # loss_datasets = {k:0 if len(v) == 0 else np.mean(v) for k,v in loss_datasets.items()}

        return loss_ave, adv_list