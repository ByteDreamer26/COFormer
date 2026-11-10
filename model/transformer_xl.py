import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import math
from torch import nn
from typing import List, Any, Union, Optional
import torch.nn.functional as F
import torch
import numpy as np
from utils.utils_data import RLTaskInput

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x):
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)
    
ACT2FN = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "geglu": GEGLU,
}

class PositionalEmbedding(nn.Module):
    '''使用正余弦方法生成位置编码'''
    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]

class PositionwiseFF(nn.Module):
    '''支持 geglu 激活的 FFD 层'''
    def __init__(self, d_model, d_inner, dropout, activation, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        if activation == "geglu":
            assert d_inner % 2 == 0

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            ACT2FN[activation](),
            nn.Linear(d_inner if activation != "geglu" else d_inner // 2, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inp, deepnorm_alpha: Optional[float] = None,):
        '''
        inp: (batch_size, qlen, d_model)
        '''
        if self.pre_lnorm:
            lnorm_out = self.layer_norm(inp)            # layer normalization
            core_out = self.CoreNet(lnorm_out)          # positionwise feed-forward
            output = core_out + inp                     # residual connection
        else:
            if deepnorm_alpha is None:
                deepnorm_alpha = 1.

            core_out = self.CoreNet(inp)                # positionwise feed-forward
            residual = core_out + inp*deepnorm_alpha    # residual connection
            output = self.layer_norm(residual)          # layer normalization

        return output                                   # (batch_size, qlen, d_model)

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, 
        n_head,                     # 注意力头数量 
        d_model,                    # 模型嵌入维度
        d_head,                     # 每个注意力头的嵌入维度，有 d_head * n_head == d_model
        dropout,                    # 注意力汇聚线性层的 pdrop
        dropatt=0,                  # 注意力分布的 pdrop
        pre_lnorm=False,            # 是否前置 layer norm 层
        r_r_bias=None,              # query_bias_position
        r_w_bias=None,              # query_bias_content
        layer_norm_epsilon=1e-5     # layer norm 参数
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        assert self.d_head * self.n_head == self.d_model

        self.qkv_net = nn.Linear(self.d_model, 3*self.n_head*self.d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(self.n_head*self.d_head, self.d_model, bias=False)

        self.scale = 1 / (self.d_head ** 0.5)

        if r_r_bias is None or r_w_bias is None: 
            # Query Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:        
            # Query Biases shared by all attention heads                          
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head*self.d_head, bias=False)   # 专门投影 key 中相对位置编码的矩阵 Wkr
        self.layer_norm = nn.LayerNorm(self.d_model, eps=layer_norm_epsilon)
        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):  
        bsz, qlen, klen, n_head = x.size()          # bsz x qlen x klen x n_head
        zero_pad_shape = (bsz, qlen, 1, n_head)     # bsz x qlen x 1 x n_head
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)  # bsz x qlen x klen + 1 x n_head

        x_padded_shape = (bsz, klen + 1, qlen, n_head)
        x_padded = x_padded.view(*x_padded_shape)   # bsz x klen+1 x qlen x n_head

        x_padded = x_padded[:, 1:]      # bsz x klen x qlen x n_head
        x = x_padded.view_as(x)         # bsz x qlen x klen x n_head

        return x

    def forward(self, w, r, mem=None, attention_mask=None, head_mask=None, output_attentions=False, deepnorm_alpha: Optional[float] = None,):
        '''
        w (hidden states):      (batch_size, qlen, n_embed) 训练时(不使用mem) qlen = seq_len = 1024; 测试时 qlen < 1024, 通常为 1
        r (reverse pos emb):    (1, klen, n_embed)          训练时(不使用mem) klen = seq_len = 1024; 测试时 klen <= 1024
        attention_mask:         (1, qlen, klen)
        mem:                    (1, mlen, n_embed)          
        '''
        qlen, rlen, bsz = w.size(1), r.size(1), w.size(0)       # 不使用mem时，qlen = rlen = seq_len; 使用时 rlen = qlen + mlen

        if mem is not None:
            cat = torch.cat([mem, w], 1)                        # (batch_size, klen, d_model)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))    # (batch_size, klen, 3*d_model)
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)                            # (batch_size, rlen, d_model) 

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)  # 都是 (batch_size, klen, d_model)
            w_head_q = w_head_q[:, -qlen:]                      # (batch_size, qlen, d_model)
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))      # (batch_size, seq_len, 3*d_model)
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)                            # (batch_size, seq_len, d_model) 
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)  # 都是 (1, klen, d_model)

        klen = w_head_k.size(1)
        assert klen == rlen

        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_head)   # hidden state 投影为 value (batch_size, klen, n_head, d_head)
        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_head)   # hidden state 投影为 query (batch_size, qlen, n_head, d_head)
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_head)   # hidden state 投影为 key   (batch_size, klen, n_head, d_head)
        r_head_k = r_head_k.view(klen, self.n_head, self.d_head)        # 相对位置编码投影为   key   (rlen, n_head, d_head)

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias                            # (batch_size, qlen, n_head, d_head)
        rr_head_q = w_head_q + self.r_r_bias                            # (batch_size, qlen, n_head, d_head)

        # (batch_size, qlen, n_head, d_head) * (batch_size, klen, n_head, d_head) -> (batch_size, qlen, klen, n_head)
        AC = torch.einsum("bind,bjnd->bijn", (rw_head_q.float(), w_head_k.float()))

        # (batch_size, qlen, n_head, d_head) * (batch_size, klen, n_head, d_head) -> (batch_size, qlen, klen, n_head)
        # 这里先对r_head_k广播为bjnd，再交换维度成bnjd，再做矩阵乘法。交换维度这步很重要，调整了相对位置编码的顺序
        BD = torch.einsum("bind,jnd->bijn", (rr_head_q.float(), r_head_k.float()))  
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)                                     # (batch_size, qlen, klen, n_head)

        # 施加 attention_mask 并计算注意力分布
        if attention_mask is not None and torch.sum(attention_mask).item():
            attention_mask = attention_mask == 1  # Switch to bool
            if attention_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[None, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
                else:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[None, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
            elif attention_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[:, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
                else:
                    # 走这个
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[:, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
        else:
            raise ValueError

        attn_prob = nn.functional.softmax(attn_score, dim=2)            # (batch_size, qlen, klen, n_head) 是下三角矩阵，每行代表一个样本的attention分布
        attn_prob = self.dropatt(attn_prob)                             # (batch_size, qlen, klen, n_head)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        if next(self.parameters()).dtype == torch.float16:
            attn_prob = attn_prob.half()

        # compute attention vector, 根据 attn_prob 提取 value
        # (batch_size, qlen, klen, n_head) * (batch_size, klen, n_head, d_head) -> (batch_size, qlen, n_head, d_head)
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, w_head_v))   # (batch_size, qlen, n_head, d_head)
        attn_vec = attn_vec.contiguous()                                    # 确保张量在内存中是连续存储
        attn_vec = attn_vec.view(
            attn_vec.size(0), 
            attn_vec.size(1), 
            self.n_head * self.d_head
        )                                                               # (batch_size, qlen, n_head*d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)                                  # (batch_size, qlen, d_model)

        if self.pre_lnorm:
            # residual connection
            outputs = (w + attn_out,)
        else:
            if deepnorm_alpha is None:
                deepnorm_alpha = 1.
            # residual connection + layer normalization
            outputs = (self.layer_norm(w * deepnorm_alpha + attn_out),)

        if output_attentions:
            outputs = outputs + (attn_prob,)

        return outputs

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, activation, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, 
            d_model, 
            d_head, 
            dropout, 
            layer_norm_epsilon=layer_norm_epsilon, 
            **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, 
            d_inner, 
            dropout, 
            activation=activation, 
            pre_lnorm=kwargs.get("pre_lnorm"), 
            layer_norm_epsilon=layer_norm_epsilon,
        )

    def forward(self, dec_inp, r, attention_mask=None, mems=None, head_mask=None, output_attentions=False, deepnorm_alpha: Optional[float]=None):

        attn_outputs = self.dec_attn(
            dec_inp, 
            r, 
            attention_mask=attention_mask, 
            mem=mems, 
            head_mask=head_mask, 
            deepnorm_alpha=deepnorm_alpha, 
            output_attentions=output_attentions,
        )
        ff_output = self.pos_ff(attn_outputs[0],  deepnorm_alpha=deepnorm_alpha)
        outputs = (ff_output,) + attn_outputs[1:]

        return outputs

class TransformerXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed                               # token 嵌入维度
        self.d_model = self.n_embed    
        self.n_position = config.n_position                         # 模型 token 位置数量（模型支持的上下文长度）
        self.n_layer = config.n_layer                               # transformer block 数量
        self.n_head = config.n_head                                 # attention head 数量
        self.d_head = self.n_embed // self.n_head                   # 每个 attention head 将 token 嵌入向量投影到的维度                      
        self.d_inner = 4 * self.d_model if config.n_inner is None else config.n_inner
        self.clamp_len = self.n_position                            # 位置索引最大值
        assert config.wd_decr_style == "constant"                   # NOTE: 现在只考虑固定的权重衰减系数
        self.weight_decay = config.wd_begin                         # 权重衰减系数（用于正则化）
        self.adam_betas = (config.adam_beta1, config.adam_beta2)    # 优化器参数
        self.adam_eps = config.adam_eps
        self.lr_begin = config.lr_begin

        self.untie_r = False                                        # 是否在所有注意力头之间共享 Query Biases 参数
        self.same_length = False
        self.pre_lnorm = True                                       # 是否前置 layernorm 层
        self.mem_len = 0.0                                            # 使用的 memory 序列长度
        self.embd_pdrop = 0.1
        self.dropattn = 0.0
        self.drop = 0.1
        self.layer_norm_epsilon = 1e-5
        self.hids = []

        # Build embedding
        self.total_vocab_size = config.num_discrete_values + config.num_continuous_bin + len(config.special_tokens)
        self.rl_separator_token_id = config.num_continuous_bin + config.num_discrete_values     # 动作分隔符

        self.token_embedding = nn.Embedding(self.total_vocab_size, self.n_embed)                # token embadding
        #self.local_pos_encoding = nn.Embedding(int(self.n_position/2) + 1, self.n_embed)       # 局部位置编码
        self.local_pos_encoding = nn.Embedding(self.total_vocab_size + 1, self.n_embed)         # 局部位置编码
        self.pos_emb = PositionalEmbedding(self.n_embed)                                        # 全局位置编码（使用相对位置编码，由正余弦公式生成）
        self.embd_drop = nn.Dropout(self.embd_pdrop)

        # Query Biases shared by all attention heads  
        if not self.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        # transformer blocks
        self.h = nn.ModuleList(
            [
                RelPartialLearnableDecoderLayer(
                    self.n_head,
                    self.d_model,
                    self.d_head,
                    self.d_inner,
                    self.drop,
                    dropatt=self.dropattn,
                    activation=config.activation_fn,
                    pre_lnorm=self.pre_lnorm,
                    r_w_bias=None if self.untie_r else self.r_w_bias,
                    r_r_bias=None if self.untie_r else self.r_r_bias,
                    layer_norm_epsilon=self.layer_norm_epsilon,
                )
                for _ in range(config.n_layer)
            ]
        )

        # The output linear layer can share parameters with token embadding layer
        self.share_input_output_embedding = config.weight_tying
        if self.share_input_output_embedding:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.n_embed, self.total_vocab_size, bias=False)

        # init parameters weight
        self.apply(self._init_weights)

        # reinit weights required by deepnorm
        self.use_deepnorm = False
        self.deepnorm_alpha = (2 * self.n_layer) ** 0.25 if self.use_deepnorm else None
        self.deepnorm_beta = (8 * self.n_layer) ** -0.25 if self.use_deepnorm else None
        if self.use_deepnorm:
            self._deepnorm_init()
        
    def _deepnorm_init(self):
        if self.use_deepnorm:
            for name, module in self.named_modules():
                if "pos_ff" in name:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=self.deepnorm_beta)
                elif "o_net" in name:
                    nn.init.xavier_uniform_(module.weight, gain=self.deepnorm_beta)
                elif "qkv_net" in name:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                    nn.init.xavier_uniform_(module.weight[2 * self.d_model:, :], gain=self.deepnorm_beta)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=np.sqrt(2/(5*self.d_model)))
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            if hasattr(module, "r_r_bias"):
                module.r_r_bias.data.normal_(mean=0.0, std=np.sqrt(2/(5*self.d_model)))
            if hasattr(module, "r_w_bias"):
                module.r_w_bias.data.normal_(mean=0.0, std=np.sqrt(2/(5*self.d_model)))

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        for name, p in module.named_parameters():
            if 'pos_ff.CoreNet.2.weight' in name:
                p.data.normal_(mean=0.0, std=(np.sqrt(2/(5*self.d_model)) / math.sqrt(2 * self.config.n_layer)))

    def configure_optimizers(self):
        """
        utilize weight decay method to avoid overfitting
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():          # 遍历所有 sub module                         
            for pn, p in m.named_parameters():      # 遍历某个 sub module 的所有 parameters
                fpn = f'{mn}.{pn}' if mn else pn  
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('r_w_bias') or pn.endswith('r_r_bias'):
                    if self.untie_r:
                        decay.add(fpn)
                elif pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if not self.untie_r:
            decay.add('r_r_bias')
            decay.add('r_w_bias')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr_begin, betas=self.adam_betas, eps=self.adam_eps)
        return optimizer

    def init_mem(self, batch_size):
        '''memory初始化为全零'''
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(
                    batch_size,
                    self.mem_len,
                    self.n_embed,
                    dtype=param.dtype,
                    device=param.device,
                )   # (batch_size, mem_len, n_embed)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mem(self, hiddens, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hiddens) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # 保留 memory 中后 self.mem_len 长度片段
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len) 
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx])

        return new_mems

    def forward(self, tasks_input:RLTaskInput, compute_loss:bool=True, batch_dataset_name:List=None, model_inputs:dict={}):
        mems = None
        assert not (compute_loss and mems is not None), "During training, Gato does not use memory mechanism."
            
        token_emb = self.token_embedding(tasks_input.tensor_seq)                # (batch_size, seq_len, self.n_embed)
        local_positional_emb = self.local_pos_encoding(tasks_input.position_id) # (batch_size, seq_len, self.n_embed) 局部位置编码
        embedding = token_emb + local_positional_emb                            # (batch_size, seq_len, self.n_embed)

        label = tasks_input.label                   # (batch_size, seq_len)
        loss_mask = tasks_input.loss_mask           # (batch_size, seq_len)
        attn_mask = tasks_input.attention_mask

        hidden_states = self.embd_drop(embedding)   # (1, qlen, n_embed)  
        qlen = hidden_states.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen

        if mlen != 0:
            assert mlen == self.mem_len

        assert self.same_length == False
        if self.same_length:
            # 测试时same_length==True, 每步预测使用相同的上下文长度 klen
            all_ones = hidden_states.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)) 
            
            # 将空的memory全部mask掉，否则这些全零嵌入会在layernorm等操作时影响注意力得分的计算
            empty_mem_len = torch.all(mems[0]== 0, dim=2).sum().item()
            attention_mask[:,:empty_mem_len] = 1
        else:
            # 训练时same_length==False, 每步预测使用相同的上下文长度随自回归过程而增长，和普通GPT一样
            attention_mask = torch.triu(hidden_states.new_ones((qlen, klen), dtype=torch.uint8), diagonal=1 + mlen)
        attention_mask = attention_mask[None, :, :]         # (1, qlen, klen)    

        # 构造反向绝对位置索引，稍后在计算attention时由 _rel_shift 方法处理为所需的相对位置编码
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)              # 控制绝对位置索引最大值
        pos_emb = self.pos_emb(pos_seq)                     # (1, klen, n_embed) 反向绝对位置编码
        pos_emb = self.embd_drop(pos_emb)                   # (1, klen, n_embed) 

        # Transformer Block
        self.hids = []
        for i, block in enumerate(self.h):
            self.hids.append(hidden_states)
            mems_i = None if mems is None else mems[i]
            outputs = block(
                hidden_states,
                pos_emb,
                mems=mems_i,
                attention_mask=attention_mask,
                head_mask=None,
                output_attentions=False,
                deepnorm_alpha=self.deepnorm_alpha
            )
            hidden_states = outputs[0]                      # (batch_size, qlen, n_embed)
        
        # 最后的线性层将 hidden_states 维度由 n_embed 调整为输出尺寸 total_vocab_size
        if self.share_input_output_embedding:
            lm_logits = F.linear(hidden_states, self.token_embedding.weight)
            assert lm_logits.shape[:-1] == hidden_states.shape[:-1]
        else:
            lm_logits = self.lm_head(hidden_states)         # (batch_size, klen, total_vocab_size)

        # update memory
        new_mems = None if mems is None else self._update_mem(self.hids, mems, mlen, qlen)

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
            
            # 直接计算所有数据集的平均损失
            loss_ave = loss.mean()
            '''
            # in order to make avoid exceeding of float16
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
            if loss.isnan() or loss.isinf():
                print("WARNING: Loss Overflow.")
            '''

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
            
            '''
            import os
            if int(os.environ['RANK']) == 0:
                print(label[0])
                print(label[0][loss_mask[0]==1][:100])
                print(lm_logits[0][loss_mask[0]==1].max(axis=1)[1][:100])
                print()
            '''
            
            '''
            with open(f'{base_path}/model/test2.txt', 'a') as file:
                file.write(str(label[0].tolist()))
                file.write('\n')

            print(label[0])
            print(label[0][loss_mask[0]==1][:100])
            print(lm_logits[0][loss_mask[0]==1].max(axis=1)[1][:100])
            print()
            '''

        '''
        with open(f'{base_path}/model/test2.txt', 'a') as file:
            for hid in self.hids:
                file.write(str(hid[0,:,0].tolist()))
                file.write('\n')
            file.write('\n')
        '''

        return lm_logits, loss_ave, loss_datasets, new_mems