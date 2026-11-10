import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import time
import numpy as np
from tqdm import tqdm
from environment.DDP_wrapper import DDP_LMPromptEnv
from data.problem_loader import ProblemLoader
from argparse import Namespace
from typing import Optional, Union
from torch.nn import functional as F
from data.dataset import get_loss_flag_and_position_id_v1, get_loss_flag_and_position_id_v2
from utils.utils_data import RLTaskInput
from utils.utils import COP_FAILED_RWD

def get_action(
    args, model, env, current_seq, sample_action, temprature=1.0,
    model_inputs: Optional[dict] = None,
    device: Optional[Union[int, str, torch.device]] = None,
    best_actions: Optional[np.ndarray] = None,
):
    prefix_dim = env.dataset.prefix_dim
    obs_dim = env.dataset.obs_dim
    act_dim = env.dataset.act_dim
    problem_batch_size_per_gpu = current_seq.shape[0]
    assert env.dataset.act_type_spec == 'int'
    batch_prefix_mask_list = env.get_prefix_mask()  # list of (act_dim, prefix_dim) or list of (prefix_dim, ) when act_dim == 1
    if act_dim > 1:
        batch_prefix_mask_list_by_act_dim = [[prefix_mask[i] for prefix_mask in batch_prefix_mask_list] for i in range(act_dim)]
    else:
        batch_prefix_mask_list_by_act_dim = [batch_prefix_mask_list, ]

    # Generate action vectors dimension by dimension
    batch_action_seq = []
    assert act_dim == 1
    for i_act in range(act_dim):
        get_loss_flag_and_position_id = get_loss_flag_and_position_id_v1 if args.local_pos_ver == 'v1' else get_loss_flag_and_position_id_v2
        _, pos_id = get_loss_flag_and_position_id(
            current_seq.shape[-1]+1, 
            prefix_dim if args.use_prefix else None,
            obs_dim, act_dim,
        )
        pos_id = pos_id[:-1]
        pos_id = pos_id[:, None].repeat(problem_batch_size_per_gpu, axis=1).T
        pos_id = torch.tensor(pos_id, dtype=torch.long)

        # Constructs current_seq as an RLTaskInput that the model can enter
        x = RLTaskInput(
            tensor_seq=current_seq, # (problem_batch_size_per_gpu, seq_len, )
            position_id=pos_id,     # (problem_batch_size_per_gpu, seq_len)
            attention_mask=torch.ones_like(current_seq, dtype=torch.int8),
            loss_mask=None,
            label=None,
            seq_len=None,
            prefix_mask=batch_prefix_mask_list_by_act_dim[i_act],
        )
        x.to(device=device)

        # model one step generating
        raw_logits, _, _, outputs = model(                              # logits: (problem_batch_size_per_gpu, seq_len, total_vocab_size)
            x, compute_loss=False, 
            batch_dataset_name=[env.dataset.dataset_name,], 
            model_inputs=model_inputs
        )

        # Set mask according to the output space of current generating dim of action 
        action_masks = env.get_action_mask(hard_action_constraint=True, generated_actions=np.array(batch_action_seq).T)[i_act],
        action_masks = torch.from_numpy(np.array(action_masks)).to(device).squeeze()
        action_masks = action_masks[None, :] if action_masks.ndim == 1 else action_masks    # (batch_size, total_vocab_size)
        logits = raw_logits[:, -1, :] - action_masks * 1e10             # (problem_batch_size_per_gpu, total_vocab_size)
        
        # get pred action token
        if best_actions is None:
            probs = F.softmax(logits / temprature, dim=-1)                  # (problem_batch_size_per_gpu, total_vocab_size)
            if sample_action:
                pred_token = torch.multinomial(probs, num_samples=1)        # (problem_batch_size_per_gpu, 1)
            else:
                _, pred_token = torch.topk(probs, k=1, dim=-1)              # (problem_batch_size_per_gpu, 1)
        else:
            pred_token = torch.tensor(best_actions.T)[:,None]               # (problem_batch_size_per_gpu, 1)

        # set pred_token as 0 for the problem which have been sloved    
        pred_token[action_masks.all(dim=1)] = 0

        # update current_seq
        current_seq = current_seq.to(device=device)
        pred_token = pred_token.to(device=device)
        current_seq = torch.cat([current_seq, pred_token], dim=1)       # (problem_batch_size_per_gpu, seq_len,)
        
        # recover model predict token to tokenizer raw
        batch_action_seq.append(pred_token.squeeze().cpu())

    # token -> action
    act = np.vstack(batch_action_seq).T                 # (problem_batch_size_per_gpu, act_dim)             
    act = env.dataset.adapter.recover_raw_act(act)      # (problem_batch_size_per_gpu, act_dim)
    return act, current_seq, outputs, raw_logits[:, -1, :]

def rollout_init(answer_list, env_name, get_best_answer):
    if env_name.startswith(('Env_TSP', 'Env_CVRP', 'Env_PCTSP', 'Env_SPCTSP')):
        best_episode_obj = np.ones(len(answer_list)) * 1e4
    elif env_name.startswith(('Env_BP', 'Env_OP')):
        best_episode_obj = np.ones(len(answer_list)) * -1e4
    else:
        raise NotImplementedError

    if get_best_answer:
        max_timestep_num = max([len(answer) for answer in answer_list])
        answers = np.array([answer + [-1] * (max_timestep_num - len(answer)) for answer in answer_list])    # (problem_batch_size_per_gpu, max_timestep_num)
        if env_name.startswith('Env_TSP'):
            answers = answers[:, 1:]
    else:
        answers = None

    return answers, best_episode_obj

def get_update_mask(env_name, episode_obj_t, best_episode_obj):
    if env_name.startswith(('Env_TSP', 'Env_CVRP', 'Env_PCTSP', 'Env_SPCTSP', 'Env_ATSP', 'Env_3DBP', 'Env_FFSP')):
        update_mask = episode_obj_t < best_episode_obj
    elif env_name.startswith(('Env_BP', 'Env_OP', 'Env_MIS')):
        update_mask = episode_obj_t > best_episode_obj
    return update_mask 

@torch.inference_mode()
def sample_batch_episode(
    args: Namespace,
    model: torch.nn.Module,
    env: DDP_LMPromptEnv,
    problemloader: ProblemLoader,
    sample_action: bool = False,
    regen_times:int = 1,
    desc: str = '',
    device: Optional[Union[int, str, torch.device]] = None,
    rl_args: Namespace = None,
    disable_tqdm: bool = False,
    add_best: bool = False,
    get_step_obj: bool=False
):
    eval_prompt_strat = args.prompt_strategy.split(";")[-1]     # moving_prompt
    rank = int(os.environ.get("RANK", default='0'))
    assert args.use_ddp_env and device is not None
    assert args.use_prefix ^ (args.use_prompt and eval_prompt_strat == "moving_prompt")
    prefix_dim = env.dataset.prefix_dim
    trans_dim = env.dataset.trans_dim
    regen_times = 1 if not sample_action else regen_times
    problem_batch_size_per_gpu = int(rl_args.q_batch_size_per_gpu/24) if env.env_name.startswith('Env_FFSP') else rl_args.q_batch_size_per_gpu
    assert problem_batch_size_per_gpu == len(problemloader)
    spliter_tokens = torch.full((rl_args.q_batch_size_per_gpu, 1), args.special_tokens['<|>'], dtype=torch.long).to(device)
    prefix_spliter_tokens = torch.full((rl_args.q_batch_size_per_gpu, 1), args.special_tokens['<X>'], dtype=torch.long).to(device)
    
    # get ready
    problemloader.reset()
    raw_model = model.module if hasattr(model, "module") else model

    # START EVALUATION
    episode_return, episode_obj, episode_slove_cnt, episode_slove_time = {'AM':[]}, [], 0, 0
    with tqdm(total=problem_batch_size_per_gpu, desc=desc, position=rank, disable=disable_tqdm) as pbar:
        for _ in range(1):
            # reset env to set the eval problem  
            problem_info, problem_obj = problemloader.get_problem(problem_batch_size_per_gpu)
            problem_idx = list(range(problem_batch_size_per_gpu))
                        
            answers, best_episode_obj = rollout_init(problem_info[2], env.env_name, add_best)
            best_episode_return = {'AM':np.ones(len(problem_idx))*-1e4,}
            time_start = time.time()
            group_returns, group_timestep, group_step_objs, group_objs, group_ids, group_answers, group_raw_logits = [], [], [], [], [], [], []
            for rt in range(regen_times):
                episode_return_t = {'AM':np.ones(len(problem_idx))*-1e4,}
                episode_obj_t = np.ones(len(problem_idx))*1e4 if env.env_name.startswith('Env_FFSP') else np.ones(len(problem_idx))*-1e4     
                episode_step_obj, episode_timestep = [], np.zeros(problem_batch_size_per_gpu, dtype=np.int32)

                # reset env 
                current_seq, info = env.reset(options={     # current_seq: (problem_batch_size_per_gpu, obs_dim)  
                    'problem_info':problem_info,
                    'problem_obj':problem_obj, 
                    'problem_idx':problem_idx,
                })       
                current_seq = current_seq.to(device)
                if get_step_obj:
                    episode_step_obj.append(info['obj'])    # (problem_batch_size_per_gpu, )

                # prepare prompt or prefix
                if args.use_prompt:
                    fixed_prompt, _ = env.get_prompt(args.strict_length, args.minimal_expert_data)  # (problem_batch_size_per_gpu, len_prompt)
                    fixed_prompt = fixed_prompt.to(device)
                    current_seq = torch.cat([fixed_prompt, current_seq, spliter_tokens], dim=-1)    # (problem_batch_size_per_gpu, len_prompt+obs_dim+1) 拼接当前 obs 和 spliter，下一步用于自回归生成 action
                elif args.use_prefix:
                    prefix_tensor, _ = env.get_prefix(with_raw=True)
                    prefix_tensor = prefix_tensor.to(device)
                    current_seq = torch.cat([prefix_tensor, prefix_spliter_tokens, current_seq, spliter_tokens], dim=-1)    # 拼接当前obs和spliter，下一步用于自回归生成action
                else:
                    current_seq = torch.cat([current_seq, spliter_tokens], dim=-1)                  # (problem_batch_size_per_gpu, obs_dim+1) 拼接当前obs和spliter，下一步用于自回归生成action                    

                # eval a batch of problem
                batch_answers, batch_raw_logits = [], []
                batch_done = np.zeros(problem_batch_size_per_gpu, dtype=bool)
                model_kwargs = {'use_cache': args.use_kvcache, 'position_ids': None, }
                t = 0
                while True:
                    # prepare model inputs
                    model_inputs = raw_model.transformer.prepare_inputs_for_generation(current_seq, **model_kwargs)

                    # Generate action
                    assert current_seq.shape[1] <= raw_model.transformer.n_position
                    best_actions = answers[:, t] if answers is not None and rt == 0 else None
                    act, current_seq, outputs, raw_logits = get_action(args, model, env, current_seq, sample_action, rl_args.temprature, model_inputs, device, best_actions)
                    act = act.squeeze()                                 # (problem_batch_size_per_gpu, act_dim) or (problem_batch_size_per_gpu, ) when act_dim==1
                    act = np.array([act]) if act.ndim == 0 else act     # (problem_batch_size_per_gpu, ) when act_dim==1
                    t += 1

                    # env one step update
                    new_seq, reward, terminated, truncated, info = env.step(act)    # new_seq: (problem_batch_size_per_gpu, obs_dim)
                    assert reward['AM'][batch_done==1].sum() == 0, 'Some episode that have ended send back reward signals'

                    act[batch_done] = -1
                    batch_answers.append(act)
                    batch_raw_logits.append(raw_logits)

                    done = terminated | truncated                                   # (problem_batch_size_per_gpu, )
                    assert (batch_done + done).max() < 2, 'Some episode end more than one times'
                    batch_done |= done                                              # (problem_batch_size_per_gpu, )

                    episode_return_t['AM'][terminated==1] = reward['AM'][terminated==1]
                    episode_obj_t[terminated==1] = info['obj'][terminated==1]
                    if get_step_obj:
                        step_obj = info['obj'].copy()
                        episode_step_obj.append(step_obj.astype(np.float32))
                    
                    # exit point
                    episode_timestep[batch_done==0] += 1
                    if batch_done.sum() >= problem_batch_size_per_gpu:
                        break

                    # update current_seq
                    new_seq = new_seq.to(device)                                                # 拼接新的观测和分隔符
                    current_seq = torch.cat([current_seq, new_seq, spliter_tokens], dim=-1)     # (problem_batch_size_per_gpu, sql_len)
                                                            # [timestep x (problem_batch_size_per_gpu, )]
                    if args.use_kvcache:
                        model_kwargs["past_key_values"] = outputs.past_key_values
                        
                    # 超长处理
                    if current_seq.shape[1] > args.n_position:                                                
                        if (args.use_prompt and eval_prompt_strat == "moving_prompt") or (not args.use_prompt and not args.use_prefix):
                            # 不用 prefix/prompt 或使用 "moving_prompt"，则随着序列增长不断保留尾部序列, 这样 prompt 序列会不断更新
                            current_seq = current_seq[:, trans_dim:]
                            if args.use_kvcache:
                                for kcache, vcache in model_kwargs["past_key_values"]:
                                    kcache = kcache[:, trans_dim:]
                                    vcache = vcache[:, trans_dim:]
                                
                        elif args.use_prefix:
                            # 如果设置 prefix, 则维持序列首部的 prefix 不变
                            window_seq_view = torch.roll(current_seq[:,prefix_dim+1:], -trans_dim)  # 将 current_seq 中除 prefix 以外的序列循环左移 prefix_dim
                            current_seq[:,prefix_dim+1:].data.copy_(window_seq_view.data)           # 除 prefix 以外的序列中，首部那个 transition 对应的序列放在尾部
                            current_seq = current_seq[:,:-trans_dim]                                # 把上一步放到尾部的（原首部）transition对应序列去除
                            
                            # kvcache只能保留prompt部分，这会导致加速效果大幅下降，最好通过调整序列长度来避免超长
                            if args.use_kvcache:
                                updated_past_key_values = []
                                for kcache, vcache in model_kwargs["past_key_values"]:
                                    kcache = kcache[:,:,:prefix_dim+1,:]
                                    vcache = vcache[:,:,:prefix_dim+1,:]
                                    updated_past_key_values.append((kcache, vcache))
                                model_kwargs["past_key_values"] = tuple(updated_past_key_values)    # 更新 model_kwargs 中的 vcache
                        else:
                            raise NotImplementedError

                if add_best and rt == 0:
                    assert np.all(episode_return_t['AM'] == 1)
                group_ids.append(current_seq)                                                   # [group_size x (problem_batch_size_per_gpu, sql_len)]
                group_returns.append((episode_return_t['AM']).tolist())                         # [group_size x (problem_batch_size_per_gpu, )]
                group_objs.append(episode_obj_t.tolist())                                       # [group_size x (problem_batch_size_per_gpu, )]
                group_answers.append(torch.from_numpy(np.stack(batch_answers).T).to(device))    # [group_size x (problem_batch_size_per_gpu, timestep_num)] 
                group_raw_logits.append(batch_raw_logits)                                       # [group_size x [timestep_num x (problem_batch_size_per_gpu, total_vocab_size)]]
                group_timestep.append(episode_timestep)                                         # [group_size x (problem_batch_size_per_gpu, )]

                if get_step_obj:
                    episode_step_obj = np.vstack(episode_step_obj)
                    episode_step_obj = (episode_step_obj[1:,:] - episode_step_obj[:-1,:]).T         # (problem_batch_size_per_gpu, timestep_num)
                    #episode_step_obj = [step_obj[:np.where(done)[0][0]+1] for step_obj, done in zip(episode_step_obj, episode_dones)]
                    group_step_objs.append(episode_step_obj)
                                
                assert np.all(episode_return_t['AM'] != -1e4) and np.all(episode_obj_t != -1e4)
                update_mask = get_update_mask(env.env_name, episode_obj_t, best_episode_obj)
                best_episode_return['AM'][update_mask] = episode_return_t['AM'][update_mask]
                best_episode_obj[update_mask] = episode_obj_t[update_mask]
            episode_slove_time += time.time() - time_start
            episode_slove_cnt += (episode_obj_t!=1e-4).sum()
            episode_return['AM'].extend(best_episode_return['AM'].tolist())
            episode_obj.extend(best_episode_obj.tolist())

            # 更新进度条
            assert (np.array(episode_return['AM']) == COP_FAILED_RWD).sum() == 0
            env_obj = 0 if len(episode_obj) == 0 else np.mean(episode_obj)
            env_obj_std = 0 if len(episode_obj) == 0 else np.std(episode_obj)
            time_used = 0 if episode_slove_cnt == 0 else episode_slove_time / episode_slove_cnt
            info = {
                'obj': f'{env_obj:.2f}',
                'std': f'{env_obj_std:.2f}',
                'time': f'{time_used:.4f}',
            }
            pbar.set_postfix(info)
            pbar.update((episode_obj_t!=1e-4).sum())
    
    group_returns = torch.tensor(group_returns, dtype=torch.float32)                    # (group_size, problem_batch_size_per_gpu)
    group_objs = torch.tensor(group_objs, dtype=torch.float32)                          # (group_size, problem_batch_size_per_gpu)
    group_timestep = torch.tensor(np.array(group_timestep), dtype=torch.int32)          # (group_size, problem_batch_size_per_gpu)
    
    # 对齐 group_step_objs 长度
    if get_step_obj:
        max_timestep = max([step_obj.shape[1] for step_obj in group_step_objs])
        for g, batch_ts in enumerate(group_timestep):
            group_step_objs[g] = np.pad(group_step_objs[g], ((0, 0), (0, max_timestep - group_step_objs[g].shape[1])), mode='constant', constant_values=0)
            for b, ts in enumerate(batch_ts):
                group_step_objs[g][b, ts+1:] = -1e5
        group_step_objs = torch.tensor(np.array(group_step_objs), dtype=torch.float32)      # (group_size, problem_batch_size_per_gpu, max_timestep)

        assert torch.allclose(
            group_objs.flatten(), 
            torch.tensor([obj[obj!=-1e5].sum() for group_objs in group_step_objs for obj in group_objs], dtype=torch.float32)
        )
    else:
        group_step_objs = torch.zeros_like(group_timestep, dtype=torch.float32)[:,:,None]   # (group_size, problem_batch_size_per_gpu, 1)
    
    return group_returns, group_objs, group_step_objs, group_ids, group_answers, group_raw_logits   # group_ids: [group_size x (q_batch_size_per_gpu, sql_len)]

def re_rollout(rl_args, problemloader, env, batch_answer, device):
    batch_answer[batch_answer==-1] = 0
    batch_answer = batch_answer.T   # (timestep_num, q_batch_size_per_gpu)
    problem_batch_size_per_gpu = int(rl_args.q_batch_size_per_gpu/24) if env.env_name.startswith('Env_FFSP') else rl_args.q_batch_size_per_gpu

    batch_prefix_masks, batch_action_masks = [], []
    
    # load problem
    problemloader.reset()
    problem_info, problem_obj = problemloader.get_problem(problem_batch_size_per_gpu)
    problem_idx = list(range(problem_batch_size_per_gpu))
    
    # reset env 
    env.reset(options={'problem_info':problem_info, 'problem_obj':problem_obj, 'problem_idx':problem_idx,})       
    
    prefix_mask_list = env.get_prefix_mask()
    action_masks = env.get_action_mask(hard_action_constraint=True)[0]
    batch_prefix_masks.append(torch.stack(prefix_mask_list))
    batch_action_masks.append(torch.from_numpy(np.vstack(action_masks)).to(device, dtype=torch.bool))

    for batch_act in batch_answer:
        batch_act = batch_act.squeeze()     # (problem_batch_size_per_gpu, act_dim) or (problem_batch_size_per_gpu, ) when act_dim==1
        
        # env one step update
        env.step(batch_act)    # new_seq: (problem_batch_size_per_gpu, obs_dim)

        prefix_mask_list = env.get_prefix_mask()
        action_masks = env.get_action_mask(hard_action_constraint=True)[0]
        batch_prefix_masks.append(torch.stack(prefix_mask_list))
        batch_action_masks.append(torch.from_numpy(np.vstack(action_masks)).to(device=device, dtype=torch.bool))

    return batch_prefix_masks[:-1], batch_action_masks[:-1]
