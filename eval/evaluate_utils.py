import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import math
import numpy as np
import torch
import time
from tqdm import tqdm
from environment.wrapper import LMPromptEnv
from environment.DDP_wrapper import DDP_LMPromptEnv
from data.problem_loader import ProblemLoader
from argparse import Namespace
from typing import Optional, Union
from torch.nn import functional as F
from data.dataset import get_loss_flag_and_position_id_v1, get_loss_flag_and_position_id_v2
from utils.utils_data import RLTaskInput
from utils.utils import COP_FAILED_RWD

def get_action(
    args, model, env, current_seq,
    sample_action: bool = False,
    device: Optional[Union[int, str, torch.device]] = None,
    model_inputs: Optional[dict] = None
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
        pos_id = pos_id[:-1, None].repeat(problem_batch_size_per_gpu, axis=1).T
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
        logits, _, _, outputs = model(                                  # logits: (problem_batch_size_per_gpu, seq_len, total_vocab_size)
            x, compute_loss=False, 
            batch_dataset_name=[env.dataset.dataset_name,], 
            model_inputs=model_inputs
        )                           
        
        # Set mask according to the output space of current generating dim of action 
        action_masks = env.get_action_mask(hard_action_constraint=True, generated_actions=np.array(batch_action_seq).T)[i_act],
        action_masks = torch.from_numpy(np.array(action_masks)).to(device).squeeze()
        action_masks = action_masks[None, :] if action_masks.ndim == 1 else action_masks    # (batch_size, total_vocab_size)
        logits = logits[:, -1, :] - action_masks * 1e10                 # (problem_batch_size_per_gpu, total_vocab_size)
        probs = F.softmax(logits, dim=-1)                               # (problem_batch_size_per_gpu, total_vocab_size)

        # get pred action token
        if sample_action:
            pred_token = torch.multinomial(probs, num_samples=1)        # (problem_batch_size_per_gpu, 1)
        else:
            _, pred_token = torch.topk(probs, k=1, dim=-1)              # (problem_batch_size_per_gpu, 1)
        
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
    return act, current_seq, outputs

#@torch.no_grad()
@torch.inference_mode()
def evalute_one_episode(
    args: Namespace,
    model: torch.nn.Module,
    env: LMPromptEnv,
    sample_action: bool = False,
    regen_times: int = 1,
    problem_info: tuple = None,
    problem_obj: tuple = None,
    device: Optional[Union[int, str, torch.device]] = None
):
    assert not args.use_ddp_env
    eval_prompt_strat = args.prompt_strategy.split(";")[-1]     # moving_prompt
    assert args.use_prefix ^ (args.use_prompt and eval_prompt_strat == "moving_prompt")
    prefix_dim = env.dataset.prefix_dim
    trans_dim = env.dataset.trans_dim
    device = device if device is not None else torch.device(f"cuda:{args.device[0]}" if torch.cuda.is_available() and torch.cuda.device_count() >= args.device[0]+1 else "cpu")
    spliter_token = torch.tensor([args.special_tokens['<|>']], dtype=torch.long).to(device)
    prefix_spliter_token = torch.tensor([args.special_tokens['<X>']], dtype=torch.long).to(device)
    regen_times = 1 if not sample_action else regen_times

    # Get ready
    raw_model = model.module if hasattr(model, "module") else model

    # 对于 COPTask 多次生成解进行自洽性投票，若为 RLTask 则 regen_times==1
    best_episode_return = {'AM':-1e4, 'DB1':-1e4,}
    best_episode_obj = 1e4 if env.env_name.startswith('Env_FFSP') else -1e4
    time_start = time.time()
    for _ in range(regen_times):   
        # reset env 
        if problem_info is None:
            assert env.env_name.startswith('Env_OBP3D')
            current_seq, info = env.reset()
        else:
            current_seq, info = env.reset(options={
                'problem_info': problem_info,
                'problem_obj': problem_obj,
            })
        current_seq = current_seq.to(device)

        # prepare prompt or prefix
        if args.use_prompt:
            fixed_prompt, raw_obs = env.get_prompt(strict_length=args.strict_length, minimal_expert_data=args.minimal_expert_data)
            fixed_prompt = fixed_prompt.to(device)
            current_seq = torch.cat([fixed_prompt, current_seq, spliter_token])         # 拼接当前obs和spliter，下一步用于自回归生成action                                 
            raw_prefix = None
            raw_obs = {k:np.vstack((v, info['obs'][k][None,:])) for k, v in raw_obs.items()}
        elif args.use_prefix:
            prefix_tensor, raw_prefix = env.get_prefix(with_raw=True)
            prefix_tensor = prefix_tensor.to(device)
            current_seq = torch.cat([prefix_tensor, prefix_spliter_token, current_seq, spliter_token])    # 拼接当前obs和spliter，下一步用于自回归生成action
            raw_obs = {k:v[None,:] if len(v.shape) != 0 else np.array([v]) for k,v in info['obs'].items()} 
        else:
            current_seq = torch.cat([current_seq, spliter_token])                       # 拼接当前obs和spliter，下一步用于自回归生成action
            raw_prefix = None
            raw_obs = {k:v[None,:] for k,v in info['obs'].items()} 
            
        trans_num = math.ceil((len(current_seq)-prefix_dim)/trans_dim)
        assert trans_num == len(list(raw_obs.values())[0])
        current_seq = current_seq[None, :]                  # (1, sql_len)

        # START EVALUATION
        model_kwargs = {'use_cache': args.use_kvcache, 'position_ids': None, }
        episode_return, episode_obj = {'AM':0, 'DB1':0}, 0
        while True:
            # prepare model inputs
            model_inputs = model.module.transformer.prepare_inputs_for_generation(current_seq, **model_kwargs)

            # Generate action
            assert current_seq.shape[1] <= raw_model.transformer.n_position
            act, current_seq, outputs = get_action(
                args=args,
                model=model,
                env=env,
                current_seq=current_seq,
                sample_action=sample_action,
                device=device,
                model_inputs=model_inputs
            )
            act = act.item()

            # env one step update
            new_seq, reward, terminated, truncated, info = env.step(act)
            new_seq = new_seq.unsqueeze(0) if new_seq.ndim == 0 else new_seq
            episode_return['AM'] += reward['AM']
            episode_return['DB1'] += reward['DB1']
            
            # exit point
            if terminated or truncated:
                episode_obj = info['obj'].item()
                break
            
            # update current_seq
            new_seq = new_seq.to(device)
            current_seq = torch.cat([current_seq.squeeze(), new_seq, spliter_token])       # (sql_len,) 拼接新的观测和分隔符
            if args.use_kvcache:
                model_kwargs["past_key_values"] = outputs.past_key_values

            # 超长处理
            if len(current_seq) > args.n_position:                                        
                if (args.use_prompt and eval_prompt_strat == "moving_prompt") or (not args.use_prompt and not args.use_prefix):
                    # 不用 prefix/prompt 或使用 "moving_prompt"，则随着序列增长不断保留尾部序列, 这样 prompt 序列会不断更新
                    current_seq = current_seq[trans_dim:]
                    if args.use_kvcache:
                        for kcache, vcache in model_kwargs["past_key_values"]:
                            kcache = kcache[:, trans_dim:]
                            vcache = vcache[:, trans_dim:]
                elif args.use_prefix:
                    # 如果设置 prefix, 则维持序列首部的 prefix 不变
                    window_seq_view = torch.roll(current_seq[prefix_dim+1:], -trans_dim)        # 将 current_seq 中除 prefix 以外的序列循环左移 prefix_dim
                    current_seq[prefix_dim+1:].data.copy_(window_seq_view.data)                 # 除 prefix 以外的序列中，首部那个 transition 对应的序列放在尾部
                    current_seq = current_seq[:-trans_dim]                                      # 把上一步放到尾部的（原首部）transition对应序列去除
                
                    # kvcache只能保留prompt部分，这会导致加速效果大幅下降，最好通过调整序列长度来避免超长
                    if args.use_kvcache:
                        updated_past_key_values = []
                        for kcache, vcache in model_kwargs["past_key_values"]:
                            kcache = kcache[:,:,:prefix_dim+1,:]
                            vcache = vcache[:,:,:prefix_dim+1,:]
                            updated_past_key_values.append((kcache, vcache))
                        model_kwargs["past_key_values"] = tuple(updated_past_key_values)        # 更新 model_kwargs 中的 vcache
                else:
                    raise NotImplementedError

                #raw_obs = {k:np.vstack((v[1:], info['obs'][k][None,:])) for k, v in raw_obs.items()}
                current_seq = current_seq[None, :]                  # (1, sql_len)
            else:
                #raw_obs = {k:np.vstack((v, info['obs'][k][None,:])) for k, v in raw_obs.items()}
                trans_num = math.ceil((len(current_seq)-prefix_dim)/trans_dim)
                current_seq = current_seq[None, :]

        # 更新最优
        if env.env_name.startswith('Env_FFSP'):
            update_best = episode_obj < best_episode_obj
        elif env.env_name.startswith('Env_MIS') or env.env_name.startswith('Env_OBP3D'):
            update_best = episode_obj > best_episode_obj
        else:
            update_best = episode_return['AM'] > best_episode_return['AM']
        
        if update_best:
            best_episode_return['AM'] = episode_return['AM']
            best_episode_return['DB1'] = episode_return['DB1']
            best_episode_obj = episode_obj

    assert best_episode_return['AM'] != -1e4 and best_episode_return['DB1'] != -1e4 and episode_obj != 1e-4
    time_end = time.time()
    time_used = time_end - time_start
    assert best_episode_return['AM'] != COP_FAILED_RWD
    return best_episode_return, best_episode_obj, time_used

@torch.inference_mode()
def evalute_batch_episode(
    args: Namespace,
    model: torch.nn.Module,
    env: DDP_LMPromptEnv,
    problemloader: ProblemLoader,
    sample_action: bool = False,
    regen_times:int = 1,
    desc: str = '',
    device: Optional[Union[int, str, torch.device]] = None,
    mean_reduce: bool = True,
):
    eval_prompt_strat = args.prompt_strategy.split(";")[-1]     # moving_prompt
    rank = int(os.environ.get("RANK", default='0'))
    assert args.use_ddp_env and device is not None
    assert args.use_prefix ^ (args.use_prompt and eval_prompt_strat == "moving_prompt")
    prefix_dim = 0 if env.dataset.prefix_dim is None else env.dataset.prefix_dim
    trans_dim = env.dataset.trans_dim
    regen_times = 1 if not sample_action else regen_times
    problem_batch_size_per_gpu = int(args.problem_batch_size_per_gpu/24) if env.env_name.startswith('Env_FFSP') else args.problem_batch_size_per_gpu
    spliter_tokens = torch.full((args.problem_batch_size_per_gpu, 1), args.special_tokens['<|>'], dtype=torch.long).to(device)
    prefix_spliter_tokens = torch.full((args.problem_batch_size_per_gpu, 1), args.special_tokens['<X>'], dtype=torch.long).to(device)

    # get ready
    problemloader.reset()
    raw_model = model.module if hasattr(model, "module") else model
        
    # START EVALUATION
    episode_return, episode_obj, episode_slove_cnt, episode_slove_time = {'AM':[], 'DB1':[]}, [], 0, 0
    episode_acts = []
    tqdm_total = args.problem_batch_num * problem_batch_size_per_gpu
    with tqdm(total=tqdm_total, desc=desc, position=rank) as pbar:
        for _ in range(args.problem_batch_num):
            # reset env to set the eval problem  
            problem_info, problem_obj = problemloader.get_problem(problem_batch_size_per_gpu)
            problem_idx = list(range(problem_batch_size_per_gpu))
            
            best_episode_return = {'AM':np.ones(len(problem_idx))*-1e4, 'DB1':np.ones(len(problem_idx))*-1e4,}
            best_episode_obj = np.ones(len(problem_idx))*1e4 if env.env_name.startswith('Env_FFSP') else np.ones(len(problem_idx))*-1e4           
            time_start = time.time()
            for _ in range(regen_times):
                episode_return_t = {'AM':np.ones(len(problem_idx))*-1e4, 'DB1':np.ones(len(problem_idx))*-1e4,}
                episode_obj_t = np.ones(len(problem_idx))*1e4 if env.env_name.startswith('Env_FFSP') else np.ones(len(problem_idx))*-1e4     
                
                # reset env 
                current_seq, info = env.reset(options={     # current_seq: (problem_batch_size_per_gpu, obs_dim)  
                    'problem_info':problem_info,
                    'problem_obj':problem_obj, 
                    'problem_idx':problem_idx,
                })       
                current_seq = current_seq.to(device)

                # prepare prompt or prefix
                if args.use_prompt:
                    fixed_prompt, raw_obs = env.get_prompt(args.strict_length, args.minimal_expert_data, with_raw=False)    # (problem_batch_size_per_gpu, len_prompt)
                    fixed_prompt = fixed_prompt.to(device)
                    current_seq = torch.cat([fixed_prompt, current_seq, spliter_tokens], dim=-1)                            # (problem_batch_size_per_gpu, len_prompt+obs_dim+1) 拼接当前 obs 和 spliter，下一步用于自回归生成 action
                    # raw_prefix = None
                    # raw_obs = {k: np.concatenate((v, info['obs'][k][:,None,:]), axis=1) if info['obs'][k].ndim==2 else \
                    #             np.concatenate((v, info['obs'][k][:,None,None]), axis=1) for k, v in raw_obs.items()}
                elif args.use_prefix:
                    prefix_tensor, raw_prefix = env.get_prefix(with_raw=False)
                    prefix_tensor = prefix_tensor.to(device)
                    current_seq = torch.cat([prefix_tensor, prefix_spliter_tokens, current_seq, spliter_tokens], dim=-1)    # 拼接当前obs和spliter，下一步用于自回归生成action
                    # raw_obs = {k: v[:,None,:] if v.ndim==2 else v[:,None,None] for k,v in info['obs'].items()} 
                else:
                    current_seq = torch.cat([current_seq, spliter_tokens], dim=-1)              # (problem_batch_size_per_gpu, obs_dim+1) 拼接当前obs和spliter，下一步用于自回归生成action
                    # raw_prefix = None
                    # raw_obs = {k: v[:,None,:] if v.ndim==2 else v[:,None,None] for k,v in info['obs'].items()} 
                # trans_num = math.ceil((current_seq.shape[1]-prefix_dim)/env.dataset.trans_dim)
                # assert trans_num == list(raw_obs.values())[0].shape[1]

                # eval a batch of problem
                batch_done = np.zeros(problem_batch_size_per_gpu, dtype=bool)
                model_kwargs = {'use_cache': args.use_kvcache, 'position_ids': None, }
                while batch_done.sum() < problem_batch_size_per_gpu:
                    # prepare model inputs
                    if args.model == 'llama':
                        model_inputs = model.module.transformer.prepare_inputs_for_generation(current_seq, **model_kwargs)
                    else:
                        model_inputs = None

                    # Generate action
                    assert current_seq.shape[1] <= raw_model.transformer.n_position
                    act, current_seq, outputs = get_action(
                        args=args,
                        model=model,
                        env=env,
                        current_seq=current_seq,
                        sample_action=sample_action,
                        device=device,
                        model_inputs=model_inputs
                    )
                    act = act.squeeze()     # (problem_batch_size_per_gpu, act_dim) or (problem_batch_size_per_gpu, ) when act_dim==1
                    episode_acts.append(act)

                    # env one step update
                    new_seq, reward, terminated, truncated, info = env.step(act)    # new_seq: (problem_batch_size_per_gpu, obs_dim)
                    assert reward['AM'][batch_done==1].sum() == 0, 'Some episode that have ended send back reward signals'

                    done = terminated | truncated                                   # (problem_batch_size_per_gpu, )
                    assert (batch_done + done).max() < 2, 'Some episode end more than one times'
                    batch_done |= done                                              # (problem_batch_size_per_gpu, )

                    episode_return_t['AM'][terminated==1] = reward['AM'][terminated==1]
                    episode_return_t['DB1'][terminated==1] = reward['DB1'][terminated==1]
                    episode_obj_t[terminated==1] = info['obj'][terminated==1]

                    # update current_seq
                    new_seq = new_seq.to(device)                                                    # 拼接新的观测和分隔符
                    current_seq = torch.cat([current_seq, new_seq, spliter_tokens], dim=-1)         # (problem_batch_size_per_gpu, sql_len)
                    if args.model == 'llama' and args.use_kvcache:
                        model_kwargs["past_key_values"] = outputs.past_key_values
                        
                    # 超长处理
                    if current_seq.shape[1] > args.n_position:                                                
                        if (args.use_prompt and eval_prompt_strat == "moving_prompt") or (not args.use_prompt and not args.use_prefix):
                            # 不用 prefix/prompt 或使用 "moving_prompt"，则随着序列增长不断保留尾部序列, 这样 prompt 序列会不断更新
                            current_seq = current_seq[:, trans_dim:]
                            if args.model == 'llama' and args.use_kvcache:
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

                        # raw_obs = {
                        #     k: np.concatenate((v[:,1:,:], info['obs'][k][:,None,:]), axis=1) if info['obs'][k].ndim==2 else \
                        #         np.concatenate((v[:,1:,:], info['obs'][k][:,None,None]), axis=1) for k, v in raw_obs.items()
                        # }
                    # else:
                        # raw_obs = {
                        #     k: np.concatenate((v, info['obs'][k][:,None,:]), axis=1) if info['obs'][k].ndim==2 else \
                        #         np.concatenate((v, info['obs'][k][:,None,None]), axis=1) for k, v in raw_obs.items()
                        # }
                        # trans_num = math.ceil((current_seq.shape[1]-prefix_dim)/env.dataset.trans_dim)
                        # assert trans_num == list(raw_obs.values())[0].shape[1]

                assert np.all(episode_return_t['AM'] != -1e4) and np.all(episode_return_t['DB1'] != -1e4) and np.all(episode_obj_t != -1e4)
                if env.env_name.startswith('Env_FFSP'):
                    update_mask = episode_obj_t < best_episode_obj
                elif env.env_name.startswith('Env_MIS'):
                    update_mask = episode_obj_t > best_episode_obj
                else:
                    update_mask = episode_return_t['AM'] > best_episode_return['AM']
                best_episode_return['AM'][update_mask] = episode_return_t['AM'][update_mask]
                best_episode_return['DB1'][update_mask] = episode_return_t['DB1'][update_mask]
                best_episode_obj[update_mask] = episode_obj_t[update_mask]
            episode_slove_time += time.time() - time_start
            episode_slove_cnt += (episode_obj_t!=1e-4).sum()
            episode_return['AM'].extend(best_episode_return['AM'].tolist())
            episode_return['DB1'].extend(best_episode_return['DB1'].tolist())
            episode_obj.extend(best_episode_obj.tolist())

            # 更新进度条
            assert (np.array(episode_return['AM']) == COP_FAILED_RWD).sum() == 0
            env_return_AM = 0 if len(episode_return['AM']) == 0 else np.mean(episode_return['AM'])
            env_return_DB1 = 0 if len(episode_return['DB1']) == 0 else np.mean(episode_return['DB1'])
            env_obj = 0 if len(episode_obj) == 0 else np.mean(episode_obj)
            env_obj_std = 0 if len(episode_obj) == 0 else np.std(episode_obj)
            time_used = 0 if episode_slove_cnt == 0 else episode_slove_time / episode_slove_cnt
            info = {
                'ret_AM': f'{env_return_AM:.4f}',
                'ret_DB1': f'{env_return_DB1:.4f}',
                'obj': f'{env_obj:.2f}',
                'std': f'{env_obj_std:.2f}',
                'time': f'{time_used:.4f}',
            }
            pbar.set_postfix(info)
            pbar.update((episode_obj_t!=1e-4).sum())

    if mean_reduce:
        env_obj = 0 if len(episode_obj) == 0 else np.mean(episode_obj)
        env_return = {
            'AM': 0 if episode_return['AM'] == [] else np.mean(episode_return['AM']),
            'DB1': 0 if episode_return['DB1'] == [] else np.mean(episode_return['DB1'])        
        }
    else:
        env_obj = episode_obj
        env_return = episode_return
    
    # actions = np.vstack(episode_acts)
    # print()
    # print('-' * 30) 
    # print(f'problem-1: start:  {problem_info[1][0]["current_position"]}')
    # print(f'problem-1: answer: {problem_info[2][0][1:]}')
    # print(f'problem-1: eval:   {list(actions[:,0])} = {env_obj[0]}')
    # print() 
    # print(f'problem-2: start:  {problem_info[1][1]["current_position"]}')
    # print(f'problem-2: answer: {problem_info[2][1][1:]}')
    # print(f'problem-2: eval:   {list(actions[:,1])} = {env_obj[1]}')
    # print('-' * 30) 
    # print()
    
    return env_return, env_obj, env_obj_std, time_used