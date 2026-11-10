import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import os
import torch
import math
import time
import json
import pprint
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from torch.nn.parallel import DistributedDataParallel as DDP
from data.dataset import BlendableDataset, RLFullDataset
from data.problem_loader import ProblemLoader
from data.sampler import build_dataloader
from data.tokenizer import ContinuousScalarTokenizer
from utils.utils import set_seed, create_file_if_not_exist
from eval.evaluate_utils import evalute_one_episode, evalute_batch_episode

class Evaluater:
    def __init__(self, args, eval_args, unico, envs, envs_problems, datasets_train, datasets_test, dataset_weights):
        self.args = args
        self.eval_args = eval_args
        self.envs = envs
        
        # problem info
        self.problem_num = {}
        self.best_obj_ave = {}
        self.random_obj_ave = {}
        self.best_obj_std = {}
        self.random_obj_std = {}
    
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) 
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # data stuff
        (
            self.dataloader_train, 
            self.dataloader_val, 
            self.dataloader_test, 
            self.problemloader_dict,
        ) = self._prepare_dataloader(datasets_train, datasets_test, dataset_weights, envs_problems)        

        # wrap the model with DDP. this step will synch model across all the processes.
        self.unico = unico.to(self.local_rank).train()   
        self.unico = DDP(self.unico, device_ids=[self.local_rank])
        self.raw_unico = self.unico.module if hasattr(self.unico, "module") else self.unico
        self.cont_tokenizer = ContinuousScalarTokenizer(self.args.tokenizer_ver, self.args.num_continuous_bin, self.args.discretize_mu, self.args.discretize_M)
        
    def _prepare_dataloader(self, datasets_train:List[RLFullDataset], datasets_test:List[RLFullDataset], dataset_weights:List,envs_problems:Dict):
        # split training set and evaluation set
        if datasets_train != []:
            train_datasets, val_datasets, test_datasets = [], [], datasets_test
            for dataset in datasets_train:
                dataset_train, dataset_val = dataset.split_dataset(self.args.split)
                val_datasets.append(dataset_val)
                train_datasets.append(dataset_train)
                
            # build BlendableDataset
            dataset_train = BlendableDataset(
                train_datasets, 
                dataset_weights,
                batch_size=self.eval_args.batch_size,
                log_data=False,
                with_dataset_info=True
            )
            dataset_val = BlendableDataset(
                val_datasets, 
                dataset_weights,
                batch_size=self.eval_args.eval_batch_size,
                with_dataset_info=True
            )

            dataloader_train = build_dataloader(self.eval_args, dataset_train, is_eval=False)
            dataloader_val = build_dataloader(self.eval_args, dataset_val, is_eval=True)

            # The test dataset is the MDP episode dataset associated with dataset_problem
            if len(datasets_test) == len(datasets_train):
                dataset_test = BlendableDataset(
                    test_datasets, 
                    dataset_weights,
                    batch_size=self.eval_args.test_batch_size,
                    with_dataset_info=True
                )

                dataloader_test = build_dataloader(self.eval_args, dataset_test, is_eval=True)
            else:
                dataset_test = dataloader_test = None
        else:
            dataloader_train = dataset_train = dataset_val = dataloader_val = dataset_test = dataloader_test = None

        # build problem loader
        problemloader_dict = {}
        for i, (name, dataset) in enumerate(envs_problems.items()):
            env = self.envs[i]
            assert env.env_name == name
            if name.startswith('Env_OBP3D'):
                problemloader_dict[name] = None
                if self.local_rank == 0:
                    self.problem_num[name] = self.eval_args.total_problem_num
                    self.random_obj_ave[name] = -1
                    self.best_obj_ave[name] = -1
                    self.random_obj_std[name] = -1
                    self.best_obj_std[name] = -1
            else:
                problemloader = ProblemLoader(dataset, env)
                problemloader_dict[name] = problemloader

                # get random_obj & best_obj
                rnd_obj_value = problemloader.random_obj_array[problemloader.random_obj_array != 0]
                best_obj_value = problemloader.best_obj_array[problemloader.best_obj_array != 0]
                obj_value = np.vstack([rnd_obj_value, best_obj_value])
                obj_value_tensor = torch.tensor(obj_value).to(self.local_rank)
                obj_value_gather_list = [torch.zeros_like(obj_value_tensor) for _ in range(self.world_size)]
                dist.barrier()
                dist.gather(obj_value_tensor, obj_value_gather_list if self.local_rank == 0 else None, dst=0)

                if self.local_rank == 0:
                    obj_info_tensor = torch.cat(obj_value_gather_list, dim=1)
                    problem_num = obj_info_tensor.shape[1]
                    random_obj_ave = obj_info_tensor[0].mean().item()
                    best_obj_ave = obj_info_tensor[1].mean().item()
                    random_obj_std = obj_info_tensor[0].std().item()
                    best_obj_std = obj_info_tensor[1].std().item()
                    print(f'[{name}] Ave obj value of [{problem_num}] samples is: \n\trandom policy: [{random_obj_ave}], std=[{random_obj_std}]\n\tbest polciy:   [{best_obj_ave}], std=[{best_obj_std}]')
                    self.problem_num[name] = problem_num
                    self.random_obj_ave[name] = random_obj_ave
                    self.best_obj_ave[name] = best_obj_ave
                    self.random_obj_std[name] = random_obj_std
                    self.best_obj_std[name] = best_obj_std
             
        return dataloader_train, dataloader_val, dataloader_test, problemloader_dict

    def _eval_policy(self, sample_action=False, regen_times=1):
        regen_times = self.eval_args.resample_times if sample_action else 1
        desc_sample = 'sample' if sample_action else 'greedy'
        desc_regen = f'-{regen_times}' if regen_times != 1 else ''
        desc = f'{desc_sample}{desc_regen}'
        episode_return = {k: {'AM':0, 'DB1':0} for k in self.args.eval_dataset_names}
        episode_obj = {k: 0 for k in self.args.eval_dataset_names}
        episode_obj_std = {k: 0 for k in self.args.eval_dataset_names}
        episode_time = {k: 0 for k in self.args.eval_dataset_names}

        for env_name, dataset_name, env in zip(self.args.eval_env_names, self.args.eval_dataset_names, self.envs):
            if self.args.use_ddp_env:                
                ave_return, ave_obj, obj_std, ave_time_used = evalute_batch_episode(
                    args=self.args, 
                    model=self.unico, 
                    env=env, 
                    problemloader=self.problemloader_dict[env_name], 
                    sample_action=sample_action, 
                    regen_times=regen_times,
                    desc=f'[GPU{self.global_rank}]: Evaluating on {env_name} ({desc})',
                    device=self.local_rank,
                )
            else:
                if not env_name.startswith('Env_OBP3D'):
                    problemloader = self.problemloader_dict[env_name]
                    problemloader.reset()
                iters = self.args.problem_batch_num * self.args.problem_batch_size_per_gpu
                epi_return, epi_obj, epi_time = {'AM':[], 'DB1':[]}, [], []
                with tqdm(total=iters, desc=f'[GPU{self.global_rank}]: Evaluating on {env_name} ({desc})', position=self.local_rank) as pbar:
                    # problems 列表中每个元素都是目标环境的一个初始观测（包含随机生成的目标COP问题）
                    # 通过 env._recover 方法还原环境的观测从而设置被评估的COP问题
                    for _ in range(iters):
                        if not env_name.startswith('Env_OBP3D'):
                            problem_info, problem_obj = problemloader.get_problem(1)
                            problem_info = (None if problem_info[0] is None else problem_info[0][0], problem_info[1][0], problem_info[2][0])
                            problem_obj = (problem_obj[0].item(), problem_obj[1].item())
                        else:
                            problem_info = problem_obj = None

                        ep_ret, ep_obj, ep_time = evalute_one_episode(
                            args=self.args, 
                            model=self.unico, 
                            env=env, 
                            sample_action=sample_action,
                            regen_times=regen_times, 
                            problem_info=problem_info, 
                            problem_obj=problem_obj, 
                            device=self.local_rank
                        )
                        epi_time.append(ep_time)
                        epi_return['AM'].append(ep_ret['AM'])
                        epi_return['DB1'].append(ep_ret['DB1'])
                        epi_obj.append(ep_obj)

                        ave_time_used = 0 if epi_time == [] else np.mean(epi_time)
                        ave_obj = 0 if epi_obj == [] else np.mean(epi_obj)
                        obj_std = 0 if epi_obj == [] else np.std(epi_obj)
                        ave_return_AM = 0 if epi_return['AM'] == [] else np.mean(epi_return['AM'])
                        ave_return_DB1 = 0 if epi_return['DB1'] == [] else np.mean(epi_return['DB1'])
                        ave_return = {'AM':ave_return_AM, 'DB1':ave_return_DB1}

                        info = {
                            'ret_AM': f'{ave_return_AM:.4f}',
                            'ret_DB1': f'{ave_return_DB1:.4f}',
                            'obj' : f'{ave_obj:.4f}',
                            'std' : f'{obj_std:.4f}',
                            'time': f'{ave_time_used:.4f}',
                        }
                        pbar.set_postfix(info)
                        pbar.update()

            episode_return[dataset_name] = ave_return
            episode_obj[dataset_name] = ave_obj
            episode_obj_std[dataset_name] = obj_std
            episode_time[dataset_name] = ave_time_used
        
        return episode_return, episode_obj, episode_obj_std, episode_time

    def _run_batch(self, rl_task_input, batch_dataset_name, compute_loss=True) -> float:
        with torch.set_grad_enabled(False):
            input_ids = rl_task_input.tensor_seq
            model_inputs = {
                "input_ids": input_ids.contiguous(),
                "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
                "use_cache": False, 
                "past_key_values": None,
            }    
            _, loss, _, _ = self.unico(rl_task_input, compute_loss, batch_dataset_name, model_inputs)
        return loss.item()
    
    def _run_macro_batch(self, desc, total_num, batch_size, dataloader):
        epoch_losses = []
        desc = f'[GPU{self.global_rank}]: ' + desc
        with tqdm(total=total_num, desc=desc, position=self.local_rank) as pbar:
            for batch in dataloader:
                rl_task_input, batch_dataset_name, _ = batch
                if self.args.auto_batch_len and rl_task_input.seq_len.max() < self.args.n_position:
                    rl_task_input.apply(lambda x: x[:, :rl_task_input.seq_len.max()] if isinstance(x, torch.Tensor) and x.dim() == 2 else x)
                    assert (rl_task_input.tensor_seq[:,-1]==self.args.special_tokens['<|>']).sum() >= 1

                rl_task_input.to(device=self.local_rank)
                batch_dataset_name = [v[0] for v in batch_dataset_name]
                loss = self._run_batch(rl_task_input, batch_dataset_name)
                epoch_losses.append(loss)

                pbar.set_postfix({'ave loss': '{:.2f}'.format(np.array(epoch_losses).mean())})
                pbar.update(batch_size)

        return np.array(epoch_losses).mean()

    def evaluate(self):    
        # check train & val_loss
        self.unico.eval()
        if self.eval_args.check_loss:
            with torch.inference_mode():
                self.dataloader_val.sampler.set_batch(0, 0, False)
                self.dataloader_train.sampler.set_batch(0, 0, True)
                train_loss = self._run_macro_batch('Calculating train loss', self.eval_args.batch_num*self.eval_args.batch_size_per_gpu, self.eval_args.batch_size_per_gpu, self.dataloader_train)
                val_loss = self._run_macro_batch('Calculating val_loss', self.eval_args.eval_batch_num*self.eval_args.eval_batch_size_per_gpu, self.eval_args.eval_batch_size_per_gpu, self.dataloader_val)
                if self.dataloader_test is not None:
                    self.dataloader_test.sampler.set_batch(0, 0, False)
                    test_loss = self._run_macro_batch('Calculating test loss', self.eval_args.eval_test_num*self.eval_args.test_batch_size_per_gpu, self.eval_args.test_batch_size_per_gpu, self.dataloader_test)
        
        # policy evaluation setting
        eval_setting = {
            # 'greedy_cst': lambda: self._eval_policy(sample_action=False),
            'sample_cst': lambda: self._eval_policy(sample_action=True),
        }

        # evaluate policy
        results = {setting: [] for setting in eval_setting.keys()}
        with torch.inference_mode():
            set_seed(self.eval_args.seed)

            # evaluate policy
            for setting, eval_func in eval_setting.items():
                if self.local_rank == 0:
                    print(f'\n\n' + '-'*20 + f' Eval policy under setting [{setting}]; seed [{self.eval_args.seed}] ' + '-'*20)
                
                # eval on all dataset
                epi_return_dict, epi_obj_dict, episode_obj_std_dict, epi_time_dict = eval_func()

                # gather result from all GPU
                result_AM, result_DB1, result_obj, result_obj_std, result_time = [], [], [], [], []
                for k in sorted(self.args.eval_dataset_names):
                    result_AM.append(epi_return_dict[k]['AM'])
                    result_DB1.append(epi_return_dict[k]['DB1'])
                    result_obj.append(epi_obj_dict[k])
                    result_obj_std.append(episode_obj_std_dict[k])
                    result_time.append(epi_time_dict[k])
                result_tensor = torch.tensor([result_AM, result_DB1, result_obj, result_obj_std, result_time]).to(self.local_rank)
                gather_list = [torch.zeros_like(result_tensor) for _ in range(self.world_size)]
                dist.barrier()
                dist.gather(result_tensor, gather_list if self.local_rank == 0 else None, dst=0)
                gather_result = torch.stack(gather_list).mean(axis=0)
                results[setting].append(gather_result)
        dist.barrier()

        # Use this structure to store eval results
        result = {
            setting: {
                env_name: {'return_AM':0, 'return_DB1':0, 'problem_num':0, 'obj_best':0, 'obj_model':0, 'obj_random':0, 'time':0} 
                for env_name in self.args.eval_env_names
            } for setting in eval_setting.keys()
        }
        if self.local_rank == 0:
            # result summary
            result_file_path = f'{self.eval_args.ckpt_performance_path}/result.txt'
            create_file_if_not_exist(result_file_path)
            with open(f'{self.eval_args.ckpt_performance_path}/eval_config.json', 'w') as f:
                f.write(json.dumps(vars(self.eval_args), indent=4))

            time.sleep(1)
            print('\n')
            for setting, res in results.items():
                res_summary = torch.stack(res).mean(axis=0)
                return_AM_summary = res_summary[0]
                return_DB1_summary = res_summary[1]
                obj_summary = res_summary[2]
                obj_std_summary = res_summary[3]
                time_summary = res_summary[4]

                print('='*20+f' {setting} '+'='*20)
                with open(result_file_path, 'a') as f:
                    f.write('='*20+f' {setting} '+'='*20 + '\n')    
                for i, env_name in enumerate(sorted(self.args.eval_env_names)):            
                    result[setting][env_name]['return_AM'] = return_AM_summary[i].item()
                    result[setting][env_name]['return_DB1'] = return_DB1_summary[i].item()
                    result[setting][env_name]['obj_model'] = obj_summary[i].item()
                    result[setting][env_name]['obj_model_std'] = obj_std_summary[i].item()
                    result[setting][env_name]['obj_best'] = self.best_obj_ave[env_name]
                    result[setting][env_name]['obj_best_std'] = self.best_obj_std[env_name]
                    result[setting][env_name]['obj_random'] = self.random_obj_ave[env_name]
                    result[setting][env_name]['obj_random_std'] = self.random_obj_std[env_name]
                    result[setting][env_name]['time'] = time_summary[i].item()
                    result[setting][env_name]['problem_num'] = self.problem_num[env_name]

                    # 打印结果
                    pp = pprint.PrettyPrinter(indent=4)
                    print(env_name)
                    pp.pprint({k: round(v,5) for k,v in result[setting][env_name].items()})

                    # 结果保存为 txt 文件
                    with open(result_file_path, 'a') as f:
                        sys.stdout = f                  # 将标准输出重定向到文件
                        pp = pprint.PrettyPrinter(indent=4)
                        print(env_name)
                        pp.pprint({k: round(v,5) for k,v in result[setting][env_name].items()})
                        sys.stdout = sys.__stdout__     # 恢复标准输出

                with open(result_file_path, 'a') as f:
                    f.write('\n') 

                # info = {
                #     'train_info':{
                #         'train_iters': self.args.train_iters,
                #         'mbatch_num': int(self.args.train_iters / self.args.batch_num),
                #         'batch_num_per_mbatch': int(self.args.batch_num / self.args.ga_begin),
                #         'batch_size': self.args.batch_size,
                #         'batch_size_per_gpu': self.args.batch_size_per_gpu,
                #         'world_size': self.args.world_size,
                #         'ga_begin': self.args.ga_begin,
                #         'ga_end': self.args.ga_end,
                #         'grad_accum_step_incr_style': self.args.grad_accum_step_incr_style
                #     },
                #     'early_stopping': {
                #         'use_early_stop': self.args.use_early_stopping,
                #         'early_stopping_patience': self.args.early_stopping_patience,
                #         'early_stopping_delta': self.args.early_stopping_delta,
                #         'eval_mb_interval': int(self.args.eval_interval / self.args.ga_begin),
                #     },
                #     'learning_rate': {
                #         'warmup_ratio': self.args.lr_warmup_ratio,
                #         'decay_ratio': self.args.lr_decay_ratio,
                #         'decay_style': self.args.lr_decay_style,
                #         'decay_factor': self.args.lr_decay_factor,
                #         'lr_begin': self.args.lr_begin,
                #         'lr_max': self.args.lr_max,
                #     }
                # }
                # with open(f'{self.eval_args.ckpt_performance_path}/info.json', 'w') as f:
                #     json.dump(info, f, indent=4)