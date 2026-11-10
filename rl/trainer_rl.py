import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import os
import copy
import torch
import time
import glob
import wandb
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict, List, Any
from torch.distributions import Categorical
from data.dataset import BlendableDataset, RLFullDataset
from contextlib import nullcontext
from data.sampler import build_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from data.problem_loader import ProblemLoader
from collections import OrderedDict
from dataclasses import dataclass, asdict
from data.dataset import get_loss_flag_and_position_id_v2
from utils.utils import set_seed, clean_print
from utils.utils_model import check_model_size
from utils.utils_data import rnd_sub_dataproblem, RLTaskInput
from scipy.stats import ttest_rel
from rl.rl_utils import sample_batch_episode, re_rollout
from train.scheduler import OptimizerParamScheduler, EarlyStopping
from eval.evaluate_utils import evalute_batch_episode, evalute_one_episode
from data.raw import get_problem_gen_func

@dataclass
class RL_Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    critic_model_state: 'OrderedDict[str, torch.Tensor]'
    critic_optimizer_state: Dict[str, Any]
    critic_scheduler_state: Dict[str, Any]
    total_steps: int        # for scheduler initialization
    trained_time: float     # for total training time
    latest_train_step: int  # for consistent optimizer-para schedule
    best_return: float      # for early_stopping
    wandb_id: str           # for resuming wandb log

class RLTrainer:
    def __init__(self, wandb_id, rl_args, args, unico, unico_ref, unico_best, unico_critic, train_envs:List, eval_envs:List, train_problems:Dict, eval_problems:Dict, datasets_sft:List, dataset_weights:List):
        self.args = args
        self.rl_args = rl_args
        self.wandb_id = wandb_id
        self.eval_envs = eval_envs
        self.train_envs = train_envs
        self.train_problems = train_problems
        self.snapshot_path = f'{self.args.out_dir}/rl/{rl_args.exp_name}/snapshot_seed{rl_args.seed}.pt'

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
        self.eval_problemloader_dict, self.dataloader_sft = self._prepare_dataloader(eval_envs, eval_problems, datasets_sft, dataset_weights)

        # wrap the model with DDP. this step will synch model across all the processes.
        self.unico = unico.to(self.local_rank).train()   
        self.unico = DDP(self.unico, device_ids=[self.local_rank])
        self.raw_unico = self.unico.module if hasattr(self.unico, "module") else self.unico        

        # optimizer
        self.optimizer = self.raw_unico.transformer.configure_optimizers()
        self.scheduler = OptimizerParamScheduler(self.args, self.optimizer) 
        self.optimizer_critic = None
        self.scheduler_critic = None
        
        # initialize train states
        self.step_now = 0
        self.step_start = 0
        self.trained_time = 0
        self.grad_batch_cnt = 0
        self.best_return = -float('inf')

        # best_obj so far for Ttest
        self.best_obj_inited = False
        self.best_obj = {k: None for k in self.args.eval_dataset_names}
        for dataset_name in sorted(self.args.eval_dataset_names):
            problem_num = len(eval_problems[f'Env_{dataset_name}'].answer_list)
            if dataset_name.startswith(('TSP', 'CVRP', 'PCTSP', 'SPCTSP', 'ATSP', '3DBP', 'FFSP')):
                self.best_obj[dataset_name] = torch.ones(problem_num).to(self.local_rank) * float('inf')    # for minimization
            elif dataset_name.startswith(('BP', 'OP', 'MIS')):
                self.best_obj[dataset_name] = torch.ones(problem_num).to(self.local_rank) * -float('inf')   # for maximization
            else:
                raise NotImplementedError(f'Unknown env name: {dataset_name}')

        # check MACs and params quantity
        clean_print('-'*30 + ' Model Size ' + '-'*30, self.local_rank)
        check_model_size(self.raw_unico, self.local_rank, '[UNICO]')
        if self.rl_args.beta > 0:   
            self.unico_ref = unico_ref.to(self.local_rank).eval()   # we don't update ref model, no need to wrap it with DDP
            self.unico_ref.requires_grad_(False)                    # disable gradient computation
            check_model_size(self.unico_ref, self.local_rank, '[UNICO_REF]')
        
        if self.rl_args.baseline_type == 'RB':
            self.unico_best = unico_best.to(self.local_rank).eval() # no need to wrap it with DDP
            self.unico_best.requires_grad_(False)                   # disable gradient computation
            check_model_size(self.unico_best, self.local_rank, '[UNICO_BEST]')
        elif self.rl_args.baseline_type == 'GAE':
            self.unico_critic = unico_critic.to(self.local_rank).train()
            self.unico_critic = DDP(self.unico_critic, device_ids=[self.local_rank])
            self.raw_unico_critic = self.unico_critic.module if hasattr(self.unico_critic, "module") else self.unico_critic 
            check_model_size(self.raw_unico_critic, self.local_rank, '[UNICO_CRITIC]')   
            self.optimizer_critic = self.raw_unico_critic.transformer.configure_optimizers()
            self.scheduler_critic = OptimizerParamScheduler(self.args, self.optimizer_critic, is_critic=True)             

        # data setting
        if self.local_rank == 0:
            print('-'*30 + ' Data Setting ' + '-'*30)
            if self.rl_args.train_dataset_type == 'sft':
                print(f'Train with SFT problems')
                for env_name, problems in self.train_problems.items():
                    print(f'> {env_name} problem num:\t{self.rl_args.q_total_problem_num} sampled in {len(problems.answer_list)}')
            elif self.rl_args.train_dataset_type == 'random':
                print(f'Train with random generated problems:\t{self.rl_args.q_total_problem_num} for each COP')
            else:
                raise NotImplementedError(f'Unknown train dataset type: {self.rl_args.train_dataset_type}')

        # load snapshot if available.
        dist.barrier()
        self._load_snapshot()

        # Mixed Precision Training
        if rl_args.use_amp:        
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16)) # bfloat16 doesn't need GradScaler cause it can be calculated in hardware drictly   
            self.scaler_critic = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16)) if self.rl_args.baseline_type == 'GAE' else None
        else:
            self.ctx = nullcontext()                                                    # null context
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)   
            self.scaler_critic = torch.cuda.amp.GradScaler(enabled=False) if self.rl_args.baseline_type == 'GAE' else None
    
    def _load_snapshot(self):
        try:
            snapshot_data = torch.load(self.snapshot_path, map_location=f"cuda:{self.local_rank}")
            snapshot = RL_Snapshot(**snapshot_data)
            
            self.raw_unico.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            step_factor = self.scheduler.load_state_dict(snapshot.scheduler_state)
            self.step_start = self.step_now = int(snapshot.latest_train_step * step_factor)
            self.trained_time = snapshot.trained_time
            self.best_return = snapshot.best_return
            self.wandb_id = snapshot.wandb_id
            
            if self.rl_args.beta > 0:
                self.unico_ref.load_state_dict(snapshot.model_state)
            if self.rl_args.baseline_type == 'RB':
                ckpt_dir = os.path.join(f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/best', '*.pt')
                ckpt_files = glob.glob(ckpt_dir)
                if ckpt_files != []:
                    ckpt_files = sorted(ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]), reverse=True)
                    best_ckpt = ckpt_files[0]
                    best_ckpt = torch.load(best_ckpt, map_location=f"cuda:{self.local_rank}")
                    self.unico_best.load_state_dict(best_ckpt)
                else:
                    self.unico_best.load_state_dict(snapshot.model_state)
            elif self.rl_args.baseline_type == 'GAE':
                self.raw_unico_critic.load_state_dict(snapshot.critic_model_state)
                self.optimizer_critic.load_state_dict(snapshot.critic_optimizer_state)
                self.scheduler_critic.load_state_dict(snapshot.critic_scheduler_state)
                
            clean_print(f"Resuming training from snapshot at step {self.step_now}", self.local_rank, '[Trainer]')
        
        except FileNotFoundError:
            self.best_return = -float('inf')
            self.step_start = 0
            self.step_now = 0
            self.trained_time = 0
            clean_print(f"Snapshot not found. Start RL from scratch", self.local_rank, '[Trainer]')

    def _save_snapshot(self):
        snapshot = RL_Snapshot(
            model_state=self.raw_unico.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            critic_model_state=self.raw_unico_critic.state_dict() if self.rl_args.baseline_type == 'GAE' else None,
            critic_optimizer_state=self.optimizer_critic.state_dict() if self.rl_args.baseline_type == 'GAE' else None,
            critic_scheduler_state=self.scheduler_critic.state_dict() if self.rl_args.baseline_type == 'GAE' else None,
            total_steps=self.rl_args.step_num,
            trained_time=self.trained_time,
            latest_train_step=self.step_now,
            best_return=self.best_return,            
            wandb_id=self.wandb_id
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.snapshot_path)

    def _ttest(self, eval_obj_tensor, ave_return=-1e5):
        update_best = False
        if ave_return > self.best_return:
            if not self.best_obj_inited or self.rl_args.ttest_alpha == 1:
                update_best = True
                self.best_obj_inited = True
            else:
                for i, dataset_name in enumerate(sorted(self.args.eval_dataset_names)):
                    best_obj = self.best_obj[dataset_name]
                    eval_obj = eval_obj_tensor[i]

                    # t = np.mean(a - b)/se indicates the ave pair difference
                    # smaller p means more significant difference
                    t, p = ttest_rel(eval_obj.cpu().numpy(), best_obj.cpu().numpy())
                    if (dataset_name.startswith(('TSP', 'CVRP', 'PCTSP', 'SPCTSP', 'ATSP', '3DBP', 'FFSP')) and t < 0) or \
                        (dataset_name.startswith(('BP', 'OP', 'MIS')) and t > 0):
                        
                        # one-sided
                        p_val = p / 2
                        if p_val < self.rl_args.ttest_alpha:
                            clean_print(f'[{dataset_name}] T-statistic: {t:.4f}, p-value: {p_val:.4f} < {self.rl_args.ttest_alpha}', self.local_rank, '[Trainer]')
                            update_best = True
                        else:
                            clean_print(f'[{dataset_name}] T-statistic: {t:.4f}, p-value: {p_val:.4f} >= {self.rl_args.ttest_alpha}', self.local_rank, '[Trainer]')
        
        if update_best:
            self.best_return = ave_return
            for i, dataset_name in enumerate(sorted(self.args.eval_dataset_names)):
                self.best_obj[dataset_name] = eval_obj_tensor[i]

        return update_best

    def _save_checkpoint(self, update_best=False, ave_return=-1e5):
        ckpt_dir = None
        if self.rl_args.save_strategy == 'interval':
            torch.save(
                self.raw_unico.state_dict(),
                f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/interval/{round(ave_return,5)}_step{self.step_now}.pt'
            )
            if self.rl_args.baseline_type == 'GAE':
                torch.save(
                    self.raw_unico_critic.state_dict(),
                    f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/interval/{round(ave_return,5)}_step{self.step_now}_critic.pt'
                )
            ckpt_dir = os.path.join(f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/interval', '*.pt')
            critic_ckpt_dir = os.path.join(f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/interval', '*_critic.pt')
        elif self.rl_args.save_strategy == 'best':
            if update_best:
                torch.save(
                    self.raw_unico.state_dict(),
                    f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/best/{round(self.best_return,5)}_step{self.step_now}.pt'
                )
                if self.rl_args.baseline_type == 'GAE':
                    torch.save(
                        self.raw_unico_critic.state_dict(),
                        f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/best/{round(self.best_return,5)}_step{self.step_now}_critic.pt'
                    )
                clean_print(f'New Best ckpt saved at step {self.step_now} with return {self.best_return}', self.local_rank, '[Trainer]')
                ckpt_dir = os.path.join(f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/best', '*.pt')
                critic_ckpt_dir = os.path.join(f'{self.args.out_dir}/rl/{self.rl_args.exp_name}/best', '*_critic.pt')
        else:
            raise NotImplementedError(f'Unknown save strategy: {self.rl_args.save_strategy}')

        # Only keep the best n ckpt
        if ckpt_dir:
            all_ckpt_files = glob.glob(ckpt_dir)    
            critic_ckpt_files = glob.glob(critic_ckpt_dir)
            actor_ckpt_files = [file for file in all_ckpt_files if file not in critic_ckpt_files]
            actor_ckpt_files = sorted(actor_ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]), reverse=True)
            for ckpt in actor_ckpt_files[self.rl_args.save_ckpt_num:]:
                os.remove(ckpt)
                critic_ckpt = ckpt.replace('.pt', '_critic.pt')
                if os.path.exists(critic_ckpt):
                    os.remove(critic_ckpt)

    def _prepare_dataloader(self, envs, envs_problems:Dict, datasets_sft:List[RLFullDataset]=None, dataset_weights:List[float]=None, is_rl=False):
        # build BlendableDataset & dataloader
        if datasets_sft is not None and dataset_weights is not None:
            dataset_sft = BlendableDataset(
                datasets_sft, 
                dataset_weights,
                batch_size=self.args.eval_batch_size,
                with_dataset_info=True
            )
            dataloader_sft = build_dataloader(self.args, dataset_sft, is_eval=True, seed=self.rl_args.seed)
        else:
            dataloader_sft = None

        # build problem loader
        problemloader_dict = {}
        for i, (name, dataset) in enumerate(envs_problems.items()):
            env = envs[i]
            assert env.env_name == name
            if name.startswith('Env_OBP3D'):
                problemloader_dict[name] = None
                if self.local_rank == 0:
                    self.problem_num[name] = self.rl_args.eval_iters_COP
                    self.random_obj_ave[name] = -1
                    self.best_obj_ave[name] = -1
                    self.random_obj_std[name] = -1
                    self.best_obj_std[name] = -1
            else:
                problemloader = ProblemLoader(dataset, env, is_rl)
                problemloader_dict[name] = problemloader

                # get random_obj & best_obj
                rnd_obj_value = problemloader.random_obj_array[problemloader.random_obj_array != 0]
                best_obj_value = problemloader.best_obj_array[problemloader.random_obj_array != 0]
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
                    #print(f'[{name}] Ave obj value of [{problem_num}] samples is: \n\trandom policy: [{random_obj_ave}], std=[{random_obj_std}]\n\tbest polciy:   [{best_obj_ave}], std=[{best_obj_std}]')
                    self.problem_num[name] = problem_num
                    self.random_obj_ave[name] = random_obj_ave
                    self.best_obj_ave[name] = best_obj_ave
                    self.random_obj_std[name] = random_obj_std
                    self.best_obj_std[name] = best_obj_std
             
        return problemloader_dict, dataloader_sft

    def _sample_episodes(self, problemloader_dict:dict=None, sample_action=False, regen_times=1, disable_tqdm=False, add_best=False, use_best_model=False):
        assert self.args.use_ddp_env

        reward_dict = {k: [] for k in self.args.eval_dataset_names}
        return_dict = {k: [] for k in self.args.eval_dataset_names}
        ids_dict = {k: None for k in self.args.eval_dataset_names}
        answers_dict = {k: [] for k in self.args.eval_dataset_names}
        for env_name, dataset_name, env in zip(self.args.eval_env_names, self.args.eval_dataset_names, self.train_envs):  
            desc = f'[GPU{self.global_rank}]: Rollout on {env_name} (group-{regen_times})' if sample_action else \
                    f'[GPU{self.global_rank}]: Rollout on {env_name} (greedy)'
            returns, objs, step_objs, ids, answers, _ = sample_batch_episode(
                args=self.args, 
                model=self.unico if not use_best_model else self.unico_best,
                env=env, 
                problemloader=problemloader_dict[env_name], 
                sample_action=sample_action, 
                regen_times=regen_times,
                desc=desc,
                device=self.local_rank,
                rl_args=self.rl_args,
                disable_tqdm=disable_tqdm,
                add_best=add_best,
                get_step_obj= self.rl_args.baseline_type=='GAE'
            )
            ids_dict[dataset_name] = ids                # [group_size x (q_batch_size_per_gpu, sql_len)]
            answers_dict[dataset_name] = answers        # [group_size x (problem_batch_size_per_gpu, timestep_num)] 
            if self.rl_args.rl_return_type == 'am':
                return_dict[dataset_name] = returns     # (group_size, q_batch_size_per_gpu)
                if self.rl_args.baseline_type == 'GAE':
                    rewards = torch.zeros_like(step_objs)   # (group_size, q_batch_size_per_gpu, max_timestep_num)
                    for g, batch_return in enumerate(returns):
                        for b, ret in enumerate(batch_return):
                            pad_num = (step_objs[g,b] == -1e5).sum().item()
                            rewards[g,b][-pad_num-1] = ret.item()
                            if pad_num > 0:
                                rewards[g,b][-pad_num:] = -1e5
                    reward_dict[dataset_name] = rewards     # [group_size x [q_batch_size_per_gpu x (timestep_num,)]])
                else:
                    reward_dict[dataset_name] = step_objs
            elif self.rl_args.rl_return_type == 'obj':
                if env_name.startswith(('Env_TSP', 'Env_CVRP', 'Env_PCTSP', 'Env_SPCTSP', 'Env_ATSP', 'Env_3DBP', 'Env_FFSP')):
                    return_dict[dataset_name] = -objs       # (group_size, q_batch_size_per_gpu)    
                    reward_dict[dataset_name] = -step_objs  # (group_size, q_batch_size_per_gpu, timestep_num)
                elif env_name.startswith(('Env_BP', 'Env_OP', 'Env_MIS')):
                    return_dict[dataset_name] = objs        # (group_size, q_batch_size_per_gpu)
                    reward_dict[dataset_name] = step_objs   # (group_size, q_batch_size_per_gpu, timestep_num)
            else:
                raise NotImplementedError(f'Unknown return type: {self.rl_args.rl_return_type}')

        return return_dict, reward_dict, ids_dict, answers_dict

    def _eval_policy(self, sample_action=False):
        regen_times = self.rl_args.resample_times if sample_action else 1
        desc_sample = 'sample' if sample_action else 'greedy'
        desc_regen = f'-{regen_times}' if regen_times != 1 else ''
        desc = f'{desc_sample}{desc_regen}'
        episode_return = {k: {'AM':[], 'DB1':[]} for k in self.args.eval_dataset_names}
        episode_obj = {k:[] for k in self.args.eval_dataset_names}
        episode_obj_std = {k: 0 for k in self.args.eval_dataset_names}
        episode_time = {k: 0 for k in self.args.eval_dataset_names}

        assert self.args.use_ddp_env
        for env_name, dataset_name, env in zip(self.args.eval_env_names, self.args.eval_dataset_names, self.eval_envs):
            if self.args.use_ddp_env:                
                env_return, env_obj, obj_std, ave_time_used = evalute_batch_episode(
                    args=self.args, 
                    model=self.unico, 
                    env=env, 
                    problemloader=self.eval_problemloader_dict[env_name], 
                    sample_action=sample_action, 
                    regen_times=regen_times,
                    desc=f'[GPU{self.global_rank}]: Evaluating on {env_name} ({desc})',
                    device=self.local_rank,
                    mean_reduce=False
                )
            
            episode_return[dataset_name] = env_return
            episode_obj[dataset_name] = env_obj
            episode_obj_std[dataset_name] = obj_std
            episode_time[dataset_name] = ave_time_used

        return episode_return, episode_obj, episode_obj_std, episode_time

    def _run_batch(self, rl_task_input, batch_dataset_name) -> float:
        input_ids = rl_task_input.tensor_seq
        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
            "use_cache": False, 
            "past_key_values": None,
        }
        _, loss, loss_datasets, _ = self.unico(rl_task_input, True, batch_dataset_name, model_inputs)
        return loss.item(), loss_datasets
    
    def _run_macro_batch(self):
        total = self.args.eval_batch_num
        desc = f'[GPU0-{self.world_size-1}]: Calculating sft_loss'
        dataloader = self.dataloader_sft
        dataloader.sampler.set_batch(0, self.step_now, False)

        mb_losses = []
        dataset_mb_losses = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
        with tqdm(total=total, desc=desc, position=self.local_rank, disable=self.local_rank!=0) as pbar:
            for batch in dataloader:
                rl_task_input, batch_data_info, _ = batch
                if self.args.auto_batch_len and rl_task_input.seq_len.max() < self.args.n_position:
                    rl_task_input.apply(lambda x: x[:, :rl_task_input.seq_len.max()] if isinstance(x, torch.Tensor) and x.dim() == 2 else x)
                    assert (rl_task_input.tensor_seq[:,-1]==self.args.special_tokens['<|>']).sum() >= 1
            
                # run batch
                batch_dataset_name = [info[0] for info in batch_data_info]
                rl_task_input.to(device=self.local_rank)
                loss, loss_datasets = self._run_batch(rl_task_input, batch_dataset_name)

                # record loss                
                mb_losses.append(loss)
                for dataset_name, dataset_loss in loss_datasets.items():
                    if dataset_loss != 0:
                        dataset_mb_losses[dataset_name].append(dataset_loss.item())

                # update progress bar
                batch_token = self.args.eval_batch_size * self.args.n_position
                pbar.set_postfix({
                    'batch_token': f'{batch_token/1e6:.4f}',
                    'loss':f'{loss:.2f}', 
                    'ave loss': f'{np.array(mb_losses).mean():.2f}',
                })
                pbar.update()

        mb_loss = np.array(mb_losses).mean()
        dataset_mb_losses = {name: 0 if len(losses) == 0 else np.array(losses).mean() for name, losses in dataset_mb_losses.items()}
        return mb_loss, dataset_mb_losses

    def _get_model_input(self, env, batch_complete_seq, batch_answer, batch_prefix_masks, batch_reward):
        '''
            重建 batch 的 RLTaskInput 对象，用于单独计算训练模型和参考模型的前向过程，
            当轨迹总长度不超过 n_position 时，只需要重新计算最后一个 timestep 的前向过程，需要完整的 token 序列 batch_complete_seq 
            由于使用动态 prefix_mask 及 action_mask, 还需要在 rollout 过程记录每个 timestep 的 prefix_mask & action_mask
        '''
        tensor_seq = batch_complete_seq[:,:-1]
        loss_flag, pos_id = get_loss_flag_and_position_id_v2(tensor_seq.shape[-1]+1, env.dataset.prefix_dim, env.dataset.obs_dim, env.dataset.act_dim,)
        pos_id = torch.tensor(pos_id[:-1, None].repeat(self.rl_args.q_batch_size_per_gpu, axis=1).T, dtype=torch.long, device=tensor_seq.device)
        loss_flag = torch.as_tensor(loss_flag[1:], dtype=torch.int8, device=tensor_seq.device)
        
        loss_mask = []
        act_idx = torch.where(loss_flag==1)[0]
        for answer in batch_answer:                        
            mask = loss_flag.clone()
            pad_timestep_num = (answer==-1).sum().item()
            if pad_timestep_num > 0:
                mask[act_idx[-pad_timestep_num:]] = 0
            loss_mask.append(mask)

        # RLTaskInput for TrajLlama model
        rl_task_input = RLTaskInput(
            tensor_seq=tensor_seq,              # (q_batch_size_per_gpu, seq_len-1)
            label=batch_complete_seq[:,1:],     # (q_batch_size_per_gpu, seq_len-1)
            rewards=batch_reward,               # (q_batch_size_per_gpu, timestep_num)
            position_id=pos_id,                 # (q_batch_size_per_gpu, seq_len-1)
            loss_mask=torch.stack(loss_mask),   # (q_batch_size_per_gpu, seq_len-1)
            prefix_mask=batch_prefix_masks,     # (q_batch_size_per_gpu, timestep_num, prefix_dim)
            attention_mask=torch.ones_like(tensor_seq, dtype=torch.bool),
            seq_len=None,
        )
        rl_task_input.to(device=self.local_rank)

        # model_input for transformers lib
        model_inputs = {
            "input_ids": tensor_seq.contiguous(),
            "position_ids": torch.arange(tensor_seq.shape[1], device=tensor_seq.device).unsqueeze(0),
            "use_cache": False,
            "past_key_values": None,
        }

        if self.rl_args.baseline_type == 'GAE':
            critic_task_input = copy.deepcopy(rl_task_input)
            critic_model_inputs = copy.deepcopy(model_inputs)

            # add last act token to critic_task_input
            assert critic_task_input.tensor_seq.min() >= 0
            critic_task_input.tensor_seq = torch.hstack((critic_task_input.tensor_seq, batch_answer[:,-1][:,None])) # (q_batch_size_per_gpu, seq_len)
            critic_task_input.tensor_seq[critic_task_input.tensor_seq==-1] = 0  # deal with the padding act
            critic_model_inputs['input_ids'] = critic_task_input.tensor_seq.contiguous()

            # add local position id for the last act token
            critic_task_input.position_id = torch.hstack((critic_task_input.position_id, torch.zeros(critic_task_input.position_id.shape[0], 1, dtype=torch.long).to(device=rl_task_input.position_id.device)))
            critic_model_inputs['position_ids'] = torch.arange(critic_task_input.tensor_seq.shape[1], device=critic_task_input.tensor_seq.device).unsqueeze(0)

            # calculate the loss at where act_spliter token is (state value)
            critic_task_input.loss_mask = torch.hstack((critic_task_input.loss_mask, torch.zeros(critic_task_input.loss_mask.shape[0],1,dtype=torch.int8).to(critic_task_input.loss_mask.device)))
            critic_task_input.attention_mask = torch.ones_like(critic_task_input.tensor_seq, dtype=torch.bool).to(device=self.local_rank)
            
            # use critic_task_input.rewards as the label for critic model
            critic_task_input.label = None
        else:
            critic_task_input = None
            critic_model_inputs = None


        return loss_flag, rl_task_input, model_inputs, critic_task_input, critic_model_inputs

    def _forward_critic(self, dataset_name, critic_task_input, critic_model_inputs):
        with self.ctx:
            loss, adv_list = self.unico_critic(
                critic_task_input, compute_loss=True, 
                batch_dataset_name=[dataset_name,]*self.rl_args.q_batch_size_per_gpu, 
                model_inputs=critic_model_inputs
            )

        return loss, adv_list

    def _get_log_prob(self, dataset_name, rl_task_input, model_inputs, loss_flag, batch_action_masks, batch_answer, model, get_entropy=False):
        with self.ctx:
            logits, _, _, _ = model(
                rl_task_input, compute_loss=True, 
                batch_dataset_name=[dataset_name,]*self.rl_args.q_batch_size_per_gpu, 
                model_inputs=model_inputs
            )      
            logits = logits[:,loss_flag==1]                 # (q_batch_size_per_gpu, timestep_num, total_vocab_size)
            logits = logits - batch_action_masks * 1e10     # (q_batch_size_per_gpu, timestep_num, total_vocab_size)
            log_probs = logits.log_softmax(dim=-1)          # (q_batch_size_per_gpu, timestep_num, total_vocab_size)

            batch_answer_temp = batch_answer.clone()
            batch_answer_temp[batch_answer_temp==-1] = 0
            act_log_probs = torch.gather(log_probs, dim=2, index=batch_answer_temp.unsqueeze(2)).squeeze(2) # (q_batch_size_per_gpu, timestep_num)

            if get_entropy:
                scaled_logits = logits / self.rl_args.temprature            # (q_batch_size_per_gpu, timestep_num, total_vocab_size)
                act_entropy = Categorical(logits=scaled_logits).entropy()   # (q_batch_size_per_gpu, timestep_num)
                return act_log_probs, act_entropy
            return act_log_probs

    def reinforce(self):
        set_seed(self.rl_args.seed)

        for step in range(self.step_start, self.rl_args.step_num+1):
            self.step_now = step
            wandb_log_dict = {}
            dist.barrier()
            clean_print('-'*50 + f' RL Step [{self.step_now}] ' + '-'*50, self.local_rank)

            # update lr
            new_lr, _, _ = self.scheduler.step()
            if self.rl_args.baseline_type == 'GAE':
                new_lr_critic, _, _ = self.scheduler_critic.step()

            # Save snapshot before the RL step process, so that there is no overlap when recover from the snapshot
            if self.rl_args.save_snapshot and self.local_rank == 0:
                self._save_snapshot()

            # Calculate validation losses at specified batch intervals
            if self.rl_args.check_sft_loss and (step % self.rl_args.eval_interval == 0 or step == self.rl_args.step_num) and (step != 0 or not self.rl_args.skip_first_eval):
                self.unico.eval()
                with torch.no_grad():
                    sft_loss, sft_loss_dataset = self._run_macro_batch()
                    wandb_log_dict.update({"losses/sft_loss": sft_loss})
                    wandb_log_dict.update({f'eval_{dataset_name[:-3]}/sft_loss': loss for dataset_name, loss in sft_loss_dataset.items()})
                dist.barrier()

            # Evaluate policy performance at specified batch intervals
            if (step % self.rl_args.eval_policy_interval == 0 or step == self.rl_args.step_num) and (step != 0 or not self.rl_args.skip_first_eval):
                self.unico.eval()
                return_list, obj_list = [], []
                with torch.inference_mode():
                    epi_return, epi_obj, episode_obj_std, epi_time = self._eval_policy(sample_action=False)
                    for dataset_name in sorted(self.args.eval_dataset_names):
                        wandb_log_dict.update({f'eval_{dataset_name[:-3]}/return_AM(greedy_cst)': np.mean(epi_return[dataset_name]['AM'])})
                        wandb_log_dict.update({f'eval_{dataset_name[:-3]}/return_DB1(greedy_cst)': np.mean(epi_return[dataset_name]['DB1'])})
                        wandb_log_dict.update({f'eval_{dataset_name[:-3]}/obj(greedy_cst)': np.mean(epi_obj[dataset_name])})
                        wandb_log_dict.update({f'eval_{dataset_name[:-3]}/obj_std(greedy_cst)': episode_obj_std[dataset_name]})
                        wandb_log_dict.update({f'eval_{dataset_name[:-3]}/time(greedy_cst)': epi_time[dataset_name]})
                        
                        obj_list.append(epi_obj[dataset_name])                          # (1, problem_num_per_gpu*problem_batch_num)
                        return_list.append(np.mean(epi_return[dataset_name]['AM']))     # 用 AM return 作为 ckpt 质量指标
                
                # gather returns and objs across all GPU
                eval_ave_return = np.mean(return_list)
                ave_return_tensor = torch.tensor([eval_ave_return,]).to(self.local_rank)
                return_gather_list = [torch.zeros_like(ave_return_tensor) for _ in range(self.world_size)]
                dist.gather(ave_return_tensor, return_gather_list if self.local_rank == 0 else None, dst=0)
                eval_obj_tensor = torch.tensor(obj_list).to(self.local_rank)
                eval_obj_gather_list = [torch.zeros_like(eval_obj_tensor) for _ in range(self.world_size)]
                dist.gather(eval_obj_tensor, eval_obj_gather_list if self.local_rank == 0 else None, dst=0)
                
                # update unico_ref & unico_best if ttest is passed
                dist.barrier()
                if self.local_rank == 0:
                    eval_ave_return = torch.vstack(return_gather_list).mean().item()
                    eval_obj_tensor = torch.hstack(eval_obj_gather_list)            # (env_num, total_problem_num)
                    update_best = self._ttest(eval_obj_tensor, eval_ave_return)
                    update_best_tensor = torch.tensor([1], device=self.local_rank) if update_best else torch.tensor([0], device=self.local_rank)
                else:
                    update_best_tensor = torch.tensor([0], device=self.local_rank)
                dist.broadcast(update_best_tensor, src=0)
                update_best = bool(update_best_tensor.item())

                if update_best:
                    if self.rl_args.beta > 0:
                        self.unico_ref.load_state_dict(self.raw_unico.state_dict())
                    if self.rl_args.baseline_type == 'RB':
                        self.unico_best.load_state_dict(self.raw_unico.state_dict())       

                dist.barrier()
                if self.local_rank == 0 and self.rl_args.save_ckpt:
                    self._save_checkpoint(update_best, eval_ave_return)

            # ========================= RL TRAINING STEP: Rollout =========================
            start_time = time.time()
            sub_step_data = []
            temp_return_dict = {'before_step': {name: [] for name in self.args.eval_dataset_names}, 'after_step': {name: [] for name in self.args.eval_dataset_names}}
            step_problemloader_all = []
            rng = np.random.default_rng(self.rl_args.seed + step)
            for sub_step_i in range(self.rl_args.q_ga_num):
                # get a batch of problems for each environment
                step_train_problems = {}
                problem_batch_size = self.world_size*self.rl_args.q_batch_size_per_gpu
                for env_name in self.args.eval_env_names:
                    if self.rl_args.train_dataset_type == 'sft':
                        problems = self.train_problems[env_name]
                        dataproblem = rnd_sub_dataproblem(problems, problem_batch_size, rng)
                    elif self.rl_args.train_dataset_type == 'random':
                        problem_gen_func = get_problem_gen_func(env_name, self.args, rng, problem_batch_size)
                        dataproblem = problem_gen_func()
                    else:
                        raise NotImplementedError(f'Unknown train dataset type: {self.rl_args.train_dataset_type}')
                    step_train_problems[env_name] = dataproblem
                step_problemloader_dict, _ = self._prepare_dataloader(self.train_envs, step_train_problems, is_rl=True)
                    
                # sample a group of answers for each problem
                self.unico.eval()
                with torch.inference_mode():
                    return_dict, reward_dict, ids_dict, answers_dict = self._sample_episodes(step_problemloader_dict, sample_action=True, regen_times=self.rl_args.q_group_size, add_best=self.rl_args.add_best_to_group,)
                    if self.rl_args.check_same_cop_improve:
                        step_problemloader_all.append(step_problemloader_dict)
                        before_return_dict, _, _, _ = self._sample_episodes(step_problemloader_dict, sample_action=False, regen_times=1)
                        for dataset_name in self.args.eval_dataset_names:
                            temp_return_dict['before_step'][dataset_name].append(before_return_dict[dataset_name])
                dist.barrier()

                # get gen_log_prob & ref_log_prob
                forward_input_dict, critic_input_dict, gen_act_entropy_dict, gen_log_probs_dict, ref_log_probs_dict = {}, {}, {}, {}, {}
                for dataset_name in sorted(self.args.eval_dataset_names):
                    answers = answers_dict[dataset_name]            # [group_size x (q_batch_size_per_gpu, timestep_num)]
                    ids = ids_dict[dataset_name]                    # [group_size x (q_batch_size_per_gpu, sql_len)]
                    rewards = reward_dict[dataset_name]             # (group_size, q_batch_size_per_gpu, max_timestep_num)
                    env = self.train_envs[self.args.eval_env_names.index(f'Env_{dataset_name}')]

                    # get log_prob of current model batch by batch
                    forward_inputs_list, gen_log_probs_list, ref_log_probs_list, critic_input_list = [], [], [], []
                    for batch_complete_seq, batch_answer, batch_reward in zip(ids, answers, rewards):
                        # re-rollout to get prefix_mask & action_mask
                        problem_loader = step_problemloader_dict[env.env_name]
                        batch_prefix_masks, batch_action_masks = re_rollout(self.rl_args, problem_loader, env, batch_answer.cpu().numpy(), device=self.local_rank)
                        batch_prefix_masks = torch.stack(batch_prefix_masks).transpose(0, 1)    # (q_batch_size_per_gpu, timestep_num, prefix_dim)
                        batch_action_masks = torch.stack(batch_action_masks).transpose(0, 1)    # (q_batch_size_per_gpu, timestep_num, vocab_size)

                        # build RLTaskInput object
                        loss_flag, rl_task_input, model_inputs, critic_task_input, critic_model_inputs = self._get_model_input(env, batch_complete_seq, batch_answer, batch_prefix_masks, batch_reward)
                        critic_input_list.append((env.dataset.dataset_name, critic_task_input, critic_model_inputs))
                        forward_input = (env.dataset.dataset_name, rl_task_input, model_inputs, loss_flag, batch_action_masks, batch_answer)
                        forward_inputs_list.append(forward_input)
                        
                        with torch.inference_mode():
                            # get ref_log_prob
                            if self.rl_args.beta > 0:
                                self.unico_ref.eval()
                                ref_log_probs = self._get_log_prob(*forward_input, self.unico_ref)
                                ref_log_probs_list.append(ref_log_probs.detach())
                            else:
                                ref_log_probs_list.append(None)

                            # get gen_log_prob
                            if self.rl_args.loss_type == 'ppo':
                                self.unico.eval()
                                gen_log_probs = self._get_log_prob(*forward_input, self.unico)
                                gen_log_probs_list.append(gen_log_probs.detach())
                            else:
                                gen_log_probs_list.append(None)

                    forward_input_dict[dataset_name] = forward_inputs_list
                    critic_input_dict[dataset_name] = critic_input_list
                    gen_log_probs_dict[dataset_name] = gen_log_probs_list   # [group_size x (q_batch_size_per_gpu, timestep_num)]
                    ref_log_probs_dict[dataset_name] = ref_log_probs_list   # [group_size x (q_batch_size_per_gpu, timestep_num)]

                # compute advantages
                advantages_dict = {}
                if self.rl_args.baseline_type == 'GM':
                    # the baseline is the average return of the sampled group
                    returns_list = []
                    for dataset_name in sorted(self.args.eval_dataset_names):
                        returns = return_dict[dataset_name]             # (group_size, q_batch_size_per_gpu)
                        advantages = (returns - returns.mean(dim=1)[:,None].detach())
                        advantages = (returns - returns.mean(dim=1)[:,None].detach()) / (returns.std(dim=1)[:,None].detach() + 1e-4)
                        returns_list.append(returns)                    # (group_size, q_batch_size_per_gpu) 
                        advantages_dict[dataset_name] = advantages      # (group_size, q_batch_size_per_gpu)
                elif self.rl_args.baseline_type in ['RB', 'RC']:
                    # the baseline is the return of greedy policy
                    use_best_model = self.rl_args.baseline_type == 'RB'
                    greedy_return_dict, _, _, _ = self._sample_episodes(step_problemloader_dict, sample_action=False, regen_times=1, disable_tqdm=True, use_best_model=use_best_model)
                    for dataset_name in sorted(self.args.eval_dataset_names):
                        sample_returns = return_dict[dataset_name]                      # (group_size, q_batch_size_per_gpu)
                        greedy_return = greedy_return_dict[dataset_name].detach()       # (1, q_batch_size_per_gpu)
                        advantages_dict[dataset_name] = sample_returns - greedy_return  # (group_size, q_batch_size_per_gpu)
                elif self.rl_args.baseline_type == 'GAE':
                    advantages = []
                    critic_loss_temp = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                    self.unico_critic.eval()
                    with torch.inference_mode():
                        for dataset_name in sorted(self.args.eval_dataset_names):
                            critic_input_list = critic_input_dict[dataset_name]             # [group_size x critic_input]
                            for critic_input in critic_input_list:
                                c_loss, adv_list = self._forward_critic(*critic_input)   
                                advantages.append(torch.tensor(adv_list))                   # (q_batch_size_per_gpu, timestep_num)
                                critic_loss_temp[dataset_name].append(c_loss.item())        # (q_batch_size_per_gpu, timestep_num)
                            advantages_dict[dataset_name] = torch.stack(advantages)         # (group_size, q_batch_size_per_gpu, timestep_num)
                else:
                    raise NotImplementedError(f'Unknown baseline type: {self.rl_args.baseline_type}')
               
                sub_step_data.append((forward_input_dict, gen_log_probs_dict, ref_log_probs_dict, advantages_dict, answers_dict))

            # ========================= RL TRAINING STEP: Advantage Batch Norm =========================
            all_step_adv_list = []
            raw_adv_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
            for step_data in sub_step_data:
                step_advantages_dict = step_data[3]
                for dataset_name in self.args.eval_dataset_names:
                    raw_adv_dict[dataset_name].append(step_advantages_dict[dataset_name])
                all_step_adv_list.extend(list(step_advantages_dict.values()))
            adv_all = torch.vstack(all_step_adv_list).to(self.local_rank)       # (dataset_num*q_ga_num*group_size, q_batch_size_per_gpu)
            raw_adv_dict = {k:torch.vstack(v) for k, v in raw_adv_dict.items()} # (dataset_num*q_ga_num*group_size, q_batch_size_per_gpu)

            gather_list = [torch.empty_like(adv_all) for _ in range(self.world_size)]
            dist.all_gather(gather_list, adv_all)
            adv_all = torch.stack(gather_list, dim=0)       # (world_size, dataset_num*q_ga_num*group_size, q_batch_size_per_gpu)
            step_adv_mean = adv_all.mean().cpu()
            step_adv_std = adv_all.std().cpu() + 1e-4

            for step_i in range(self.rl_args.q_ga_num):
                for dataset_name in sorted(self.args.eval_dataset_names):
                    sub_step_data[step_i][3][dataset_name] = (sub_step_data[step_i][3][dataset_name] - step_adv_mean) / step_adv_std

            for dataset_name in self.args.eval_dataset_names:
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/max_adv': raw_adv_dict[dataset_name].max().item()})
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/min_adv': raw_adv_dict[dataset_name].min().item()})
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/mean_adv': raw_adv_dict[dataset_name].mean().item()})
                
            # ========================= RL TRAINING STEP: Update model multiple times =========================
            self.unico.train()
            with tqdm(total=self.rl_args.update_num, desc=f'[GPU0-{self.world_size-1}]: Update Model', position=self.local_rank, disable=self.local_rank != 0) as pbar:
                for update_i in range(self.rl_args.update_num):
                    rl_loss_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                    improve_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                    kl_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                    gen_act_entropy_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                    temp_return_dict['after_step'] = {name: [] for name in self.args.eval_dataset_names}
                    
                    # loop for different cop
                    for dataset_name in sorted(self.args.eval_dataset_names):
                        # loop for grad accumulation
                        for forward_input_dict, gen_log_probs_dict, ref_log_probs_dict, advantages_dict, answers_dict in sub_step_data:                
                            forward_inputs_list = forward_input_dict[dataset_name]
                            gen_log_probs_list = gen_log_probs_dict[dataset_name]   # [group_size x (q_batch_size_per_gpu, timestep_num)]
                            ref_log_probs_list = ref_log_probs_dict[dataset_name]   # [group_size x (q_batch_size_per_gpu, timestep_num)]
                            advantages = advantages_dict[dataset_name]              # (group_size, q_batch_size_per_gpu)
                            answers = answers_dict[dataset_name]                    # [group_size x (q_batch_size_per_gpu, timestep_num)]
                            
                            # loop for different group
                            gen_act_entropy_list = []
                            for forward_input, gen_log_probs, ref_log_probs, advantage, answer in zip(forward_inputs_list, gen_log_probs_list, ref_log_probs_list, advantages, answers):
                                # compute log_probs
                                log_probs, act_entropy = self._get_log_prob(*forward_input, self.unico, get_entropy=True)           # (q_batch_size_per_gpu, timestep_num)
                                                                
                                # compute loss
                                loss_mask = answer != -1                                                                            # (q_batch_size_per_gpu, timestep_num) 
                                advantage = advantage.to(self.local_rank)                       
                                advantage = advantage[:,None].repeat(1, log_probs.shape[1]) if advantage.ndim == 1 else advantage   # (q_batch_size_per_gpu, timestep_num)
                                if self.rl_args.loss_type == 'ppo':
                                    ratio = torch.exp(log_probs - gen_log_probs)                                            # (q_batch_size_per_gpu, timestep_num)
                                    clipped_ratio = torch.clamp(ratio, 1-self.rl_args.clip_adv[0], 1+self.rl_args.clip_adv[1])
                                    surrogate_loss = torch.min(ratio * advantage, clipped_ratio * advantage)                # (q_batch_size_per_gpu, timestep_num)

                                    if self.rl_args.beta > 0:
                                        assert log_probs.shape == gen_log_probs.shape == ref_log_probs.shape
                                        per_token_kl = torch.exp(ref_log_probs-log_probs) - (ref_log_probs-log_probs) - 1   # (q_batch_size_per_gpu, timestep_num)
                                        per_token_loss = -(surrogate_loss - self.rl_args.beta * per_token_kl)               # (q_batch_size_per_gpu, timestep_num)
                                        kl_dict[dataset_name].append(per_token_kl.mean().item())
                                    else:
                                        per_token_loss = -surrogate_loss
                         
                                    loss = ((per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()  
                                elif self.rl_args.loss_type == 'reinforce':
                                    log_probs[loss_mask == 0] = 0
                                    log_likelihood = log_probs.sum(1)                           # (q_batch_size_per_gpu, )
                                    loss = -(advantage * log_likelihood[:,None]).mean()
                                else:
                                    raise NotImplementedError(f'Unknown loss type: {self.rl_args.loss_type}')

                                # compute entropy loss
                                entropy_bonus = ((act_entropy * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean() 
                                gen_act_entropy_list.append(entropy_bonus.item())
                                loss -= self.rl_args.entropy_coef * entropy_bonus
                                
                                # backward pass, with gradient scaling if training in fp16
                                rl_loss_dict[dataset_name].append(loss.item())
                                loss_ave = loss / (len(self.args.eval_dataset_names) * len(sub_step_data) * self.rl_args.q_group_size)
                                with self.unico.no_sync():
                                    self.scaler.scale(loss_ave).backward()    
                        gen_act_entropy_dict[dataset_name] = np.mean(gen_act_entropy_list).item()
                        
                    # update actor model
                    if self.args.clip_grad != 0:
                        self.scaler.unscale_(self.optimizer)    # When using AMP, the gradients are stored in scaled form, We need to unscale the gradients to perform the correct clipping operation.
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.unico.parameters(), self.args.clip_grad)
                    self.scaler.step(self.optimizer)            # use scaler to adjust gradients and skip update step if necessary to avoid numerical instability when using AMP.
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)  # flush the gradients as soon as we can, no need for this memory anymore
                    
                    # update critic model
                    if self.rl_args.baseline_type == 'GAE':
                        # release memory
                        torch.cuda.empty_cache()

                        # compute critic loss
                        self.unico_critic.train()
                        critic_loss_dict = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
                        for dataset_name in sorted(self.args.eval_dataset_names):
                            critic_input_list = critic_input_dict[dataset_name]
                            for i, critic_input in enumerate(critic_input_list):
                                critic_loss, _ = self._forward_critic(*critic_input)
                                critic_loss_dict[dataset_name].append(critic_loss)    
                                assert critic_loss.item() == critic_loss_temp[dataset_name][i]

                                loss_ave_crtic = critic_loss / (len(self.args.eval_dataset_names) * len(sub_step_data) * self.rl_args.q_group_size)
                                with self.unico_critic.no_sync():
                                    self.scaler_critic.scale(loss_ave_crtic).backward()

                        # update critic model       
                        if self.args.clip_grad != 0:
                            self.scaler_critic.unscale_(self.optimizer_critic)
                            grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.unico_critic.parameters(), self.args.clip_grad)
                        self.scaler_critic.step(self.optimizer_critic)
                        self.scaler_critic.update()
                        self.optimizer_critic.zero_grad(set_to_none=True)
                    
                    # check_same_cop_improve
                    dist.barrier()
                    if self.rl_args.check_same_cop_improve:
                        for sub_step_i in range(self.rl_args.q_ga_num):
                            # prepare dataloader
                            step_problemloader_dict = step_problemloader_all[sub_step_i]
                            for problemloader in step_problemloader_dict.values():
                                problemloader.reset()

                            # sample a group of answers for each problem
                            self.unico.eval()
                            with torch.inference_mode():
                                after_return_dict, _, _, _ = self._sample_episodes(step_problemloader_dict, sample_action=False, regen_times=1, disable_tqdm=True)
                                for dataset_name in self.args.eval_dataset_names:
                                    temp_return_dict['after_step'][dataset_name].extend(after_return_dict[dataset_name])

                        for dataset_name in sorted(self.args.eval_dataset_names):
                            am_before_step = torch.vstack(temp_return_dict['before_step'][dataset_name]).mean().item()
                            am_after_step = torch.vstack(temp_return_dict['after_step'][dataset_name]).mean().item()
                            improve_dict[dataset_name].append(am_after_step - am_before_step)

                    # 更新进度条
                    rl_loss_dict = {k: np.mean(v) for k, v in rl_loss_dict.items()}
                    rl_loss = np.mean(list(rl_loss_dict.values()))
                    kl_dict = {k: np.mean(v) if self.rl_args.beta > 0 else 0 for k, v in kl_dict.items()}
                    kl = np.mean(list(kl_dict.values())) if self.rl_args.beta > 0 else 0
                    gen_act_entropy = np.mean(list(gen_act_entropy_dict.values()))
                    pbar_info = {
                        'act_entropy':  f'{gen_act_entropy:.4f}',
                        'grad_norm': f'{grad_norm:.2f}',
                        'kl': f'{kl:.3f}',
                        'actor_loss': f'{rl_loss:.4f}',
                    }
                    if self.rl_args.check_same_cop_improve:
                        improve_dict = {k: np.mean(v) for k, v in improve_dict.items()}
                        improve = np.mean(list(improve_dict.values()))
                        pbar_info.update({'improved': f'{improve:.4f}'})
                    if self.rl_args.baseline_type == 'GAE':
                        critic_loss_dict = {k: torch.hstack(v).mean().item() for k, v in critic_loss_dict.items()}
                        critic_loss = np.mean(list(critic_loss_dict.values()))
                        pbar_info.update({'critic_loss': f'{critic_loss:.4f}'})
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)

                    if self.rl_args.use_dynamic_temprature:
                        if gen_act_entropy < 0.5:
                            self.rl_args.temprature += 0.1
                        elif gen_act_entropy > 1.5:
                            self.rl_args.temprature -= 0.1

            # wandb log
            self.trained_time += time.time() - start_time
            for dataset_name in self.args.eval_dataset_names:
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/kl': kl_dict[dataset_name]})
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/rl_loss': rl_loss_dict[dataset_name]})
                wandb_log_dict.update({f'eval_{dataset_name[:-3]}/act_entropy': gen_act_entropy_dict[dataset_name]})
            wandb_log_dict.update({
                'info/lr': new_lr,
                'info/temprature': self.rl_args.temprature,
                'info/grad_norm': grad_norm.item(),
                "info/grpo_step": step,
                "info/act_entropy": gen_act_entropy,
                'info/trained_time': self.trained_time,
                "losses/rl_loss": rl_loss,
            })
            if self.rl_args.check_same_cop_improve:
                wandb_log_dict.update({'info/AM_improved': improve})
                for dataset_name in self.args.eval_dataset_names:
                    wandb_log_dict.update({f'eval_{dataset_name[:-3]}/AM_improved': improve_dict[dataset_name]})
            if self.rl_args.baseline_type == 'GAE':
                wandb_log_dict.update({
                    "info_crtic/lr": new_lr_critic,
                    "info_crtic/grad_norm_critic": grad_norm_critic.item(),
                    "losses/critic_loss": critic_loss
                })
                for dataset_name in self.args.eval_dataset_names:
                    wandb_log_dict.update({f'eval_{dataset_name[:-3]}/critic_loss': critic_loss_dict[dataset_name]})

            # gather info from all GPU
            log_value_tensor = torch.tensor([wandb_log_dict[k] for k in sorted(wandb_log_dict)]).to(self.local_rank)
            log_gather_list = [torch.zeros_like(log_value_tensor) for _ in range(self.world_size)]
            dist.gather(log_value_tensor, log_gather_list if self.local_rank == 0 else None, dst=0)

            # only do file saving and logging at rank0
            if self.local_rank == 0:
                # get log info 
                log_value_tensor = torch.mean(torch.stack(log_gather_list), axis=0, dtype=torch.float32)
                for i, key in enumerate(sorted(wandb_log_dict)):
                    wandb_log_dict[key] = log_value_tensor[i].item()
                wandb.log(wandb_log_dict)