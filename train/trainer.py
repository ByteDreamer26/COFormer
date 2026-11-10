import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import wandb
import time
import glob
import pprint
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from model import UniCO
from contextlib import nullcontext
from data.problem_loader import ProblemLoader
from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Any, Dict, List
from train.scheduler import OptimizerParamScheduler, EarlyStopping
from data.dataset import BlendableDataset, RLFullDataset
from data.sampler import build_dataloader
from data.tokenizer import ContinuousScalarTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils_model import check_MAC
from utils.utils import set_seed, clean_print
from eval.evaluate_utils import evalute_one_episode, evalute_batch_episode

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    total_steps: int        # for scheduler initialization
    trained_time: float     # for total training time
    latest_train_batch: int # for consistent optimizer-para schedule
    best_val_loss: float    # for early_stopping
    best_return: float      # for early_stopping
    wandb_id: str           # for resuming wandb log

class Trainer:
    def __init__(self, args, seed, wandb_id, envs, datasets, dataset_weights, envs_problems):
        self.seed = seed
        self.args = args
        self.wandb_id = wandb_id
        self.envs = envs
        self.snapshot_path = f'{self.args.out_dir}/sft/snapshot_seed{self.seed}.pt'
        set_seed(seed)

        # policy evaluation setting
        self.eval_setting = {
            'greedy_cst': lambda: self._eval_policy(sample_action=False),
            #'sample_cst': lambda: self._eval_policy(sample_action=True),
        }

        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) 
        self.global_rank = int(os.environ["RANK"])      
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        # wrap the model with DDP. this step will synch model across all the processes.
        self.unico = UniCO(args).to(self.local_rank).train()
        self.unico = DDP(self.unico, device_ids=[self.local_rank])
        self.raw_unico = self.unico.module if hasattr(self.unico, "module") else self.unico
        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=args.early_stopping_delta)

        # initialize train states
        self.optimizer = self.raw_unico.transformer.configure_optimizers()  
        self.scheduler = OptimizerParamScheduler(self.args, self.optimizer) 
        self.best_return = -float('inf')
        self.best_val_loss = float('inf')
        self.ga_current = self.args.ga_begin
        self.batch_start = 0
        self.batch_now = 0
        self.trained_time = 0
        self.grad_batch_cnt = 1

        # load snapshot if available.
        self._load_snapshot()

        # compile the model
        if self.args.compile:
            clean_print("compiling the model... (takes a ~minute)", self.local_rank, '[Trainer]')
            self.raw_unico = torch.compile(self.raw_unico)      # requires PyTorch 2.0
        
        # data stuff
        (
            self.dataloader_train, 
            self.dataloader_val, 
            self.problemloader_dict,
            self.dataset_train, 
            self.dataset_val,
        ) = self._prepare_dataloader(datasets, dataset_weights, envs_problems)        

        # check MACs and params quantity
        check_MAC(args, self.dataloader_val, self.local_rank, self.world_size)
        dist.barrier()

        # Mixed Precision Training
        if args.use_amp:        
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16)) # bfloat16 doesn't need GradScaler cause it can be calculated in hardware drictly   
        else:
            self.ctx = nullcontext()                                                    # null context
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)      

    def _prepare_dataloader(self, datasets:List[RLFullDataset], dataset_weights:List[float], envs_problems:Dict):
        # split training set and evaluation set
        datasets_train, datasets_val = [], []
        for dataset in datasets:
            dataset_train, dataset_val = dataset.split_dataset(self.args.split)
            datasets_train.append(dataset_train)
            datasets_val.append(dataset_val)
            
        # build BlendableDataset & dataloader
        dataset_train = BlendableDataset(
            datasets_train, 
            dataset_weights,
            batch_size=self.args.batch_size_per_gpu,
            with_dataset_info=True
        )
        dataset_val = BlendableDataset(
            datasets_val, 
            dataset_weights,
            batch_size=self.args.eval_batch_size_per_gpu,
            with_dataset_info=True
        )
        dataloader_train = build_dataloader(self.args, dataset_train, is_eval=False, current_batch=self.batch_now, seed=self.seed)
        dataloader_val = build_dataloader(self.args, dataset_val, is_eval=True, seed=self.seed)
        
        # build problemloader
        problemloader_dict = {}
        for i, (name, dataset) in enumerate(envs_problems.items()):
            env = self.envs[i]
            assert env.env_name == name
            problemloader_dict[name] = ProblemLoader(dataset, env) if not env.env_name.startswith('Env_OBP3D') else None

        if self.local_rank == 0:
            train_batch_size_begin = self.args.batch_size_per_gpu * self.world_size * self.args.ga_begin
            train_batch_size_end = self.args.batch_size_per_gpu * self.world_size * self.args.ga_end
            train_sample_all = self.args.batch_size_per_gpu * self.world_size * self.args.train_iters

            print('-'*100)
            print(f"> Val/Eval batch_size:    {self.args.eval_batch_size} ({self.args.eval_batch_size_per_gpu}*{self.world_size})")
            print(f"> Val Dataset size:       {len(dataset_val)}")
            print(f"> Train batch_size:       {train_batch_size_begin} ({self.args.batch_size_per_gpu}*{self.world_size}*{self.args.ga_begin}) ---{self.args.grad_accum_step_incr_style}---> {train_batch_size_end} ({self.args.batch_size_per_gpu}*{self.world_size}*{self.args.ga_end})")
            print(f"> Train sample all:       {train_sample_all}")
            print(f'> Train dataset size:     {len(dataset_train)}')
            pp = pprint.PrettyPrinter(indent=4)
            for i, dataset in enumerate(dataset_train.datasets):
                env_name = dataset.env_name
                if not env.env_name.startswith('Env_OBP3D'):
                    problemloader = problemloader_dict[env_name]
                    rnd_obj_value = problemloader.random_obj_array[problemloader.random_obj_array != 0].mean()
                    best_obj_value = problemloader.best_obj_array[problemloader.best_obj_array != 0].mean()
                    info = {'size': len(dataset), 'weight': dataset_weights[i]}
                    info[f'rnd_obj[{len(problemloader)}]'] = rnd_obj_value
                    info[f'best_obj[{len(problemloader)}]'] = best_obj_value
                    print(f'\t{dataset.dataset_name:10}', end='\t')
                    pp.pprint({k: round(v,3) for k,v in info.items()})
            print('-'*100)
        
        dist.barrier()
        return dataloader_train, dataloader_val, problemloader_dict, dataset_train, dataset_val

    def _load_snapshot(self):
        try:
            snapshot_data = torch.load(self.snapshot_path, map_location=f"cuda:{self.local_rank}")
            snapshot = Snapshot(**snapshot_data)
            self.raw_unico.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            batch_factor = self.scheduler.load_state_dict(snapshot.scheduler_state)
            self.best_val_loss = snapshot.best_val_loss
            self.best_return = snapshot.best_return
            self.batch_start = self.batch_now = int(snapshot.latest_train_batch * batch_factor)
            macro_batch_now = self.batch_now // self.args.batch_num
            self.wandb_id = snapshot.wandb_id
            self.trained_time = snapshot.trained_time
            self.ga_current = self.scheduler.get_ga_step()
            self.args.batch_size = self.args.batch_size_per_gpu * self.world_size * self.ga_current
            clean_print(f"Resuming training from snapshot at macro_batch {macro_batch_now}({self.batch_now}) with grad_accum_step={self.ga_current}", self.local_rank, '[Trainer]')
        
        except FileNotFoundError:
            self.best_return = -float('inf')
            self.best_val_loss = float('inf')
            self.batch_start = 0
            self.batch_now = 0
            self.trained_time = 0
            self.ga_current = self.args.ga_begin
            self.args.batch_size = self.args.batch_size_per_gpu * self.world_size * self.ga_current

            if self.args.init_from is not None:
                self.raw_unico.load_state_dict(torch.load(self.args.init_from, map_location=f"cuda:{self.local_rank}"))
                clean_print(f"Pretrained model {self.args.init_from} loaded.\nFine-tuning from scratch with grad_accum_step={self.ga_current}", self.local_rank, '[Trainer]')
            else:
                clean_print(f"Snapshot not found. Training model from scratch with grad_accum_step={self.ga_current}", self.local_rank, '[Trainer]')

    def _save_snapshot(self):
        snapshot = Snapshot(
            model_state=self.raw_unico.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            total_steps=self.args.train_iters,
            trained_time=self.trained_time,
            latest_train_batch=self.batch_now,
            best_val_loss=self.best_val_loss,
            best_return=self.best_return,
            wandb_id=self.wandb_id
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.snapshot_path)

    def _save_checkpoint(self, ave_return=-1e5, val_loss=1e5):
        ckpt_dir = None
        macro_batch_now = self.batch_now // self.args.batch_num
        if self.args.save_strategy == 'interval':
            torch.save(
                self.raw_unico.state_dict(),
                f'{self.args.out_dir}/sft/interval/{self.seed}/{round(ave_return,3)}_seed{self.seed}_mb{macro_batch_now}.pt'
            )
            ckpt_dir = os.path.join(f'{self.args.out_dir}/sft/interval/{self.seed}', '*.pt')
        elif self.args.save_strategy == 'best':
            if not self.args.is_obs_pretrain and ave_return > self.best_return:
                self.best_return = ave_return
                torch.save(
                    self.raw_unico.state_dict(),
                    f'{self.args.out_dir}/sft/best/{round(self.best_return,3)}_seed{self.seed}_mb{macro_batch_now}.pt'
                )
            if self.args.is_obs_pretrain and val_loss < self.best_val_loss: 
                self.best_val_loss = val_loss         
                torch.save(
                    self.raw_unico.state_dict(),
                    f'{self.args.out_dir}/sft/best/{round(self.best_val_loss,3)}_seed{self.seed}_mb{macro_batch_now}.pt'
                )
            ckpt_dir = os.path.join(f'{self.args.out_dir}/sft/best', '*.pt')
        else:
            raise NotImplementedError(f'Unknown save strategy: {self.args.save_strategy}')

        # Only keep the best n ckpt
        if ckpt_dir:
            ckpt_files = glob.glob(ckpt_dir)
            if len(ckpt_files) > self.args.save_ckpt_num:
                ckpt_files = sorted(ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]))
                for ckpt in ckpt_files[:-self.args.save_ckpt_num]:
                    os.remove(ckpt)

    def _eval_policy(self, sample_action=False):
        desc_sample = 'sample' if sample_action else 'greedy'
        desc = f'{desc_sample}'
        episode_return = {k: {'AM':0, 'DB1':0} for k in self.args.eval_dataset_names}
        episode_obj = {k: 0 for k in self.args.eval_dataset_names}
        episode_time = {k: [] for k in self.args.eval_dataset_names}

        for env_name, dataset_name, env in zip(self.args.eval_env_names, self.args.eval_dataset_names, self.envs):
            if self.args.use_ddp_env:                
                ave_return, ave_obj, obj_std, ave_time_used = evalute_batch_episode(
                    args=self.args, 
                    model=self.unico, 
                    env=env,
                    problemloader=self.problemloader_dict[env_name],
                    sample_action=sample_action,
                    desc=f'[GPU{self.global_rank}]: Evaluating on {env_name} ({desc})',
                    device=self.local_rank        
                )
            else:
                if not env_name.startswith('Env_OBP3D'):
                    problemloader = self.problemloader_dict[env_name]
                    problemloader.reset()
                iters = self.args.problem_batch_num * self.args.problem_batch_size_per_gpu
                epi_return, epi_obj, epi_time = {'AM':[], 'DB1':[]}, [], []
                with tqdm(total=iters, desc=f'[GPU{self.global_rank}]: Evaluating on {env_name} ({desc})', position=self.local_rank) as pbar:
                    for _ in range(iters):
                        if not env_name.startswith('Env_OBP3D'):
                            problem_info, problem_obj = problemloader.get_problem(1)
                            problem_info = (None if problem_info[0] is None else problem_info[0][0], problem_info[1][0], problem_info[2][0])
                            problem_obj = (problem_obj[0].item(), problem_obj[1].item())
                        else:
                            problem_info = problem_obj = None

                        # eval tasks
                        ep_ret, ep_obj, ep_time = evalute_one_episode(
                            args=self.args, 
                            model=self.unico, 
                            env=env, 
                            sample_action=sample_action,
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
                            'std': f'{obj_std:.4f}',
                            'time': f'{ave_time_used:.2f}',
                        }
                        pbar.set_postfix(info)
                        pbar.update()
                
            episode_return[dataset_name] = ave_return
            episode_obj[dataset_name] = ave_obj
            episode_time[dataset_name] = ave_time_used

        return episode_return, episode_obj, episode_time

    def _run_batch(self, rl_task_input, batch_dataset_name, is_train=False) -> float:
        with self.ctx:
            input_ids = rl_task_input.tensor_seq
            model_inputs = {
                "input_ids": input_ids.contiguous(),
                "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
                "use_cache": False, 
                "past_key_values": None,
            }
            _, loss, loss_datasets, _ = self.unico(rl_task_input, True, batch_dataset_name, model_inputs)
        
        new_lr = new_wd = grad_norm = None
        if is_train:
            loss_ave = loss / self.ga_current  
            if self.grad_batch_cnt % self.ga_current != 0:
                # Only sync gradients at the last micro step.
                with self.unico.no_sync():
                    self.scaler.scale(loss_ave).backward()
            else:
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss_ave).backward()

                # clip the gradient
                if self.args.clip_grad != 0:
                    # When using AMP, the gradients are stored in scaled form, We need to unscale the gradients to perform the correct clipping operation.
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.unico.parameters(), self.args.clip_grad)
                
                # When using AMP, scaler can adjust gradients and skip update step if necessary to avoid numerical instability.
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # flush the gradients as soon as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)
                
                # update lr, grad_accum_step and weight_decay
                new_lr, new_ga_step, new_wd = self.scheduler.step(self.ga_current)
                self.ga_current = new_ga_step
                self.args.batch_size = self.args.batch_size_per_gpu * self.world_size * self.ga_current
                self.grad_batch_cnt = 0

            self.grad_batch_cnt += 1

        return loss.item(), loss_datasets, grad_norm, new_lr, new_wd

    def _run_macro_batch(self, is_train=False):
        macro_batch_now = self.batch_now // self.args.batch_num
        if is_train:
            total = self.args.batch_num
            desc = f'[GPU0-{self.world_size-1}]: Trianing batch {macro_batch_now}({self.batch_now}-{self.batch_now + self.args.batch_num})'
            dataloader = self.dataloader_train
            dataloader.sampler.set_batch(self.batch_now, macro_batch_now, is_train=True)
        else:
            total = self.args.eval_batch_num
            desc = f'[GPU0-{self.world_size-1}]: Calculating val_loss'
            dataloader = self.dataloader_val
            dataloader.sampler.set_batch(self.batch_now, macro_batch_now, is_train=False)     

        mb_losses = []
        dataset_mb_losses = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
        dataset_visited_info = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}

        #self.optimizer.zero_grad()
        new_lr = new_wd = grad_norm = 0
        with tqdm(total=total, desc=desc, position=self.local_rank, disable=self.local_rank!=0) as pbar:
            for batch in dataloader:
                # 加载 batch data
                rl_task_input, batch_data_info, _ = batch
                if self.args.auto_batch_len and rl_task_input.seq_len.max() < self.args.n_position:
                    rl_task_input.apply(lambda x: x[:, :rl_task_input.seq_len.max()] if isinstance(x, torch.Tensor) and x.dim() == 2 else x)
                    assert (rl_task_input.tensor_seq[:,-1]==self.args.special_tokens['<|>']).sum() >= 1

                # 记录数据集访问信息
                batch_dataset_name = []
                for dataset_name, dataset_idx in batch_data_info:
                    dataset_visited_info[dataset_name].append(dataset_idx)
                    batch_dataset_name.append(dataset_name)

                # run batch
                rl_task_input.to(device=self.local_rank)
                loss, loss_datasets, norm, lr, wd = self._run_batch(rl_task_input, batch_dataset_name, is_train)
                new_lr = lr if lr is not None else new_lr
                new_wd = wd if wd is not None else new_wd
                grad_norm = norm.item() if norm is not None else grad_norm

                # 记录总损失和各数据集上损失信息
                mb_losses.append(loss)
                for dataset_name, dataset_loss in loss_datasets.items():
                    if dataset_loss != 0:
                        dataset_mb_losses[dataset_name].append(dataset_loss.item())

                # 更新进度条
                batch_token = self.args.batch_size * self.args.n_position if is_train else self.args.eval_batch_size * self.args.n_position
                pbar.set_postfix({
                    'grad_norm': f'{grad_norm:.4f}',
                    'batch_token': f'{batch_token/1e6:.4f}',
                    'loss':f'{loss:.2f}', 
                    'ave loss (latest 20)': f'{np.array(mb_losses[-20:]).mean():.2f}',
                })
                pbar.update()
                
                # 归集部分 batch 运行信息上传到 wandb
                batch_info_tensor = torch.tensor([new_lr, grad_norm]).to(self.local_rank)
                batch_info_gather_list = [torch.zeros_like(batch_info_tensor) for _ in range(self.world_size)]
                dist.gather(batch_info_tensor, batch_info_gather_list if self.local_rank == 0 else None, dst=0)

                if self.local_rank == 0:
                    log_value_tensor = torch.mean(torch.stack(batch_info_gather_list), axis=0, dtype=torch.float32)
                    batch_log_dict = {
                        'info/grad_norm': log_value_tensor[1].item(),
                        'info/batch_token': batch_token/1e6
                    }
                    wandb.log(batch_log_dict)

        mb_loss = np.array(mb_losses).mean()
        dataset_mb_losses = {name: 0 if len(losses) == 0 else np.array(losses).mean() for name, losses in dataset_mb_losses.items()}
        return mb_loss, dataset_mb_losses, new_lr, dataset_visited_info

    def train(self):
        early_stop_signal = torch.tensor([0,]).to(self.local_rank)
        if self.local_rank == 0:
            data_visited_cnt = {dataset.dataset_name: np.zeros(len(dataset), dtype=np.int8) for dataset in self.dataset_train.datasets}   
        
        # start training
        for batch in range(self.batch_start, self.args.train_iters + 1, self.args.batch_num):   
            self.batch_now = batch
            eval_ave_return = 0
            wandb_log_dict = {}

            # Save snapshot before the macro batch training process, so that there is no overlap when recover from the snapshot
            #if self.args.save_snapshot and batch % self.args.save_interval == 0 and batch != 0 and self.local_rank == 0:
            if self.args.save_snapshot and batch % self.args.save_interval == 0 and self.local_rank == 0:
                self._save_snapshot()
            
            # Calculate validation losses at specified batch intervals
            if (batch % self.args.eval_interval == 0 or batch == self.args.train_iters) and (batch != 0 or not self.args.skip_first_eval):
                self.unico.eval()
                with torch.no_grad():
                    val_loss, val_loss_dataset, _, _ = self._run_macro_batch(is_train=False)
                    wandb_log_dict.update({"losses/val_loss": val_loss})
                    wandb_log_dict.update({f'eval_{dataset_name[:-3]}/val_loss': loss for dataset_name, loss in val_loss_dataset.items()})

            # Evaluate policy performance at specified batch intervals
            if (batch % self.args.eval_policy_interval == 0 or batch == self.args.train_iters) and (batch != 0 or not self.args.skip_first_eval):
                self.unico.eval()
                with torch.inference_mode():  
                    return_list = []
                    for setting, eval_func in self.eval_setting.items():
                        epi_return, epi_obj, epi_time = eval_func()
                        for dataset_name in self.args.eval_dataset_names:
                            wandb_log_dict.update({f'eval_{dataset_name[:-3]}/return_AM({setting})': np.mean(epi_return[dataset_name]['AM'])})
                            wandb_log_dict.update({f'eval_{dataset_name[:-3]}/return_DB1({setting})': np.mean(epi_return[dataset_name]['DB1'])})
                            wandb_log_dict.update({f'eval_{dataset_name[:-3]}/obj({setting})': np.mean(epi_obj[dataset_name])})
                            wandb_log_dict.update({f'eval_{dataset_name[:-3]}/time({setting})': np.mean(epi_time[dataset_name])})
                            
                        return_list.extend([v['AM'] for v in epi_return.values()])  # 用 AM return 作为 ckpt 质量指标
                    eval_ave_return = np.mean(return_list)
                    
            if batch == self.args.train_iters:
                break

            # one training macro batch
            self.unico.train()
            start_time = time.time()
            train_loss, train_loss_dataset, new_lr, dataset_visited_info = self._run_macro_batch(is_train=True)
            self.trained_time += time.time() - start_time
            
            wandb_log_dict.update(
                {f'eval_{dataset_name[:-3]}/train_loss': loss
                for dataset_name, loss in train_loss_dataset.items()}
            )
            wandb_log_dict.update({
                "info/batch": batch,
                "info/m_batch": int(batch/self.args.batch_num),
                'info/lr': new_lr,
                "losses/train_loss": train_loss, 
                'info/trained_time': self.trained_time,
            })
            
            # gather info from all GPU
            log_value_tensor = torch.tensor([wandb_log_dict[k] for k in sorted(wandb_log_dict)] + [eval_ave_return,]).to(self.local_rank)
            log_gather_list = [torch.zeros_like(log_value_tensor) for _ in range(self.world_size)]
            dist.gather(log_value_tensor, log_gather_list if self.local_rank == 0 else None, dst=0)

            # gather data info from all GPU
            dataset_info = [len(dataset_visited_info[dataset_name]) for dataset_name in self.args.eval_dataset_names]
            for dataset_name in self.args.eval_dataset_names:
                dataset_info += dataset_visited_info[dataset_name]

            dataset_info_tensor = torch.tensor(dataset_info).to(self.local_rank)
            dataset_info_gather_list = [torch.zeros_like(dataset_info_tensor) for _ in range(self.world_size)]
            dist.gather(dataset_info_tensor, dataset_info_gather_list if self.local_rank == 0 else None, dst = 0)

            if self.local_rank == 0:
                data_info_gather = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}   
                for info in dataset_info_gather_list:
                    info_len, info = info[:len(self.dataset_train.datasets)], info[len(self.dataset_train.datasets):]
                    for l, dataset_name in zip(info_len, self.args.eval_dataset_names):
                        data_info_gather[dataset_name].append(info[:l])
                        info = info[l:]
                
                macro_batch_size = 0
                for dataset_name in self.args.eval_dataset_names:
                    dataset_idxs = torch.cat(data_info_gather[dataset_name])        # 合并各个卡在当前 macro batch 中访问的数据索引
                    macro_batch_size += len(dataset_idxs)
                    data_visited_cnt[dataset_name][dataset_idxs.cpu()] += 1
                assert macro_batch_size >= self.args.batch_size_per_gpu * self.world_size * self.args.batch_num

            # only do file saving and logging at rank0
            if self.local_rank == 0:
                # get log info 
                log_value_tensor = torch.mean(torch.stack(log_gather_list), axis=0, dtype=torch.float32)
                for i, key in enumerate(sorted(wandb_log_dict)):
                    wandb_log_dict[key] = log_value_tensor[i].item()

                # save ckpt for policy learning
                if self.args.save_ckpt and batch != 0:
                    if self.args.is_obs_pretrain:
                        self._save_checkpoint(val_loss=wandb_log_dict['losses/val_loss'])
                    elif batch % self.args.save_interval == 0:
                        eval_ave_return = log_value_tensor[-1].item()
                        self._save_checkpoint(ave_return=eval_ave_return)

                # early stopping by val_loss
                if 'losses/val_loss' in wandb_log_dict:
                    self.early_stopping(wandb_log_dict['losses/val_loss'])
                
                # log info to wandb
                for dataset_name in self.args.eval_dataset_names:
                    dataset_visited_cnt = data_visited_cnt[dataset_name]
                    visited_part = dataset_visited_cnt[dataset_visited_cnt!=0]

                    log_name = dataset_name[:dataset_name.find('_')]
                    wandb_log_dict.update({f'info/{log_name}_ratio': len(visited_part)/len(dataset_visited_cnt)})   # 训练数据集被访问比例
                    wandb.run.summary[f"info/{log_name}_visited"] = len(visited_part)                               # 训练数据集中被访问子集的尺寸        
                    wandb.run.summary[f"info/{log_name}_num"] = np.sum(visited_part)                                # 训练数据集访问计数    
                    wandb.run.summary[f"info/{log_name}_times"] = np.mean(visited_part)                             # 训练数据集中被访问子集的访问次数
                wandb.log(wandb_log_dict)

                # early stopping
                if self.args.use_early_stopping and self.early_stopping.early_stop:
                    early_stop_signal = torch.tensor([1,]).to(self.local_rank)
            
            # early stopping exit point
            dist.broadcast(tensor=early_stop_signal, src=0)
            if early_stop_signal:
                print(f'[GPU{self.local_rank}] Early Stop')
                break