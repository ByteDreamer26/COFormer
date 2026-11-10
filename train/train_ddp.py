import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import shutil
import wandb
import json
import torch
import setproctitle
import torch.distributed as dist
from tqdm import tqdm
from functools import partial
from train.trainer import Trainer
from environment.wrapper import LMPromptEnv
from environment.DDP_wrapper import DDP_LMPromptEnv
from train.config import parse_args
from data.make_data import * 
from environment.used import get_data_info_dict
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import create_folder_if_not_exist, create_folder_overwrite_if_exist
setproctitle.setproctitle("UniCO@Single")

# CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=gpu ./train/train_ddp.py 
def ddp_setup():
    num_cores = os.cpu_count()
    num_threads = max(1, min(4, num_cores // 4))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MASTER_ADDR"] = "localhost"         # localhost for single node
    os.environ["MASTER_PORT"] = "21663"             # Any free port
    init_process_group(backend="nccl")              # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def get_args_ready(WORLD_SIZE:int, RANK:int):
    args = parse_args()
    args.world_size = WORLD_SIZE

    # model setting
    args.model = 'llama'
    args.n_embed = 256              # embedding dimension
    args.n_q_head = 4              # attention query head num (for llama GQA)
    args.n_kv_head = 4              # attention key/value head num (for llama GQA)
    args.n_head = args.n_q_head     # attention head num (for TransformerXL)
    args.n_position = 500          # model input sequence length (max context length)
    args.n_layer = 6               # transformer block num
    args.rms_norm_eps = 1e-6
    args.dropout = 0.1                          # for pretraining 0 is good, for finetuning try 0.1+
    if args.model == 'llama':
        assert args.n_q_head % args.n_kv_head == 0

    # optimizer setting
    args.lr_begin = 0                                       
    args.lr_max = 2.5e-4                          # with baby networks can afford to go a bit higher
    args.lr_decay_factor = 10.0                 # min learning rate equals to (learning_rate / 10) usually
    args.lr_warmup_ratio = 0.05
    args.lr_decay_ratio = 0.95
    args.lr_decay_style = "cosine"
    args.wd_begin = 0.1                        # with baby networks can afford to go a bit higher (1e-4 ~ 1e-2)
    args.wd_end = args.wd_begin                 # For most of situation, keep the weight decay coefficient 'constant' is suitable
    args.wd_decr_style = "constant"            
    args.ga_begin = 2                           # batch_grad_accum is used to simulate larger batch sizes              
    args.ga_end = args.ga_begin                 # with baby networks we can simply use 'constant' grad_accum_step, but for large networks sometimes increase to 2x~10x
    args.grad_accum_step_incr_style = "constant"
    args.adam_beta2 = 0.95                      # make a bit bigger because number of tokens per iter is small

    # prefix & prompt paras
    args.prompt_strategy = "stochastic_subseq;moving_prompt"
    args.prompt_prob = 0.25               
    args.prompt_ratio = 0.5                      
    args.prompt_at_final_transition_prob = 0.5
    args.use_prefix = True
    args.use_prompt = False
    assert args.use_prompt ^ args.use_prefix

    # training setting
    args.batch_size_per_gpu = 75               # training batch_size (per GPU)
    args.batch_size = args.batch_size_per_gpu * WORLD_SIZE * args.ga_begin  # equivalent training batch_size
    args.batch_num = 75*args.ga_begin           # batch num of each macro batch
    args.train_iters = 200*args.batch_num       # total batch_num
    args.eval_batch_num = 5
    args.eval_batch_size_per_gpu = 800
    args.eval_batch_size = args.eval_batch_size_per_gpu * WORLD_SIZE        # equivalent eval batch_size
    args.problem_batch_num = 5
    args.problem_batch_size_per_gpu = 200       # eval problem batch_size (per GPU)
    args.total_problem_num = args.problem_batch_size_per_gpu * WORLD_SIZE * args.problem_batch_num
    args.clip_grad = 1.0                        # clip gradients at this value, or disable if == 0.0
    args.num_workers = 0                        # dataloader workers
    args.early_stopping_patience = 6
    args.early_stopping_delta = 0

    # embedding paras
    args.num_continuous_bin = 1800
    args.num_discrete_values = 200
    args.special_tokens = {
        "<|>": args.num_discrete_values + args.num_continuous_bin,
        "<X>": args.num_discrete_values + args.num_continuous_bin + 1
    }
    args.adj_range = False                      # Whether to adjust the original data value range to be symmetric about point O
    args.local_pos_ver = 'v1'
    if args.adj_range:
        args.tokenizer_ver = 'v1'
        args.discretize_mu = 15
        args.discretize_M = 2
    else:
        args.tokenizer_ver = 'v2'
        args.discretize_mu = 15
        args.discretize_M = 4

    # env paras
    common_scale = 20
    args.bp_item_num = common_scale
    args.tsp_city_num = common_scale
    args.atsp_city_num = common_scale
    args.pctsp_node_num = common_scale
    args.spctsp_node_num = common_scale
    args.op_node_num = common_scale
    args.cvrp_node_num = common_scale
    args.ffsp_job_num = common_scale
    args.mis_node_num = common_scale

    common_data_num = 100000
    args.data_num_bp = common_data_num
    args.data_num_atsp = common_data_num
    args.data_num_tsp = common_data_num
    args.data_num_op = common_data_num
    args.data_num_cvrp = common_data_num
    args.data_num_pctsp = common_data_num
    args.data_num_spctsp = common_data_num
    args.data_num_ffsp = common_data_num
    args.data_num_mis = common_data_num
    args.data_num_obp3d = common_data_num

    common_data_source = None
    args.data_source_spctsp = common_data_source
    args.data_source_atsp = common_data_source
    args.data_source_bp = common_data_source
    args.data_source_tsp = common_data_source
    args.data_source_op = common_data_source
    args.data_source_cvrp = common_data_source
    args.data_source_pctsp = common_data_source
    args.data_source_ffsp = common_data_source
    args.data_source_mis = common_data_source
    args.data_source_obp3d = common_data_source

    # ctrl setting, which are usually changed
    args.seeds = [42, ]
    args.disable_visited_obs = True
    args.is_obs_pretrain = False
    args.eval_problem_set = 'problem'               # 'train_problem' or 'problem'
    args.init_from = None                           # None for training from scratch or resuming from latest snapshot within out-dir
    #args.eval_env_names = ['Env_BP_V1', 'Env_TSP_V3', 'Env_PCTSP_V3', 'Env_OP_V4', 'Env_CVRP_V3', 'Env_ATSP_V1', 'Env_SPCTSP_V3']
    args.eval_env_names = ['Env_TSP_V3', ]
    args.dataset_weights_type = 'proportion'        # 'equal' or 'proportion'
    args.use_ddp_env = False
    args.auto_batch_len = True                      # Automatically clip the sample length to the maximum length in the batch
    args.weight_tying = False                        # Tie the word embedding and softmax weights, like in GPT-2
    args.override_opt_param_scheduler = True       # Set 'True' to override all scheduler setting, otherwise the scheduler will be set by checkpoint
    args.skip_first_eval = True                    # skip the first evaluation to at batch 0
    args.wandb = False                              # use wandb to log training info
    args.use_early_stopping = True                 # use the early stopping mechanism to aviod overfitting
    args.save_ckpt = True                           # update ckpt by save_interval and save_strategy
    args.save_ckpt_num = 3                          # the number of ckpt to save, 0 for saving all ckpt
    args.save_snapshot = True                       # save the latest traing snapshot, from which we can resume training 
    args.save_strategy = 'best'                     # 'best' or 'interval'
    args.use_kvcache = True                        # use kv cache to speed up evaluation          
    args.use_amp = False                            # use automatic mixed precision (AMP) to speed up training, which may hurt the performance
    args.compile = False                            # compile the model to speed up training
    args.eval_policy_interval = args.batch_num * 100  # batch interval for policy evaluation
    args.eval_interval = args.batch_num * 1         # batch interval for eval loss calculating, keep frequent because we'll overfit
    args.save_interval = args.batch_num * 1         # batch interval for saving ckpt
    args.compile = args.compile and torch.__version__ >= "2.0"  # only support torch 2.0+
    assert args.train_iters % args.batch_num == 0
    if args.is_obs_pretrain:
        args.eval_policy_interval = args.train_iters

    # IO setting
    args.exp_name = 'Random'
    args.wandb_project = 'Rebuttal'
    args.exp_profile = f'{args.exp_name}_{args.n_position}_{args.n_embed}_{args.n_head}_{args.n_layer}'
    args.exp_profile = f'{args.exp_profile}_compiled' if args.compile else args.exp_profile
    args.exp_profile = f'{args.exp_profile}_ampd' if args.use_amp else args.exp_profile
    args.out_dir = f'{base_path}/out/{args.exp_profile}'
    args.data_base_path = '/data1/autoco/unico'

    # get ready for wandb logging
    if RANK == 0:
        create_folder_if_not_exist(f'{base_path}/Wandb')
    if not args.wandb:       
        os.environ['WANDB_MODE'] = 'offline'

    return args

def get_train_objs(args, RANK, check_tokenized:bool=True):
    basic_env_builders, ddp_env_builders = [], []
    datasets_train, datasets_prompt, datasets_prompt_ddp, envs_problems = [], [], [], {}
    data_info_map = get_data_info_dict(args)
    for env_name in args.eval_env_names:
        info = data_info_map[env_name]
        
        # get problem for evaluation
        envs_problems[env_name] = get_data(
            args, env_name, args.eval_problem_set,
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
        ) if env_name != 'Env_OBP3D_V1' else None

        # get train dataset
        datasets_train += get_data(
            args, env_name, 'train',
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
        )[0]

        # get prompt dataset
        dataset, ddp_datset = get_data(
            args, env_name, 'prompt',
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
            get_dataset=not args.use_ddp_env, get_ddp_dataset=args.use_ddp_env
        )
        datasets_prompt += dataset
        datasets_prompt_ddp += ddp_datset
        
        # get env builders
        problem_batch_size = int(args.problem_batch_size_per_gpu / 24) if env_name.startswith('Env_FFSP') else args.problem_batch_size_per_gpu 
        ddp_env_builders.append(
            None if info['ddp_env'] is None else \
            partial(info['ddp_env'], scale=info['problem_scale'], batch_size=problem_batch_size, adj_range=args.adj_range)
        )
        basic_env_builders.append(
            None if info['baisc_env'] is None else \
            partial(info['baisc_env'], scale=info['problem_scale'], adj_range=args.adj_range)
        )
       
    # build envs for evaluation
    eval_prompt_strat = args.prompt_strategy.split(";")[-1] # moving_prompt
    if args.use_ddp_env:
        envs = [DDP_LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(ddp_env_builders, datasets_prompt_ddp)]
    else:
        envs = [LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(basic_env_builders, datasets_prompt)]

    #dataset_weights = list(weight.values())
    if args.dataset_weights_type == 'proportion':
        dataset_weights = [len(dataset) for dataset in datasets_train]
    elif args.dataset_weights_type == 'equal':
        dataset_weights = [1] * len(datasets_train)
    else:
        raise NotImplementedError('dataset_weights_type should be one of [proportion, equal]')
        
    args.eval_dataset_names = [dataset.dataset_name for dataset in datasets_train]    
    assert [env_name[4:] for env_name in args.eval_env_names] == args.eval_dataset_names

    # make sure all data are tokenized properly (only for use_prompt without mlp_emb)
    if check_tokenized:
        if RANK == 0:
            for dataset in datasets_train:
                for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Checking tokenize format of {dataset.dataset_name}_train'):
                    dataset.check_token_list_format(dataset.get(i, with_raw_obs=False))
            if not args.use_ddp_env:
                for dataset in datasets_prompt:
                    for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Checking tokenize format of {dataset.dataset_name}_prompt'):
                        dataset.check_token_list_format(dataset.get(i, with_raw_obs=False))
        dist.barrier()
 
    return datasets_train, dataset_weights, envs, envs_problems

def save_setting(args):
    if (args.save_ckpt or args.save_snapshot) and RANK == 0:
        # create floder to save ckpts and hyperparas if we need
        create_folder_if_not_exist(f'{args.out_dir}/sft/{args.save_strategy}')
        with open(f'{args.out_dir}/sft/config.json', 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
        
        # save the training script
        script_path = os.path.abspath(__file__)
        shutil.copy2(
            src=script_path,
            dst=f"{args.out_dir}/sft/{script_path[script_path.rfind('/')+1:]}",
        )    

if __name__ == "__main__":
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    # get hyper paras ready
    args = get_args_ready(WORLD_SIZE, RANK)

    # build training objs
    datasets_train, dataset_weights, envs, envs_problems = get_train_objs(args, RANK, check_tokenized=False)

    # save setting
    save_setting(args)

    # train
    for seed in args.seeds:
        args.seed = seed
        if args.save_ckpt and args.save_strategy == 'interval' and RANK == 0:
            create_folder_overwrite_if_exist(f'{args.out_dir}/sft/interval/{seed}')

        # This unique id is necessary for log resuming
        wandb_id = wandb.util.generate_id() 
        
        # build trainer
        trainer = Trainer(args, seed, wandb_id, envs, datasets_train, dataset_weights, envs_problems)

        # wandb log only on rank0
        if RANK == 0:
            with wandb.init(
                project=args.wandb_project,
                group = args.exp_profile,
                name = f"seed_{seed}",
                id = trainer.wandb_id,
                resume = 'allow',
                dir = f'{base_path}/Wandb',
                config=args
            ):
                raw_model = trainer.unico.module if hasattr(trainer.unico, "module") else trainer.unico
                wandb.watch(raw_model, log='all', log_freq=100)
                trainer.train()
        else:
            trainer.train()

        wandb.finish()
        assert wandb.run is None

    # destroy DDP process group
    destroy_process_group()