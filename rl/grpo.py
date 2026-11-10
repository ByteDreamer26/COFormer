import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import wandb
import torch
import json
import shutil
import setproctitle
import torch.distributed as dist
from functools import partial
from environment.wrapper import LMPromptEnv
from environment.DDP_wrapper import DDP_LMPromptEnv
from environment.used import get_data_info_dict
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_model import load_model
from rl.trainer_rl import RLTrainer
from rl.config import parse_args
from data.make_data import * 
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import create_folder_if_not_exist, create_folder_overwrite_if_exist
setproctitle.setproctitle("UniCO@RL")

# CUDA_VISIBLE_DEVICES=3,5,6,7 torchrun --standalone --nproc_per_node=gpu --master_port=21666 ./evaluate_test/DDP_evaluate2.py 
def ddp_setup():
    num_cores = os.cpu_count()
    num_threads = max(1, min(4, num_cores // 4))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MASTER_ADDR"] = "localhost"         # localhost for single node
    os.environ["MASTER_PORT"] = "22367"             # Any free port
    init_process_group(backend="nccl")              # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def get_args_ready(WORLD_SIZE, RANK):
    rl_args = parse_args()
    rl_args.seed = 42
    rl_args.num_workers = 0

    # IO setting
    rl_args.data_base_path = base_path
    rl_args.sft_snapshot_path = None
    rl_args.sft_ckpt_path = 'Cmp-6All50(embv2-6x20w)_500_768_8_10/sft/best/XXX_seed42_mb60.pt'
    #rl_args.eval_envs = ['Env_BP_V1', 'Env_TSP_V3', 'Env_PCTSP_V3', 'Env_OP_V4', 'Env_CVRP_V3']
    rl_args.eval_envs = ['Env_OP_V4', ]
    rl_args.wandb_project = 'UniCO_RL2'

    # data para
    common_data_num = 200000
    rl_args.data_num_spctsp = common_data_num
    rl_args.data_num_atsp = common_data_num
    rl_args.data_num_bp = common_data_num
    rl_args.data_num_tsp = common_data_num
    rl_args.data_num_op = common_data_num
    rl_args.data_num_cvrp = common_data_num
    rl_args.data_num_pctsp = common_data_num
    rl_args.data_num_obp3d = common_data_num
    
    # optimizer setting
    rl_args.lr_begin = 5.0e-5                                      
    rl_args.lr_max = 5.0e-5                 # with baby networks can afford to go a bit higher
    rl_args.lr_decay_factor = 10.0          # min learning rate equals to (learning_rate / 10) usually
    rl_args.lr_warmup_ratio = 0.05
    rl_args.lr_decay_ratio = 0.95
    rl_args.lr_decay_style = "cosine"
    rl_args.wd_begin = 0.0                  # with baby networks can afford to go a bit higher (1e-4 ~ 1e-2)
    rl_args.wd_end = rl_args.wd_begin       # For most of situation, keep the weight decay coefficient 'constant' is suitable
    rl_args.wd_decr_style = "constant"            
    rl_args.adam_beta2 = 0.95               # make a bit bigger because number of tokens per iter is small

    rl_args.lr_begin_critic = 5.0e-5                                       
    rl_args.lr_max_critic = 2.0e-4          # with baby networks can afford to go a bit higher
    rl_args.lr_decay_factor_critic = 2.0    # min learning rate equals to (learning_rate / 10) usually
    rl_args.lr_warmup_ratio_critic = 0.05
    rl_args.lr_decay_ratio_critic = 0.95
    rl_args.lr_decay_style_critic = "cosine"

    # 强化学习问题
    rl_args.step_num = 400                  # 每个 step 进行一次 rollout 和 update_num 次策略更新
    rl_args.update_num = 1                  # 每个 step 进行策略更新的次数
    rl_args.clip_adv = (0.2, 0.3)           # clip advantage (Clip-Higher, ref DAPO)
    rl_args.beta = 0.00                     # KL coefficient
    rl_args.gamma_gae = 1.00                # GAE reward discount factor
    rl_args.lambda_gae = 0.95               # GAE lambda
    rl_args.entropy_coef = 0.00             # entropy coefficient
    rl_args.temprature = 1.0                # rollout temprature
    rl_args.q_group_size = 16               # rollout times
    rl_args.q_batch_size_per_gpu = 100      # question batch size per gpu
    rl_args.q_ga_num = 6                    # batch_grad_accum
    rl_args.q_batch_size = WORLD_SIZE * rl_args.q_batch_size_per_gpu * rl_args.q_ga_num
    rl_args.q_total_problem_num = rl_args.step_num * rl_args.q_batch_size
    rl_args.ttest_alpha = 0.05              # ttest alpha, only ckpt which pass the ttest will be saved and use as greedy baseline (for RB)
    rl_args.loss_type = 'ppo'         # [ppo, reinforce]
    rl_args.rl_return_type = 'obj'          # [am, obj]
    rl_args.train_dataset_type = 'random'   # [sft, random]
    rl_args.baseline_type = 'RB'           # [GM:GroupMean, RC:RolloutCurrent, RB:RolloutBest, GAE:Generalized Advantage Estimation]
    rl_args.critic_feature = 'last'         # [mean, last]
    if rl_args.train_dataset_type == 'random':
        rl_args.rl_return_type = 'obj'
    
    # 评估训练损失、验证损失
    rl_args.eval_interval = 1
    rl_args.eval_batch_size_per_gpu = 1000  
    rl_args.eval_batch_size = rl_args.eval_batch_size_per_gpu * WORLD_SIZE
    rl_args.eval_batch_num = 2
    
    # 评估问题
    rl_args.eval_policy_interval = 2
    rl_args.problem_batch_size_per_gpu = 500
    rl_args.problem_batch_num = 4
    rl_args.total_problem_num = WORLD_SIZE * rl_args.problem_batch_size_per_gpu * rl_args.problem_batch_num
    rl_args.resample_times = 16
    #assert rl_args.problem_batch_size_per_gpu % 24 == 0

    # 控制参数
    rl_args.n_position = None           # 模型输入长度，None 表示和预训练时相同
    rl_args.use_kvcache = True
    rl_args.use_ddp_env = True
    rl_args.use_amp = False
    rl_args.add_best_to_group = False
    rl_args.save_snapshot = True
    rl_args.save_ckpt = True
    rl_args.save_ckpt_num = 3           # the number of ckpt to save, 0 for saving all ckpt
    rl_args.save_strategy = 'best'      # 'best' or 'interval'
    rl_args.skip_first_eval = False
    rl_args.wandb = True
    rl_args.check_same_cop_improve = False
    rl_args.check_sft_loss = True
    rl_args.use_dynamic_temprature = False
    rl_args.ttest_for_best_model = True
    rl_args.exp_name = 'Improve_OP'
    
    # FFSP需要更长的序列长度，否则kvcahce加速效果有限
    if rl_args.use_kvcache:
        for env_name in rl_args.eval_envs:
            if env_name.startswith('Env_FFSP'):
                rl_args.n_position = 4000
                break
    
    # get ready for wandb logging
    if RANK == 0:
        create_folder_if_not_exist(f'{base_path}/Wandb')
    if not rl_args.wandb:       
        os.environ['WANDB_MODE'] = 'offline'

    return rl_args

def get_train_components(rl_args, WORLD_SIZE, RANK, check_tokenized:bool=True):
    # load model and ref model
    exp_name = rl_args.sft_ckpt_path[:rl_args.sft_ckpt_path.find('/')] if rl_args.sft_ckpt_path is not None else \
                rl_args.sft_snapshot_path[:rl_args.sft_snapshot_path.find('/')]
    config_path = f'{base_path}/out/{exp_name}/sft/config.json'
    ckpt_path = None if rl_args.sft_ckpt_path is None else f'{base_path}/out/{rl_args.sft_ckpt_path}'
    snapshot_path = None if rl_args.sft_snapshot_path is None else f'{base_path}/out/{rl_args.sft_snapshot_path}'
    args, unico = load_model(config_path, ckpt_path, snapshot_path, rl_args.n_position)
    
    # load ref model
    if rl_args.beta > 0:
        _, unico_ref = load_model(config_path, ckpt_path, snapshot_path, rl_args.n_position)
        unico_ref.config.eval_dataset_names = [name[4:] for name in rl_args.eval_envs]  
    else:
        unico_ref = None

    # load best model
    if rl_args.baseline_type == 'RB':
        _, unico_best = load_model(config_path, ckpt_path, snapshot_path, rl_args.n_position)
    else:
        unico_best = None

    # load critic model
    if rl_args.baseline_type == 'GAE':
        _, unico_critic = load_model(config_path, ckpt_path, snapshot_path, rl_args.n_position, is_critic=True)
        unico_critic.config.critic_feature = rl_args.critic_feature
        unico_critic.config.gamma_gae = rl_args.gamma_gae
        unico_critic.config.lambda_gae = rl_args.lambda_gae
        unico_critic.config.eval_dataset_names = [name[4:] for name in rl_args.eval_envs]
    else:
        unico_critic = None

    # if start from a RL checkpoint, set the seed according to step to avioid data overlap
    if rl_args.sft_ckpt_path is not None and '/rl/' in rl_args.sft_ckpt_path:
        start_step = int(rl_args.sft_ckpt_path[rl_args.sft_ckpt_path.rfind('step')+4:-3])
        rl_args.seed = args.seeds[0] + start_step

    # update args for evaluation
    args.world_size = WORLD_SIZE
    args.adj_range = getattr(args, 'adj_range', False)
    args.local_pos_ver = getattr(args, 'local_pos_ver', 'v1')
    args.worker_num = rl_args.worker_num
    args.use_kvcache = rl_args.use_kvcache
    args.use_ddp_env = rl_args.use_ddp_env
    args.data_base_path = rl_args.data_base_path

    args.data_num_atsp = rl_args.data_num_atsp
    args.data_num_spctsp = rl_args.data_num_spctsp
    args.data_num_tsp = rl_args.data_num_tsp
    args.data_num_bp = rl_args.data_num_bp
    args.data_num_op = rl_args.data_num_op
    args.data_num_pctsp = rl_args.data_num_pctsp
    args.data_num_cvrp = rl_args.data_num_cvrp
    args.data_num_obp3d = rl_args.data_num_obp3d

    args.problem_batch_size_per_gpu = rl_args.problem_batch_size_per_gpu
    args.problem_batch_num = rl_args.problem_batch_num
    args.total_problem_num = rl_args.total_problem_num
    args.q_total_problem_num = rl_args.q_total_problem_num

    args.eval_batch_size_per_gpu = rl_args.eval_batch_size_per_gpu
    args.eval_batch_size = rl_args.eval_batch_size
    args.eval_batch_num = rl_args.eval_batch_num
    args.eval_env_names = rl_args.eval_envs
    args.eval_dataset_names = [name[4:] for name in args.eval_env_names]  

    args.train_iters = rl_args.step_num
    args.lr_begin = rl_args.lr_begin
    args.lr_max = rl_args.lr_max
    args.lr_decay_factor = rl_args.lr_decay_factor
    args.lr_warmup_ratio = rl_args.lr_warmup_ratio
    args.lr_decay_ratio = rl_args.lr_decay_ratio
    args.lr_decay_style = rl_args.lr_decay_style
    args.wd_begin = rl_args.wd_begin
    args.wd_end = rl_args.wd_end
    args.wd_decr_style = rl_args.wd_decr_style
    args.adam_beta2 = rl_args.adam_beta2

    args.lr_begin_critic = rl_args.lr_begin_critic
    args.lr_max_critic = rl_args.lr_max_critic
    args.lr_decay_factor_critic = rl_args.lr_decay_factor_critic
    args.lr_warmup_ratio_critic = rl_args.lr_warmup_ratio_critic
    args.lr_decay_ratio_critic = rl_args.lr_decay_ratio_critic
    args.lr_decay_style_critic = rl_args.lr_decay_style_critic
    
    # load prompt dataset and eval problem
    eval_basic_env_builders, eval_ddp_env_builders = [], []
    train_basic_env_builders, train_ddp_env_builders = [], []
    datasets_sft, datasets_prompt, datasets_prompt_ddp = [], [], []
    train_problems, eval_problems = {}, {}
    data_info_map = get_data_info_dict(args)
    for env_name in args.eval_env_names:
        info = data_info_map[env_name]

        # get problem for evaluation
        eval_problems[env_name] = get_data(
            args, env_name, 'problem',
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
        ) if env_name != 'Env_OBP3D_V1' else None

        # get train problem for RL
        if rl_args.train_dataset_type == 'sft':
            train_problems[env_name] = get_data(
                args, env_name, 'train_problem',
                info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'], is_rl=True
            ) if env_name != 'Env_OBP3D_V1' else None

            if rl_args.debug:
                eval_problems[env_name] = train_problems[env_name]

        # get train dataset
        if rl_args.check_sft_loss:
            datasets_sft += get_data(
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
        eval_problem_batch_size_per_gpu = int(args.problem_batch_size_per_gpu/24) if env_name.startswith('Env_FFSP') else args.problem_batch_size_per_gpu
        eval_ddp_env_builders.append(None if info['ddp_env'] is None else partial(info['ddp_env'], scale=info['problem_scale'], batch_size=eval_problem_batch_size_per_gpu, adj_range=args.adj_range))
        eval_basic_env_builders.append(None if info['baisc_env'] is None else partial(info['baisc_env'], scale=info['problem_scale'], adj_range=args.adj_range))

        train_problem_batch_size_per_gpu = int(rl_args.q_batch_size_per_gpu/24) if env_name.startswith('Env_FFSP') else rl_args.q_batch_size_per_gpu
        train_ddp_env_builders.append(None if info['ddp_env'] is None else partial(info['ddp_env'], scale=info['problem_scale'], batch_size=train_problem_batch_size_per_gpu, adj_range=args.adj_range))
        train_basic_env_builders.append(None if info['baisc_env'] is None else partial(info['baisc_env'], scale=info['problem_scale'], adj_range=args.adj_range))

    # make sure all data are tokenized properly 
    if check_tokenized:
        if RANK == 0:
            for dataset in datasets_prompt:
                for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Checking tokenize format of {dataset.dataset_name}_prompt'):
                    dataset.check_token_list_format(dataset.get(i, with_raw_obs=False))
        dist.barrier()

    # build envs for evaluation
    eval_prompt_strat = rl_args.prompt_strategy.split(";")[-1]     # moving_prompt
    if rl_args.use_ddp_env:
        eval_envs = [DDP_LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(eval_ddp_env_builders, datasets_prompt_ddp)]
        train_envs = [DDP_LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(train_ddp_env_builders, datasets_prompt_ddp)]
    else:
        eval_envs = [LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(eval_basic_env_builders, datasets_prompt)]
        train_envs = [LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(train_basic_env_builders, datasets_prompt)]

    # set dataset weights
    if rl_args.check_sft_loss:
        if args.dataset_weights_type == 'proportion':
            dataset_weights = [len(dataset) for dataset in datasets_sft]
        elif args.dataset_weights_type == 'equal':
            dataset_weights = [1] * len(datasets_sft)
        else:
            raise NotImplementedError('dataset_weights_type should be one of [proportion, equal]')
    else:
        dataset_weights = None

    return args, unico, unico_ref, unico_best, unico_critic, train_envs, eval_envs, train_problems, eval_problems, datasets_sft, dataset_weights

def save_setting(args, rl_args, RANK):
    if (rl_args.save_ckpt or rl_args.save_snapshot) and RANK == 0:
        # create floder to save ckpts and hyperparas if we need
        create_folder_if_not_exist(f'{args.out_dir}/rl/{rl_args.exp_name}/{args.save_strategy}')
        with open(f'{args.out_dir}/rl/{rl_args.exp_name}/config.json', 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
        
        # save the training script
        script_path = os.path.abspath(__file__)
        shutil.copy2(
            src=script_path,
            dst=f"{args.out_dir}/rl/{rl_args.exp_name}/{script_path[script_path.rfind('/')+1:]}",
        )   

    if rl_args.save_ckpt and rl_args.save_strategy == 'interval' and RANK == 0:
        create_folder_overwrite_if_exist(f'{args.out_dir}/rl/{rl_args.exp_name}/interval/')

if __name__ == "__main__":
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 

    # get hyper paras ready
    rl_args = get_args_ready(WORLD_SIZE, RANK)

    # load training component
    components = get_train_components(rl_args, WORLD_SIZE, RANK, check_tokenized=False)
    args = components[0]

    # save setting
    save_setting(args, rl_args, RANK)

    # This unique id is necessary for log resuming
    wandb_id = wandb.util.generate_id() 

    # build trainer
    trainer = RLTrainer(wandb_id, rl_args, *components)

    # wandb log only on rank0
    if RANK == 0:
        with wandb.init(
            project = rl_args.wandb_project,
            group = f'{rl_args.exp_name}-{args.exp_profile}',
            name = f"seed_{rl_args.seed}",
            id = trainer.wandb_id,
            resume = 'allow',
            dir = f'{base_path}/Wandb',
            config=rl_args
        ):
            raw_model = trainer.unico.module if hasattr(trainer.unico, "module") else trainer.unico
            wandb.watch(raw_model, log='all', log_freq=100)
            trainer.reinforce()
    else:
        trainer.reinforce()

    wandb.finish()
    assert wandb.run is None

    # destroy DDP process group
    destroy_process_group()