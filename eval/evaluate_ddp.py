import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import setproctitle
import torch.distributed as dist
from functools import partial
from environment.wrapper import LMPromptEnv
from environment.DDP_wrapper import DDP_LMPromptEnv
from environment.used import get_data_info_dict
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_model import load_model
from eval.evaluater import Evaluater
from eval.config import parse_args
from data.make_data import * 
from torch.distributed import init_process_group, destroy_process_group 
setproctitle.setproctitle("UniCO@eval")

# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu ./eval/evaluate_ddp.py 
def ddp_setup():
    num_cores = os.cpu_count()
    num_threads = max(1, min(4, num_cores // 4))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MASTER_ADDR"] = "localhost"         # localhost for single node
    os.environ["MASTER_PORT"] = "22367"             # Any free port
    init_process_group(backend="nccl")              # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def get_args_ready(WORLD_SIZE):
    eval_args = parse_args()
    eval_args.seed = 42
    eval_args.num_workers = 0

    eval_args.snapshot_path = None
    eval_args.data_base_path = base_path
    
    eval_args.ckpt_path = 'Cmp-6All100(GATO)_1000_768_8_10/sft/best/xxx_seed42_mb100.pt'
    eval_args.eval_envs = ['Env_TSP_V2', 'Env_BP_V2', 'Env_PCTSP_V1', 'Env_OP_V2', 'Env_CVRP_V1', 'Env_SPCTSP_V1',] 
    eval_args.dataset_weights_type = 'proportion'

    common_data_num = 10000
    eval_args.data_num_spctsp = common_data_num
    eval_args.data_num_atsp = common_data_num
    eval_args.data_num_bp = common_data_num
    eval_args.data_num_tsp = common_data_num
    eval_args.data_num_op = common_data_num
    eval_args.data_num_cvrp = common_data_num
    eval_args.data_num_pctsp = common_data_num
    eval_args.data_num_ffsp = common_data_num
    eval_args.data_num_mis = common_data_num
    eval_args.data_num_obp3d = common_data_num

    common_data_source = None
    eval_args.data_source_spctsp = 'am_sampling'
    eval_args.data_source_op = 'am_sampling'
    eval_args.data_source_cvrp = 'am_sampling'
    eval_args.data_source_pctsp = 'am_sampling'
    eval_args.data_source_atsp = common_data_source
    eval_args.data_source_bp = common_data_source
    eval_args.data_source_tsp = common_data_source
    eval_args.data_source_ffsp = common_data_source
    eval_args.data_source_mis = common_data_source
    eval_args.data_source_obp3d = common_data_source
    
    # 评估训练损失、验证损失、测试损失
    eval_args.n_position = None             # None 表示和训练时相同 
    eval_args.eval_problem_set = 'problem'  # train_problem or problem
    eval_args.batch_size_per_gpu = 40
    eval_args.batch_size = eval_args.batch_size_per_gpu * WORLD_SIZE
    eval_args.batch_num = 5
    eval_args.eval_batch_size_per_gpu = 40  
    eval_args.eval_batch_size = eval_args.eval_batch_size_per_gpu * WORLD_SIZE
    eval_args.eval_batch_num = 20
    eval_args.test_batch_size_per_gpu = 40
    eval_args.test_batch_size = eval_args.test_batch_size_per_gpu * WORLD_SIZE
    eval_args.test_batch_num = 5
    assert eval_args.batch_num != 0 and eval_args.eval_batch_num != 0 and eval_args.test_batch_num != 0
    
    # 评估问题
    eval_args.problem_batch_size_per_gpu = 200
    eval_args.problem_batch_num = 4
    eval_args.total_problem_num = WORLD_SIZE * eval_args.problem_batch_size_per_gpu * eval_args.problem_batch_num
    eval_args.resample_times = 16
    #assert eval_args.problem_batch_size_per_gpu % 24 == 0
        
    # 控制参数
    eval_args.use_kvcache = True
    eval_args.use_ddp_env = True
    eval_args.check_loss = False

    # FFSP需要更长的序列长度，否则kvcahce加速效果有限
    # if eval_args.use_kvcache:
    #     for env_name in eval_args.eval_envs:
    #         if env_name.startswith('Env_FFSP'):
    #             eval_args.n_position = 4000
    #             break

    # eval_args.n_position = 1000

    assert (eval_args.ckpt_path is None) ^ (eval_args.snapshot_path is None) 
    if eval_args.ckpt_path is not None:
        exp_profile = eval_args.ckpt_path[:eval_args.ckpt_path.find('/')]
    else:
        exp_profile = eval_args.snapshot_path[:eval_args.snapshot_path.find('/')]
    eval_args.ckpt_performance_path = f'{base_path}/out/{exp_profile}/sft/performance'
    return eval_args

def get_eval_components(eval_args, WORLD_SIZE, RANK, check_tokenized:bool=True):
    # load model
    exp_name = eval_args.ckpt_path[:eval_args.ckpt_path.find('/')] if eval_args.ckpt_path is not None else \
                eval_args.snapshot_path[:eval_args.snapshot_path.find('/')]
    args, unico = load_model(
        config_path=f'{base_path}/out/{exp_name}/sft/config.json', 
        ckpt_path=None if eval_args.ckpt_path is None else f'{base_path}/out/{eval_args.ckpt_path}',
        snapshot_path=None if eval_args.snapshot_path is None else f'{base_path}/out/{eval_args.snapshot_path}',
        n_position=eval_args.n_position
    )
    args.adj_range = getattr(args, 'adj_range', False)
    args.local_pos_ver = getattr(args, 'local_pos_ver', 'v1')
    args.worker_num = eval_args.worker_num
    args.seed = eval_args.seed
    args.use_kvcache = eval_args.use_kvcache
    args.use_ddp_env = eval_args.use_ddp_env
    args.data_base_path = eval_args.data_base_path

    args.data_num_atsp = eval_args.data_num_atsp
    args.data_num_spctsp = eval_args.data_num_spctsp
    args.data_num_tsp = eval_args.data_num_tsp
    args.data_num_bp = eval_args.data_num_bp
    args.data_num_op = eval_args.data_num_op
    args.data_num_pctsp = eval_args.data_num_pctsp
    args.data_num_cvrp = eval_args.data_num_cvrp
    args.data_num_ffsp = eval_args.data_num_ffsp
    args.data_num_mis = eval_args.data_num_mis
    args.data_num_obp3d = eval_args.data_num_obp3d
    args.dataset_weights_type = eval_args.dataset_weights_type
    
    args.data_source_atsp = eval_args.data_source_atsp
    args.data_source_spctsp = eval_args.data_source_spctsp
    args.data_source_tsp = eval_args.data_source_tsp
    args.data_source_bp = eval_args.data_source_bp
    args.data_source_op = eval_args.data_source_op
    args.data_source_pctsp = eval_args.data_source_pctsp
    args.data_source_cvrp = eval_args.data_source_cvrp
    args.data_source_ffsp = eval_args.data_source_ffsp
    args.data_source_mis = eval_args.data_source_mis
    args.data_source_obp3d = eval_args.data_source_obp3d

    args.problem_batch_size_per_gpu = eval_args.problem_batch_size_per_gpu
    args.problem_batch_num = eval_args.problem_batch_num
    args.total_problem_num = eval_args.total_problem_num
    
    args.eval_env_names = eval_args.eval_envs
    args.eval_dataset_names = [name[4:] for name in args.eval_env_names]    

    # load prompt dataset and eval problem
    basic_env_builders, ddp_env_builders = [], []
    datasets_train, datasets_prompt, datasets_prompt_ddp, datasets_test, envs_problems = [], [], [], [], {}
    data_info_map = get_data_info_dict(args)
    for env_name in args.eval_env_names:
        info = data_info_map[env_name]

        # get problem for evaluation
        envs_problems[env_name] = get_data(
            args, env_name, eval_args.eval_problem_set,
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
        ) if env_name != 'Env_OBP3D_V1' else None

        # get prompt dataset
        dataset, ddp_datset = get_data(
            args, env_name, 'prompt',
            info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
            get_dataset=not args.use_ddp_env, get_ddp_dataset=args.use_ddp_env
        )
        datasets_prompt += dataset
        datasets_prompt_ddp += ddp_datset
        
        # get env builders
        problem_batch_size_per_gpu = int(args.problem_batch_size_per_gpu/24) if env_name.startswith('Env_FFSP') else args.problem_batch_size_per_gpu
        ddp_env_builders.append(
            None if info['ddp_env'] is None else \
            partial(info['ddp_env'], scale=info['problem_scale'], batch_size=problem_batch_size_per_gpu, adj_range=args.adj_range)
        )
        basic_env_builders.append(
            None if info['baisc_env'] is None else \
            partial(info['baisc_env'], scale=info['problem_scale'], adj_range=args.adj_range)
        )

        if eval_args.check_loss:
            datasets_train += get_data(
                args, env_name, 'train',
                info['cop_type'], info['use_prefix'], info['problem_scale'], info['data_num'], info['data_source'],
            )[0]

    # make sure all data are tokenized properly 
    if check_tokenized:
        if RANK == 0:
            for dataset in datasets_train:
                for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Checking tokenize format of {dataset.dataset_name}_train'):
                    dataset.check_token_list_format(dataset.get(i, with_raw_obs=False))
            for dataset in datasets_prompt:
                for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Checking tokenize format of {dataset.dataset_name}_prompt'):
                    dataset.check_token_list_format(dataset.get(i, with_raw_obs=False))
        dist.barrier()

    # build envs for evaluation
    eval_prompt_strat = eval_args.prompt_strategy.split(";")[-1]     # moving_prompt
    if eval_args.use_ddp_env:
        envs = [DDP_LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(ddp_env_builders, datasets_prompt_ddp)]
    else:
        envs = [LMPromptEnv(env_builer(), args, prompt_dataset, eval_prompt_strat) for env_builer, prompt_dataset in zip(basic_env_builders, datasets_prompt)]

    # build dataset weights if we need to check loss
    dataset_weights = None
    if eval_args.check_loss:
        if args.dataset_weights_type == 'proportion':
            weight = {name: len(dataset) for name, dataset in zip(args.eval_env_names, datasets_train)}
        elif args.dataset_weights_type == 'equal':
            weight = {env_name: 1 for env_name in args.eval_env_names}
        else:
            raise NotImplementedError(f'Unsupported dataset_weights_type {args.dataset_weights_type}')
        dataset_weights = list(weight.values())
        
    if RANK == 0:
        create_folder_overwrite_if_exist(eval_args.ckpt_performance_path)
        
    return args, unico, datasets_train, datasets_test, dataset_weights, envs, envs_problems

def main():
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # get hyper paras ready
    eval_args = get_args_ready(WORLD_SIZE)

    # load training component
    args, unico, datasets_train, datasets_test, dataset_weights, envs, envs_problems = get_eval_components(eval_args, WORLD_SIZE, RANK, check_tokenized=False)

    # build evaluater
    evaluater = Evaluater(
        args, eval_args, unico, envs, envs_problems,
        datasets_train, datasets_test, dataset_weights, 
    )

    # evaluate
    evaluater.evaluate()

    # destroy DDP process group
    destroy_process_group()

if __name__ == "__main__":
    main()