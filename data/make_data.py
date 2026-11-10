import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import os
from tqdm import tqdm
from typing import List, Union
from environment.used.BaseEnv_COP import DataProblem
from data.dataset import RLFullDataset
from data.adapter import DatasetAdapter
from data.DDP_dataset import DDP_RLFullDataset
from environment.used.Env_bp_v1 import *
from environment.used.Env_tsp_v1 import *
from environment.used.Env_tsp_v2 import *
from environment.used.Env_tsp_v3 import *
from environment.used.Env_pctsp_v3 import *
from environment.used.Env_op_v4 import *
from environment.used.Env_cvrp_v3 import *
from environment.used.Env_spctsp_v3 import *
from utils.utils_data import split_dataproblem, load_data, to_d4rl_format

class DatasetAdapterGeneral(DatasetAdapter):
    def __init__(self, dataset_name, epi_obs, epi_act, epi_prefix=None, disable_visited_obs=False):
        super().__init__(dataset_name, epi_obs, epi_act, epi_prefix, disable_visited_obs)

def adjust_range(episodes:List, scale:int, env_name:str) -> List:
    if env_name == 'Env_TSP_V3':
        TSP_V3.get_range_offset()
        offsets = DDP_TSP_V3.get_range_offset()
    elif env_name == 'Env_PCTSP_V3':
        PCTSP_V3.get_range_offset(scale)
        offsets = DDP_PCTSP_V3.get_range_offset(scale)
    elif env_name == 'Env_SPCTSP_V3':
        SPCTSP_V3.get_range_offset(scale)
        offsets = DDP_SPCTSP_V3.get_range_offset(scale)
    elif env_name == 'Env_OP_V4':
        OP_V4.get_range_offset(scale)
        offsets = DDP_OP_V4.get_range_offset(scale)
    elif env_name == 'Env_CVRP_V3':
        CVRP_V3.get_range_offset()
        offsets = DDP_CVRP_V3.get_range_offset()
    elif env_name == 'Env_BP_V1':
        offsets = {}
    else:
        raise NotImplementedError
    
    if offsets != {}:
        for i in tqdm(range(len(episodes)), desc=f'[{env_name}] Adjusting value range'):
            for obs_item, offset in offsets['observation'].items():
                episodes[i]['observations'][obs_item] += offset
            for prefix_item, offset in offsets['prefix'].items():
                episodes[i]['prefix'][prefix_item] += offset

def get_data_by_num(data:Union[List, DataProblem], data_type:str, eval_iter:int, data_num:int, is_train=True, is_rl=False):
    def _get_data_num():
        world_size = int(os.environ.get("WORLD_SIZE", default='1'))

        # 问题数据，手动调整数据量使其可以被平均分配到多卡
        if data_type == 'problem':
            num = eval_iter                             # 加载未见问题时，直接加载评估所需数量
            num -= (num % world_size)                   # 调整为可以被GPU数量整除
        elif data_type == 'train_problem':
            train_data_num = int(data_num * 0.9)        # 加载训练问题时（debug），先计算训练数据量，再从中加载所需数量 
            num = eval_iter if is_rl else min(eval_iter, train_data_num)
            num -= (num % world_size)                   # 调整为可以被GPU数量整除
        
        # 轨迹数据，之后由 dataloader 调整数据量使其可以被平均分配到多卡
        elif data_type == 'train':
            num = data_num                          # 训练集轨迹数据
        elif data_type == 'prompt':
            num = min(int(data_num * 0.1), 1500)    # 用作提示序列的轨迹数据
        elif data_type == 'test':                   
            num = eval_iter                         # 未见问题对应的轨迹数据
        else:
            raise NotImplementedError

        return num

    # 确定加载数据量
    if data_num == 0:
        assert not data_type.endswith('problem')    # 加载全部轨迹
        data_num = len(data)
    if eval_iter == 0:
        assert data_type.endswith('problem')        # 加载全部测试问题
        eval_iter = len(data.answer_list)
    num = _get_data_num()

    # 返回指定量数据
    if data_type == 'train_problem':
        return split_dataproblem(data, 0, num)
    elif data_type == 'problem':
        if is_train:
            return split_dataproblem(data, 0, num)      # 训练时使用前部分问题作为验证集
        else:
            return split_dataproblem(data, -num, None)  # 评估时使用后部分问题作为测试集，和验证集不重叠
    else:
        assert num <= len(data), f'{data_type} data num {num} > {len(data)}'
        return data[:num]

def get_RLFullDataset(args, data, data_name, env_name, scale=20, get_dataset=True, get_ddp_dataset=False, adapter_builder=DatasetAdapterGeneral):
    if args.adj_range:
        adjust_range(data, scale, env_name)
    data = to_d4rl_format(data, env_name)
    adapter = adapter_builder(
        dataset_name = data_name, 
        epi_obs = data['observations'][0], 
        epi_act = data['actions'][0], 
        epi_prefix = data['prefixes'][0] if data['prefixes'] is not None else None,
        disable_visited_obs=args.disable_visited_obs
    )
    dataset = RLFullDataset(args, data, adapter, data_name, env_name) if get_dataset else None
    ddp_dataset = DDP_RLFullDataset(args, data, adapter, data_name, env_name) if get_ddp_dataset else None
    return [dataset, ], [ddp_dataset, ]   

def get_data(
    args, env_name, data_type, cop_type, use_prefix, scale, data_num, data_source=None,
    get_dataset=True, get_ddp_dataset=False, 
    is_train=True, is_rl=False,
):
    assert data_type in ['train', 'prompt', 'test', 'problem', 'train_problem']
    data_name = env_name[4:]
    scale = '' if scale == -1 else scale
    data_file_name = f'{cop_type}{scale}_{data_type}'
    
    # 加载数据
    data = load_data(data_name, data_file_name, data_base_path=args.data_base_path, data_source=data_source)

    # 对于 Prefix 模型，用未使用的 prompt 数据扩展评估问题 problem 数据数据集
    if use_prefix and data_type == 'problem':
        try:
            prompt_data = load_data(data_name=data_name, data_file_name=f'{cop_type}{scale}_prompt_problem', data_base_path=args.data_base_path)
            data.problem_list.extend(prompt_data.problem_list)
            data.answer_list.extend(prompt_data.answer_list)
            if data.prefix_list is not None:
                data.prefix_list.extend(prompt_data.prefix_list)
        except FileNotFoundError:
            pass

    # 调整加载数据量并返回
    if is_rl:
        total_problem_num = args.q_total_problem_num
    else:
        total_problem_num = int(args.total_problem_num / 24) if env_name.startswith('Env_FFSP') else args.total_problem_num
    data = get_data_by_num(data, data_type, total_problem_num, data_num, is_train, is_rl)
    if data_type.endswith('problem'):
        return data 
    return get_RLFullDataset(args, data, data_name, env_name, scale, get_dataset, get_ddp_dataset)