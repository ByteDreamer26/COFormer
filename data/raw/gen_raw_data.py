import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

import math
import multiprocessing
import concurrent.futures
from functools import partial
from utils.utils_data import load_all_raw_dataset, merge_to_dataset
from utils.utils import create_folder_overwrite_if_exist, create_folder_if_not_exist
from data.raw.bp import check_bp_data, gen_raw_bp
from data.raw.tsp import check_tsp_data, gen_raw_tsp
from data.raw.cvrp import check_cvrp_data, gen_raw_cvrp
from data.raw.op import check_op_data, gen_raw_op
from data.raw.pctsp import check_pctsp_data, gen_raw_pctsp
from data.raw.spctsp import check_spctsp_data, gen_raw_spctsp

FUNC_DICT = {
    'bp': (check_bp_data, gen_raw_bp),
    'tsp': (check_tsp_data, gen_raw_tsp),
    'cvrp': (check_cvrp_data, gen_raw_cvrp),
    'op': (check_op_data, gen_raw_op),
    'pctsp': (check_pctsp_data, gen_raw_pctsp),
    'spctsp': (check_spctsp_data, gen_raw_spctsp),
}

if __name__ == "__main__":
    # 设置参数
    scale = 20       
    cop_type = 'bp'                 # bp, tsp
    data_num_per_process = 100      # 每个进程保存的样本数量
    worker_num = 25                 # 同时执行的生成进程数量
    data_num_dict = {
        'train': 1000,
        'problem': 1000,
        'prompt': 1000,
    }

    #data_base_path = '/data3/xxx/unico'
    data_base_path = base_path
    create_folder_overwrite_if_exist(f'{data_base_path}/_temp') 
    create_folder_if_not_exist(f'{data_base_path}/data/raw/{cop_type}/_temp') 

    # 加载已经存在的数据
    ex_datasets = load_all_raw_dataset(data_base_path, cop_type, scale)
    ex_seeds = [dataset.seed_list for dataset in ex_datasets.values()]
    ex_seeds = [seed for seeds in ex_seeds for seed in seeds]

    # 生成新数据
    check_data_func, gen_raw_data_func = FUNC_DICT[cop_type]
    for data_type in ['problem', 'train', 'prompt']:
        target_data_num = data_num_dict[data_type]
        ex_dataset = ex_datasets[data_type]
        ex_data_num = len(ex_dataset.answer_list)
        data_to_go = target_data_num - ex_data_num
        total_processes_num = math.ceil(data_to_go/data_num_per_process)

        # 打印数据集信息
        if data_to_go > 0:    
            print('-'*100 + f'\nThere are [{ex_data_num}] raw data for [{data_type}] generated, [{data_to_go}] to go\n' + '-'*100)
        else:
            print('-'*100 + f'\nThere are [{ex_data_num}] raw data for [{data_type}] generated, the requirements of [{target_data_num}] have already been met\n' + '-'*100)
            continue

        # 分阶段生成所有数据，每个阶段最多worker_num个进程并行，直到总进程数量达到processes_num
        # 每个阶段最后集合该阶段所有并行进程生成的数据，保存到target_dataset_path
        target_dataset_path = f'{data_base_path}/data/raw/{cop_type}/{cop_type}{scale}/{cop_type}{scale}_{data_type}.pkl'
        target_dataset_backup_path = f'{data_base_path}/data/raw/{cop_type}/{cop_type}{scale}/{cop_type}{scale}_{data_type}_backup.pkl'
        gen_data_num = ex_data_num
        for stage_idx, process_idx in enumerate(range(0, total_processes_num, worker_num)):
            # 本阶段使用的进程数量
            stage_worker_num = min(worker_num, total_processes_num-process_idx)
            stage_data_num = data_num_per_process * stage_worker_num
            
            # 给每个进程分配不同的随机种子
            max_seed = -1 if ex_seeds == [] else max(ex_seeds)
            seeds = list(range(max_seed+1, max_seed+1+stage_worker_num))
            ex_seeds.extend(seeds)
            
            # 创建一个 ProcessPoolExecutor 对象，在循环中提交任务
            print('='*50 + f' STAGE {stage_idx}: seeds = [{max_seed+1}, {max_seed+1+stage_worker_num}) data {gen_data_num} ~ {gen_data_num+stage_data_num}/{target_data_num}' + '='*50)
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            dataset_path_list = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=stage_worker_num) as executor:
                # 提交任务
                futures = {}
                for i, seed in enumerate(seeds):
                    futures[executor.submit(partial(gen_raw_data_func, data_base_path, data_num_per_process, seed, scale, i%worker_num, lock))] = seed

                # 等待所有进程执行完毕
                for future in concurrent.futures.as_completed(futures):
                    try:
                        dataset_path = future.result()
                        dataset_path_list.append(dataset_path)
                    except Exception as e:
                        print(f"{e}")
            del executor

            # 将该阶段生成的所有子数据集合并
            gen_data_num += stage_data_num
            merge_to_dataset(dataset_path_list, target_dataset_path, target_dataset_backup_path)

            # 数据合法性检查
            check_data_func(data_base_path, scale, show_info=False, check_data_num=200)
            os.remove(target_dataset_backup_path)   # 通过检查则删除备份文件

    print('='*55 + f' ALL DONE ' + '='*55)
    check_data_func(data_base_path, scale, show_info=True, check_data_num=200)