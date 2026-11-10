import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import numpy as np
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_data import load_data_and_check_quantity
from utils.utils_slover import calc_pctsp_cost

# 数据量
ENV_NAME = 'SPCTSP_V2'
cop_type = 'spctsp'
scale = 100
data_source = 'am_sampling'
num_problem = 0
num_episode = 0

# 获取指定环境的所有数据集   
file_dir = os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}') if data_source is None else \
            os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}_{data_source}')
file_list = glob.glob(os.path.join(file_dir, '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
#file_names = ['spctsp20_problem.pkl',]

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True, data_source=data_source)
