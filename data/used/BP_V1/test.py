import os
import sys
import glob
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(base_path)

import glob
import random
from tqdm import tqdm
from utils.utils import create_folder_overwrite_if_exist
from utils.utils_data import load_data_and_check_quantity
from utils.utils_slover import knapsack_dp, calc_bp_total

# 数据量
ENV_NAME = 'BP_V1'
cop_type = 'bp'
scale = 100
num_problem = 0
num_episode = 100

# 获取指定环境的所有数据集   
file_list = glob.glob(os.path.join(f'{base_path}/data/used/{ENV_NAME}/{cop_type}{scale}', '*.pkl'))
file_names = [os.path.basename(file) for file in file_list]
file_names = ['bp20_problem.pkl']

# 加载数据 & 打印样本量
file_data = load_data_and_check_quantity(file_names, ENV_NAME, num_problem, num_episode, check_ave_obj=True)\

for problem, answer in zip(file_data['bp20_problem'].problem_list, file_data['bp20_problem'].answer_list):
    # 计算最优解
    opt_value, opt_answer = knapsack_dp(problem['capacity_left'].item(), problem['item_volumes'], problem['item_values'])
    value = calc_bp_total(problem['item_values'], answer)
    assert value == opt_value, f'计算错误: {value} != {opt_value}'
