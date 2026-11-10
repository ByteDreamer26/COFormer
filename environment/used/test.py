import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

from environment.used.Env_cvrp_v1 import CVRP_V1, CVRP_logger_V1
from environment.used.Env_op_v1 import OP_V1, OP_logger_V1
from environment.used.Env_pctsp_v1 import PCTSP_V1, PCTSP_logger_V1
from utils.utils import COP_FAILED_RWD
from gym.utils.env_checker import check_env
import random
import numpy as np
from tqdm import tqdm

node_num = 20
logger = CVRP_logger_V1(env_name='Env_CVRP_V1', dataset_name='CVRP_V1')
env = CVRP_V1(node_num=node_num) 
#logger = OP_logger_V1(env_name='Env_OP_V1', dataset_name='OP_V1')
#env = OP_V1(node_num=node_num) 
#logger = PCTSP_logger_V1(env_name='Env_PCTSP_V1', dataset_name='PCTSP_V1')
#env = PCTSP_V1(node_num=node_num) 
check_env(env)
seed = 42
env.action_space.seed(seed)
env.reset(seed=seed)

return_list = []
epsode_num = 200
epi_cnt = 0

with tqdm(total=epsode_num, desc='collecting data') as pbar:
    epi_cnt = 0
    safe_log = []
    while epi_cnt < epsode_num:
        # 生成随机 episode
        obs, _ = env.reset()
        value_space = env.get_action_value_space(hard_action_constraint=True)[0]
        obss, acts, rewards, value_spaces = [obs, ], [], [], [value_space, ]
        while True:
            action = random.choice(value_space)
            obs, reward, terminated, truncated, _ = env.step(action)

            acts.append(action)
            rewards.append(reward)
            obss.append(obs)
            value_space = env.get_action_value_space(hard_action_constraint=True)[0]
            value_spaces.append(value_space)

            if terminated or truncated:
                obss.pop(-1)
                value_spaces.pop(-1)
                break
        
        # 构造成轨迹形式
        assert terminated and not truncated
        safe_log.append(terminated and not truncated)
            
        obs_dict = {}
        for k in obss[0].keys():
            if isinstance(obss[0][k], int) or isinstance(obss[0][k], float):
                obs_dict[k] = np.array([obs[k] for obs in obss], dtype=np.int32)
            elif obss[0][k].shape[0] == 1:
                obs_dict[k] = np.concatenate([obs[k] for obs in obss])
            else:
                obs_dict[k] = np.vstack([obs[k] for obs in obss])
        episode = {
            'prefix': None,
            'observations': obs_dict,
            'actions': np.array(acts).astype(np.int32),
            'rewards': np.array(rewards).astype(np.float32),
            'act_value_space': value_spaces
        }
        return_list.append(sum(rewards))

        # log episode
        logger.log_episode(
            desc='test',
            is_eval=True,
            episode=episode, 
            epoch_num=0, 
            episode_num=epi_cnt,
            seed=seed
        )

        pbar.set_postfix({
            'ave return': '%.3f' % np.mean(return_list),
            'safe': '%.3f' % np.mean(safe_log)
        })
        pbar.update()
        epi_cnt += 1

# Close the environment rendering if used
env.close()