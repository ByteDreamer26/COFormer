from environment.used.Env_bp_v1 import BP_V1, DDP_BP_V1
from environment.used.Env_bp_v2 import BP_V2, DDP_BP_V2
from environment.used.Env_cvrp_v1 import CVRP_V1, DDP_CVRP_V1
from environment.used.Env_cvrp_v2 import CVRP_V2, DDP_CVRP_V2
from environment.used.Env_cvrp_v3 import CVRP_V3, DDP_CVRP_V3
from environment.used.Env_atsp_v1 import ATSP_V1, DDP_ATSP_V1
from environment.used.Env_atsp_v2 import ATSP_V2, DDP_ATSP_V2
from environment.used.Env_atsp_v3 import ATSP_V3, DDP_ATSP_V3
from environment.used.Env_tsp_v1 import TSP_V1, DDP_TSP_V1
from environment.used.Env_tsp_v2 import TSP_V2, DDP_TSP_V2
from environment.used.Env_tsp_v3 import TSP_V3, DDP_TSP_V3
from environment.used.Env_op_v1 import OP_V1, DDP_OP_V1
from environment.used.Env_op_v2 import OP_V2, DDP_OP_V2
from environment.used.Env_op_v3 import OP_V3, DDP_OP_V3
from environment.used.Env_op_v4 import OP_V4, DDP_OP_V4
from environment.used.Env_pctsp_v1 import PCTSP_V1, DDP_PCTSP_V1
from environment.used.Env_pctsp_v2 import PCTSP_V2, DDP_PCTSP_V2
from environment.used.Env_pctsp_v3 import PCTSP_V3, DDP_PCTSP_V3
from environment.used.Env_ffsp_v1 import DDP_FFSP_V1
from environment.used.Env_ffsp_v2 import DDP_FFSP_V2
from environment.used.Env_ffsp_v3 import DDP_FFSP_V3
from environment.used.Env_mis_v1 import DDP_MIS_V1
from environment.used.Env_mis_v2 import DDP_MIS_V2
from environment.used.Env_spctsp_v1 import SPCTSP_V1, DDP_SPCTSP_V1
from environment.used.Env_spctsp_v2 import SPCTSP_V2, DDP_SPCTSP_V2
from environment.used.Env_spctsp_v3 import SPCTSP_V3, DDP_SPCTSP_V3
from environment.used.Env_obp3d_v1 import OBP3D_V1
from data.make_data import *

def get_data_info_dict(args):
    data_info = {
        # ---------------------------- bp ----------------------------
        'Env_BP_V1': {
            'ddp_env': DDP_BP_V1,
            'baisc_env': BP_V1,
            'cop_type': 'bp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'bp_item_num', 20),
            'data_num': getattr(args, 'data_num_bp', 200),
            'data_source': getattr(args, 'data_source_bp', None),
        },
        'Env_BP_V2': {
            'ddp_env': DDP_BP_V2,
            'baisc_env': BP_V2,
            'cop_type': 'bp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'bp_item_num', 20),
            'data_num': getattr(args, 'data_num_bp', 200),
            'data_source': getattr(args, 'data_source_bp', None),
        },

        # ---------------------------- tsp ----------------------------
        'Env_TSP_V1': {
            'ddp_env': DDP_TSP_V1,
            'baisc_env': TSP_V1,
            'cop_type': 'tsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'tsp_city_num', 20),
            'data_num': getattr(args, 'data_num_tsp', 200),
            'data_source': getattr(args, 'data_source_tsp', None),
        },
        'Env_TSP_V2': {
            'ddp_env': DDP_TSP_V2,
            'baisc_env': TSP_V2,
            'cop_type': 'tsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'tsp_city_num', 20),
            'data_num': getattr(args, 'data_num_tsp', 200),
            'data_source': getattr(args, 'data_source_tsp', None),
        },
        'Env_TSP_V3': {
            'ddp_env': DDP_TSP_V3,
            'baisc_env': TSP_V3,
            'cop_type': 'tsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'tsp_city_num', 20),
            'data_num': getattr(args, 'data_num_tsp', 200),
            'data_source': getattr(args, 'data_source_tsp', None),
        },

        # ---------------------------- pctsp ----------------------------
        'Env_PCTSP_V1': {
            'ddp_env': DDP_PCTSP_V1,
            'baisc_env': PCTSP_V1,
            'cop_type': 'pctsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'pctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_pctsp', 200),
            'data_source': getattr(args, 'data_source_pctsp', None),
        },
        'Env_PCTSP_V2': {
            'ddp_env': DDP_PCTSP_V2,
            'baisc_env': PCTSP_V2,
            'cop_type': 'pctsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'pctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_pctsp', 200),
            'data_source': getattr(args, 'data_source_pctsp', None),
        },
        'Env_PCTSP_V3': {
            'ddp_env': DDP_PCTSP_V3,
            'baisc_env': PCTSP_V3,
            'cop_type': 'pctsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'pctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_pctsp', 200),
            'data_source': getattr(args, 'data_source_pctsp', None),
        },
        
        # ----------------------------- op -----------------------------
        'Env_OP_V1': {
            'ddp_env': DDP_OP_V1,
            'baisc_env': OP_V1,
            'cop_type': 'op',
            'use_prefix': False,
            'problem_scale': getattr(args, 'op_node_num', 20),
            'data_num': getattr(args, 'data_num_op', 200),
            'data_source': getattr(args, 'data_source_op', None),
        },
        'Env_OP_V2': {
            'ddp_env': DDP_OP_V2,
            'baisc_env': OP_V2,
            'cop_type': 'op',
            'use_prefix': False,
            'problem_scale': getattr(args, 'op_node_num', 20),
            'data_num': getattr(args, 'data_num_op', 200),
            'data_source': getattr(args, 'data_source_op', None),
        },
        'Env_OP_V3': {
            'ddp_env': DDP_OP_V3,
            'baisc_env': OP_V3,
            'cop_type': 'op',
            'use_prefix': False,
            'problem_scale': getattr(args, 'op_node_num', 20),
            'data_num': getattr(args, 'data_num_op', 200),
            'data_source': getattr(args, 'data_source_op', None),
        },
        'Env_OP_V4': {
            'ddp_env': DDP_OP_V4,
            'baisc_env': OP_V4,
            'cop_type': 'op',
            'use_prefix': True,
            'problem_scale': getattr(args, 'op_node_num', 20),
            'data_num': getattr(args, 'data_num_op', 200),
            'data_source': getattr(args, 'data_source_op', None),
        },

        # ----------------------------- cvrp -----------------------------
        'Env_CVRP_V1': {
            'ddp_env': DDP_CVRP_V1,
            'baisc_env': CVRP_V1,
            'cop_type': 'cvrp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'cvrp_node_num', 20),
            'data_num': getattr(args, 'data_num_cvrp', 200),
            'data_source': getattr(args, 'data_source_cvrp', None),
        },
        'Env_CVRP_V2': {
            'ddp_env': DDP_CVRP_V2,
            'baisc_env': CVRP_V2,
            'cop_type': 'cvrp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'cvrp_node_num', 20),
            'data_num': getattr(args, 'data_num_cvrp', 200),
            'data_source': getattr(args, 'data_source_cvrp', None),
        },
        'Env_CVRP_V3': {
            'ddp_env': DDP_CVRP_V3,
            'baisc_env': CVRP_V3,
            'cop_type': 'cvrp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'cvrp_node_num', 20),
            'data_num': getattr(args, 'data_num_cvrp', 200),
            'data_source': getattr(args, 'data_source_cvrp', None),
        },

        # ----------------------------- atsp -----------------------------
        'Env_ATSP_V1': {
            'ddp_env': DDP_ATSP_V1,
            'baisc_env': ATSP_V1,
            'cop_type': 'atsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'atsp_city_num', 20),
            'data_num': getattr(args, 'data_num_atsp', 200),
            'data_source': getattr(args, 'data_source_atsp', None),
        },
        'Env_ATSP_V2': {
            'ddp_env': DDP_ATSP_V2,
            'baisc_env': ATSP_V2,
            'cop_type': 'atsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'atsp_city_num', 20),
            'data_num': getattr(args, 'data_num_atsp', 200),
            'data_source': getattr(args, 'data_source_atsp', None),
        },
        'Env_ATSP_V3': {
            'ddp_env': DDP_ATSP_V3,
            'baisc_env': ATSP_V3,
            'cop_type': 'atsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'atsp_city_num', 20),
            'data_num': getattr(args, 'data_num_atsp', 200),
            'data_source': getattr(args, 'data_source_atsp', None),
        },

        # ----------------------------- spctsp -----------------------------
        'Env_SPCTSP_V1': {
            'ddp_env': DDP_SPCTSP_V1,
            'baisc_env': SPCTSP_V1,
            'cop_type': 'spctsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'spctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_spctsp', 200),
            'data_source': getattr(args, 'data_source_spctsp', None),
        },
        'Env_SPCTSP_V2': {
            'ddp_env': DDP_SPCTSP_V2,
            'baisc_env': SPCTSP_V2,
            'cop_type': 'spctsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'spctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_spctsp', 200),
            'data_source': getattr(args, 'data_source_spctsp', None),
        },
        'Env_SPCTSP_V3': {
            'ddp_env': DDP_SPCTSP_V3,
            'baisc_env': SPCTSP_V3,
            'cop_type': 'spctsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'spctsp_node_num', 20),
            'data_num': getattr(args, 'data_num_spctsp', 200),
            'data_source': getattr(args, 'data_source_spctsp', None),
        },

        # ----------------------------- ffsp -----------------------------
        'Env_FFSP_V1': {
            'ddp_env': DDP_FFSP_V1,
            'baisc_env': None,
            'cop_type': 'ffsp',
            'use_prefix': True,
            'problem_scale': getattr(args, 'ffsp_job_num', 20),
            'data_num': getattr(args, 'data_num_ffsp', 200),
            'data_source': getattr(args, 'data_source_ffsp', None),
        },
        'Env_FFSP_V2': {
            'ddp_env': DDP_FFSP_V2,
            'baisc_env': None,
            'cop_type': 'ffsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'ffsp_job_num', 20),
            'data_num': getattr(args, 'data_num_ffsp', 200),
            'data_source': getattr(args, 'data_source_ffsp', None),
        },
        'Env_FFSP_V3': {
            'ddp_env': DDP_FFSP_V3,
            'baisc_env': None,
            'cop_type': 'ffsp',
            'use_prefix': False,
            'problem_scale': getattr(args, 'ffsp_job_num', 20),
            'data_num': getattr(args, 'data_num_ffsp', 200),
            'data_source': getattr(args, 'data_source_ffsp', None),
        },
    
        # ----------------------------- mis -----------------------------
        'Env_MIS_V1': {
            'ddp_env': DDP_MIS_V1,
            'baisc_env': None,
            'cop_type': 'mis',
            'use_prefix': False,
            'problem_scale': getattr(args, 'mis_node_num', 20),
            'data_num': getattr(args, 'data_num_mis', 200),
            'data_source': getattr(args, 'data_source_mis', None),
        },
        'Env_MIS_V2': {
            'ddp_env': DDP_MIS_V2,
            'baisc_env': None,
            'cop_type': 'mis',
            'use_prefix': False,
            'problem_scale': getattr(args, 'mis_node_num', 20),
            'data_num': getattr(args, 'data_num_mis', 200),
            'data_source': getattr(args, 'data_source_mis', None),
        },

        # ----------------------------- obp3d -----------------------------
        'Env_OBP3D_V1': {
            'ddp_env': None,
            'baisc_env': OBP3D_V1,
            'cop_type': 'obp3d',
            'use_prefix': False,
            'problem_scale': -1,
            'data_num': getattr(args, 'data_num_obp3d', 200),
            'data_source': getattr(args, 'data_source_obp3d', None),
        },
    }
    return data_info