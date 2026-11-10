from data.raw.tsp import gen_rl_tsp
from data.raw.bp import get_rl_bp
from data.raw.cvrp import get_rl_cvrp
from data.raw.op import get_rl_op
from data.raw.pctsp import get_rl_pctsp
from data.raw.spctsp import get_rl_spctsp

def get_problem_gen_func(env_name, args, rng, batch_size):
    if env_name.startswith('Env_TSP'):
        return lambda: gen_rl_tsp(rng, batch_size, args.tsp_city_num)
    elif env_name.startswith('Env_BP'):
        return lambda: get_rl_bp(rng, batch_size, args.bp_item_num)
    elif env_name.startswith('Env_CVRP'):
        return lambda: get_rl_cvrp(rng, batch_size, args.cvrp_node_num)
    elif env_name.startswith('Env_OP'):
        return lambda: get_rl_op(rng, batch_size, args.op_node_num)
    elif env_name.startswith('Env_PCTSP'):
        return lambda: get_rl_pctsp(rng, batch_size, args.pctsp_node_num)
    elif env_name.startswith('Env_SPCTSP'):
        return lambda: get_rl_spctsp(rng, batch_size, args.spctsp_node_num)
    else:
        raise NotImplementedError(f"Environment {env_name} not implemented for problem generation.")