import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import subprocess
#from gurobipy import GRB, tuplelist, quicksum, itertools, math, Model
from datetime import datetime
from subprocess import check_call, check_output
from scipy.spatial import distance_matrix
from typing import List, Union
import os
import time
import numpy as np

# ------------------------ DP & DC for 01BP ------------------------
def knapsack_dp(capacity, volumes, values):
    """ 使用动态规划求解01背包问题, 返回最大价值和选择的物品索引, 要求 capacity 和 volumes 都是整数"""
    num_items = len(volumes)
    dp = [[0] * (capacity + 1) for _ in range(num_items + 1)]

    for i in range(1, num_items + 1):
        for j in range(1, capacity + 1):
            if volumes[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - volumes[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]

    # 回溯选择的物品索引
    selected_items = []
    i, j = num_items, capacity
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            selected_items.append(i - 1)
            j -= volumes[i - 1]
        i -= 1

    return dp[num_items][capacity], selected_items[::-1]

def knapsack_dc(capacity, volumes, values):
    """ 使用遍历法求解01背包问题, 返回最大价值和选择的物品索引 """
    num_items = len(volumes)
    max_value = 0
    max_selection = []

    for i in range(1 << num_items):
        selection = []
        total_volume = 0
        total_value = 0

        for j in range(num_items):
            if (i >> j) & 1:
                selection.append(j)
                total_volume += volumes[j]
                total_value += values[j]

        if total_volume <= capacity and total_value > max_value:
            max_value = total_value
            max_selection = selection

    return max_value, max_selection

def calc_bp_total(item_values:np.ndarray, selection:Union[np.ndarray, List]):
    return item_values[selection].sum()

# ------------------------ LKH & DC for TSP ------------------------
def calc_tsp_distance(position, answer):
    if isinstance(answer, list):
        sorted_answer = position[np.array(answer+[0], dtype=np.int32)]
        distance = np.linalg.norm(sorted_answer[1:] - sorted_answer[:-1], axis=-1).sum()
    elif isinstance(answer, np.ndarray):
        # position: (problem_batch_size_per_gpu, node_num, 2)
        # answer: (problem_batch_size_per_gpu, node_num)
        problem_batch_size_per_gpu = answer.shape[0]
        answer_idxs = np.hstack((answer, np.zeros((problem_batch_size_per_gpu, 1), dtype=np.int32)))    # (problem_batch_size_per_gpu, node_num+1)
        position_idxs = np.arange(problem_batch_size_per_gpu).reshape(problem_batch_size_per_gpu, 1)
        sorted_answer = position[position_idxs, answer_idxs]                                            # (problem_batch_size_per_gpu, node_num+1, 2)
        distance = np.linalg.norm(sorted_answer[:,1:] - sorted_answer[:,:-1], axis=-1).sum(axis=1)      # (problem_batch_size_per_gpu, )
    return distance

def TSP_lkh(data, l_opt_limit=6, scale=100000, log_only=False, always=False):
    coord = data # nx2
    coord = (coord * scale).astype(int)
    generate_path = f'{os.path.dirname(os.path.abspath(__file__))}/LKH-revised/LKH'
    ### we change the tour record of original LKH in WriteTour.c:
    # 1. index start from 0
    # 2. new logic to identify the end of tour
    saved_path = f'{base_path}/utils/tsp_saved_model'   

    always = always and not log_only
    if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
        raise ValueError()
    f_par = "SPECIAL`PROBLEM_FILE = -`OUTPUT_TOUR_FILE = "+ saved_path+"`"
    f_vrp = "NAME : xxx`TYPE : TSP`DIMENSION :"+ str(coord.shape[0])+"`EDGE_WEIGHT_TYPE : EUC_2D`NODE_COORD_SECTION`"
    f_vrp += '`'.join(str(index+1)+' '+str(xy[0])+' '+str(xy[1]) for (index, xy) in enumerate(coord)) + '`'
    f_vrp += 'EOF`'

    subprocess.run([f'{generate_path}', f_par, f_vrp])
    with open(saved_path, 'r') as f:
        solution = f.readlines()
        solution = [int(t) for t in solution]
    del coord

    ### obtain the obs and rewards
    # obs: dimension: (size*3) (coords+visited) + 1 (first node index) + 1(last node index)) 
    distance = calc_tsp_distance(data, solution)

    return distance, solution

def TSP_gurobi(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour

# ------------------------ LKH for VRP ------------------------
'''
def calc_vrp_distance(depot, loc, tour):
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
'''

def calc_vrp_distance(pos, tour:List):
    if len(tour) == 0:
        return 0
    sorted_locs = pos[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def CVRP_lkh(depot, loc, demand, capacity, grid_size=1):
    executable = f'{base_path}/utils/LKH-original/LKH-3.0.8/LKH'
    problem_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    #problem_id = 'test'
    problem_filename = f'{base_path}/temp/{problem_id}.lkh.vrp'
    tour_filename = f'{base_path}/temp/{problem_id}.lkh.tour'
    param_filename = f'{base_path}/temp/{problem_id}.lkh.par'
    log_filename = f'{base_path}/temp/{problem_id}.lkh.log'

    try:
        # VRP 问题
        write_vrplib(
            problem_filename, depot, loc, demand, capacity, 
            grid_size, name=problem_id
        )

        # LKH 参数
        write_lkh_par(
            filename = param_filename, 
            parameters = {
                "PROBLEM_FILE": problem_filename, 
                "OUTPUT_TOUR_FILE": tour_filename, 
                "RUNS": 1
            }
        )

        # 调用 LKH 方法求解
        with open(log_filename, 'w') as f:
            start = time.time()
            check_call([executable, param_filename], stdout=f, stderr=f)
            duration = time.time() - start
        tour = read_vrplib(tour_filename, n=len(demand))

        # 验证解的合法性
        try:
            assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
            cap_left = capacity
            for idx in tour:
                cap_left = capacity if idx == 0 else cap_left - demand[idx-1]
                assert cap_left >= 0
        except AssertionError:
            return None, None, None

        # 删除临时文件
        os.remove(problem_filename)
        os.remove(tour_filename)
        os.remove(param_filename)
        os.remove(log_filename)

        # 返回总路程
        pos = np.vstack((np.array(depot)[None,:], loc))
        return calc_vrp_distance(pos, tour), tour, duration

    except Exception as e:
        #print(e)
        try:
            os.remove(problem_filename)
            os.remove(tour_filename)
            os.remove(param_filename)
            os.remove(log_filename)
        except Exception:
            pass
        return None, None, None

def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 1234
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))

def read_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()

def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem"):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")



# ------------------------ gurobi for OP ------------------------
def calc_op_total(prize:np.ndarray, tour:Union[np.ndarray, List]):
    #assert len(set(tour)) == len(tour), "Tour cannot contain duplicates"
    return 0 if len(tour) == 0 else prize[tour].sum()

def calc_op_distance(pos:np.ndarray, tour:np.ndarray):
    sorted_locs = pos[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def solve_euclidian_op(depot, loc, prize, max_length, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan op problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """
    points = [depot] + loc
    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if tour is not None:
                # add subtour elimination constraint for every pair of cities in tour
                # model.cbLazy(quicksum(model._vars[i, j]
                #                       for i, j in itertools.combinations(tour, 2))
                #              <= len(tour) - 1)

                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour)))

    # Given a tuplelist of edges, find the shortest subtour
    def subtour(edges, exclude_depot=True):
        unvisited = list(range(n))
        #cycle = range(n + 1)  # initial length has 1 more city
        cycle = None
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            # If we do not yet have a cycle or this is the shorter cycle, keep this cycle
            # Unless it contains the depot while we do not want the depot
            if (
                (cycle is None or len(cycle) > len(thiscycle))
                    and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
            ):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points
    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables
    vars = m.addVars(dist.keys(), vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # Depot vars can be 2
    for i,j in vars.keys():
        if i == 0 or j == 0:
            vars[i,j].vtype = GRB.INTEGER
            vars[i,j].ub = 2

    prize_dict = {
        i + 1: -p  # We need to maximize so negate
        for i, p in enumerate(prize)
    }
    delta = m.addVars(range(1, n), obj=prize_dict, vtype=GRB.BINARY, name='delta')

    # Add degree-2 constraint (2 * delta for nodes which are not the depot)
    m.addConstrs(vars.sum(i,'*') == (2 if i == 0 else 2 * delta[i]) for i in range(n))

    # Length of tour constraint
    m.addConstr(quicksum(var * dist[i, j] for (i, j), var in vars.items() if j < i) <= max_length)

    # Optimize model
    m._vars = vars
    m._dvars = delta
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected, exclude_depot=False)
    assert tour[0] == 0, "Tour should start with depot"

    return m.objVal, tour

MAX_LENGTH_TOL = 1e-5
def OP_gurobi(
    depot:List, loc:List, prize:List, max_length:float, 
    timeout=None, gap=None
):
    # 求解得等的 tour 中 node 从 1 开始索引，含有出发仓库 idx=0，不含终点仓库
    cost, tour = solve_euclidian_op(
        depot, loc, prize, max_length, threads=1, timeout=timeout, gap=gap
    )
    assert tour[0] == 0
    
    # 返回的 tour 中去除出发仓库 idx=0
    tour = tour[1:]
    assert calc_op_distance(np.array([depot,]+loc), tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
    assert abs(-calc_op_total(np.array(prize), np.array(tour)-1) - cost) <= 1e-4, "Cost is incorrect"
    
    # node 索引调整为从 0 开始
    tour = [idx-1 for idx in tour]
    return cost, tour

'''
# ------------------------ compass for op ------------------------
def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum()
'''
def calc_op_length(depot, loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def write_oplib(filename, depot, loc, prize, max_length, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc) + 1),
                ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")

def read_oplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("NODE_SEQUENCE_SECTION"):
                started = True
    
    assert len(tour) > 0, "Unexpected length"
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()

def OP_compass(depot, loc, prize, max_length):
    MAX_LENGTH_TOL = 1e-5
    executable = f'{base_path}/utils/my_compass/compass/compass'
    problem_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    #problem_id = '0000'
    problem_filename = f'{base_path}/temp/{problem_id}.oplib'
    tour_filename = f'{base_path}/temp/{problem_id}.compass.tour'
    log_filename = f'{base_path}/temp/{problem_id}.compass.log'

    try:
        write_oplib(problem_filename, depot, loc, prize, max_length, name=problem_id)

        with open(log_filename, 'w') as f:
            check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename],
                        stdout=f, stderr=f)

        # 返回的 tour 中不含出发仓库 idx=0
        tour = read_oplib(tour_filename, n=len(prize))
        if not calc_op_length(depot, loc, tour) <= max_length:
            print("Warning: length exceeds max length:", calc_op_length(depot, loc, tour), max_length)
        assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"    
        cost = -calc_op_total(np.array(prize), np.array(tour)-1)

        # 删除临时文件
        os.remove(problem_filename)
        os.remove(tour_filename)
        os.remove(log_filename)

        # node 索引调整为从 0 开始
        tour = [idx-1 for idx in tour]
        return cost, tour

    except Exception as e:
        try:
            os.remove(log_filename)
            os.remove(problem_filename)
            os.remove(tour_filename)
            
        except Exception:
            pass
        #print("Exception occured")
        #print(e)
        return None, None

# ------------------------ ILS for PCTSP ------------------------
def calc_pctsp_total(vals, tour:List):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    #assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    if len(tour) == 0:
        return 0
    return np.array(vals)[np.array(tour) - 1].sum()

def calc_pctsp_length(pos:np.ndarray, tour:np.ndarray):
    sorted_locs = pos[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

def calc_pctsp_cost(pos:np.ndarray, penalty:np.ndarray, prize:np.ndarray, tour:np.ndarray):
    # With some tolerance we should satisfy minimum prize
    #assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    if not calc_pctsp_total(prize, tour) >= 1-1e-5:
        return None
    # Penalty is only incurred for locations not visited, so charge total penalty minus penalty of locations visited
    return calc_pctsp_length(pos, tour) + np.sum(penalty) - calc_pctsp_total(penalty, tour)

def write_pctsp(filename, depot, loc, penalty, prize):
    coord = [depot] + loc
    return write_pctsp_dist(filename, distance_matrix(coord, coord), penalty, prize)

def float_to_scaled_int_str(v):  # Program only accepts ints so scale everything by 10^7
    return str(float_to_scaled_int(v))

def float_to_scaled_int(v):
    return int(v * 10000000 + 0.5)

def write_pctsp_dist(filename, dist, penalty, prize):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(prize)]),
            "",
            "",
            " ".join([float_to_scaled_int_str(p) for p in [0] + list(penalty)]),
            "",
            "",
            *(
                " ".join(float_to_scaled_int_str(d) for d in d_row)
                for d_row in dist
            )
        ]))

def PCTSP_ILS(depot, loc, penalty, deterministic_prize, runs=2):
    def _get_pctsp_executable():
        execfile = f'{base_path}/utils/PCTSP/PCPTSP/main.out'
        if not os.path.isfile(execfile):
            sourcefile = f'{base_path}/utils/PCTSP/PCPTSP/main.cpp'
            print ("Compiling...")
            check_call(["g++", "-g", "-Wall", sourcefile, "-std=c++11", "-o", execfile])
            print ("Done!")
        assert os.path.isfile(execfile), "{} does not exist! Compilation failed?".format(execfile)
        return os.path.abspath(execfile)

    executable = _get_pctsp_executable()
    problem_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    #problem_id = 'test'
    problem_filename = f'{base_path}/temp/{problem_id}.ils.pctsp'
    log_filename = f'{base_path}/temp/{problem_id}.ils.log'

    write_pctsp(problem_filename, depot, loc, penalty, deterministic_prize)
    try:
        with open(log_filename, 'w') as f:
            start = time.time()
            output = check_output(
                # exe, filename, min_total_prize (=1), num_runs
                [executable, problem_filename, float_to_scaled_int_str(1.), str(runs)],
                stderr=f
            ).decode('utf-8')
            duration = time.time() - start
            f.write(output)
    except subprocess.CalledProcessError:
        #print('求解出现段错误', depot)
        return None, None, None

    # Now parse output
    tour = None
    for line in output.splitlines():
        heading = "Best Result Route: "
        if line[:len(heading)] == heading:
            tour = np.array(line[len(heading):].split(" ")).astype(int)
            break
    assert tour is not None, "Could not find tour in output!"
    assert tour[0] == 0, "Tour should start with depot"
    assert tour[-1] == 0, "Tour should end with depot"
    tour = tour[1:-1]  # Strip off depot

    # check cost
    cost = calc_pctsp_cost(np.array([depot,]+loc), penalty, deterministic_prize, tour)
    if cost is None:
        #print('解不符合约束', depot)
        return None, None, None

    # 删除临时文件
    try:
        os.remove(problem_filename)
        os.remove(log_filename)
    except FileNotFoundError:
        pass

    return cost, tour.tolist(), duration

def SPCTSP_REOPT(depot, loc, penalty, deterministic_prize, stochastic_prize, runs=2):
    def _get_spctsp_executable():
        # use the same cpp file as PCTSP
        execfile = f'{base_path}/utils/PCTSP/PCPTSP/main.out'
        if not os.path.isfile(execfile):
            sourcefile = f'{base_path}/utils/PCTSP/PCPTSP/main.cpp'
            print ("Compiling...")
            check_call(["g++", "-g", "-Wall", sourcefile, "-std=c++11", "-o", execfile])
            print ("Done!")
        assert os.path.isfile(execfile), "{} does not exist! Compilation failed?".format(execfile)
        return os.path.abspath(execfile)

    executable = _get_spctsp_executable()
    problem_id = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    problem_filename = f'{base_path}/temp/{problem_id}.ils.spctsp'
    log_filename = f'{base_path}/temp/{problem_id}.ils.log'
    
    total_start = time.time()
    outputs = []
    durations = []
    final_tour = []

    coord = [depot] + loc
    mask = np.zeros(len(coord), dtype=bool)
    dist = distance_matrix(coord, coord)
    penalty = np.array(penalty)
    deterministic_prize = np.array(deterministic_prize)
    
    it = 0
    total_collected_prize = 0
    while len(final_tour) < len(stochastic_prize):
        mask[final_tour] = True
        if len(final_tour) > 0:
            dist[0, :] = dist[final_tour[-1], :]
        remaining_deterministic_prize = deterministic_prize[~mask[1:]]
        write_pctsp_dist(problem_filename,
                                 dist[np.ix_(~mask, ~mask)], penalty[~mask[1:]], remaining_deterministic_prize)
        min_prize_int = max(0, min(
                    float_to_scaled_int(1. - total_collected_prize),
                    sum([float_to_scaled_int(v) for v in remaining_deterministic_prize])
                ))
        try:
            with open(log_filename, 'a') as f:
                start = time.time()
                output = check_output(
                    # exe, filename, min_total_prize (=1), num_runs
                    [executable, problem_filename, str(min_prize_int), str(runs)],
                    stderr=f
                ).decode('utf-8')
                durations.append(time.time() - start)
                outputs.append(output)
        except subprocess.CalledProcessError:
            #print('求解出现段错误', depot)
            return None, None, None
        
        tour = None
        for line in output.splitlines():
            heading = "Best Result Route: "
            if line[:len(heading)] == heading:
                tour = np.array(line[len(heading):].split(" ")).astype(int)
                break
        assert tour is not None, "Could not find tour in output!"

        assert tour[0] == 0, "Tour should start with depot"
        assert tour[-1] == 0, "Tour should end with depot"
        tour = tour[1:-1]  # Strip off depot

        try: 
            tour_node_ids = np.arange(len(coord), dtype=int)[~mask][tour]
        except Exception as e:
            #print('解不符合约束', depot)
            return None, None, None
        
        if len(tour_node_ids) == 0:
            # The inner algorithm can decide to stop, but does not have to
            assert total_collected_prize > 1 - 1e-5, "Collected prize should be one"
            break
        
        ### first reopt
        final_tour.append(tour_node_ids[0])


        total_collected_prize = calc_pctsp_total(stochastic_prize, final_tour)
        it = it + 1

    final_cost = calc_pctsp_cost(np.array([depot,]+loc), penalty, stochastic_prize, final_tour)
    total_duration = time.time() - total_start

    if final_cost is None:
        #print('解不符合约束', depot)
        return None, None, None

    # 删除临时文件
    try:
        os.remove(problem_filename)
        os.remove(log_filename)
    except FileNotFoundError:
        pass

    return final_cost, final_tour, total_duration