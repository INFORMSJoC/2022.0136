from collections import namedtuple
import numpy as np
import time
import pickle
from sys import argv
from itertools import product
from numba import njit
from numba import types
from numba.typed import Dict, List

data = namedtuple('data', 'c_e c_r l_e l_r h b demand')
demand = namedtuple('demand', 'min max support')


def load_data():
    """
    Sample file structure:
    100 (c_e)
    0   (c_r)
    0   (le)
    2   (lr)
    5   (h)
    95  (b)
    0   (min demand)
    4   (max demand)
    """
    with open(filename, 'r') as f:
        data.c_e = float(f.readline())
        data.c_r = float(f.readline())
        data.l_e = float(f.readline())
        data.l_r = float(f.readline())
        data.h = float(f.readline())
        data.b = float(f.readline())
        d_min = float(f.readline())
        d_max = float(f.readline())
        # demand can be modeled in a better way (needs to be a class)
        # I will keep it like this for now
        demand.min = d_min
        demand.max = d_max
        support = d_max - d_min
        demand.support = support
        demand.prob = dict(zip(np.arange(d_min, d_max + 1), np.repeat(1 / (support + 1), support + 1)))
        if 'fc' in filename:
            data.f_e = float(f.readline())
            data.f_r = float(f.readline())
    data.demand = demand
    return data


@njit
def vf_update(state, vf, demand_prob, actions):
    """
    Calculation of value iteration for a single update.
    state:   tuple of integers of length l=lf-le: [IPe, qr(t-l+1), ..., qr(t-1)]
    vf:      dictionary of value function per state

    The value iteration update is
    vf[state] <-- min_{(qe, qr)} {sum(s in states) prob(s| (qe, qr, state))*vf[s]}
    """

    best_action, best_cost = None, 10e9

    for qe, qr in actions:
        # Immediate cost of action
        cost = qe * ce
        if qe:
            cost += fe
        if qr:
            cost += fr        
        # Partial state update
        ip_e = state[0] + qe + state[1]

        pipeline = state[2:] if lr > 2 else qr

        for dem in range(d_min, d_max + 1):
            ipe_new = ip_e - dem
            this_state = (ipe_new, *pipeline, qr) if lr > 2 else (ipe_new, qr)

            # If we jump to a state that is not in our list, we are not playing optimal
            # so we can safely get out of here. 
            if (this_state not in vf) or (vf[this_state] > 10e9 - 1.):
                cost = 10e9
                break
            else:
                # Careful: qr(t-1) has not arrived yet, we need to take it out
                inv_on_hand = ipe_new - state[1]
                inv_cost = inv_on_hand * h if inv_on_hand >= 0 else -inv_on_hand * b
                cost += demand_prob[dem] * (inv_cost + vf[this_state])
        if cost < best_cost:
            best_cost = cost
            best_action = (qr, qe)

    return best_cost, best_action


def main():
    """
    Value iteration function.
    
    Parameters: 
    filename (str): filename of parameter file.
    
    """
    # Note that some of the states should never be reached (the ones with high inventory and high qr)
    # If we land in such a state we will ignore it
    dim_pipeline = lr - le - 1
    min_ip = int(d_max * lr)
    max_ip = int((lr + 1) * (d_max + 1) + d_max)
    states_ = list(product(range(-min_ip, max_ip + 1), *(range(int(d_max) + 1),) * int(dim_pipeline)))
    states = List()
    for state in states_:
        states.append(state)
    # SW mention we never need to order more than max demand for any mode
    actions_ = list(product(range(int(d_max)*2 + 1), range(int(d_max)*2 + 1)))
    actions = List()
    for action in actions_:
        actions.append(action)
    # Values can be initiated arbitrarily
    vals = np.repeat(1., len(states))

    vf_ = dict(zip(states, vals))

    vf = Dict.empty(key_type=types.UniTuple(types.int64, lr), value_type=types.float64)
    for k, v in vf_.items():
        vf[k] = v

    demand_prob_ = demand.prob

    demand_prob = Dict.empty(key_type=types.float64, value_type=types.float64)
    for k, v in demand_prob_.items():
        demand_prob[k] = v

    max_iterations, tolerance, delta = 1000000, 10e-9, 10.
    all_values = np.zeros(max_iterations)
    these_values = np.zeros(len(states))

    start_time = time.time()
    
    iteration_arr = []
    time_arr = []
    value_arr = []
    qf = {}
    
    # Main value iteration loop
    for iteration in range(max_iterations):
        # We first store each newly updated state

        for idx, state in enumerate(states):
            # print(idx, state, len(states), vf_update(state, vf, demand_prob, actions, states)[0])
            these_values[idx] = vf_update(state, vf, demand_prob, actions)[0]
        # After the minimum for each state has been calculated, we update the states
        # If done in the same loop, converge sucks even more
        for idx, state in enumerate(states):
            vf[state] = these_values[idx]

        this_average = np.mean([val for val in vf.values() if val < 10e8])

        all_values[iteration] = this_average / (iteration + 1)

        if iteration > 1 and iteration % 100 == 0:
            
            iteration_arr.append(iteration)
            time_arr.append(time.time() - start_time)
            value_arr.append(all_values[iteration])

            print('iteration: %d average cost: %1.3f' % (iteration, all_values[iteration]))
            
            delta = all_values[iteration - 1] - all_values[iteration]
            
            if delta <= tolerance:
                for state in states:
                    qa = vf_update(state, vf, demand_prob, actions)[1]
                    if qa:
                        qf[state] = qa
                ilr, ice, ib, ih, iu = int(lr), int(ce), int(b), int(h), int(d_max)
                if fe==0.:
                    f_name = f'dp_state_output_lr={ilr}ce={ice}b={ib}h={ih}u={iu}.p'
                else:
                    f_name = f'dp_state_output_lr={ilr}ce={ice}b={ib}h={ih}u={iu}fe={fe}fr={fr}.p'
                pickle.dump(qf, open(f_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                break

    end_time = time.time() - start_time
    # np.savetxt('recursion_output/iteration_output_lr=%d_ce=%d.csv'%(lr,ce),np.c_[iteration_arr,time_arr,value_arr],fmt="%d,%1.2f,%1.9f",header="iteration,run time [seconds],cost")
    print('DP terminated after %1.2f seconds. Tolerance: %1.9f Iterations: %d' % (end_time, delta, iteration))


if __name__ == '__main__':
    filename = 'ds1.in' if not len(argv) > 1 else argv[1]
    instance_data = load_data()
    ce, cr, lr, le = instance_data.c_e, instance_data.c_r, instance_data.l_r, instance_data.l_e
    d_min, d_max, demand = instance_data.demand.min, instance_data.demand.max, instance_data.demand
    h, b = instance_data.h, instance_data.b
    fe, fr = 0., 0.
    if 'fc' in filename:
        fe, fr = instance_data.f_e, instance_data.f_r    
    main()

