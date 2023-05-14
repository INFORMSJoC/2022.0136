from collections import namedtuple
import numpy as np
import time
from sys import argv
from itertools import product

data = namedtuple('data', 'c_e c_r l_e l_r h b demand')
demand = namedtuple('demand', 'min max support')


def load_data(filename):
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
        data.c_e = int(f.readline())
        data.c_r = int(f.readline())
        data.l_e = int(f.readline())
        data.l_r = int(f.readline())
        data.h = int(f.readline())
        data.b = int(f.readline())
        d_min = int(f.readline())
        d_max = int(f.readline())
        # demand can be modeled in a better way (needs to be a class)
        # I will keep it like this for now
        demand.min = d_min
        demand.max = d_max
        support = d_max - d_min
        demand.support = support
        demand.prob = dict(zip(np.arange(d_min, d_max + 1), np.repeat(1 / (support+1), support+1)))
    data.demand = demand
    return data

def vf_update(state, vf, actions, states, this_data):
    """
    Calculation of value iteration for a single update.
    state:   tuple of integers of length l=lf-le: [IPe, qr(t-l+1), ..., qr(t-1)]
    vf:      dictionary of value function per state

    The value iteration update is
    vf[state] <-- min_{(qe, qr)} {sum(s in states) prob(s| (qe, qr, state))*vf[s]}
    """

    best_action, best_cost = None, 10e9

    max_d, min_d, prob = this_data.demand.max, this_data.demand.min, this_data.demand.prob

    for qe, qr in actions:
        # Immediate cost of action
        cost = qe * this_data.c_e
        # Partial state update
        ip_e = state[0] + qe + state[1]

        pipeline = [*state[2:], qr] if state[2:] else qr

        for dem in range(min_d, max_d + 1):
            ipe_new = ip_e - dem
            # This should work for the general case
            this_state = (ipe_new, *pipeline) if state[2:] else (ipe_new, qr)
            # If we jump to a state that is not in our list, we are not playing optimal
            # so we can safely get out of here. 
            if this_state not in states:
                cost = 10e9
                break
            else:
                # Careful: qr(t-1) has not arrived yet, we need to take it out
                inv_on_hand = ipe_new - state[1]
                inv_cost = inv_on_hand * this_data.h if inv_on_hand >= 0 else -inv_on_hand * this_data.b
                cost += prob[dem] * (inv_cost + vf[this_state])
        if cost < best_cost:
            best_cost = cost
            best_action = (qe, qr)

    if not best_action:
        # If there is no best action it means we were left in a state we were not supposed 
        # to ever get there with optimal play; we can remove it
        vf.pop(state, None)
        states.remove(state)

    return best_cost, best_action

def main(filename='ds1.in'):
    instance_data = load_data(filename)
    # In problems where demand in [0, 4], the expedited inventory position is between -8 and 13
    # Note that some of the states should never be reached (the ones with high inventory and high qr)
    # If we land in such a state we will remove it
    dim_pipeline = instance_data.l_r - instance_data.l_e - 1
    states = list(product(range(-8, 15 + 1), *(range(5 + 1),) * dim_pipeline))
    # SW mention we never need to order more than max demand for any mode
    actions = list(product(range(5+1), range(5+1)))
    # Values can be initiated arbitrarily
    vals = np.repeat(1, len(states))
    vf = dict(zip(states, vals))

    max_iterations, tolerance, delta = 200000, 10e-9, 10.
    all_values = np.zeros(max_iterations)
    these_values = np.zeros(len(states))

    stat_time = time.time()

    # Main value iteration loop
    for iteration in range(max_iterations):
        # We first store the each newly updated state
        for idx, state in enumerate(states):
            these_values[idx], best_action = vf_update(state, vf, actions, states, instance_data)
        # After the minimum for each state has been calculated, we update the states
        # If done in the same loop, converge sucks even more
        for idx, state in enumerate(states):
            vf[state] = these_values[idx]

        values = np.array([val for val in vf.values()])
        this_average = values.mean()
        all_values[iteration] = this_average/(iteration+1)

        if (iteration > 99) and (iteration % 100 == 0):
            print(f'iteration: {iteration} average cost: {all_values[iteration]}')
            delta = all_values[iteration - 1] - all_values[iteration]
            if delta <= tolerance:
                break

    end_time = time.time() - stat_time
    print(f'DP terminated after {end_time} seconds. Tolerance: {delta} Iterations: {iteration}')


if __name__ == '__main__':
    filename = 'ds1.in' if not len(argv) > 1 else argv[1]
    main(filename)


