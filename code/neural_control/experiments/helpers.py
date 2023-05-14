import numpy as np
from collections import namedtuple
from copy import deepcopy
import torch

from neural_control.demand_generators import FileBasedDemandGenerator
from neural_control.dynamics import SequentialDualSourcingModel



def roll_boundary(tensor, roll_dim, roll_shift, default_value=0):
    if roll_shift == 0:
        return tensor
    new_tensor = tensor.roll(dims=(roll_dim), shifts=(roll_shift))
    zero_slice = [slice(None, None)]*len(tensor.shape)
    zero_slice[roll_dim] = slice(None, roll_shift) if roll_shift>0 else slice(roll_shift, None)
    new_tensor[zero_slice] = new_tensor[zero_slice]*0 + default_value
    return new_tensor

def load_experiment_conf(filename: str):
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
    data = namedtuple('data', 'c_e c_r l_e l_r h b demand')
    demand = namedtuple('demand', 'min max support')

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


current_results = [
    dict(baselines='NNC-RNN', b=495, mean_cost=747117, median_cost=692118, method='current'),
    dict(baselines='CDI', b=495, mean_cost=773993, median_cost=770362, method='current'),
    dict(baselines='NNC-RNN', b=495, mean_cost=666600, median_cost=620968, method='future'),
    dict(baselines='CDI', b=495, mean_cost=722346, median_cost=716001, method='future'),
    dict(baselines='NNC-RNN', b=95, mean_cost=583873, median_cost=563711, method='current'),
    dict(baselines='CDI', b=95, mean_cost=736018, median_cost=735265, method='current'),
    dict(baselines='NNC-RNN', b=95, mean_cost=564003, median_cost=541150, method='future'),
    dict(baselines='CDI', b=95, mean_cost=684495, median_cost=682958, method='future')
]

base_cofing = dict(h=5, b=495, cr=0, ce=20, lr=2, le=0)
service_configs = dict(high_service=dict(b=495), low_service=dict(b=95))


def get_config(use_low_service: bool) -> dict:
    """
    Get the current experiment configuration.

    Parameters
    ----------
    use_low_service: bool
        Whether to use low service :math:`b=95` or high service :math:`b=495` configuration.

    Returns
    -------
    experiment_config: dict
        The experiment configuration dictionary.
    """
    experiment_config = deepcopy(base_cofing)
    if use_low_service:
        experiment_config.update(service_configs['low_service'])
    else:
        experiment_config.update(service_configs['high_service'])
    return experiment_config


class MSOMDemandLSTMExperiment:
    def __init__(self,
                 file_demand_gen: FileBasedDemandGenerator,
                 sourcing_parameters: dict,
                 initial_inventory: float = 0.0,
                 ):
        self.file_demand_gen = file_demand_gen
        self.T = self.file_demand_gen.max_weeks

        self.mean_vector = torch.tensor(self.file_demand_gen.mean_array).float()
        self.std_vector = torch.tensor(self.file_demand_gen.std_array).float()

        self.sourcing_parameters = sourcing_parameters
        self.dynamics = SequentialDualSourcingModel(**sourcing_parameters)
        self.dynamics.I_0 = torch.tensor(initial_inventory)

    def generate_state_sequence(self, N=16, T=None, use_mean=False, use_std=False, mean_shift=1, std_shift=1):
        if T is None:
            T = self.T
        self.dynamics.reset(N)
        inital_inventories = torch.randint(low=0, high=100 + 1, size=(N, 1)).float()  # sds.I_0.repeat(N, 1)
        initial_demands = self.dynamics.all_demands[0]
        initial_qr = torch.cat(self.dynamics.previous_qr, dim=-1)
        initial_qe = torch.cat(self.dynamics.previous_qe, dim=-1)
        future_demands = self.file_demand_gen.sample_trajectory(t=0, n_samples=N, n_timesteps=T)
        demands = torch.cat([initial_demands, future_demands], dim=-1).float()
        past_demands = demands[:, :-1]
        nn_input = (inital_inventories - past_demands).unsqueeze(
            -1)  # (inital_inventories - past_demands).diff(dim=1, prepend=torch.zeros(N, 1)).unsqueeze(-1)

        nn_input_2 = torch.zeros_like(self.mean_vector.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1))
        nn_input_3 = torch.zeros_like(self.std_vector.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1))

        if use_mean:
            nn_input_2 = self.mean_vector.unsqueeze(0).unsqueeze(-1).roll(mean_shift, dims=1)
            if mean_shift > 0:
                nn_input_2[:, :mean_shift, :] = 0
            if mean_shift < 0:
                nn_input_2[:, mean_shift:, :] = 0
            nn_input_2 = nn_input_2.repeat(N, 1, 1)

        if use_std:
            nn_input_3 = self.std_vector.unsqueeze(0).unsqueeze(-1).roll(std_shift, dims=1)
            if std_shift > 0:
                nn_input_3[:, :std_shift, :] = 0
            if std_shift < 0:
                nn_input_3[:, std_shift:, :] = 0
            nn_input_3 = nn_input_3.repeat(N, 1, 1)

        nn_input = torch.cat([nn_input,
                              nn_input_2,
                              nn_input_3
                              ], dim=-1)
        return nn_input, initial_qr, initial_qe, inital_inventories, demands, past_demands

    def validate_model(self, controller, val_N=512, test_T=None, dynamics=None,
                       **state_sequence_params):
        if test_T is None:
            test_T = self.T
        if dynamics is None:
            dynamics = self.dynamics
        with torch.no_grad():
            controller.eval()
            dynamics.reset(val_N)
            nn_input, initial_qr, initial_qe, initial_inventories, demands, past_demands = self.generate_state_sequence(
                N=val_N, **state_sequence_params)

            costs, invs, qr, qe = lstm_model_step(controller, nn_input, initial_qr, initial_qe, initial_inventories,
                                                  dynamics, demands, test_T)

            mean_costs = costs.mean().item()
            median_costs = costs.median().item()

            return (mean_costs,
                    median_costs,
                    qr.cpu().detach().numpy(),
                    qe.cpu().detach().numpy(),
                    invs.cpu().detach().numpy(),
                    demands.cpu().detach().numpy(),
                    initial_inventories.cpu().detach().numpy(),
                    costs.cpu().detach().numpy()
                    )


def lstm_model_step(controller, nn_input, initial_qr, initial_qe, initial_inventories, dynamics, demands, T):
    new_qr, new_qe = controller(nn_input)

    qr = torch.cat([initial_qr, new_qr], dim=-1) if dynamics.lr > 0 else new_qr
    qe = torch.cat([initial_qe, new_qe], dim=-1) if dynamics.le > 0 else new_qe

    qe_arrived = qe[:, :T]
    qr_arrived = qr[:, :T]

    qe_ordered = qe[:, dynamics.le:]
    qr_ordered = qr[:, dynamics.lr:]

    costs, invs = dynamics.replay_multisteps(initial_inventories,
                                             qra=qr_arrived,
                                             qea=qe_arrived,
                                             qro=qr_ordered,
                                             qeo=qe_ordered,
                                             all_demands=demands
                                             )
    return costs, invs, qr, qe
