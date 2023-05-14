import torch

from neural_control.demand_generators import AbstractDemandGenerator, TorchDistDemandGenerator


class SequentialDualSourcing(torch.nn.Module):
    def __init__(self,
                 controller: torch.nn.Module,
                 ce: int=20,
                 cr: int=0,
                 le: int=0,
                 lr: int=2,
                 h: int=5,
                 b: int=495,
                 T: int=50,
                 I_0: int=0,
                 fe: int = 100,
                 fr: int = 50,
                 learn_I_0: bool=True,
                 demand_generator: AbstractDemandGenerator = TorchDistDemandGenerator(),
                 ):

        super().__init__()
        self.controller = controller
        self.ce = ce
        self.cr = cr
        self.le = le
        self.lr = lr
        self.h = h
        self.b = b
        self.T = T

        I_0 = torch.tensor([I_0], requires_grad=learn_I_0, dtype=torch.float)
        if learn_I_0:
            self.I_0 = torch.nn.Parameter(I_0)
        else:
            self.I_0 = I_0
        self.fe = fe
        self.fr = fr
        self.demand_generator = demand_generator

        self.controller = controller

    def simulate(self, controller_kwargs: dict = {}):
        if self.current_timestep > self.T:
            raise ValueError("Timestep coutner surpassed value of T. Please reset dynamics.")

        # demands up to current timestep, that is already known to the model.
        past_demands = self.all_demands[:, 0:self.current_timestep+1, :]

        #inventories up to current timestep, that is already known to the model
        past_inventories = self.all_inventories[:, 0:self.current_timestep+1, :]
        past_costs = self.all_costs[:, :self.current_timestep+1, :]

        # find previous orders from order trajectory tensors
        previous_qr = self.all_qr[:, :self.lr + self.current_timestep, :]
        previous_qe = self.all_qe[:, :self.le + self.current_timestep, :]

        # orders that are just placed.
        qr, qe, nn_context = self.controller(past_demands, past_inventories, previous_qr, previous_qe, past_costs, **controller_kwargs)

        # orders are persisted to respective tensors
        self.all_qr[:, self.lr + self.current_timestep, 0:1] = qr
        self.all_qe[:, self.le + self.current_timestep, 0:1] = qe

        # orders that arrive
        qra = self.all_qr[:, self.current_timestep, :]
        qea = self.all_qe[:, self.current_timestep, :]

        # Get new demand from the demand trajectory tensor
        D = self.all_demands[:, self.current_timestep + 1, :]

        # inventory update with arrived orders
        self.I_i = self.I_i + qra + qea
        # cost update with sourcing, holding and backlogging costs and new demand
        c_i = self.ce * qe + self.cr * qr + self.h * torch.relu(self.I_i - D) \
              + self.b * torch.relu(D - self.I_i)
        # new cost update with fixed costs and identity straight-through positivity check
        has_fe_cost = qe - torch.relu(qe).detach() + torch.ones_like(qe)*(qe > 0.1)
        has_fr_cost = qr - torch.relu(qr).detach() + torch.ones_like(qr)*(qr > 0.1)

        c_i = c_i + self.fe*has_fe_cost + self.fr*has_fr_cost
        self.all_costs[:, self.current_timestep+1, :] = c_i

        # inventory update with new demand
        self.I_i = self.I_i - D

        self.all_inventories[:, self.current_timestep + 1, :] = self.I_i

        # timestep update
        self.current_timestep += 1

        return c_i, D, self.I_i, qr, qra, qe, qea, nn_context

    def replay_step(self,
                    previous_inventory,
                    current_demand,
                    qra,
                    qea,
                    qr,
                    qe):

        current_inventory = previous_inventory.round() + qra.round() + qea.round()

        c_i = self.ce * qe.round() + self.cr * qr.round() + \
              self.h * torch.relu(current_inventory.round() - current_demand.round()) + \
              self.b * torch.relu(current_demand.round() - current_inventory.round())

        current_inventory = current_inventory.round() - current_demand.round()

        return c_i, current_inventory.round()

    def reset(self,
              minibatch_size=16,
              seed=None):

        if not seed is None:
            torch.manual_seed(seed)
        self.current_timestep = 0

        self.all_qr = torch.zeros([minibatch_size, self.T + self.lr, 1])
        # N x T + lr x 1
        self.all_qe = torch.zeros([minibatch_size, self.T + self.le, 1])
        # N x T + le x 1
        self.all_inventories = torch.zeros([minibatch_size, self.T+1, 1])
        # N x T x 1
        self.all_demands = torch.zeros([minibatch_size, self.T+1, 1])
        # N x T + 1 x 1
        self.all_demands[:, 1:, :] = self.demand_generator.sample_trajectory(t=0,
                                                                             n_samples=minibatch_size,
                                                                             n_timesteps=self.T
                                                                             )
        self.all_costs = torch.zeros([minibatch_size, self.T+1, 1])
        # N x T x 1

        self.learned_I_0 = self.I_0.repeat([minibatch_size, 1]) - torch.frac(self.I_0).clone().detach()
        self.I_i = self.learned_I_0

        self.all_inventories[:, 0, 0:1] = self.I_i
