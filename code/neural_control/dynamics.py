import torch
import numpy as np

from neural_control.demand_generators import AbstractDemandGenerator, TorchDistDemandGenerator
from neural_control.straight_through import StraightThroughEstimator, BinaryDecoupling, FractionalDecoupling, StraighThroughReLU


class DualSourcingModel(torch.nn.Module):
    def __init__(self,
                 controller: torch.nn.Module,
                 ce: int = 20,
                 cr: int = 0,
                 le: int = 0,
                 lr: int = 2,
                 h: int = 5,
                 b: int = 495,
                 T: int = 50,
                 I_0: int = 0,
                 learn_I_0: bool = True,
                 demand_generator=torch.distributions.Uniform(low=0, high=4 + 1),
                 demand_distribution=[-1],
                 empirical_baseline=1,
                 fr=0,
                 fe=0
                 # high is exclusive
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
        self.I_0 = torch.tensor([I_0], requires_grad=learn_I_0, dtype=torch.float)
        self.empirical_baseline = empirical_baseline

        self.demand_flag = -1
        self.demand_generator = demand_generator

        if demand_distribution[0] != -1:
            self.demand_flag = 1
            self.demand_generator = lambda i, num_samples: \
                torch.from_numpy(demand_distribution[0](i, num_samples)).int().unsqueeze(-1)

        self.controller = controller
        self.fr = fr
        self.fe = fe

    def simulate(self, mean=0, std=0):

        if self.demand_flag == 1 and self.current_timestep == 0:
            np.random.seed(10)

        sample_size = self.I_i.shape[0]
        D = self.all_demands[-1]

        if self.demand_flag == -1:
            qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe)
        else:
            if self.empirical_baseline == 1:
                qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe, mean, std)
            elif self.empirical_baseline == 2:
                min_demands = [max(0, mean[i] - 2.58 * std[i]) for i in range(len(mean))]
                max_demands = [mean[i] + 2.58 * std[i] for i in range(len(mean))]
                qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe, min_demands, max_demands)
            else:
                self.demand_arr[:len(self.all_demands)] = self.all_demands
                qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe, self.demand_arr, mean, std)

        # orders are added to corresponding vectors
        self.previous_qr.append(qr)
        self.previous_qe.append(qe)

        # orders arrive
        qra = self.previous_qr[-self.lr - 1]
        qea = self.previous_qe[-self.le - 1]

        # demand is generated
        if self.demand_flag == -1:
            D = self.demand_generator.sample(
                [sample_size, 1]).int()  # here we round a continuous sample
            self.all_demands.append(D)
        else:
            D = self.demand_generator(self.current_timestep, sample_size)
            self.all_demands.append(D)

        # inventory and cost updates
        self.I_i = self.I_i + qra + qea

        c_i = self.ce * qe + self.cr * qr + self.h * torch.relu(self.I_i - D) \
              + self.b * torch.relu(D - self.I_i)

        # new cost update with fixed costs and identity straight-through positivity check
        has_fe_cost = qe - qe.detach() + torch.ones_like(qe) * (
                qe > 0.1)  # 0.5*(torch.ones_like(qe)+torch.tanh(4*(qe-0.5*torch.ones_like(qe))))#qe - qe.detach() + torch.ones_like(qe) * (qe >= 1)#0.5*(torch.ones_like(qe)+torch.tanh(5*(qe-0.5*torch.ones_like(qe))))#qe - torch.relu(qe).detach() + torch.ones_like(qe) * (qe > 0.1)
        has_fr_cost = qr - qr.detach() + torch.ones_like(qr) * (
                qr > 0.1)  # 0.5*(torch.ones_like(qr)+torch.tanh(4*(qr-0.5*torch.ones_like(qr))))#qr - qr.detach() + torch.ones_like(qr) * (qr >= 1)#0.5*(torch.ones_like(qr)+torch.tanh(5*(qr-0.5*torch.ones_like(qr))))

        c_i = c_i + self.fe * has_fe_cost + self.fr * has_fr_cost

        self.I_i = self.I_i - D

        self.current_timestep += 1

        return c_i, D, self.I_i, qr, qra, qe, qea

    def replay_step(self,
                    previous_inventory,
                    current_demand,
                    qra,
                    qea,
                    qr,
                    qe):

        current_inventory = previous_inventory + qra + qea

        c_i = self.ce * qe + self.cr * qr + \
              self.h * torch.relu(current_inventory - current_demand) + \
              self.b * torch.relu(current_demand - current_inventory)

        # new cost update with fixed costs and identity straight-through positivity check
        has_fe_cost = qe - qe.detach() + torch.ones_like(qe) * (
                qe > 0.1)
        has_fr_cost = qr - qr.detach() + torch.ones_like(qr) * (
                qr > 0.1)

        c_i = c_i + self.fe * has_fe_cost + self.fr * has_fr_cost

        current_inventory = current_inventory - current_demand

        return c_i, current_inventory

    def reset(self,
              minibatch_size=16,
              seed=None):

        if not seed is None:
            torch.manual_seed(seed)
        self.current_timestep = 0
        self.previous_qr = [torch.zeros([minibatch_size, 1])]
        self.previous_qe = [torch.zeros([minibatch_size, 1])]

        if self.lr > 0:
            self.previous_qr = self.previous_qr * self.lr

        if self.le > 0:
            self.previous_qe = self.previous_qe * self.le

        self.learned_I_0 = self.I_0.repeat([minibatch_size, 1]) - torch.frac(self.I_0).clone().detach()
        self.I_i = self.learned_I_0
        self.all_demands = [torch.zeros([minibatch_size, 1])]

        self.demand_arr = 0
        if self.empirical_baseline == 3:
            self.demand_arr = [-1.0 * torch.ones([minibatch_size, 1]) for i in range(self.T)]


class SingleSourcingModel(torch.nn.Module):
    def __init__(self,
                 controller: torch.nn.Module,
                 l=2,
                 h=5,
                 b=495,
                 T=50,
                 I_0=0,
                 learn_I_0=True,
                 demand_generator=torch.distributions.Uniform(low=0, high=4 + 1)
                 # high is exclusive
                 ):

        super().__init__()
        self.controller = controller
        self.l = l
        self.h = h
        self.b = b
        self.T = T
        self.I_0 = torch.tensor([I_0], requires_grad=learn_I_0, dtype=torch.float)
        self.demand_generator = demand_generator
        self.controller = controller

    def simulate(self):
        sample_size = self.I_i.shape[0]
        D = self.all_demands[-1]
        q = self.controller(D, self.I_i, self.previous_q)

        # orders are added to corresponding vectors
        self.previous_q.append(q)

        # orders arrive
        qa = self.previous_q[self.current_timestep - self.l - 1]

        # demand is generated
        D = self.demand_generator.sample(
            [sample_size, 1]).int()  # here we round a continuous sample
        self.all_demands.append(D)

        # inventory and cost updates
        self.I_i = self.I_i + qa

        c_i = self.h * torch.relu(self.I_i - D) \
              + self.b * torch.relu(D - self.I_i)

        self.I_i = self.I_i - D

        return c_i, D, self.I_i, q, qa

    def replay_step(self,
                    previous_inventory,
                    current_demand,
                    qa,
                    q):

        current_inventory = previous_inventory + qa

        c_i = self.h * torch.relu(current_inventory - current_demand) + \
              self.b * torch.relu(current_demand - current_inventory)

        current_inventory = current_inventory - current_demand

        return c_i, current_inventory

    def reset(self,
              minibatch_size=16,
              seed=None):

        if not seed is None:
            torch.manual_seed(seed)

        self.current_timestep = 0
        self.previous_q = [torch.zeros([minibatch_size, 1])]

        if self.l > 0:
            self.previous_q = self.previous_q * self.l

        self.learned_I_0 = self.I_0.repeat([minibatch_size, 1]) - torch.frac(self.I_0).clone().detach()
        self.I_i = self.learned_I_0
        self.all_demands = [torch.zeros([minibatch_size, 1])]


fractional_decoupling = StraightThroughEstimator(FractionalDecoupling.apply)

binary_decoupling = StraightThroughEstimator(BinaryDecoupling.apply)

straight_through_relu = StraightThroughEstimator(StraighThroughReLU.apply)


class SequentialDualSourcingModel(torch.nn.Module):
    def __init__(self,
                 ce: int = 20,
                 cr: int = 0,
                 le: int = 0,
                 lr: int = 2,
                 h: int = 5,
                 b: int = 495,
                 T: int = 50,
                 I_0: int = 0,
                 learn_I_0: bool = True,
                 fr=0,
                 fe=0
                 ):

        super().__init__()
        self.ce = ce
        self.cr = cr
        self.le = le
        self.lr = lr
        self.h = h
        self.b = b
        self.T = T
        self.I_0 = torch.tensor([I_0], requires_grad=learn_I_0, dtype=torch.float)

        self.fr = fr
        self.fe = fe

    def replay_step(self,
                    previous_inventory,
                    current_demand,
                    qra,
                    qea,
                    qr,
                    qe):

        current_inventory = previous_inventory + qra + qea

        c_i = self.ce * qe + self.cr * qr + \
              self.h * torch.relu(current_inventory - current_demand) + \
              self.b * torch.relu(current_demand - current_inventory)

        # new cost update with fixed costs and identity straight-through positivity check
        has_fe_cost = binary_decoupling(qe)
        has_fr_cost = binary_decoupling(qr)

        c_i = c_i + self.fe * has_fe_cost + self.fr * has_fr_cost

        current_inventory = current_inventory - current_demand

        return c_i, current_inventory

    def replay_multisteps(self,
                          initial_inventories: torch.Tensor,
                          qra: torch.Tensor,
                          qea: torch.Tensor,
                          qro: torch.Tensor,
                          qeo: torch.Tensor,
                          all_demands: torch.Tensor,
                          decouple=True
                          ):
        """
        Calculates the costs and inventory updates over a sample of multiple multi-step trajectories.

        Parameters
        ----------
        initial_inventories: torch.Tensor
            The initial inventories of shape math:`N \times 1`
        qr: torch.Tensor
            The trajectory of regular orders, of shape  math:`N \times T + L_r`
        qe: torch.Tensor
            The trajectory of expendited orders, of shape  math:`N \times T + L_e`

        all_demands: torch.Tensor
            Demands trajectory of shape math:`N \times T + 1`

        Returns
        -------


        """
        if decouple:
            initial_inventories = fractional_decoupling(initial_inventories) # I_0
            qra = fractional_decoupling(qra)
            qea = fractional_decoupling(qea)
            qro = fractional_decoupling(qro)
            qeo = fractional_decoupling(qeo)


        past_demands = all_demands[:, :-1]
        current_demands = all_demands[:, 1:] #D_0=0



        inventories =  qra + qea - past_demands

        I_t_prev = initial_inventories + inventories.cumsum(dim=-1)

        order_costs = self.ce * qeo + self.cr * qro

        unmatched_costs = self.h * torch.relu(I_t_prev - current_demands) + \
                          self.b * torch.relu(current_demands - I_t_prev)

        all_costs = order_costs + unmatched_costs

        # new cost update with fixed costs and identity straight-through positivity check
        has_fe_cost = binary_decoupling(qeo)
        has_fr_cost = binary_decoupling(qro)
        all_costs = all_costs + self.fe * has_fe_cost + self.fr * has_fr_cost
        #final_inventories = inventories[:, -1:] - all_demands[:, -1:]
        inventories = qra + qea - current_demands
        I_t = initial_inventories + inventories.cumsum(dim=-1)
        return all_costs, I_t

    def reset(self,
              minibatch_size=16,
              seed=None):

        if not seed is None:
            torch.manual_seed(seed)
        self.current_timestep = 0
        self.previous_qr = [torch.zeros([minibatch_size, 1])]
        self.previous_qe = [torch.zeros([minibatch_size, 1])]

        if self.lr > 0:
            self.previous_qr = self.previous_qr * self.lr

        if self.le > 0:
            self.previous_qe = self.previous_qe * self.le

        self.learned_I_0 = fractional_decoupling(self.I_0.repeat([minibatch_size, 1]))
        self.I_i = self.learned_I_0
        self.all_demands = [torch.zeros([minibatch_size, 1])]


def test_implementations(lr=3, le=2, N=2, T=5, I_0=10.0):
    N = N
    T = T

    sds = SequentialDualSourcingModel(lr=3, le=2)
    sds.I_0 = torch.tensor(I_0)

    sds.reset(N)
    inital_inventories = sds.I_0.repeat(N, 1)
    initial_demands = sds.all_demands[0]
    initial_qr = torch.cat(sds.previous_qr, dim=-1)
    initial_qe = torch.cat(sds.previous_qe, dim=-1)

    future_demands = torch.randint(low=1, high=4, size=(N, T))
    demands = torch.cat([initial_demands, future_demands], dim=-1)

    new_qe = torch.randn([N, T]).abs().round()
    new_qr = torch.randn([N, T]).abs().round()

    qr = torch.cat([initial_qr, new_qr], dim=-1) if sds.lr > 0 else new_qr
    qe = torch.cat([initial_qe, new_qe], dim=-1) if sds.le > 0 else new_qe

    qe_arrived = qe[:, :T]
    qr_arrived = qr[:, :T]

    qe_ordered = qe[:, sds.le:]
    qr_ordered = qr[:, sds.lr:]

    costs, invs = sds.replay_multisteps(inital_inventories,
                                        qra=qr_arrived,
                                        qea=qe_arrived,
                                        qro=qr_ordered,
                                        qeo=qe_ordered,
                                        all_demands=demands
                                        )

    all_costs = []
    all_invs = []
    sds.reset(N)
    previous_inventory = sds.I_0.repeat(N)

    for i in range(T):
        current_demand = demands[:, i + 1]
        qra = qr_arrived[:, i]
        qea = qe_arrived[:, i]
        qr_i = qr_ordered[:, i]
        qe_i = qe_ordered[:, i]
        cs, previous_inventory = sds.replay_step(
            previous_inventory,
            current_demand,
            qra=qra,
            qea=qea,
            qr=qr_i,
            qe=qe_i
        )
        all_invs.append(previous_inventory)
        all_costs.append(cs)
    step_wise_invs = torch.stack(all_invs, -1)[:, :]
    step_wise_costs = torch.stack(all_costs, -1)

    assert torch.all(step_wise_invs == invs)
    assert torch.all(step_wise_costs == costs)

if __name__ == '__main__':
    test_implementations()