import torch
import numpy as np

class DualSourcingModel(torch.nn.Module):
    def __init__(self,
                 controller: torch.nn.Module,
                 ce=20,
                 cr=0,
                 le=0,
                 lr=2,
                 h=5,
                 b=495,
                 T=50,
                 I_0=0,
                 learn_I_0=True,
                 demand_generator=torch.distributions.Uniform(low=0, high=4 + 1),
                 demand_distribution=[-1],
                 empirical_baseline=1
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
            self.demand_generator = lambda i,num_samples: \
                torch.from_numpy(demand_distribution[0](i,num_samples)).int().unsqueeze(-1)
        
        self.controller = controller

    def simulate(self,mean=0,std=0):

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
                min_demands = [max(0,mean[i]-2.58*std[i]) for i in range(len(mean))]
                max_demands = [mean[i]+2.58*std[i] for i in range(len(mean))]
                qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe, min_demands, max_demands)
            else:
                self.demand_arr[:len(self.all_demands)] = self.all_demands
                qr, qe = self.controller(D, self.I_i, self.previous_qr, self.previous_qe, self.demand_arr, mean, std)

        # orders are added to corresponding vectors
        self.previous_qr.append(qr)
        self.previous_qe.append(qe)

        # orders arrive
        qra = self.previous_qr[-self.lr-1]
        qea = self.previous_qe[-self.le-1]

        # demand is generated
        if self.demand_flag == -1:
            D = self.demand_generator.sample(
                [sample_size, 1]).int() # here we round a continuous sample
            self.all_demands.append(D)
        else:
            D = self.demand_generator(self.current_timestep,sample_size)
            self.all_demands.append(D)

        # inventory and cost updates
        self.I_i = self.I_i + qra + qea

        c_i = self.ce * qe + self.cr * qr + self.h * torch.relu(self.I_i - D) \
              + self.b * torch.relu(D - self.I_i)
              
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
            self.previous_qr = self.previous_qr  * self.lr

        if self.le > 0:
            self.previous_qe = self.previous_qe * self.le

        self.learned_I_0 =  self.I_0.repeat([minibatch_size, 1]) - torch.frac(self.I_0).clone().detach()
        self.I_i = self.learned_I_0
        self.all_demands = [torch.zeros([minibatch_size, 1])]
        
        self.demand_arr = 0
        if self.empirical_baseline == 3:
            self.demand_arr = [-1.0*torch.ones([minibatch_size, 1]) for i in range(self.T)]
        
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
        qa = self.previous_q[self.current_timestep-self.l-1]
	
        # demand is generated
        D = self.demand_generator.sample(
            [sample_size, 1]).int() # here we round a continuous sample
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
            self.previous_q = self.previous_q  * self.l

        self.learned_I_0 =  self.I_0.repeat([minibatch_size, 1]) - torch.frac(self.I_0).clone().detach()
        self.I_i = self.learned_I_0
        self.all_demands = [torch.zeros([minibatch_size, 1])]

