import torch
import abc

def imitatation_learn(controller,
                      lr,
                      le,
                      optimizer,
                      state_trajectories, # shape: minibatch_size x T + 1
                      qr_trajectories,    # shape: minibatch_size x T + 1 + lr
                      qe_trajectories,    # shape: minibatch_size x T + 1 + le
                      minibatch_size=256,
                      epochs=1000
                      ):
    all_losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        minibatch_indices = torch.randint(low=0,
                                          high=state_trajectories.shape[0],
                                          size=[minibatch_size]
                                          )

        end_time_indices = torch.randint(low=1, # because of provided trajectory being T+1 steps
                                         high=state_trajectories.shape[1],
                                         size=[minibatch_size]
                                         )
        ds_inv    = state_trajectories[minibatch_indices, end_time_indices, 0].unsqueeze(-1)
        ds_demand = state_trajectories[minibatch_indices, end_time_indices, 1].unsqueeze(-1)

        qr_steps = end_time_indices.unsqueeze(-1) - torch.arange(lr, 0, step=-1).unsqueeze(-0) - 1
        ds_past_qr = qr_trajectories[minibatch_indices.unsqueeze(-1), qr_steps]
        qe_steps = end_time_indices.unsqueeze(-1) - torch.arange(le, 0, step=-1).unsqueeze(-0) - 1
        ds_past_qe = qe_trajectories[minibatch_indices.unsqueeze(-1), qe_steps]
        target_qr  = qr_trajectories[minibatch_indices, end_time_indices]
        target_qe  = qe_trajectories[minibatch_indices, end_time_indices]
        Q = controller(ds_inv, ds_demand, ds_past_qr, ds_past_qe)
        loss = ((torch.stack(Q, -1) - torch.stack([target_qr, target_qe], -1)) ** 2).mean()
        loss.backward()
        all_losses.append(loss.cpu().detach().item())
        optimizer.step()
    return all_losses


class DualSourcingController(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                time = 0
                ):
        pass

class DualFullyConnectedRegressionController(DualSourcingController):
    def __init__(self,
                 lr,
                 le,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.lr = lr
        self.le = le

        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=2)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=lr + le + 1,
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders
                ):
        observation_list = [
            current_inventory
        ]
        if isinstance(past_regular_orders, list):
            reg_order_obs = torch.cat(past_regular_orders[-self.lr:], dim=-1)
        else:
            reg_order_obs = past_regular_orders[:,-self.lr:]
        observation_list.append(reg_order_obs)

        if isinstance(past_expedited_orders, list):
            exp_order_obs = torch.cat(past_expedited_orders[-self.le:], dim=-1)
        else:
            exp_order_obs = past_expedited_orders[:,-self.le:]
        
        if self.le > 0:
            observation_list.append(exp_order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                h = torch.relu(h)  # we need the outputs to always be positive
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe


class DualFullyConnectedRegressionControllerCompressed(DualSourcingController):
    def __init__(self,
                 lr,
                 le,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.lr = lr
        self.le = le

        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=2)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=lr + le,
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders
                ):
        observation_list = []
      
        if isinstance(past_regular_orders, list):
            reg_order_obs = torch.cat(past_regular_orders[-self.lr:], dim=-1)
        else:
            reg_order_obs = past_regular_orders[:,-self.lr:]
        reg_order_obs[:,0] += current_inventory.flatten()
        observation_list.append(reg_order_obs)

        if isinstance(past_expedited_orders, list):
            exp_order_obs = torch.cat(past_expedited_orders[-self.le:], dim=-1)
        else:
            exp_order_obs = past_expedited_orders[:,-self.le:]
        
        if self.le > 0:
            observation_list.append(exp_order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                #h = torch.relu(h)# we need the outputs to always be positive
                h = h - h.detach()*(h <= 0)
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe

class DualFullyConnectedRegressionControllerTime(DualSourcingController):
    def __init__(self,
                 lr,
                 le,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.lr = lr
        self.le = le

        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=2)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=lr + le + 2,
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                mean = 0,
                std = 0,
                ):
        observation_list = []
        #    current_inventory
        #]
        
        observation_list.append(mean*torch.ones_like(current_inventory,requires_grad=True))
        observation_list.append(std*torch.ones_like(current_inventory,requires_grad=True))
        
        if isinstance(past_regular_orders, list):
            reg_order_obs = torch.cat(past_regular_orders[-self.lr:], dim=-1)
        else:
            reg_order_obs = past_regular_orders[:,-self.lr:]
        reg_order_obs[:,0] += current_inventory.flatten()
        observation_list.append(reg_order_obs)

        if isinstance(past_expedited_orders, list):
            exp_order_obs = torch.cat(past_expedited_orders[-self.le:], dim=-1)
        else:
            exp_order_obs = past_expedited_orders[:,-self.le:]
        
        if self.le > 0:
            observation_list.append(exp_order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                h = torch.relu(h)  # we need the outputs to always be positive
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe

class DualFullyConnectedRegressionControllerTime2(DualSourcingController):
    def __init__(self,
                 lr,
                 le,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.lr = lr
        self.le = le

        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=2)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=lr + le + 2*(lr-le),
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                min_demands = 0,
                max_demands = 0,
                ):
        observation_list = []
        #    current_inventory
        #]
        
        min_d = torch.tensor([])
        max_d = torch.tensor([])
        for i in range(len(min_demands)):
            min_d = torch.hstack((min_d,min_demands[i]*torch.ones_like(current_inventory,requires_grad=True)))
            max_d = torch.hstack((max_d,max_demands[i]*torch.ones_like(current_inventory,requires_grad=True)))
        
        observation_list.append(min_d)
        observation_list.append(max_d)

        if isinstance(past_regular_orders, list):
            reg_order_obs = torch.cat(past_regular_orders[-self.lr:], dim=-1)
        else:
            reg_order_obs = past_regular_orders[:,-self.lr:]
        reg_order_obs[:,0] += current_inventory.flatten()
        observation_list.append(reg_order_obs)

        if isinstance(past_expedited_orders, list):
            exp_order_obs = torch.cat(past_expedited_orders[-self.le:], dim=-1)
        else:
            exp_order_obs = past_expedited_orders[:,-self.le:]
        
        if self.le > 0:
            observation_list.append(exp_order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                h = torch.relu(h)  # we need the outputs to always be positive
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe

class DualFullyConnectedRegressionControllerTime3(DualSourcingController):
    def __init__(self,
                 lr,
                 le,
                 T,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.lr = lr
        self.le = le

        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=2)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=lr + le + T + 2,
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                demand_data,
                mean = 0,
                std = 0
                ):
        observation_list = []
        #    current_inventory
        #]

        observation_list.append(mean*torch.ones_like(current_inventory,requires_grad=True))
        observation_list.append(std*torch.ones_like(current_inventory,requires_grad=True))
                
        if isinstance(demand_data, list):
            demands = torch.cat(demand_data, dim=-1)
            observation_list.append(demands)
        else:
            observation_list.append(demand_data)
        
        if isinstance(past_regular_orders, list):
            reg_order_obs = torch.cat(past_regular_orders[-self.lr:], dim=-1)
        else:
            reg_order_obs = past_regular_orders[:,-self.lr:]
        reg_order_obs[:,0] += current_inventory.flatten()
        observation_list.append(reg_order_obs)

        if isinstance(past_expedited_orders, list):
            exp_order_obs = torch.cat(past_expedited_orders[-self.le:], dim=-1)
        else:
            exp_order_obs = past_expedited_orders[:,-self.le:]
        
        if self.le > 0:
            observation_list.append(exp_order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                h = torch.relu(h)  # we need the outputs to always be positive
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe
                   
class SingleSourcingController(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self,
                current_demand,
                current_inventory,
                past_orders
                ):
        pass

class SingleFullyConnectedRegressionController(SingleSourcingController):
    def __init__(self,
                 l,
                 n_hidden_units=None,
                 activations=None,
                 initializations=None,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        is_shallow = (n_hidden_units is None)
        out_features_l0 = 1
        self.layers = []

        self.l = l
        
        if not is_shallow:
            out_features_l0 = n_hidden_units[0]
            for i in range(len(n_hidden_units) - 1):
                intermediate_layer = torch.nn.Linear(in_features=n_hidden_units[i],
                                                     out_features=n_hidden_units[i + 1]
                                                     )
                self.layers.append(intermediate_layer)
            output_layer = torch.nn.Linear(in_features=n_hidden_units[-1], out_features=1, bias=False)
            self.layers.append(output_layer)

        input_layer = torch.nn.Linear(in_features=l + 1,
                                      out_features=out_features_l0)
        self.layers.insert(0, input_layer)
        self.layers = torch.nn.ModuleList(self.layers)

        if activations is not None:
            self.activations = activations
        elif isinstance(n_hidden_units, list):
            self.activations = [torch.relu] * (len(n_hidden_units) + 1)
        else:
            self.activations = [torch.relu]

        self.initializations = initializations
        if self.initializations is not None:
            for layer in self.layers:
                for j, name, param in enumerate(layer.named_parameters()):
                    if not ('bias' in name or (torch.tensor(param.shape) < 2).all()):
                        self.initializations[j](param)
        self.allow_rounding_correction = allow_rounding_correction

    def forward(self,
                current_demand,
                current_inventory,
                past_orders,
                ):
        observation_list = [
            current_inventory
        ]
        if isinstance(past_orders, list):
            order_obs = torch.cat(past_orders[-self.l:], dim=-1)
        else:
            order_obs = past_orders[:,-self.l:]
        
        if self.l > 0:
            observation_list.append(order_obs)
        
        observation = torch.cat(observation_list, dim=-1)
        h = observation
        for j, layer in enumerate(self.layers):
            h = layer(h)
            if j < len(self.layers) - 1:
                h = self.activations[j](h)
            else:
                h = torch.relu(h)  # we need the outputs to always be positive
        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()#torch.round(h)
            
        q = h[:, 0].unsqueeze(-1)
        return q
    
