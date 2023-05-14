import torch

from neural_control.controllers import DualSourcingController

class FCController(torch.nn.Module):
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

        input_layer = torch.nn.Linear(in_features = lr + le + 1, #+ 1 + 1, # ones for demands, invs, costs
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
                past_demands,
                past_invetories,
                past_regular_orders,
                past_expedited_orders,
                past_costs
                ):

        observation_list = [
            #past_demands[:, -1:, :],
            past_invetories[:, -1:, :],
            #past_costs[:, -1:, :]
        ]

        if self.lr > 0:
            if past_regular_orders.shape[1] == 0:
                past_regular_orders = torch.zeros([past_regular_orders.shape[0], self.lr, 1])
            observation_list.append(past_regular_orders[:, -self.lr:, :])
        if self.le > 0:
            if past_expedited_orders.shape[1] == 0:
                past_expedited_orders = torch.zeros([past_demands.shape[0], self.le, 1])
            observation_list.append(past_expedited_orders[:, -self.le:, :])


        observation = torch.cat(observation_list, dim=1).squeeze(-1)
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
        return qr, qe, None


class LSTMController(DualSourcingController):

    def __init__(self,
                 lr,
                 le,
                 hidden_size=6,
                 num_layers=1,
                 allow_rounding_correction=True
                 ):
        super().__init__()
        self.lr = lr
        self.le = le
        self.n_hidden_units = None
        self.allow_rounding_correction = allow_rounding_correction

        self.input_size = (self.lr > 0) + (self.le > 0) + 1 # regular order, expedited order, current inventory

        self.lstm_layer = torch.nn.LSTM(input_size=self.input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        batch_first=True
                                        )
        self.output_layer = torch.nn.Linear(hidden_size, 2)



    def forward(self,
                current_demand,
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                time = 0,
                h_previous = None,
                c_previous = None
                ):
        batch_size = current_inventory.shape[0]
        max_history = max(current_inventory.shape[1],
                          past_regular_orders.shape[1],
                          past_expedited_orders.shape[1]
                          )

        observations = torch.zeros([batch_size, max_history, self.input_size])

        observations[:, -current_inventory.shape[1]:, 0:1] = current_inventory
        if self.lr > 0:
            observations[:, -past_regular_orders.shape[1]:, 1:2] = past_regular_orders
        if self.le > 0:
            observations[:, -past_expedited_orders.shape[1]:, 2:3] = past_expedited_orders

        h = observations

        nn_context=None
        if not h_previous is None and not c_previous is None:
            nn_context = (h_previous, c_previous)

        h, (hn, cn) = self.lstm_layer(h, nn_context)

        h = self.output_layer(h[:, -1, :])

        h = torch.relu(h)

        if self.allow_rounding_correction:
            h = h - torch.frac(h).clone().detach()

        qr = h[:, 0].unsqueeze(-1)
        qe = h[:, 1].unsqueeze(-1)
        return qr, qe, dict(h_previous=hn, c_previous=cn)

