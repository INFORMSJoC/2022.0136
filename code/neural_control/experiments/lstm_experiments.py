import json
import os
from logging import info

import pandas as pd
import torch
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import time

from plotly import graph_objs as go

from neural_control.demand_generators import FileBasedDemandGenerator
from neural_control.dynamics import fractional_decoupling, binary_decoupling, straight_through_relu
from neural_control.experiments.helpers import MSOMDemandLSTMExperiment, lstm_model_step, get_config


class LSTMTrainer:

    def __init__(self,
                 experiment_data_loader: MSOMDemandLSTMExperiment,
                 live_plot_losses: bool = False):
        self.experiment_data_loader = experiment_data_loader
        self.live_plot_losses = live_plot_losses
        if live_plot_losses:
            # use only in notebooks
            self.fig = go.FigureWidget()
            self.fig.add_scatter()
            self.fig.add_scatter()
            self.fig.data[0].name = 'Training Costs'
            self.fig.data[0].showlegend = True
            self.fig.data[1].name = 'Validation Costs'
            self.fig.data[1].showlegend = True
            self.fig.layout.xaxis.title.text = 'Epoch'
            self.fig.layout.yaxis.title.text = 'Cost'
            self.fig.show()

    def pretrain(self,
                 controller: torch.nn.Module,
                 optimizer_type="torch.optim.RMSprop",
                 lr=1e-3,
                 N=16,
                 n_epochs=10,
                 logging_frequency: int = 10,
                 resampling_freq: int = 1,
                 qe_qr_loss_ratio: float = 0.001,
                 use_progress: bool = True,
                 **state_sequence_params
                 ):
        optimizer = eval(optimizer_type)(controller.parameters(), lr=lr)
        nn_input, initial_qr, initial_qe, inital_inventories, demands, past_demands = \
            self.experiment_data_loader.generate_state_sequence(N=N,
                                                                **state_sequence_params
                                                                )
        info('Pretraining starts: ')

        progress = range(n_epochs)
        if use_progress:
            progress = tqdm(progress, desc='Pre-training Epochs')

        for i in progress:
            optimizer.zero_grad()
            new_qr, new_qe = controller(nn_input)
            new_qr = new_qr.roll(self.experiment_data_loader.sourcing_parameters['lr'], dims=1)
            new_qr[:, :self.experiment_data_loader.sourcing_parameters['lr']] = 0 * new_qr[:, :
                                                                                              self.experiment_data_loader.sourcing_parameters[
                                                                                                  'lr']]

            new_qe = new_qe.roll(self.experiment_data_loader.sourcing_parameters['le'], dims=1)
            new_qe[:, :self.experiment_data_loader.sourcing_parameters['le']] = 0 * new_qe[:, :
                                                                                              self.experiment_data_loader.sourcing_parameters[
                                                                                                  'le']]

            loss = ((new_qr - (past_demands - new_qe.detach())) ** 2).mean() + (
                    (new_qe - (past_demands - new_qr.detach())) ** 2).mean() * qe_qr_loss_ratio
            if i % logging_frequency == 0:
                info('Epoch: ' + str(i) + ' Training Loss: ' + str(loss))
            if i > 0 and i % resampling_freq == 0:
                nn_input, initial_qr, initial_qe, inital_inventories, demands, past_demands = \
                    self.experiment_data_loader.generate_state_sequence(N=N, **state_sequence_params)

            loss.backward()
            optimizer.step()
            if use_progress:
                progress.set_postfix({'Training Loss': np.round(loss.detach().item(), decimals=0)})

            new_qr, new_qe = controller(nn_input)
        return controller, new_qr, new_qe

    def train_model(self,
                    controller,
                    initial_minibatch_size: int = 128,
                    minibatch_size: int = 16,
                    n_epochs: int = 10000,
                    warmup_epochs: int = 3000,

                    resampling_frequency: int = 1000,
                    optimizer_reset_frequency: int = 1000,

                    reset_to_best_tolerance_rate: float = 2.0,
                    performance_hist_window: int = 5,

                    fig_update_frequency: int = 100,
                    optimizer_type="torch.optim.RMSprop",
                    optimizer_params={'lr': 1e-3},
                    use_progress: bool = True
                    ):
        dynamics = self.experiment_data_loader.dynamics
        T = self.experiment_data_loader.T
        training_costs = []
        val_costs = []
        best_val_loss = [np.infty]
        best_model = [deepcopy(controller.state_dict())]

        nn_input, initial_qr, initial_qe, initial_inventories, demands, past_demands = \
            self.experiment_data_loader.generate_state_sequence(N=initial_minibatch_size, **state_sequence_params)
        optimizer =eval(optimizer_type)(controller.parameters(), **optimizer_params)
        progress = range(n_epochs)
        if use_progress:
            progress = tqdm(progress, desc='Training Epochs')

        for i in progress:
            controller.train()
            optimizer.zero_grad()
            costs, invs, qr, qe = lstm_model_step(controller, nn_input, initial_qr, initial_qe, initial_inventories,
                                                  dynamics, demands, T)
            mean_costs = costs.mean()
            J = mean_costs
            # J = (costs[costs > costs.quantile(0.3, dim=0, keepdim=True)]).mean()  # + (0.5*costs.max(dim=0).values).mean()
            J.backward()
            optimizer.step()
            training_costs.append(costs.mean().item())
            val_cost, _, _, _, _, _, _, _ = self.experiment_data_loader.validate_model(controller,
                                                                                       **state_sequence_params
                                                                                       )
            val_costs.append(val_cost)

            if val_cost < best_val_loss[0]:
                best_val_loss[0] = val_cost
                best_model[0] = deepcopy(controller.state_dict())
            elif not np.isfinite(val_cost):
                controller.load_state_dict(best_model[0])

            if i % resampling_frequency and i > warmup_epochs:
                nn_input, initial_qr, initial_qe, initial_inventories, demands, past_demands = \
                    self.experiment_data_loader.generate_state_sequence(N=minibatch_size,  **state_sequence_params)

            if i % resampling_frequency and i > warmup_epochs:
                optimizer = eval(optimizer_type)(controller.parameters(), **optimizer_params)
                if np.mean(val_costs[-performance_hist_window:]) > reset_to_best_tolerance_rate * best_val_loss[0]:
                    controller.load_state_dict(best_model[0])

            if i % fig_update_frequency == 0 and self.live_plot_losses:
                time.sleep(0.001)
                with self.fig.batch_update():
                    self.fig.data[0].y = training_costs
                    self.fig.data[1].y = val_costs
            if use_progress:
                progress.set_postfix({'Validation Costs': np.round(val_cost, decimals=0)})

        return best_model[0], best_val_loss[0], pd.DataFrame(dict(train_cost=training_costs, val_cost=val_costs))

class SimpleLSTMController(torch.nn.Module):
    def __init__(self,
                 n_input_features=3,
                 hidden_size=256,
                 proj_size=250,
                 num_lstm_layers: int = 1,
                 lstm_dropout: float = 0.0
                 ):
        super().__init__()
        self.n_input_features = n_input_features
        self.input_layer = torch.nn.Linear(n_input_features, n_input_features)

        self.lstm_layer = torch.nn.LSTM(input_size=n_input_features,
                                        hidden_size=hidden_size,
                                        num_layers=num_lstm_layers,
                                        batch_first=True,
                                        bidirectional=False,
                                        dropout=lstm_dropout,
                                        proj_size=proj_size
                                        )
        self.projection_skip_layer = torch.nn.Linear(n_input_features, proj_size)
        self.output_layer = torch.nn.Linear(proj_size, 4)

    def forward(self, x):
        x = torch.nn.functional.relu(self.input_layer(x))
        h, (h_T, c_T) = self.lstm_layer(x)
        skip_h = self.projection_skip_layer(x)
        y = self.output_layer(torch.relu(h + skip_h))
        y_int = fractional_decoupling(y[:, :, [0, 1]])
        y_bin = binary_decoupling(y[:, :, [2, 3]])
        y = straight_through_relu(y_int) * y_bin

        return y[:, :, 0], y[:, :, 1]

def dump_to_json(dir, filename, dictionary):
    to_write = deepcopy(dictionary)
    with open(os.path.join(dir, filename), 'w') as f:
        json.dump(to_write,f, sort_keys=True, indent=4)

def get_parameters(results_folder):
    file_demand_gen_parameters = dict(demand_file_path='../../../data/MSOM_data/msom.2020.0933.csv',
                                      max_weeks=115,
                                      skus=("SKU-A-3",
                                            "SKU-B-3",
                                            "SKU-C-3",
                                            "SKU-E-3",
                                            "SKU-H-3",
                                            "SKU-I-3",
                                            "SKU-J-3",
                                            "SKU-N-3",
                                            "SKU-S-3",
                                            "SKU-T-3"
                                            ),
                                      scaling_factor=1e5,
                                      impute_val=0,
                                      myclip_a=0,
                                      myclip_b=1e3
                                      )
    dump_to_json(results_folder, 'file_demand_gen_parameters.json', file_demand_gen_parameters)
    lstm_controller_parameters = dict(
                n_input_features=3,
                 hidden_size=256,
                 proj_size=250,
                 num_lstm_layers = 1,
                 lstm_dropout = 0.0
    )
    dump_to_json(results_folder, 'lstm_controller_parameters.json', lstm_controller_parameters)

    sourcing_parameters = get_config(use_low_service=False)
    dump_to_json(results_folder, 'sourcing_parameters.json', sourcing_parameters)

    pretrain_parameters = dict(
        lr=1e-3,
        N=16,
        n_epochs=10,
        logging_frequency=10,
        resampling_freq=1,
        qe_qr_loss_ratio=0.001,
        optimizer_type="torch.optim.RMSprop",
    )
    dump_to_json(results_folder, 'pretrain_parameters.json', pretrain_parameters)

    train_parameters = dict(initial_minibatch_size=128,
                            minibatch_size=128,
                            n_epochs=5000,
                            warmup_epochs=5000,
                            resampling_frequency=1000,
                            optimizer_reset_frequency=1000,
                            reset_to_best_tolerance_rate=2.0,
                            performance_hist_window=5,

                            fig_update_frequency=100,
                            optimizer_type="torch.optim.RMSprop",
                            optimizer_params={'lr': 1e-3},
                            use_progress=True
                            )
    dump_to_json(results_folder, 'train_parameters.json', train_parameters)

    state_sequence_params = dict(use_mean=True, use_std=True, mean_shift=0, std_shift=0)
    dump_to_json(results_folder, 'state_sequence_params.json', state_sequence_params)

    return file_demand_gen_parameters, lstm_controller_parameters,sourcing_parameters,pretrain_parameters,train_parameters, state_sequence_params

if __name__ == '__main__':
    run_id = 'lstm_' + pd.Timestamp.today().strftime('%Y-%m-%d_%H-%M')
    n_trials = 10
    results_folder = os.path.join('lstm_results', run_id)
    os.makedirs(results_folder, exist_ok=True)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    file_demand_gen_parameters, lstm_controller_parameters, sourcing_parameters, pretrain_parameters, train_parameters,\
        state_sequence_params =\
    get_parameters(results_folder)

    file_demand_gen = FileBasedDemandGenerator(**file_demand_gen_parameters)

    initial_inventory: float = 0.0  # only for object construction, we use random initialization for inventories in

    msom_data_loader = MSOMDemandLSTMExperiment(file_demand_gen, sourcing_parameters, initial_inventory)
    trainer = LSTMTrainer(msom_data_loader, False)


    overall_best_model = None
    overall_best_loss = np.infty
    overall_learning_df = None

    for i in range(n_trials):
        qe_mean = 0
        qr_mean = 0
        while qe_mean < 1e3 and qr_mean < 1e4:
            controller = SimpleLSTMController(**lstm_controller_parameters)
            controller, qr, qe = trainer.pretrain(controller, **pretrain_parameters)
            qe_mean = qe.mean().detach().item()
            qr_mean = qe.mean().detach().item()
        current_best_model, current_best_loss, learning_df = trainer.train_model(controller,
                                                                    **train_parameters
                                                                    )
        if current_best_loss < overall_best_loss:
            overall_best_loss = current_best_loss
            overall_best_model = current_best_model
            overall_learning_df = learning_df

    torch.save(overall_best_model, os.path.join(results_folder, run_id + '.pt'))
    overall_learning_df.to_csv(os.path.join(results_folder, 'learning_curves.csv'))
    validation_controller = controller.load_state_dict(overall_best_model)
    mean_costs, median_costs, qr,  qe, invs, demands, initial_inventories, costs = msom_data_loader.validate_model(controller,
                                                                                                                   **state_sequence_params
                                                                                       )
    results = dict(
        mean_costs = mean_costs,
        median_costs = median_costs,
        qr = qr,
        qe = qe,
        invs = invs,
        demands = demands,
        initial_inventories = initial_inventories,
        costs = costs
    )
    torch.save(results,os.path.join(results_folder, 'validation_results.pt'))
