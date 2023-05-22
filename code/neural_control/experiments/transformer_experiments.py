import os

from tqdm.auto import tqdm

from neural_control.dynamics import SequentialDualSourcingModel, fractional_decoupling, binary_decoupling, \
    straight_through_relu
from neural_control.demand_generators import FileBasedDemandGenerator
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

import time
from plotly import graph_objects as go

from neural_control.experiments.helpers import roll_boundary, get_config

from collections import namedtuple

from neural_control.experiments.lstm_experiments import dump_to_json

SequentialState = namedtuple("SequentialState", [
    'initial_inventories',
    'initial_demands',
    'past_demands',
    'future_demands',
    'all_demands',
    'initial_qr',
    'initial_qe',
    'means',
    'stds'
])


class InputGenerator:
    def __init__(self,
                 dynamics,
                 demand_generator,
                 use_mean: bool,
                 use_std: bool,
                 mean_shift: int = 0,
                 std_shift: int = 0
                 ):
        self.dynamics = dynamics
        self.demand_generator = demand_generator
        self.use_mean = use_mean
        self.use_std = use_std
        self.mean_shift = mean_shift
        self.std_shift = std_shift

    def generate_nn_input(self,
                          state_tuple,
                          initial_guess_qr,
                          initial_guess_qe
                          ):
        inventory_state = -state_tuple.past_demands
        inventory_state[:, 0:1] = state_tuple.initial_inventories
        observation_list = [inventory_state, initial_guess_qr, initial_guess_qe]
        if self.use_mean:
            observation_list.append(state_tuple.means.squeeze(1))
            observation_list.append(state_tuple.stds.squeeze(1))

        nn_input = torch.stack(observation_list, dim=-1)
        return nn_input

    def generate_transformer_inputs(self,
                                    nn_input,
                                    enc_dec_split: int,
                                    dec_end: int = None
                                    ):
        sequence_token = torch.ones_like(nn_input[:, :1, :])
        encoder_input = torch.cat([sequence_token, nn_input[:, :enc_dec_split, :]], dim=1)
        decoder_input = torch.cat([sequence_token, nn_input[:, enc_dec_split:dec_end, :]], dim=1)
        return encoder_input, decoder_input


def generate_state(inventory_dynamics,
                   demand_generator,
                   N=16,
                   T=None,
                   mean_shift=1,
                   std_shift=1,
                   ):
    if T is None:
        T = demand_generator.max_weeks
    inventory_dynamics.reset(N)
    initial_inventories = torch.randint(low=0, high=100 + 1, size=(N, 1)).float()  # sds.I_0.repeat(N, 1)
    initial_demands = inventory_dynamics.all_demands[0]
    initial_qr = torch.cat(inventory_dynamics.previous_qr, dim=-1)
    initial_qe = torch.cat(inventory_dynamics.previous_qe, dim=-1)
    future_demands = demand_generator.sample_trajectory(t=0, n_samples=N, n_timesteps=T)
    all_demands = torch.cat([initial_demands, future_demands], dim=-1).float()
    past_demands = all_demands[:, :-1]

    means = roll_boundary(demand_generator.mean_vector, roll_dim=0, roll_shift=mean_shift)
    means = means[:T].unsqueeze(0).repeat(N, 1, 1)
    stds = roll_boundary(demand_generator.std_vector, roll_dim=0, roll_shift=std_shift)
    stds = stds[:T].unsqueeze(0).repeat(N, 1, 1)

    return SequentialState(
        initial_inventories,
        initial_demands,
        past_demands,
        future_demands,
        all_demands,
        initial_qr,
        initial_qe,
        means,
        stds
    )


def get_random_qr_qe(past_demands, lr, le):
    qr_part = torch.rand_like(past_demands)
    qr_arrived = fractional_decoupling(torch.relu(past_demands * qr_part))
    qr = roll_boundary(qr_arrived, roll_dim=1, roll_shift=-lr)
    qe_arrived = fractional_decoupling(torch.relu(past_demands * (1 - qr_part)))
    qe = roll_boundary(qe_arrived, roll_dim=1, roll_shift=-le)
    return qr, qe


def unroll_transformer(transformer_controller, state_tuple, input_generator, preserve_causality=True):
    N = state_tuple.past_demands.shape[0]
    T = state_tuple.past_demands.shape[1]
    new_qr = torch.zeros(N, T)
    new_qe = torch.zeros(N, T)
    nn_input = input_generator.generate_nn_input(state_tuple, new_qr, new_qe)

    for i in range(T):
        enc_dec_split = i
        dec_end = i + 1
        encoder_input, decoder_input = input_generator.generate_transformer_inputs(nn_input, enc_dec_split,
                                                                                   dec_end=dec_end)
        dec_qr, dec_qe = transformer_controller(encoder_input, decoder_input, use_subsequent_mask=preserve_causality)
        new_qr[:, i] = dec_qr[:, -1]
        new_qe[:, i] = dec_qe[:, -1]
    return new_qr, new_qe


def unroll_transformer_w_teacher(transformer_controller, state_tuple, input_generator, preserve_causality=True):
    N = state_tuple.past_demands.shape[0]
    T = state_tuple.past_demands.shape[1]
    new_qr = torch.zeros(N, T)
    new_qe = torch.zeros(N, T)
    nn_input = input_generator.generate_nn_input(state_tuple, new_qr, new_qe)

    for i in range(T):
        enc_dec_split = i
        dec_end = None
        encoder_input, decoder_input = input_generator.generate_transformer_inputs(nn_input, enc_dec_split,
                                                                                   dec_end=dec_end)
        dec_qr, dec_qe = transformer_controller(encoder_input, decoder_input, use_subsequent_mask=preserve_causality)
        # new_qr[:, i] = dec_qr[:, (T-i)]
        # new_qe[:, i] = dec_qe[:, (T-i)]
        new_qr[:, i:] = dec_qr[:, 1:]
        new_qe[:, i:] = dec_qe[:, 1:]
    return new_qr, new_qe


def validate_model(transformer_controller,
                   dynamics,
                   demand_generator,
                   input_generator,
                   val_N=1000,
                   test_T=None
                   ):
    with torch.no_grad():
        transformer_controller.eval()
        dynamics.reset(val_N)

        if test_T is None:
            test_T = demand_generator.max_weeks

        state_tuple = generate_state(dynamics, demand_generator, N=val_N)
        new_qr, new_qe = unroll_transformer(transformer_controller, state_tuple, input_generator,
                                            preserve_causality=True)

        qr = torch.cat([state_tuple.initial_qr, new_qr], dim=-1) if dynamics.lr > 0 else new_qr
        qe = torch.cat([state_tuple.initial_qe, new_qe], dim=-1) if dynamics.le > 0 else new_qe

        qe_arrived = qe[:, :test_T]
        qr_arrived = qr[:, :test_T]

        qe_ordered = qe[:, dynamics.le:]
        qr_ordered = qr[:, dynamics.lr:]

        costs, invs = dynamics.replay_multisteps(state_tuple.initial_inventories,
                                                 qra=qr_arrived,
                                                 qea=qe_arrived,
                                                 qro=qr_ordered,
                                                 qeo=qe_ordered,
                                                 all_demands=state_tuple.all_demands
                                                 )
        mean_costs = costs.mean().item()
        median_costs = costs.median().item()

        return (mean_costs,
                median_costs,
                qr.cpu().detach().numpy(),
                qe.cpu().detach().numpy(),
                invs.cpu().detach().numpy(),
                state_tuple.all_demands.cpu().detach().numpy(),
                state_tuple.initial_inventories.cpu().detach().numpy()
                )


class SimpleTransformerController(torch.nn.Module):
    def __init__(self,
                 n_input_features=3,
                 input_hidden_layer_sizes=[16, 16, 16],
                 output_hidden_layer_sizes=[16],
                 transformer_heads=8,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 transformer_hidden_dimensions=16,
                 transformer_activation=torch.nn.functional.relu,
                 transformer_dropout=0.0
                 ):
        super().__init__()
        self.n_input_features = n_input_features
        self.input_hidden_layers = []
        prev_input_size = n_input_features
        for ihs in input_hidden_layer_sizes:
            self.input_hidden_layers.append(torch.nn.Linear(prev_input_size, ihs))
            self.input_hidden_layers.append(torch.nn.ReLU())
            prev_input_size = ihs

        if isinstance(transformer_activation, str):
            self.transformer_activation = eval(transformer_activation)
        else:
            self.transformer_activation = transformer_activation

        self.input_network = torch.nn.Sequential(
            *self.input_hidden_layers
        )
        self.transformer_block = torch.nn.Transformer(d_model=prev_input_size,
                                                      nhead=transformer_heads,
                                                      num_encoder_layers=num_encoder_layers,
                                                      num_decoder_layers=num_decoder_layers,
                                                      dim_feedforward=transformer_hidden_dimensions,
                                                      dropout=transformer_dropout,
                                                      activation=self.transformer_activation,
                                                      batch_first=True
                                                      )
        self.output_hidden_layers = []
        for ihs in output_hidden_layer_sizes:
            self.output_hidden_layers.append(torch.nn.Linear(prev_input_size, ihs))
            self.output_hidden_layers.append(torch.nn.ReLU())
            prev_input_size = ihs
        self.output_network = torch.nn.Sequential(
            *self.output_hidden_layers
        )
    def forward(self, x_enc, x_dec, use_subsequent_mask=True):
        x_enc_t = self.input_network(x_enc)
        x_dec_t = self.input_network(x_dec)
        subsequent_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x_dec.shape[1]) if use_subsequent_mask else None
        h = self.transformer_block(x_enc_t, x_dec_t, tgt_mask=subsequent_mask)
        y = self.output_network(torch.relu(h*x_dec_t + x_dec_t.detach()))
        y = fractional_decoupling(y)
        y = straight_through_relu(y)

        return y[:, :, 0], y[:, :, 1]


def test_mask_causality(dynamics, demand_generator):
    tt = torch.nn.Transformer(d_model=3,
                              nhead=1,
                              num_encoder_layers=1,
                              num_decoder_layers=1,
                              dim_feedforward=52,
                              dropout=0.0,
                              activation=torch.nn.functional.relu,
                              batch_first=True
                              )

    tt = tt.double()

    state_tuple = generate_state(dynamics, demand_generator)
    inventory_state = -state_tuple.past_demands
    inventory_state[:, 0:1] = state_tuple.initial_inventories
    inventory_state = inventory_state

    initial_guess_qr, initial_guess_qe = get_random_qr_qe(state_tuple.past_demands)

    nn_input = torch.stack([inventory_state, initial_guess_qr, initial_guess_qe], dim=-1).double()

    encoder_input = nn_input[:, :10, :]
    decoder_input = nn_input[:, 10:, :]
    mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1])
    mask2 = torch.nn.Transformer.generate_square_subsequent_mask(4)

    tt.eval()
    a = tt.forward(encoder_input, decoder_input, tgt_mask=mask)
    b = tt.forward(encoder_input, decoder_input[:, :4], tgt_mask=mask2)
    # causality test
    assert torch.allclose(a[:, :4], b[:, :4])

class TransformerTrainer:
    def __init__(self,
                 dynamics,
                 demand_generator,
                 input_generator,
                 optimizer_class=torch.optim.RMSprop,
                 use_live_training_figure = False,
                 figure_update_frequency=10
                 ):
        self.dynamics = dynamics
        self.demand_generator = demand_generator
        self.input_generator = input_generator
        if isinstance(optimizer_class, str):
            self.optimizer_class = eval(optimizer_class)
        else:
            self.optimizer_class = optimizer_class

        self.use_live_training_figure = use_live_training_figure
        if use_live_training_figure:
            self.fig = go.FigureWidget()
            self.fig.add_scatter()
            self.fig.add_scatter()
        self.figure_update_frequency=figure_update_frequency

        self.training_costs = []
        self.val_costs = []

    def train_step(self, controller, optimizer, state_tuple):
        controller.train()
        optimizer.zero_grad()

        fnew_qr, fnew_qe = unroll_transformer(controller, state_tuple, self.input_generator, preserve_causality=True)

        qr = torch.cat([state_tuple.initial_qr, fnew_qr], dim=-1) if self.dynamics.lr > 0 else fnew_qr
        qe = torch.cat([state_tuple.initial_qe, fnew_qe], dim=-1) if self.dynamics.le > 0 else fnew_qe

        qe_arrived = qe[:, :self.demand_generator.max_weeks]
        qr_arrived = qr[:, :self.demand_generator.max_weeks]

        qe_ordered = qe[:, self.dynamics.le:]
        qr_ordered = qr[:, self.dynamics.lr:]

        costs, invs = self.dynamics.replay_multisteps(state_tuple.initial_inventories,
                                            qra=qr_arrived,
                                            qea=qe_arrived,
                                            qro=qr_ordered,
                                            qeo=qe_ordered,
                                            all_demands=state_tuple.all_demands
                                            )
        J = costs.mean()
        J.backward()
        optimizer.step()

        return J.item()


    def train(self, controller, minibatch_size, total_epochs,
              no_validation_period, learning_rate, optimizer_reset_frequency, val_frequency):


        optimizer = self.optimizer_class(controller.parameters(), lr=learning_rate)
        state_tuple = generate_state(self.dynamics, self.demand_generator, N=minibatch_size)
        best_mean_training_cost = np.infty

        best_val_loss = [np.infty]
        best_model = [deepcopy(controller.state_dict())]

        val_cost = np.infty
        progress = tqdm(range(total_epochs))
        progress.set_description('Training Epochs: ')
        for i in progress:
            mean_training_cost = self.train_step(controller, optimizer, state_tuple)
            self.training_costs.append(mean_training_cost)

            if (mean_training_cost < best_mean_training_cost and i > no_validation_period) or i%val_frequency==0:
                best_mean_training_cost = mean_training_cost
                val_cost, _, _, _, _, _, _ = validate_model(controller, self.dynamics, self.demand_generator, self.input_generator)
                self.val_costs.append(val_cost)
            else:
                self.val_costs.append(val_cost)


            if val_cost < best_val_loss[0]:
                best_val_loss[0] = val_cost
                best_model[0] = deepcopy(controller.state_dict())

            if i % optimizer_reset_frequency == 0 and i > 0:
                lr = learning_rate
                optimizer = self.optimizer_class(controller.parameters(), lr=lr)

            if self.use_live_training_figure  and i % self.figure_update_frequency == 0:
                time.sleep(0.01)
                with self.fig.batch_update():
                    self.fig.data[0].y = self.training_costs
                    self.fig.data[1].y = self.val_costs
            progress.set_postfix({'best_val_cost' : best_val_loss[0]})
        return best_val_loss[0], best_model[0]


def get_parameters(results_folder,
                   use_low_service=True,
                   use_mean_std=False,
                   mean_std_shift=1,
                   ):
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
    transformer_controller_parameters = dict(n_input_features=3 + [0,2][use_mean_std],
                                             input_hidden_layer_sizes=[16, 16, 16],
                                             output_hidden_layer_sizes=[16],
                                             transformer_heads=8,
                                             num_encoder_layers=1,
                                             num_decoder_layers=1,
                                             transformer_hidden_dimensions=64,
                                             transformer_activation='torch.nn.functional.relu',
                                             transformer_dropout=0.0
                                             )

    dump_to_json(results_folder, 'transformer_controller_parameters.json', transformer_controller_parameters)

    sourcing_parameters = get_config(use_low_service=use_low_service)
    dump_to_json(results_folder, 'sourcing_parameters.json', sourcing_parameters)

    train_parameters = dict(optimizer_type="torch.optim.RMSprop",
                            minibatch_size = 32,
                            total_epochs = 1500,
                            no_validation_period = 100,
                            learning_rate = 7e-4,
                            optimizer_reset_frequency = 50,
                            val_frequency = 50,
                            pretrain_minibatch_size=32,
                            pretrain_total_epochs = 200,
                            pretrain_no_validation_period = 150,
                            pretrain_learning_rate = 7e-4,
                            pretrain_optimizer_reset_frequency = 50,
                            pretrain_val_frequency = 50,
                            pretrain_samples=10
                            )
    dump_to_json(results_folder, 'train_parameters.json', train_parameters)

    state_sequence_params = dict(use_mean=use_mean_std, use_std=use_mean_std, mean_shift=mean_std_shift, std_shift=mean_std_shift)
    dump_to_json(results_folder, 'state_sequence_params.json', state_sequence_params)

    return file_demand_gen_parameters, transformer_controller_parameters, sourcing_parameters, train_parameters, state_sequence_params

def run_experiment(is_low_service=True, use_mean_std=False, mean_std_shift=1):
    experiment_id = 'transformer_' + pd.Timestamp.today().strftime('%Y-%m-%d_%H-%M')

    experiment_id += ['_high','_low'][is_low_service]+"_sevice_"
    if use_mean_std:
        experiment_id+=['prev_','future_'][mean_std_shift<=0]
    experiment_id += ['nomeanstd', 'withmeanstd'][use_mean_std]
    results_folder = os.path.join('transformer_results', experiment_id)
    os.makedirs(results_folder, exist_ok=True)
    file_demand_gen_parameters, transformer_controller_parameters, sourcing_parameters, train_parameters, \
    state_sequence_params = get_parameters(results_folder,
                   use_low_service=is_low_service,
                   use_mean_std=use_mean_std,
                   mean_std_shift=mean_std_shift,
                   )
    demand_generator = FileBasedDemandGenerator(**file_demand_gen_parameters)
    dynamics = SequentialDualSourcingModel(**sourcing_parameters)
    dynamics.I_0 = torch.tensor(0.0)
    input_generator = InputGenerator(dynamics, demand_generator, use_mean=use_mean_std, use_std=use_mean_std,
                                     mean_shift=mean_std_shift, std_shift=mean_std_shift)
    overall_best_loss = [np.infty]
    overall_best_model = [None]
    overall_best_models_learning_curves = {'pretrain_train_cost' : None, 'pretrain_val_cost' :  None}

    for i in range(train_parameters['pretrain_samples']):
        controller = SimpleTransformerController(**transformer_controller_parameters)
        trainer = TransformerTrainer(dynamics, demand_generator, input_generator, train_parameters['optimizer_type'], False)
        best_loss, best_model = trainer.train(controller,
                                              minibatch_size=train_parameters['pretrain_minibatch_size'],
                                              total_epochs=train_parameters['pretrain_total_epochs'],
                                              no_validation_period=train_parameters['pretrain_no_validation_period'],
                                              learning_rate=train_parameters['pretrain_learning_rate'],
                                              optimizer_reset_frequency=train_parameters['pretrain_optimizer_reset_frequency'],
                                              val_frequency=train_parameters['pretrain_val_frequency']
                                              )
        if best_loss < overall_best_loss[0]:
            overall_best_loss[0] = best_loss
            overall_best_model[0] = best_model
            overall_best_models_learning_curves['pretrain_train_cost'] = trainer.training_costs
            overall_best_models_learning_curves['pretrain_val_cost'] = trainer.val_costs


    controller.load_state_dict(overall_best_model[0])
    pd.DataFrame(overall_best_models_learning_curves).to_csv(os.path.join(results_folder,'pretraining_curves.csv'))

    trainer = TransformerTrainer(dynamics, demand_generator, input_generator, train_parameters['optimizer_type'], False)
    best_loss, best_model = trainer.train(controller,
                                          minibatch_size=train_parameters['minibatch_size'],
                                          total_epochs=train_parameters['total_epochs'],
                                          no_validation_period=train_parameters['no_validation_period'],
                                          learning_rate=train_parameters['learning_rate'],
                                          optimizer_reset_frequency=train_parameters['optimizer_reset_frequency'],
                                          val_frequency=train_parameters['val_frequency']
                                          )

    fine_tune_learning_curves = dict(fine_tune_train_cost=trainer.training_costs,
                                     fine_tune_val_cost=trainer.val_costs,
                                     )
    pd.DataFrame(fine_tune_learning_curves).to_csv(os.path.join(results_folder,'fine_tune_learning_curves.csv'))

    controller.load_state_dict(best_model)
    torch.save(best_model, os.path.join(results_folder,'transformer.pt'))
    test_cost, test_median_cost, new_qr, new_qe, invs, demands, I_0 = validate_model(controller,
                                                                                     dynamics,
                                                                                     demand_generator,
                                                                                     input_generator,
                                                                                     val_N=1000
                                                                                     )
    validation_resuts = dict(
        test_mean_cost = test_cost,
        test_median_cost = test_median_cost,
        new_qr = new_qr,
        new_qe = new_qe,
        invs = invs,
        demands = demands,
        I_0 = I_0
    )
    torch.save(validation_resuts, os.path.join(results_folder,'validation_results.pt'))


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # No means or stds...
    run_experiment(is_low_service=True, use_mean_std=False, mean_std_shift=0)
    run_experiment(is_low_service=False, use_mean_std=False, mean_std_shift=0)

    # Past means or stds...
    run_experiment(is_low_service=True, use_mean_std=True, mean_std_shift=1)
    run_experiment(is_low_service=False, use_mean_std=True, mean_std_shift=1)

    # Future means or stds
    run_experiment(is_low_service=True, use_mean_std=True, mean_std_shift=0)
    run_experiment(is_low_service=False, use_mean_std=True, mean_std_shift=0)
