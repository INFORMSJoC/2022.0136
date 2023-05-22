from abc import abstractmethod

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import truncnorm


class AbstractDemandGenerator:

    @abstractmethod
    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:
        """
        Generates a demand sample from the provided generator.

        Parameters
        ----------
        t: int
            The timestep to generate samples for. This will be used for time-dependent demand generators.
        n_samples: int
            The number of samples :math:`N` to generate.

        Returns
        -------
        D: torch.Tensor
            As we assume our dynamics are sampled in mini-batches for training,
            then this method will generate a column  vector of size :math:`N \times 1`.

        """
        pass

    @abstractmethod
    def sample_trajectory(self, t: int, n_samples: int, n_timesteps: int):
        """
        Samples a whole demand trajectory, to speed up training and loading.

        Parameters
        ----------
        t: int
            Start time to sample from, often 0.
        n_samples: int
            Number of samples to generate.
        n_timesteps: int
            Number of timesteps in trajectory.

        Returns
        -------
        D-traj: torch.Tensor
            The sampled trajectories tensor, which has shape :math:`N \times T \times 1`
        """
        pass


class TorchDistDemandGenerator(AbstractDemandGenerator):

    def __init__(self, distribution: torch.distributions.Distribution = torch.distributions.Uniform(low=0, high=4 + 1)):
        """
        Generates demand based on a toch distribution

        Parameters
        ----------
        distribution: torch.distributions.Distribution
        """

        self.distribution = distribution

    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:
        D = self.distribution.sample([n_samples, 1]).round().int()
        return D

    def sample_trajectory(self, t: int, n_samples: int, n_timesteps: int) -> torch.Tensor:
        D_traj = self.distribution.sample([n_samples, n_timesteps, 1]).round().int()
        return D_traj


class FileBasedDemandGenerator(AbstractDemandGenerator):
    # TODO: load df and test
    def __init__(self,
                 demand_file_path='../../MSOM_data/msom.2020.0933.csv',
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
                 myclip_b=1e3,
                 ):
        """
        Generates demand based on a toch distribution

        Parameters
        ----------
        distribution: torch.distributions.Distribution
        """

        self.demand_file_path = demand_file_path
        self.demand_df = pd.read_csv(demand_file_path)
        if max_weeks is None:
            self.max_weeks = max(self.demand_df["Week"].to_numpy())
        else:
            self.max_weeks = max_weeks

        self.date_index = np.arange(1, self.max_weeks + 1)

        self.skus = skus
        self.impute_val = impute_val
        self.scaling_factor = scaling_factor

        self.imputed_data, self.mean_array, self.std_array = self.__preprocess_file__()
        self.mean_vector = torch.tensor(self.mean_array).float()
        self.std_vector = torch.tensor(self.std_array).float()

        self.myclip_a = myclip_a
        self.myclip_b = myclip_b


    def __preprocess_file__(self):
        y = []
        for SKU in self.skus:

            data_ = self.demand_df[self.demand_df["SKU"] == SKU]
            weeks = data_[data_["SKU"] == SKU]["Week"].to_numpy()
            orders = data_[data_["SKU"] == SKU]["Customer Orders"].to_numpy()
            # data_["Total Customer Orders"] = data_.groupby(["Week"])['Customer Orders'].transform('sum')
            # total_orders = data_["Total Customer Orders"].to_numpy()
            total_orders = data_.groupby(["Week"])['Customer Orders'].transform('sum').to_numpy()
            weeks_total_orders = sorted(set([(x[0], x[1]) for x in zip(weeks, total_orders)]))

            weeks_ = np.array([x[0] for x in weeks_total_orders])
            total_orders_ = np.array([x[1] for x in weeks_total_orders])

            y_ = []
            for i in range(len(self.date_index)):

                if self.date_index[i] in weeks_:
                    y_.append(total_orders_[self.date_index[i] == weeks_][0])
                else:
                    y_.append(np.nan)

            y.append(np.array(y_))

        y = np.array(y)
        y_impute = np.nan_to_num(y, nan=self.impute_val)
        mean_arr = np.array([np.mean(y_impute[:, i] / self.scaling_factor) for i in range(len(self.date_index))])
        std_arr = np.array([np.std(y_impute[:, i] / self.scaling_factor, ddof=1) for i in range(len(self.date_index))])
        return y_impute, mean_arr, std_arr

    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:

        sample = self.scaling_factor * truncnorm.rvs(
            (self.myclip_a - self.mean_array[t]) / (1e-9 + self.std_array[t]),
            (self.myclip_b - self.mean_array[t]) / (1e-9 + self.std_array[t]),
            loc=self.mean_array[t], scale=self.std_array[t],
            size=n_samples)
        return torch.tensor(sample)

    def sample_trajectory(self, t: int, n_samples: int, n_timesteps: int) -> torch.Tensor:
        T = t + n_timesteps
        a = (self.myclip_a - self.mean_array[t:T]) / (1e-9 + self.std_array[t:T])
        b = (self.myclip_b - self.mean_array[t:T]) / (1e-9 + self.std_array[t:T])
        samples = self.scaling_factor * \
                 truncnorm.rvs(a,
                               b,
                               loc=self.mean_array[t:T],
                               scale=self.std_array[t:T],
                               size=(n_samples, n_timesteps)
                               )
        return torch.tensor(samples)

    def plot_gaussian_process_vs_model(self):
        fig, ax = plt.subplots(facecolor='white')
        for i in range(len(self.skus)):
            plt.plot(self.date_index, self.imputed_data[i] / 1e5, alpha=0.3, color='Grey')
        plt.plot(self.date_index, self.mean_array, color='tab:blue', alpha=0.8)
        plt.fill_between(self.date_index, self.mean_array + 1.96 * self.std_array, self.mean_array - 1.96 * self.std_array, alpha=0.3)
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        plt.xlim(0, 120)
        plt.ylim(0, 2.5)
        plt.xlabel(r"week")
        plt.ylabel(r"customer orders [$\times 10^5$]")
        plt.tight_layout()
        plt.savefig("gaussian_sampling.png", dpi=480)
        plt.show()


    def plot_distributions(self):
        x = np.linspace(0, 4, 100)
        plt.figure()
        for i in range(40):
            myclip_a = 0
            myclip_b = 1e2
            my_mean = self.mean_array[i]
            my_std = self.std_array[i]
            a, b = (myclip_a - my_mean) / (1e-9 + my_std), (myclip_b - my_mean) / (1e-9 + my_std)
            rv = truncnorm(a, b, loc=my_mean, scale=my_std)
            r = lambda x: rv.pdf(x)
            plt.plot(x, r(x))
        plt.xlabel(r"orders")
        plt.ylabel(r"PDF")
        plt.show()

    def plot_samples_vs_data(self, sample_size=1000):
        fig, ax = plt.subplots(facecolor="white")
        samples = self.sample_trajectory(0, sample_size, self.max_weeks).cpu().detach().numpy()/self.scaling_factor
        for i in range(sample_size):
            plt.plot(self.date_index, samples[i], alpha=0.05, color="Grey")

        for i in range(len(self.skus)):
            plt.plot(self.date_index, self.imputed_data[i] / self.scaling_factor, alpha=1.0, color="tab:cyan")
        plt.plot(100, 100, alpha=0.3, color="tab:blue", label="data")
        plt.plot(100, 100, alpha=0.3, color="Grey", label="sample")
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        plt.xlim(0, 120)
        plt.ylim(0, 2.5)
        plt.xlabel(r"week")
        plt.ylabel(r"customer orders [$\times 10^5$]")
        plt.legend(loc=2, frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig("gaussian_sampling_2.png", dpi=480)
        plt.show()
