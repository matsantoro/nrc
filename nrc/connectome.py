from typing import Optional, Union, List
from pathlib import Path

import elephant.conversion
import numpy as np
import pandas as pd
import scipy.sparse as sp
import neo
import quantities as qt
from functools import partial

import nrc.reliability
from nrc.config import registry_property, autosave_method, pickle_dump_to_path, pickle_load_from_path
from tqdm import tqdm


class SpikeTrainsCollection:
    spikes_array = registry_property('spikes_array')
    t_start = registry_property('t_start')
    t_stop = registry_property('t_stop')
    gids = registry_property('gids')

    def __init__(self, root: Optional[Path], s_array: Optional[np.ndarray] = None,
                 t_start: Optional[Union[qt.quantity.Quantity, int]] = None,
                 t_stop: Optional[Union[qt.quantity.Quantity, int]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts spike train data for a single simulation seed, for a collection of neurons.
        :param root: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param s_array: Spike array containing spikes to store. Assumes an N x 2 array, where N is
            the number of spikes. Column 0 contains spike times, column 1 neuron GIDS.
        :param t_start: Spike trains start. If an integer and not a quantities.quantity.Quantity object,
            assumes milliseconds.
        :param t_stop: Spike trains stop. If an integer and not a quantities.quantity.Quantity object,
            assumes milliseconds.
        :param gids: The gids of the neurons to consider.
        """
        self.root = root
        if self.root is not None:  # if object is rooted, initialize autosave structure
            self.root.mkdir(exist_ok=True, parents=True)
            # target paths
            self.spikes_array_path = root / "spikes.npy"
            self.t_start_path = root / "t_start.pkl"
            self.t_stop_path = root / "t_stop.pkl"
            self.gids_path = root / "gids.npy"
            # helper dictionary. See config.registry_property for details.
            self.property_registry = {
                'spikes_array': [partial(np.load, file=self.spikes_array_path),
                                 lambda x: np.save(arr=x, file=str(self.spikes_array_path)),
                                 self.spikes_array_path.exists],
                't_start': [partial(pickle_load_from_path, path=self.t_start_path),
                            lambda x: pickle_dump_to_path(obj=x, path=self.t_start_path),
                            self.t_start_path.exists],
                't_stop': [partial(pickle_load_from_path, path=self.t_stop_path),
                           lambda x: pickle_dump_to_path(obj=x, path=self.t_stop_path),
                           self.t_stop_path.exists],
                'gids': [partial(np.load, file=self.gids_path),
                         lambda x: np.save(arr=x, file=str(self.gids_path)),
                         self.gids_path.exists]
            }
        # initalize attributes if present.
        # uninitialized registry_property's will be loaded from memory
        if s_array is not None:
            self.spikes_array = s_array
        self.neo_spike_trains = None
        if t_start is not None:
            if type(t_start) is int:
                self.t_start = t_start * qt.ms
            else:
                self.t_start = t_start
        if t_stop is not None:
            if type(t_stop) is int:
                self.t_stop = t_stop * qt.ms
            else:
                self.t_stop = t_stop
        if gids is not None:
            self.gids = gids

    @autosave_method
    def get_neo_spike_trains(self) -> List[neo.SpikeTrain]:
        """
        Function to get neo.SpikeTrain objects to wrap single neuron spikes.

        :return list_of_spike_trains:
        """
        if self.neo_spike_trains is not None:
            return self.neo_spike_trains
        else:
            self.neo_spike_trains = [
                neo.SpikeTrain(times=self.spikes_array[self.spikes_array[:, 1] == gid][:, 0] * qt.ms,
                               t_start=self.t_start, t_stop=self.t_stop, units=qt.ms)
                for gid in self.gids
            ]
            return self.neo_spike_trains

    @autosave_method
    def get_binned_spike_trains(self, time_bin: Union[qt.quantity.Quantity, int]) \
            -> elephant.conversion.BinnedSpikeTrain:
        """
        Function to retrieve the binned version of the spike trains of the object.

        :param time_bin: Time bin for binning.
        :return binned_spike_trains: BinnedSpikeTrain object containing binned spike trains.
        """
        if type(time_bin) is int:
            _time_bin = time_bin * qt.ms
        else:
            _time_bin = time_bin
        return elephant.conversion.BinnedSpikeTrain(self.get_neo_spike_trains(), bin_size=_time_bin)

    def convolve_with_kernel(self, time_bin: Union[qt.quantity.Quantity, int], kernel: elephant.kernels.Kernel) \
            -> np.ndarray:
        """
        Function to convolve the spike trains with a generic elephant kernel.

        :param time_bin: size of the time bin to bin the spike trains with.
        :param kernel: elephant.kernels.Kernel object to convolve the binned spike trains with.
        :return convolved_spike_trains: np.ndarray object containing binned spike trains convolved with kernel.
        """
        if type(time_bin) is int:
            _time_bin = time_bin * qt.ms
        else:
            _time_bin = time_bin
        return nrc.reliability.convolve_with_kernel(self.get_binned_spike_trains(time_bin=time_bin), kernel)

    @autosave_method
    def convolve_with_gaussian_kernel(self, time_bin: Union[qt.quantity.Quantity, int],
                                      sigma: Union[qt.quantity.Quantity, int]):
        """
        Function to specifically convolve the spike trains with a gaussian kernel. STs are first binned,
        then convolved.
        :param time_bin: time bin of the binned spike trains.
        :param sigma: sigma of the gaussian kernel to use
        :return convolved_spike_trains: np.ndarray object containing binned spike trains convolved with kernel
        """
        if type(time_bin) is int:
            _time_bin = time_bin * qt.ms
        else:
            _time_bin = time_bin
        if type(sigma) is int:
            _sigma = sigma * qt.ms
        else:
            _sigma = sigma
        kernel = elephant.kernels.GaussianKernel(sigma=_sigma)
        return self.convolve_with_kernel(time_bin=time_bin, kernel=kernel)

    def get_number_of_spikes(self) -> np.ndarray:
        """
        Returns an array with the total number of spikes fired during the simulation per neuron.

        :return n_spikes: np.ndarray containing a number per GID representing the total number of spikes fired.
        """
        return np.array([len(x) for x in self.get_neo_spike_trains()])

    def get_firing_rates(self) -> qt.quantity.Quantity:
        """
        Returns an array with the total number of spikes fired during the simulation per neuron.

        :return n_spikes: np.ndarray containing a number per GID representing the total number of spikes fired.
        """
        return self.get_number_of_spikes() / (self.t_stop - self.t_start)

    def unroot(self):
        self.root = None


class Simulation:
    seeds = registry_property('seeds')
    gids = registry_property('gids')

    def __init__(self, root: Optional[Path], spike_train_collection_list: Optional[List[SpikeTrainsCollection]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts simulation data for multiple simulation seeds, for a collection of neurons.

        :param root: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param spike_train_collection_list: list of SpikeTrainCollection objects. Each represents a seed.
        :param gids: gids of the considered neurons.
        """
        self.root = root
        if self.root is not None:
            self.root.mkdir(exist_ok=True, parents=True)
            self.gids_path = self.root / "gids.npy"

            # seed memory save, retrieve and check function. See config.registry_property
            def retrieve_seeds_from_memory():
                st_list = []
                for seed_path in sorted(list(self.root.glob("seed*/")), key=lambda x: int(x.stem[4:])):
                    st_list.append(SpikeTrainsCollection(seed_path))
                return st_list

            def store_seeds_to_memory(seeds):
                for i, seed in enumerate(seeds):
                    target_for_seed = self.root / ("seed" + str(i))
                    SpikeTrainsCollection(target_for_seed, seed.spikes_array,
                                          seed.t_start, seed.t_stop, seed.gids)

            def check_if_seed():
                return self.root.glob("seed*") is not None

            # property registry initialization. See config/registry_property
            self.property_registry = {
                'seeds': [retrieve_seeds_from_memory, store_seeds_to_memory, check_if_seed],
                'gids': [partial(np.load, file=self.gids_path),
                         lambda x: np.save(arr=x, file=str(self.gids_path)),
                         self.gids_path.exists]
            }

        # initialize attributes.
        if spike_train_collection_list is not None:
            try:
                assert np.min([st.t_stop for st in spike_train_collection_list]) == \
                       np.max([st.t_stop for st in spike_train_collection_list])
                assert np.min([st.t_start for st in spike_train_collection_list]) == \
                       np.max([st.t_start for st in spike_train_collection_list])
                self.seeds = spike_train_collection_list
            except AssertionError:
                print("Different start and stop times for spike trains.")

        if gids is not None:
            self.gids = gids

    def convolve_with_kernel(self, time_bin: Union[int, qt.quantity.Quantity], kernel: elephant.kernels.Kernel) \
            -> np.ndarray:
        """
        Convolve all seeds with a given kernel. Returns a N_seeds x N_neurons x time_bins array.

        :param time_bin: time bin used to bin spike trains
        :param kernel: elephant.kernel.Kernel to use for the convolution
        :return convolved_spike_trains: np.ndarray with N_seeds x N_neurons x time_bins entries containing the
            convolved spike trains.
        """
        return np.stack([st.convolve_with_kernel(time_bin=time_bin, kernel=kernel) for st in self.seeds])

    @autosave_method
    def convolve_with_gk(self, time_bin: Union[int, qt.quantity.Quantity], sigma: Union[int, qt.quantity.Quantity], ) \
            -> np.ndarray:
        """
        Convolve all the seeds with a gaussian kernel. Returns a N_seeds x N_neurons x time_bins array
        :param time_bin: time bin used to bin spike trains
        :param sigma: sigma of the gaussian kernel
        :return convolved_spike_trains: np.ndarray with N_seeds x N_neurons x time_bins entries containing the
            convolved spike trains.
        """
        return np.stack([st.convolve_with_gaussian_kernel(time_bin=time_bin, sigma=sigma) for st in self.seeds])

    @autosave_method
    def get_binned_spike_trains(self, time_bin: Union[int, qt.quantity.Quantity]) -> np.ndarray:
        """
        Retrieve binned spike trains for all seeds.

        :param time_bin: time bin to use for binning the spike trains
        :return binned_spike_trains: np.ndarray with N_seeds x N_neurons x time_bins entries containing the
            binned spike trains.
        """
        return np.stack([st.get_binned_spike_trains(time_bin=time_bin) for st in self.seeds])

    @autosave_method
    def gk_rel_scores(self, time_bin: Union[int, qt.quantity.Quantity], sigma: Union[int, qt.quantity.Quantity]) \
            -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute gaussian kernel reliability for all neurons across seeds.

        :param time_bin: time bin to use for binning the spike trains
        :param sigma: sigma of the gaussian kernel to use to convolve the spike trains
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        convolved_sts = self.convolve_with_gk(time_bin=time_bin, sigma=sigma)
        return [nrc.reliability.cosine_reliability(convolved_sts[:, i, :]) for i in tqdm(range(len(self.gids)))]

    @autosave_method
    def gk_rel_pearson_scores(self, time_bin: Union[int, qt.quantity.Quantity],
                              sigma: Union[int, qt.quantity.Quantity]) -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute gaussian kernel reliability with pearson similarity for all neurons across seeds.

        :param time_bin: time bin to use for binning the spike trains
        :param sigma: sigma of the gaussian kernel to use to convolve the spike trains
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        convolved_sts = self.convolve_with_gk(time_bin=time_bin, sigma=sigma)
        return [nrc.reliability.pearson_reliability(convolved_sts[:, i, :]) for i in tqdm(range(len(self.gids)))]

    @autosave_method
    def average_number_of_spikes(self) -> np.ndarray:
        """
        Return the average number of spikes fired per seed for all neurons.

        :return average_number_of_spikes: avearge number of spikes fired across seeds.
        """
        return np.mean(np.stack([seed.get_number_of_spikes() for seed in self.seeds]), axis=0)

    def average_firing_rate(self):
        """
        Return the average firing rate of all neurons.

        :return average_firing_rates: avearge firing rates across seeds.
        """
        return np.mean(np.stack(
            [seed.get_firing_rates().rescale(1 / qt.ms) for seed in self.seeds]
        ), axis=0) * (1 / qt.ms)

    def unroot(self):
        self.root = None
        for seed in self.seeds:
            seed.unroot()

class Connectome:
    adjacency = registry_property('adjacency', )
    neuron_data = registry_property('neuron_data', )
    simulations = registry_property('simulations', )
    gids = registry_property('gids', )

    def __init__(self, root: Optional[Path], adjacency_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 ndata: Optional[pd.DataFrame] = None, simulations: Optional[List[Simulation]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts connectivity and simulation data for multiple simulation seeds, for a given connectome.

        :param root: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param adjacency_matrix: np.ndarray or scipy.sparse.spmatrix with the adjacency matrix of neurons.
            Assumes A_ij != 0 <=> neuron i is presynaptic to neuron j.
        :param ndata: pandas.DataFrame containing neuron info.
        :param simulations: list of Simulation objects.
        :param gids: gids of the considered neurons.
        """
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.adjacency_matrix_path = self.root / "adj.npy"
            self.neuron_data_path = self.root / "ndata.pkl"
            self.gids_path = self.root / "gids.npy"

            # property registry functions for simulation list. See config.registry_property for info.
            def retrieve_sims_from_memory():
                sim_list = []
                for sim_path in sorted(list(self.root.glob("sim*/")), key=lambda x: int(x.stem[3:])):
                    sim_list.append(Simulation(sim_path))
                return sim_list

            def store_sims_to_memory(sims):
                for i, sim in enumerate(sims):
                    target_for_sim = self.root / ("sim" + str(i))
                    Simulation(target_for_sim,
                                          sim.seeds)

            def check_if_sims():
                return self.root.glob("sim*") is not None

            self.property_registry = {
                'adjacency': [partial(np.load, file=self.adjacency_matrix_path),
                              lambda x: np.save(arr=x, file=str(self.adjacency_matrix_path)),
                              self.adjacency_matrix_path.exists],
                'neuron_data': [partial(pd.read_pickle, filepath_or_buffer=self.neuron_data_path),
                                lambda x: pd.to_pickle(obj=x, filepath_or_buffer=self.neuron_data_path),
                                self.neuron_data_path.exists],
                'simulations': [retrieve_sims_from_memory, store_sims_to_memory, check_if_sims],
                'gids': [partial(np.load, file=self.gids_path),
                         lambda x: np.save(arr=x, file=str(self.gids_path)),
                         self.gids_path.exists]
            }

            # attribute assignment
            if adjacency_matrix is not None:
                self.adjacency = adjacency_matrix

            if ndata is not None:
                self.neuron_data = ndata

            if simulations is not None:
                self.simulations = simulations

            if gids is not None:
                self.gids = gids

    def unroot(self):
        self.root = None
        for sim in self.simulations:
            sim.unroot()