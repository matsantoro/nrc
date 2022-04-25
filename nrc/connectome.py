import pdb
from typing import Optional, Union, List, Dict
from pathlib import Path

import elephant.conversion
import multiprocessing
import numpy as np
import pandas as pd
import scipy.sparse as sp
import neo
import quantities as qt
from functools import partial
import time

import nrc.reliability
from nrc.config import registry_property, autosave_method, pickle_dump_to_path, pickle_load_from_path, RootedObject
import nrc.poisson_generation
from nrc.structural import (
    structural_function_list,
    biggest_cc
)

import pyflagsercount

from tqdm import tqdm


class SpikeTrainsCollection(RootedObject):
    spikes_array = registry_property('spikes_array')
    t_start = registry_property('t_start')
    t_stop = registry_property('t_stop')
    gids = registry_property('gids')

    def __init__(self, root_path: Optional[Path], s_array: Optional[np.ndarray] = None,
                 t_start: Optional[Union[qt.quantity.Quantity, int]] = None,
                 t_stop: Optional[Union[qt.quantity.Quantity, int]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts spike train data for a single simulation seed, for a collection of neurons.
        :param root_path: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param s_array: Spike array containing spikes to store. Assumes an N x 2 array, where N is
            the number of spikes. Column 0 contains spike times, column 1 neuron GIDS.
        :param t_start: Spike trains start. If an integer and not a quantities.quantity.Quantity object,
            assumes milliseconds.
        :param t_stop: Spike trains stop. If an integer and not a quantities.quantity.Quantity object,
            assumes milliseconds.
        :param gids: The gids of the neurons to consider.
        """
        RootedObject.__init__(self, root_path)
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

    @classmethod
    def from_neo_spike_trains(cls, spike_trains: List[neo.SpikeTrain], root: Optional[Path], gids: np.ndarray):
        spikes_array = np.concatenate(
            [np.stack(
                [spike_train.as_array(qt.ms), np.ones(len(spike_train)) * gids[i]]
            ).T for i, spike_train in enumerate(spike_trains)]
        )
        return SpikeTrainsCollection(root, spikes_array, spike_trains[0].t_start, spike_trains[0].t_stop, gids)

    def root(self, root_path: Path):
        super(SpikeTrainsCollection, self).root(root_path)
        # target paths
        self.spikes_array_path = root_path / "spikes.npy"
        self.t_start_path = root_path / "t_start.pkl"
        self.t_stop_path = root_path / "t_stop.pkl"
        self.gids_path = root_path / "gids.npy"
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

    def unroot(self):
        self.spikes_array
        self.gids
        self.t_start
        self.t_stop
        super(SpikeTrainsCollection, self).unroot()


class Simulation(RootedObject):
    seeds = registry_property('seeds')
    gids = registry_property('gids')

    def __init__(self, root_path: Optional[Path],
                 spike_train_collection_list: Optional[List[SpikeTrainsCollection]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts simulation data for multiple simulation seeds, for a collection of neurons.

        :param root: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param spike_train_collection_list: list of SpikeTrainCollection objects. Each represents a seed.
        :param gids: gids of the considered neurons.
        """
        RootedObject.__init__(self, root_path)
        # initialize attributes.
        if spike_train_collection_list is not None:
            try:
                assert np.min([st.t_stop for st in spike_train_collection_list]) == \
                       np.max([st.t_stop for st in spike_train_collection_list])
                assert np.min([st.t_start for st in spike_train_collection_list]) == \
                       np.max([st.t_start for st in spike_train_collection_list])
                self.seeds = spike_train_collection_list
            except AssertionError:
                raise ValueError("Different start and stop times for spike trains.")

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
    def gk_rel_scores(self, time_bin: Union[int, qt.quantity.Quantity], sigma: Union[int, qt.quantity.Quantity],
                      save_convolved_sts: bool = False) -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute gaussian kernel reliability for all neurons across seeds.

        :param time_bin: time bin to use for binning the spike trains
        :param sigma: sigma of the gaussian kernel to use to convolve the spike trains
        :param save_convolved_sts: whether to save the convolved spike trains
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        convolved_sts = self.convolve_with_gk(time_bin=time_bin, sigma=sigma, autosave=save_convolved_sts)
        return [nrc.reliability.cosine_reliability(convolved_sts[:, i, :]) for i in tqdm(range(len(self.gids)))]

    @autosave_method
    def gk_rel_pearson_scores(self, time_bin: Union[int, qt.quantity.Quantity],
                              sigma: Union[int, qt.quantity.Quantity],
                              save_convolved_sts: bool = False) -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute gaussian kernel reliability with pearson similarity for all neurons across seeds.

        :param time_bin: time bin to use for binning the spike trains
        :param sigma: sigma of the gaussian kernel to use to convolve the spike trains
        :param save_convolved_sts: whether to save convolved spike trains.
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        convolved_sts = self.convolve_with_gk(time_bin=time_bin, sigma=sigma, autosave=save_convolved_sts)
        return [nrc.reliability.pearson_reliability(convolved_sts[:, i, :]) for i in tqdm(range(len(self.gids)))]

    @autosave_method
    def vp_rel_scores(self, time_constant: Union[int, qt.quantity.Quantity]) -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute victor-purpura reliability for all neurons across seeds.

        :param time_constant: time bin to use for binning the spike trains
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        if type(time_constant) is int:
            _time_constant = time_constant * qt.ms
        else:
            _time_constant = time_constant
        return [nrc.reliability.get_vp_reliability(
            spike_trains=[sts.get_neo_spike_trains()[i] for sts in self.seeds],
            time_constant=_time_constant,
            convert_to_similarity=True,
        ) for i in tqdm(range(len(self.gids)))
        ]

    @autosave_method
    def vr_rel_scores(self, time_constant: Union[int, qt.quantity.Quantity]) -> list[nrc.reliability.ReliabilityScore]:
        """
        Compute van Rossum reliability for all neurons across seeds.

        :param time_constant: time bin to use for binning the spike trains
        :return list_of_reliability_scores: list of ReliabilityScore objects containing the score of all neurons.
        """
        if type(time_constant) is int:
            _time_constant = time_constant * qt.ms
        else:
            _time_constant = time_constant
        return [nrc.reliability.get_vr_reliability(
            spike_trains=[sts.get_neo_spike_trains()[i] for sts in self.seeds],
            time_constant=_time_constant,
            convert_to_similarity=True,
        ) for i in tqdm(range(len(self.gids)))
        ]

    @autosave_method
    def average_number_of_spikes(self) -> np.ndarray:
        """
        Return the average number of spikes fired per seed for all neurons.

        :return average_number_of_spikes: avearge number of spikes fired across seeds.
        """
        return np.mean(np.stack([seed.get_number_of_spikes() for seed in self.seeds]), axis=0)

    @autosave_method
    def average_firing_rate(self):
        """
        Return the average firing rate of all neurons.

        :return average_firing_rates: avearge firing rates across seeds.
        """
        return np.mean(np.stack(
            [seed.get_firing_rates().rescale(1 / qt.ms) for seed in self.seeds]
        ), axis=0) * (1 / qt.ms)

    @autosave_method
    def average_firing_rate_profiles(self, sigma: int):
        """
        Return the avera<age firing rate profiles for all neurons, computed with a gaussian kernel

        :param sigma: width of the kernel
        :return: an array containing the firing rate profiles of neurons.
        """
        neuron_sts = []
        for i, neuron in enumerate(self.gids):
            neuron_sts.append(self.seeds[0].get_neo_spike_trains()[i].merge(
                *[seed.get_neo_spike_trains()[i] for seed in self.seeds[1:]]
            ))
        return nrc.poisson_generation.retrieve_fr_profile_from_sts(
            neuron_sts,
            elephant.kernels.GaussianKernel(sigma * qt.ms)
        ).T

    @classmethod
    def from_firing_rate_profiles(cls, firing_rate_profiles: neo.AnalogSignal, repetitions: int,
                                  seeds: int, root_path: Optional[Path], gids: np.ndarray):
        """
        Generate another simulation object with poisson process generation using given firing rate profiles.

        :param firing_rate_profiles: a signal containing the firing rate profiles of the simulation
        :param repetitions: number of repetitions of the generated experiments
        :param seeds: seed to use for the generation. Will produce seeds for the simulation
        :param root_path: root_path of the new simulation object
        :param gids: gids to use
        :return: a simulation object with the generated spike trains.
        """
        all_spike_trains = []
        for neuron in range(len(firing_rate_profiles)):
            single_neuron_firing_rate = neo.AnalogSignal(
                firing_rate_profiles[neuron, :],
                units=firing_rate_profiles.units,
                t_start=firing_rate_profiles.t_start,
                sampling_rate=firing_rate_profiles.sampling_rate
            )
            all_spike_trains.append(nrc.poisson_generation.generate_spike_trains_from_profile(
                single_neuron_firing_rate, repetitions, seeds
            ))
        seeds = []
        for i in range(repetitions):
            seeds.append(SpikeTrainsCollection.from_neo_spike_trains(
                [all_spike_trains[neuron][i] for neuron in range(len(firing_rate_profiles))],
                root=None,
                gids=gids,
            ))
        return Simulation(root_path, seeds, gids)

    def root(self, root_path: Path):
        super(Simulation, self).root(root_path)
        self.gids_path = self.root_path / "gids.npy"

        # seed memory save, retrieve and check function. See config.registry_property
        def retrieve_seeds_from_memory():
            st_list = []
            for seed_path in sorted(list(self.root_path.glob("seed*/")), key=lambda x: int(x.stem[4:])):
                st_list.append(SpikeTrainsCollection(seed_path))
            return st_list

        def store_seeds_to_memory(seeds):
            for i, seed in enumerate(seeds):
                target_for_seed = self.root_path / ("seed" + str(i))
                SpikeTrainsCollection(target_for_seed, seed.spikes_array,
                                      seed.t_start, seed.t_stop, seed.gids)

        def check_if_seed():
            return self.root_path.glob("seed*") is not None

        # property registry initialization. See config/registry_property
        self.property_registry = {
            'seeds': [retrieve_seeds_from_memory, store_seeds_to_memory, check_if_seed],
            'gids': [partial(np.load, file=self.gids_path),
                     lambda x: np.save(arr=x, file=str(self.gids_path)),
                     self.gids_path.exists]
        }

    def unroot(self):
        self.seeds
        self.gids
        super(Simulation, self).unroot()
        for seed in self.seeds:
            seed.unroot()


class Connectome(RootedObject):
    adjacency = registry_property('adjacency', )
    neuron_data = registry_property('neuron_data', )
    simulations = registry_property('simulations', )
    gids = registry_property('gids', )

    def __init__(self, root_path: Optional[Path], adjacency_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 ndata: Optional[pd.DataFrame] = None, simulations: Optional[List[Simulation]] = None,
                 gids: Optional[np.ndarray] = None):
        """
        Object that hosts connectivity and simulation data for multiple simulation seeds, for a given connectome.

        :param root_path: Where to root the object. Object whose root is None do not automatically save data
            or computation results.
        :param adjacency_matrix: np.ndarray or scipy.sparse.spmatrix with the adjacency matrix of neurons.
            Assumes A_ij != 0 <=> neuron i is presynaptic to neuron j.
        :param ndata: pandas.DataFrame containing neuron info.
        :param simulations: list of Simulation objects.
        :param gids: gids of the considered neurons.
        """
        RootedObject.__init__(self, root_path)

        # attribute assignment
        if adjacency_matrix is not None:
            self.adjacency = adjacency_matrix

        if gids is not None:
            self.gids = gids

        if ndata is not None:
            self.neuron_data = ndata

        if simulations is not None:
            self.simulations = simulations

    def root(self, root_path: Path):
        super(Connectome, self).root(root_path)
        self.adjacency_matrix_path = self.root_path / "adj.npy"
        self.neuron_data_path = self.root_path / "ndata.pkl"
        self.gids_path = self.root_path / "gids.npy"

        # property registry functions for simulation list. See config.registry_property for info.
        def retrieve_sims_from_memory():
            sim_list = []
            for sim_path in sorted(list(self.root_path.glob("sim*/")), key=lambda x: int(x.stem[3:])):
                sim_list.append(Simulation(sim_path))
            return sim_list

        def store_sims_to_memory(sims):
            for i, sim in enumerate(sims):
                target_for_sim = self.root_path / ("sim" + str(i))
                Simulation(target_for_sim,
                           sim.seeds, self.gids)

        def check_if_sims():
            return self.root_path.glob("sim*") is not None

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

    @autosave_method
    def simplex_list(self, threads: int = 30, flagser_args: Optional[Dict]=None):
        """
        Computes the simplex lists and returns them.
        :param threads: number of thread to use.
        :param flagser_args: other arguments for flagser. min_dim_print and threads already specified.
        :return simplex_lists: a list containing one array per dimension with the simplices of that dimension
        """
        if flagser_args is None:
            flagser_args = dict()
        lists = pyflagsercount.flagser_count(self.adjacency,
                                             return_simplices=True, threads=threads, min_dim_print=0,
                                             **flagser_args)['simplices']
        return [np.array(l) for l in lists[1:]]

    @autosave_method
    def reciprocal_connections_in_simplices(self, threads=30):
        """
        Return a list of array, where for each dimension there is the number of bidirectional edges
        a given simplex contains
        :param threads: number of threads to compute the simplex lists with
        :return: a list of arrays where each array contains the reciprocal connections of all simplices
        """
        lists = self.simplex_list(threads=threads)
        bedges = np.multiply(self.adjacency, self.adjacency.T)
        bedges_lists = []
        for i in range(len(lists)):
            bedge_array = np.zeros(len(lists[i]))
            for j in range(len(lists[i])):
                bedge_array = int(np.sum(bedges[lists[i][j]][:, lists[i][j]])/2)
        return bedge_array

    def unroot(self):
        self.simulations
        self.gids
        self.adjacency
        self.neuron_data
        super(Connectome, self).unroot()
        for sim in self.simulations:
            sim.unroot()


class TribeView(RootedObject):
    analysis_data = registry_property("analysis_data")
    analysis_data_error = registry_property("analysis_data_error")

    def __init__(self, transform_method_list: List[callable],
                 connectome_object: Optional[Connectome],
                 tribe_type: str = ''):
        """
        Object that allows manipulation and usage of tribes.

        :param transform_method_list: list of transformation methods to use with this tribe.
        :param connectome_object:  connectome object to use for tribe manipulation
        :param tribe_type: type of initial tribe. "in", "out", "", are allowed, with the obvious meaning.
        """
        self.transform_method_list = transform_method_list
        self.connectome_object = connectome_object
        self.analysis_method_list = structural_function_list
        self.tribe_type = tribe_type
        self.tribes()
        RootedObject.__init__(self, connectome_object.root_path / (self.tribe_type + "tribes"))

    def _tribes_for_chief(self):
        """
        Helper function for initial tribe retrieval.
        :return tribes: an array object containing in elemen A_ij whether neuron j is part of the tribe of i
        """
        if self.tribe_type is '':
            return (self.connectome_object.adjacency + self.connectome_object.adjacency.T +
                np.diag(np.ones(len(self.connectome_object.adjacency)))).astype(bool)
        elif self.tribe_type is 'in':
            return (self.connectome_object.adjacency.T +
                    np.diag(np.ones(len(self.connectome_object.adjacency)))).astype(bool)
        elif self.tribe_type is 'out':
            return (self.connectome_object.adjacency +
                    np.diag(np.ones(len(self.connectome_object.adjacency)))).astype(bool)

    def tribes(self, root: bool = True):
        """
        Return the tribes with the current transform method list.

        :param root: whether to root the tribes in the process. The TribeView will be rooted according to the
        transforms in it. For example, second degree out tribes of in tribes will be stored as
        "original_root/intribes/second_degree_out/_tribes.pkl"
        :return: an ndarray containing in A_ij whether neuron j belongs to the tribe of neuron i
        """
        if root:
            self.root(self.connectome_object.root_path / (self.tribe_type + 'tribes/' +
                                                          '/'.join([x.__name__ for x in self.transform_method_list])))
        tribes = self._tribes()
        return tribes

    @autosave_method
    def _tribes(self):
        tribes = self._tribes_for_chief()
        for method in self.transform_method_list:
            tribes = method(tribes, self.connectome_object)
        return tribes

    def threshold(self, bool_array: np.ndarray, threshold_name: str):
        """
        Add a threshold for neurons in the TribeView object.

        :param bool_array: The boolean array representing whether the neuron passed the threshold
        :param threshold_name: The name of the threshold
        """
        self.add_transform(
            lambda x, conn: np.apply_along_axis(arr=x, func1d=lambda y: y * bool_array, axis=1),
            'threshold_' + threshold_name
        )

    def second_degree(self, kind: str):
        """
        Add a second degree transformation for neurons in the TribeView object.

        :param kind: Which kind of transformation to apply.
        """
        if kind == 'all':
            def f(y, connectome):
                return np.logical_and(
                    np.sum(connectome.adjacency[y], axis=0, dtype=bool) +
                    np.sum(connectome.adjacency.T[y].T, axis=1, dtype=bool),
                    np.logical_not(y)
                )
        elif kind == 'in':
            def f(y, connectome):
                return np.logical_and(
                    np.sum(connectome.adjacency.T[y].T, axis=1, dtype=bool),
                    np.logical_not(y)
                )
        elif kind == 'out':
            def f(y, connectome):
                return np.logical_and(
                    np.sum(connectome.adjacency[y], axis=0, dtype=bool),
                    np.logical_not(y)
                )
        else:
            raise NotImplementedError(kind + " as a second degree kind is not implemented.")

        def second_degree_transform(x: np.ndarray, conn: Connectome):
            return np.apply_along_axis(
                arr=x,
                axis=1,
                func1d=partial(f, connectome=conn)
            )

        self.add_transform(
            second_degree_transform,
            'second_degree_' + kind
        )

    def biggest_cc(self):
        """
        Add a biggest_cc transform to the TribeView object. weak connection.
        """
        def biggest_cc_transform(x: np.ndarray, conn: Connectome):
            def turn_tribe_in_biggest_cc(boolean_array):
                indices = np.nonzero(boolean_array)[0]
                tribe_matrix = conn.adjacency[indices][:, indices]
                indices_submatrix = indices[biggest_cc(tribe_matrix)[1]]
                mask = np.zeros((len(boolean_array),), dtype=bool)
                mask[indices_submatrix] = True
                return np.logical_and(boolean_array, mask)

            return np.apply_along_axis(
                arr=x,
                axis=1,
                func1d=lambda y: turn_tribe_in_biggest_cc(y)
            )

        self.add_transform(
            biggest_cc_transform,
            'biggest_cc'
        )

    def generate_control(self, layer_profile: bool = False, mtype_profile: bool = False, seed: int = 0):
        """
        Add a control generation step to the tribe retrieval process.

        :param layer_profile: bool indicating whether the newly generated tribes should have the
        same layer profile as the previous.
        :param mtype_profile: bool indicating whether the newly generated tribes should have the
        same mtype profile as the previous
        :param seed: seed for the control generation.
        """
        gen = np.random.Generator(np.random.PCG64(seed))

        def control_transform(x: np.ndarray, conn: Connectome):
            def turn_tribe_to_control(boolean_array):
                indices = np.nonzero(boolean_array)[0]
                if layer_profile or mtype_profile:
                    if layer_profile:
                        profile_type = 'layer'
                    else:
                        profile_type = 'mtype'
                    categories = conn.neuron_data[profile_type]
                    category_counts = np.unique(conn.neuron_data.iloc[indices][profile_type], return_counts=True)
                    nhbds = []
                    for cat, value in zip(*category_counts):
                        neuron_pool = conn.neuron_data.iloc[categories == cat].index
                        layer_neurons = gen.choice(neuron_pool, size=value, replace=False)
                        nhbds.append(layer_neurons)
                    try:
                        nhbd = np.concatenate(nhbds)
                    except ValueError: #empty tribe
                        nhbd = []
                    array = np.zeros((len(boolean_array),), dtype=bool)
                    array[nhbd] = True
                    return array
                else:
                    indices_submatrix = gen.choice(len(indices), np.count_nonzero(boolean_array), replace=False)
                mask = np.zeros((len(boolean_array),), dtype=bool)
                mask[indices_submatrix] = True
                return np.logical_and(boolean_array, mask)

            return np.apply_along_axis(
                arr=x,
                axis=1,
                func1d=lambda y: turn_tribe_to_control(y)
            )

        func_name = 'control_transform'
        if layer_profile:
            func_name += '_layer'
        elif mtype_profile:
            func_name += '_mtype'
        self.add_transform(
            control_transform,
            func_name
        )

    def add_transform(self, method: callable, method_name: Optional[str]):
        """
        Add a generic transformation.

        :param method: the method to use in transformation. Should have two inputs: the tribe matrix and the connectome
        object
        :param method_name: name to use in saving the data
        """
        if self.root_path is not None:
            print("Object unrooted")
            self.unroot()
        if method_name is not None:
            method.__name__ = method_name
        self.transform_method_list.append(method)

    def pop_transform(self, method_name: Optional[str]):
        """
        Remove the last transformation or a transformation with a given name.

        :param method_name: if present, removes the first transformation with a given name.
        """
        if method_name is not None:
            for i, method in enumerate(self.transform_method_list):
                if method.__name__ == method_name:
                    self.transform_method_list.pop(i)
                    if self.root_path is not None:
                        print("object unrooted")
                        self.unroot()
                    break
        else:
            self.transform_method_list.pop()
            if self.root_path is not None:
                print("object unrooted")
                self.unroot()

    def add_analysis(self, method: callable, method_name: Optional[str]):
        """
        Add a generic analysis method.

        :param method: the analysis method. Should have three inputs: the boolean array expressing the tribe,
        the submatrix containing the tribe connectivity, and the connectome object.
        :param method_name: name of the analysis method.
        """
        if method_name is not None:
            method.__name__ = method_name
        self.analysis_method_list.append(method)

    def pop_analysis(self, method_name: Optional[str]):
        """
        Remove the last analysis or an analysis with a given name.

        :param method_name: if present, removes the first analysis with a given name.
        """
        if method_name is not None:
            for i, method in enumerate(self.transform_method_list):
                if method.__name__ == method_name:
                    self.analysis_method_list.pop(i)
                    break
        else:
            self.transform_method_list.pop()

    def _store_data(self, analysis_list: List[callable], results: List):
        self.analysis_data.loc[results[0], [method.__name__ for method in analysis_list]] = results[1:]

    def _store_errors(self, analysis_list: List[callable], results: List):
        self.analysis_data_error.loc[results[0], [method.__name__ for method in analysis_list]] = results[1:]

    def analyse(self, analysis_list: Optional[List[callable]] = None):
        """
        Compute tribe data and stores it in the data dataframe.
        :param analysis_list: if present, only compute the passed analysis functions.
        """
        #  In: function, string
        if analysis_list is None:
            analysis_list = self.analysis_method_list
        vertices_passed = self.analysis_data_error[[x.__name__ for x in analysis_list]].dropna().index.values
        tribes = self.tribes()

        def compute_for_chief(chief: int, queue: multiprocessing.Queue):
            tribe_gids = self.connectome_object.gids[tribes[chief]]
            kwargs = {'gids': tribe_gids,
                      'connectivity': self.connectome_object.adjacency[tribe_gids][:, tribe_gids],
                      'adjacency': self.connectome_object.adjacency}
            param_list = [chief]
            error_list = [chief]
            for function in analysis_list:
                try:
                    param_list.append(function(**kwargs))
                    error_list.append(0)
                except Exception as e:
                    param_list.append(np.nan)
                    error_list.append(1)
            queue.put([param_list, error_list])

        def compute_for_chiefs(chief_list, queue):
            for chief in chief_list:
                compute_for_chief(chief, queue)

        vertices_to_do = np.setdiff1d(self.connectome_object.gids, vertices_passed)
        threads = []
        chunks = list(range(0, len(vertices_to_do), 10)) + [len(vertices_to_do)]
        chunks = [[start, end] for start, end in zip(chunks[:-1], chunks[1:])]
        mpqueue = multiprocessing.Queue()
        for j, chunk in enumerate(chunks):
            while len(threads) >= 30:
                for i, thread in enumerate(threads):
                    if not thread.is_alive():
                        threads.pop(i)
                        break
                while not mpqueue.empty():
                    a, b = mpqueue.get()
                    self._store_data(analysis_list, a)
                    self._store_errors(analysis_list, b)
                time.sleep(0.5)
            threads.append(
                multiprocessing.Process(target=compute_for_chiefs, args=(vertices_to_do[chunk[0]:chunk[1]], mpqueue)))
            threads[-1].start()
            if not j % 10:
                print('Updating partial results..' + str(j) + "/" + str(len(chunks)))
                self.analysis_data = self.analysis_data
                self.analysis_data_error = self.analysis_data_error
            if chunk == chunks[-1]:
                print("Assigned all targets.")
                for thread in threads:
                    thread.join()
                while not mpqueue.empty():
                    a, b = mpqueue.get()
                    self._store_data(analysis_list, a)
                    self._store_errors(analysis_list, b)
                self.analysis_data = self.analysis_data
                self.analysis_data_error = self.analysis_data_error

    def root(self, root_path: Path):
        super(TribeView, self).root(root_path)
        self.analysis_data_path = self.root_path / "features.pkl"
        self.analysis_data_error_path = self.root_path / "errors.pkl"
        self.property_registry = {
            "analysis_data": [
                partial(pd.read_pickle, filepath_or_buffer=self.analysis_data_path),
                lambda x: x.to_pickle(self.analysis_data_path),
                self.analysis_data_path.exists
            ],
            "analysis_data_error": [
                partial(pd.read_pickle, filepath_or_buffer=self.analysis_data_error_path),
                lambda x: x.to_pickle(self.analysis_data_error_path),
                self.analysis_data_error_path.exists
            ]
        }
        if self.analysis_data_path.exists():
            self.analysis_data.reindex(columns=[m.__name__ for m in self.analysis_method_list])
            self.analysis_data = self.analysis_data  # store to hd
        else:
            self.analysis_data = pd.DataFrame(np.nan, index=self.connectome_object.gids, dtype=object,
                                              columns=[m.__name__ for m in self.analysis_method_list])

        if self.analysis_data_error_path.exists():
            self.analysis_data_error.reindex(columns=[m.__name__ for m in self.analysis_method_list])
            self.analysis_data_error = self.analysis_data_error  # store to hd
        else:
            self.analysis_data_error = pd.DataFrame(np.nan, index=self.connectome_object.gids, dtype=object,
                                                    columns=[m.__name__ for m in self.analysis_method_list])

    def unroot(self):
        super(TribeView, self).unroot()
