from typing import Optional, Union, List
from pathlib import Path

import elephant.conversion
import numpy as np
import pandas as pd
import scipy.sparse as sp
import neo
import quantities as qt
from functools import partial, wraps

import nrc.reliability
from nrc.config import registry_property
from nrc.reliability import convolve_with_kernel
import pickle


def autosave_method(method):
    @wraps(method)
    def autosave(*args, **kwargs):
        obj = args[0]
        root = obj.root
        if root is None:
            print("Object is unrooted. Cannot save.")
            return method(*args, **kwargs)
        if len(args) > 1:
            print("Each argument must be explicitly stated to have autosave. " +
                  "Result of " + str(method.__name__) + " not saved")
            return method(*args, **kwargs)
        else:
            target_string = '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs)])
            target = root / (method.__name__ + '_' + target_string + '.pkl')
            if target.exists():
                print("Target already computed. Retrieved from " + str(target))
                return pickle.load(target.open('rb'))
            else:
                print("Target doesn't exist. Saving to " + str(target))
                res = method(*args, **kwargs)
                pickle.dump(res, target.open('wb'))
                return res
    return autosave


class SpikeTrainsCollection:
    spikes_array = registry_property('spikes_array')
    t_start = registry_property('t_start')
    t_stop = registry_property('t_stop')
    gids = registry_property('gids')

    def __init__(self, root: Optional[Path], s_array: Optional[np.ndarray] = None,
                 t_start: Optional[Union[qt.quantity.Quantity, int]] = None,
                 t_stop: Optional[Union[qt.quantity.Quantity, int]] = None,
                 gids: Optional[np.ndarray] = None):
        self.root = root
        if self.root is not None:
            self.root.mkdir(exist_ok=True, parents=True)
            self.spikes_array_path = root / "spikes.npy"
            self.t_start_path = root / "t_start.npy"
            self.t_stop_path = root / "t_stop.npy"
            self.gids_path = root / "gids.npy"
            self.property_registry = {
                'spikes_array': [partial(np.load, file=self.spikes_array_path),
                                  lambda x: np.save(arr=x, file=str(self.spikes_array_path)),
                                  self.spikes_array_path.exists],
                't_start': [partial(np.load, file=self.t_start_path),
                             lambda x: np.save(arr=x, file=str(self.t_start_path)),
                             self.t_start_path.exists],
                't_stop': [partial(np.load, file=self.t_stop_path),
                           lambda x: np.save(arr=x, file=str(self.t_stop_path)),
                           self.t_stop_path.exists],
                'gids' : [partial(np.load, file=self.gids_path),
                          lambda x: np.save(arr=x, file=str(self.gids_path)),
                          self.gids_path.exists]
            }
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
    def get_neo_spike_trains(self):
        if self.neo_spike_trains is not None:
            return self.neo_spike_trains
        else:
            self.neo_spike_trains = [
                neo.SpikeTrain(times=self.spikes_array[self.spikes_array[:, 0] == gid][:, 1],
                               t_start=self.t_start, t_stop=self.t_stop, units=qt.ms)
                for gid in self.gids
            ]
            return self.neo_spike_trains

    @autosave_method
    def get_binned_spike_trains(self, time_bin: Union[qt.quantity.Quantity, int]):
        if type(time_bin) is int:
            _time_bin = time_bin * qt.ms
        else:
            _time_bin = time_bin
        return elephant.conversion.BinnedSpikeTrain(self.get_neo_spike_trains(), bin_size=_time_bin)

    def convolve_with_kernel(self, time_bin: Union[qt.quantity.Quantity, int], kernel: elephant.kernels.Kernel):
        if type(time_bin) is int:
            _time_bin = time_bin * qt.ms
        else:
            _time_bin = time_bin
        return nrc.reliability.convolve_with_kernel(self.get_binned_spike_trains(time_bin=time_bin), kernel)

    @autosave_method
    def convolve_with_gaussian_kernel(self, time_bin: Union[qt.quantity.Quantity, int],
                                      sigma: Union[qt.quantity.Quantity, int]):
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

    def __hash__(self):
        if self.root is not None:
            return hash(str(self.root))
        else:
            raise NotImplementedError("Unrooted objects cannot be hashed.")


class Simulation:
    seeds = registry_property('seeds')
    gids = registry_property('gids')

    def __init__(self, root: Optional[Path], spike_train_collection_list: Optional[List[SpikeTrainsCollection]] = None,
                 gids: Optional[np.ndarray] = None):
        self.root = root
        if self.root is not None:
            self.root.mkdir(exist_ok=True, parents=True)
            self.gids_path = self.root / "gids.npy"

            def retrieve_seeds_from_memory():
                st_list = []
                for seed_path in sorted(list(self.root.glob("seed*/")), key=lambda x: int(x.stem[4:])):
                    st_list.append(SpikeTrainsCollection(seed_path))
                return st_list

            def store_seeds_to_memory(seeds):
                for i, seed in enumerate(seeds):
                    target_for_seed = self.root / ("seed" + str(i))
                    instance = SpikeTrainsCollection(target_for_seed, seed.spikes_array,
                                                     seed.t_start, seed.t_stop, seed.gids)

            def check_if_seed():
                return self.root.glob("seed*") is not None

            self.property_registry = {
                'seeds': [retrieve_seeds_from_memory, store_seeds_to_memory, check_if_seed],
                'gids': [partial(np.load, file=self.gids_path),
                         lambda x: np.save(arr=x, file=str(self.gids_path)),
                         self.gids_path.exists]
            }
        if spike_train_collection_list is not None:
            self.seeds = spike_train_collection_list

        if gids is not None:
            self.gids = gids

    def __hash__(self):
        if self.root is not None:
            return hash(str(self.root))
        else:
            raise NotImplementedError("Unrooted objects cannot be hashed.")

    def convolve_with_kernel(self, time_bin: Union[int, qt.quantity.Quantity], kernel: elephant.kernels.Kernel):
        return np.stack([st.convolve_with_kernel(time_bin=time_bin, kernel=kernel) for st in self.seeds])

    @autosave_method
    def convolve_with_gk(self, time_bin: Union[int, qt.quantity.Quantity], sigma: Union[int, qt.quantity.Quantity], ):
        return np.stack([st.convolve_with_gaussian_kernel(time_bin=time_bin, sigma=sigma) for st in self.seeds])

    @autosave_method
    def get_binned_spike_trains(self, time_bin: Union[int, qt.quantity.Quantity]):
        return np.stack([st.get_binned_spike_trains(time_bin=time_bin) for st in self.seeds])

    @autosave_method
    def gk_rel_scores(self, time_bin: Union[int, qt.quantity.Quantity], sigma: Union[int, qt.quantity.Quantity]):
        convolved_sts = self.convolve_with_gk(time_bin, sigma)
        return [nrc.reliability.cosine_reliability(convolved_sts[:, i, :]) for i in range(len(self.gids))]


class Connectome:
    adjacency = registry_property('adjacency', )
    neuron_data = registry_property('neuron_data', )
    simulations = registry_property('simulations', )
    gids = registry_property('gids', )

    def __init__(self, root: Optional[Path], adjacency_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 ndata: Optional[pd.DataFrame] = None, simulations: Optional[List[Simulation]] = None,
                 gids: Optional[np.ndarray] = None):
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.adjacency_matrix_path = self.root / "adj.npy"
            self.neuron_data_path = self.root / "ndata.pkl"
            self.gids_path = self.root / "gids.npy"

            def retrieve_sims_from_memory():
                sim_list = []
                for sim_path in sorted(list(self.root.glob("sim*/")), key=lambda x: int(x.stem[3:])):
                    sim_list.append(Simulation(sim_path))
                return sim_list

            def store_sims_to_memory(sims):
                for i, sim in enumerate(sims):
                    target_for_sim = self.root / ("sim" + str(i))
                    instance = Simulation(target_for_sim,
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

            if adjacency_matrix is not None:
                self.adjacency = adjacency_matrix

            if ndata is not None:
                self.neuron_data = ndata

            if simulations is not None:
                self.simulations = simulations

            if gids is not None:
                self.gids = gids

    def __hash__(self):
        if self.root is not None:
            return hash(str(self.root))
        else:
            raise NotImplementedError("Unrooted objects cannot be hashed.")