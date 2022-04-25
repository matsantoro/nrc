import elephant.kernels
import neo
from typing import List, Union
import quantities as qt
import numpy as np


def retrieve_fr_profile_from_sts(spike_trains_list: List[neo.SpikeTrain], kernel: elephant.kernels.Kernel,):
    """
    Firing rate profiles from spike trains objects.
    :param spike_trains_list: list of eno.SpikeTrains objects, one per neuron.
    :param kernel: kernel to use.
    :return: istantaneous rate of all neurons.
    """
    ir = elephant.statistics.instantaneous_rate(spike_trains_list, 1 * qt.ms, kernel=kernel,
                                                t_start=spike_trains_list[0].t_start,
                                                t_stop=spike_trains_list[0].t_stop)
    return ir


def generate_spike_trains_from_profile(profile: neo.AnalogSignal, repetitions: int, seed: int) -> List[neo.SpikeTrain]:
    """
    Generate a number of spike train objects from a given firing rate profile.

    :param profile: firing rate profile for a single neuron.
    :param repetitions: number of repetitions
    :param seed: seed for the whole process.
    :return: list of neo.SpikeTrain objects.
    """
    np.random.seed(seed)
    return [elephant.spike_train_generation.inhomogeneous_poisson_process(profile) for i in range(repetitions)]


def generate_spike_trains_from_firing_rate(firing_rate: qt.quantity.Quantity, repetitions: int, seed: int,
                                           t_start: Union[int, qt.quantity.Quantity],
                                           t_stop: Union[int, qt.quantity.Quantity]) -> List[neo.SpikeTrain]:
    """
    Generate list of spike trains from homogeneous poisson process.

    :param firing_rate: quantities object containing the firing rate to generate spike trains with.
    :param repetitions: number of sts to generate
    :param seed: seed for the process
    :param t_start: start in milliseconds of the new spike trains
    :param t_stop: end in milliseconds of the new spike trains
    """
    np.random.seed(seed)
    return [
        elephant.spike_train_generation.homogeneous_poisson_process(
            firing_rate,
            t_start=t_start,
            t_stop=t_stop,
        ) for i in range(repetitions)
    ]

