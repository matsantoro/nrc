from typing import List

import elephant.kernels
import neo
import quantities as qt
from elephant.conversion import BinnedSpikeTrain
from elephant.kernels import GaussianKernel
from elephant.spike_train_correlation import (
    corrcoef,
    cross_correlation_histogram,
    spike_time_tiling_coefficient,
)
from elephant.spike_train_dissimilarity import (
    van_rossum_distance,
    victor_purpura_distance,
)
from scipy.ndimage import convolve1d

import numpy as np


def _normalize(matrix: np.ndarray):
    """
    Normalizes matrix by dividing by its diagonal values.
    :param matrix: matrix to normalize
    :return nmatrix: normalized matrix.
    """
    variances = np.sqrt(np.diag(matrix))
    m = (matrix / np.expand_dims(variances,0)) / np.expand_dims(variances,1)
    return np.nan_to_num(m)


def _average_triu(matrix):
    """
    Takes the average of the upper triangular part of the matrix.

    :param matrix: matrix to do the average
    :return: a number.
    """
    return np.sum(np.triu(matrix, 1)) / matrix.shape[0] / (matrix.shape[0]-1) * 2


def average_pearson(traces: np.array) -> float:
    """
    Average pearson correlation of signals.

    :param traces: np.array containing the signals to make the correlation of.
    :return: a number.
    """
    # Each row is a trace.
    traces_noavg = traces - np.expand_dims(np.mean(traces, axis = 1), -1)
    similarity_matrix = np.dot(traces_noavg, traces_noavg.T)
    correlation_matrix = _normalize(similarity_matrix)
    return _average_triu(correlation_matrix)


def pearson_matrix(traces: np.array) -> np.ndarray:
    """
    Pearson matrix of a set of signals.

    :param traces: np.ndarray containing the signals
    :return: pearson correlation matrix
    """
    traces_noavg = traces - np.expand_dims(np.mean(traces, axis = 1), -1)
    similarity_matrix = np.dot(traces_noavg, traces_noavg.T)
    correlation_matrix = _normalize(similarity_matrix)
    return correlation_matrix


def average_cosine_distance(traces: np.array) -> float:
    """
    Average cosine distance of a set of signals.

    :param traces: np.array containing the signals
    :return: a number
    """
    similarity_matrix = np.dot(traces, traces.T)
    correlation_matrix = _normalize(similarity_matrix)
    return _average_triu(correlation_matrix)


def cosine_matrix(traces: np.array) -> np.ndarray:
    """
    Cosine similarity matrix of given signals.

    :param traces: The signals to compute the cosine similarity of.
    :return s_matrix: Similarity matrix of the signals.
    """
    similarity_matrix = np.dot(traces, traces.T)
    correlation_matrix = _normalize(similarity_matrix)
    return correlation_matrix


class ReliabilityScore:
    def __init__(self, seed_correlation_matrix: np.ndarray):
        """
        Interface class to work with reliability scores. Hosts original matrix of similarity of seed pairs.

        :param seed_correlation_matrix: matrix of seed pair similarity.
        """
        self.n_seeds = seed_correlation_matrix.shape[0]
        correlations = seed_correlation_matrix[
            np.triu_indices(self.n_seeds, 1)
        ]
        self.value = np.average(correlations)
        self.matrix = seed_correlation_matrix
        self.std = np.std(correlations)


def get_kernel_reliability(
    spike_trains: List[neo.SpikeTrain],
    binsize: qt.quantity.Quantity = 1.0 * qt.ms,
    sigma: qt.quantity.Quantity = 1.0 * qt.ms,
) -> ReliabilityScore:
    """
    Gaussian kernel reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param binsize: bin size to bin the spike trains with
    :param sigma: sigma of the gaussian kernel to use.
    :return score: ReliabilityScore instance of the spike trains.
    """
    binned_spike_trains = BinnedSpikeTrain(
        spike_trains,
        t_start=spike_trains[0].t_start,
        t_stop=spike_trains[0].t_stop,
        bin_size=binsize,
    )
    kernel = GaussianKernel(sigma)
    kernel_values = get_kernel_values(kernel, binsize)
    convolved_signals = convolve1d(
        binned_spike_trains.to_array().astype(float), kernel_values, axis=1
    )
    seed_correlation_matrix = cosine_matrix(convolved_signals)
    return ReliabilityScore(seed_correlation_matrix)


def get_kernel_values(kernel: elephant.kernels.Kernel, binsize: qt.quantity.Quantity):
    """
    Helper function to retrieve the values of a kernel.
    :param kernel: kernel to retrieve the values of.
    :param binsize: size of a bin in milliseconds.
    :return: an array containing the kernel values.
    """
    kernel_bins = np.ceil(2 * kernel.sigma / binsize)
    kernel_values = kernel(np.arange(-kernel_bins, kernel_bins + 1) * binsize)
    return kernel_values


def get_cc_reliability(
    spike_trains: List[neo.SpikeTrain],
    binsize: qt.quantity.Quantity = 1.0 * qt.ms,
    n_lags: int = 2,
):
    """
    Cross-correlogram reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param binsize: bin size to bin the spike trains with
    :param n_lags: number of lags to use
    :return score: ReliabilityScore instance of the spike trains.
    """
    binned_spike_trains = [
        BinnedSpikeTrain(
            spike_train,
            t_start=spike_trains[0].t_start,
            t_stop=spike_trains[0].t_stop,
            bin_size=binsize,
        )
        for spike_train in spike_trains
    ]
    matrix = np.zeros((len(spike_trains), len(spike_trains)))
    for i in range(len(spike_trains)):
        for j in range(i + 1, len(spike_trains)):
            matrix[i, j] = np.average(
                cross_correlation_histogram(
                    binned_spike_trains[i],
                    binned_spike_trains[j],
                    (-n_lags, n_lags),
                    cross_correlation_coefficient=True,
                )[0]
            )
    return ReliabilityScore(matrix)


def get_vr_reliability(
    spike_trains: List[neo.SpikeTrain],
    time_constant: qt.quantity.Quantity = 1.0 * qt.ms,
    convert_to_similarity: bool = False,
):
    """
    Van rossum reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param time_constant: time constant to use
    :param convert_to_similarity: whether to convert the distance to a similarity measure
    :return score: ReliabilityScore instance of the spike trains.
    """
    distance = van_rossum_distance(spike_trains, time_constant)
    if convert_to_similarity:
        distance = 1 / (1 + distance)
    return ReliabilityScore(distance)


def get_vp_reliability(
    spike_trains: List[neo.SpikeTrain],
    time_constant: qt.quantity.Quantity = 1.0 * qt.ms,
    convert_to_similarity: bool = False,
):
    """
    Victor-purpura reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param time_constant: time constant to use
    :param convert_to_similarity: whether to convert the distance to a similarity measure
    :return score: ReliabilityScore instance of the spike trains.
    """
    distance = victor_purpura_distance(spike_trains, 1 / time_constant)
    if convert_to_similarity:
        distance = 1 / (1 + distance)
    return ReliabilityScore(distance)


def get_stt_reliability(
    spike_trains: List[neo.SpikeTrain],
    time_constant: qt.quantity.Quantity = 1.0 * qt.ms,
):
    """
    STT reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param time_constant: time constant to use
    :return score: ReliabilityScore instance of the spike trains.
    """
    matrix = np.zeros((len(spike_trains), len(spike_trains)))
    for i in range(len(spike_trains)):
        for j in range(i + 1, len(spike_trains)):
            if len(spike_trains[i]) and len(spike_trains[j]):
                matrix[i, j] = spike_time_tiling_coefficient(
                    spike_trains[i], spike_trains[j], time_constant
                )
            else:
                matrix[i, j] = 0
    return ReliabilityScore(matrix)


def get_cor_reliability(
    spike_trains: List[neo.SpikeTrain],
    binsize: qt.quantity.Quantity = 1.0 * qt.ms,
):
    """
    Correlation reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param binsize: binsize size.
    :return score: ReliabilityScore instance of the spike trains.
    """
    binned_spike_trains = BinnedSpikeTrain(
        spike_trains,
        t_start=spike_trains[0].t_start,
        t_stop=spike_trains[0].t_stop,
        bin_size=binsize,
    )
    seed_correlation_matrix = corrcoef(binned_spike_trains)
    return ReliabilityScore(seed_correlation_matrix)


def get_cos_reliability(
    spike_trains: List[neo.SpikeTrain],
    binsize: qt.quantity.Quantity = 1.0 * qt.ms,
):
    """
    cosine similarity reliability of a list of neo.SpikeTrain objects.

    :param spike_trains: list of neo.SpikeTrain objects to compute the reliability of.
    :param binsize: binsize size.
    :return score: ReliabilityScore instance of the spike trains.
    """

    binned_spike_trains = BinnedSpikeTrain(
        spike_trains,
        t_start=spike_trains[0].t_start,
        t_stop=spike_trains[0].t_stop,
        bin_size=binsize,
    )
    seed_correlation_matrix = cosine_matrix(binned_spike_trains.to_array())
    return ReliabilityScore(seed_correlation_matrix)


def pearson_reliability(
        convolved_spike_trains: np.ndarray,
):
    seed_correlation_matrix = pearson_matrix(convolved_spike_trains)
    return ReliabilityScore(seed_correlation_matrix)


def cosine_reliability(
        convolved_spike_trains: np.ndarray,
):
    seed_correlation_matrix = cosine_matrix(convolved_spike_trains)
    return ReliabilityScore(seed_correlation_matrix)


def convolve_with_kernel(
        binned_spike_trains: elephant.conversion.BinnedSpikeTrain,
        kernel: elephant.kernels.Kernel,
):
    kernel_values = get_kernel_values(kernel, binned_spike_trains.bin_size)
    return convolve1d(binned_spike_trains.to_array().astype(float), kernel_values, axis=1)