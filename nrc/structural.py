import subprocess
import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg
from scipy.sparse import load_npz
import scipy.sparse.csgraph as csgraph
from numpy.linalg import inv
from pathlib import Path
from functools import wraps
from pyflagser.flagser_count import flagser_count_unweighted
from pyflagsercontain import compute_cell_count
from pyflagsercount import pyflagsercount


def from_adjacency(local_function):
    @wraps(local_function)
    def global_function(gids, connectivity=None, adjacency=None):
        if adjacency is None:
            return global_function(gids, connectivity)
        else:
            return local_function(gids, adjacency[gids].T[gids].T)
    return global_function

######### helper stuff to be removed
def np_to_nx(adjacency_matrix):
#  In: numpy array
# Out: networkx directed graph
    return nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)


def nx_to_np(directed_graph):
#  In: networkx directed graph
# Out: numpy array
    return nx.to_numpy_array(directed_graph,dtype=int)


@from_adjacency
def cell_count_at_v0(gids, connectivity):
#  In: adjacency matrix
# Out: list of integers
    simplexcontainment = compute_cell_count(connectivity.shape[0], np.transpose(np.array(np.nonzero(connectivity))))
    return simplexcontainment[0][1:]  # exclude ec

@from_adjacency
def chief_indegree(gids, connectivity):
    return np.sum(connectivity[:,0])

@from_adjacency
def chief_outdegree(gids, connectivity):
    return np.sum(connectivity[0])


def spectral_gap(matrix, thresh=10, param='low'):
    #  In: matrix
    # Out: float
    current_spectrum = spectrum_make(matrix)
    current_spectrum = spectrum_trim_and_sort(current_spectrum, threshold_decimal=thresh)
    return spectrum_param(current_spectrum, parameter=param)


def spectrum_make(matrix):
    #  In: matrix
    # Out: list of complex floats
    assert np.any(matrix), 'Error (eigenvalues): matrix is empty'
    eigenvalues = scipy.linalg.eigvals(matrix)
    return eigenvalues


def spectrum_trim_and_sort(spectrum, modulus=True, threshold_decimal=10):
    #  In: list of complex floats
    # Out: list of unique (real or complex) floats, sorted by modulus
    if modulus:
        return np.sort(np.unique(abs(spectrum).round(decimals=threshold_decimal)))
    else:
        return np.sort(np.unique(spectrum.round(decimals=threshold_decimal)))


def spectrum_param(spectrum, parameter):
    #  In: list of complex floats
    # Out: float
    assert len(spectrum) != 0, 'Error (eigenvalues): no eigenvalues (spectrum is empty)'
    if parameter == 'low':
        if spectrum[0]:
            return spectrum[0]
        else:
            assert len(spectrum) > 1, 'Error (low spectral gap): spectrum has only zeros, cannot return nonzero eigval'
            return spectrum[1]
    elif parameter == 'high':
        assert len(
            spectrum) > 1, 'Error (high spectral gap): spectrum has one eigval, cannot return difference of top two'
        return spectrum[-1] - spectrum[-2]
    elif parameter == 'radius':
        return spectrum[-1]


@from_adjacency
def tcc(gids, connectivity):
    outdeg = np.count_nonzero(connectivity[0])
    indeg = np.count_nonzero(np.transpose(connectivity)[0])
    repdeg = np.count_nonzero(np.multiply(connectivity, connectivity.T)[0])
    totdeg = indeg + outdeg
    chief_containment = cell_count_at_v0(connectivity)
    numerator = 0 if len(chief_containment) < 3 else chief_containment[2]
    denominator = (totdeg * (totdeg - 1) - (indeg * outdeg + repdeg))
    if denominator == 0:
        return 0
    return numerator / denominator


@from_adjacency
def ccc(gids, connectivity):
    deg = chief_outdegree(gids, connectivity) + chief_indegree(gids, connectivity)
    numerator = np.linalg.matrix_power(connectivity + np.transpose(connectivity), 3)[0][0]
    denominator = 2 * (deg * (deg - 1) - 2 * np.count_nonzero(np.multiply(connectivity, connectivity.T)[0]))
    if denominator == 0:
        return 0
    return numerator / denominator


@from_adjacency
def tribe_size(gids, connectivity):
    return len(connectivity)


@from_adjacency
def reciprocal_connections(gids, connectivity):
    rc_count = np.count_nonzero(np.multiply(connectivity, np.transpose(connectivity))) // 2
    return rc_count


@from_adjacency
def number_of_0_indegree_nodes(gids, connectivity):
    return np.sum(np.sum(connectivity, axis=1) == 0)

@from_adjacency
def number_of_0_outdegree_nodes(gids, connectivity):
    return np.sum(np.sum(connectivity, axis=0) == 0)


def average_indegree(gids, connectivity, adjacency):
    return np.mean(np.sum(adjacency[:, gids], axis=0))


def average_outdegree(gids, connectivity, adjacency):
    return np.mean(np.sum(adjacency[gids, :], axis=1))


def average_degree(gids, connectivity, adjacency):
    return average_outdegree(gids, connectivity, adjacency) + average_indegree(gids, connectivity, adjacency)


@from_adjacency
def average_in_tribe_indegree(gids, connectivity):
    return np.mean(np.sum(connectivity, axis=1))


def in_edge_boundary(gids, connectivity, adjacency):
    return np.sum(adjacency[:, gids]) - np.sum(connectivity)


def out_edge_boundary(gids, connectivity, adjacency):
    return np.sum(adjacency[gids, :]) - np.sum(connectivity)


@from_adjacency
def edge_volume(gids, connectivity):
    return np.sum(connectivity)


@from_adjacency
def cell_counts(gids, connectivity):
    return flagser_count_unweighted(connectivity)[1:]


@from_adjacency
def connected_components(gids, connectivity):
    return csgraph.connected_components(connectivity)


@from_adjacency
def asg_low(gids, connectivity):
    return spectral_gap(connectivity, param='low')


@from_adjacency
def asg_high(gids, connectivity):
    return spectral_gap(connectivity, param='high')


@from_adjacency
def asg_radius(gids, connectivity):
    return spectral_gap(connectivity, param='radius')


@from_adjacency
def tpsg_high(gids, connectivity):
    return spectral_gap(tps_matrix(connectivity), param='high')


@from_adjacency
def tpsg_low(gids, connectivity):
    return spectral_gap(tps_matrix(connectivity), param='low')


@from_adjacency
def tpsg_radius(gids, connectivity):
    return spectral_gap(tps_matrix(connectivity), param='radius')


def tps_matrix(connectivity):
    #  in: tribe matrix
    # out: transition probability matrix
    degree_vector = np.sum(connectivity, axis=1)
    inverted_degree_vector = np.nan_to_num(1/degree_vector)
    return np.matmul(np.diagflat(inverted_degree_vector), connectivity)


@from_adjacency
def clsg_high(gids, connectivity):
    return spectral_gap(cls_matrix_fromadjacency(connectivity), param='high')


@from_adjacency
def clsg_low(gids, connectivity):
    return spectral_gap(cls_matrix_fromadjacency(connectivity), param='radius')


@from_adjacency
def clsg_radius(gids, connectivity):
    #  in: index
    # out: float
    return spectral_gap(cls_matrix_fromadjacency(connectivity), param='radius')


def cls_matrix_fromadjacency(matrix, is_strongly_conn=False):
    #  in: numpy array
    # out: numpy array
    matrix_nx = np_to_nx(matrix)
    return cls_matrix_fromdigraph(matrix_nx, matrix=matrix, matrix_given=True, is_strongly_conn=is_strongly_conn)


def cls_matrix_fromdigraph(digraph, matrix=np.array([]), matrix_given=False, is_strongly_conn=False):
    #  in: networkx digraph
    # out: numpy array
    digraph_sc = digraph
    matrix_sc = matrix
    # Make sure is strongly connected
    if not is_strongly_conn:
        largest_comp = max(nx.strongly_connected_components(digraph), key=len)
        digraph_sc = digraph.subgraph(largest_comp)
        matrix_sc = nx_to_np(digraph_sc)
    elif not matrix_given:
        matrix_sc = nx_to_np(digraph_sc)
    # Degeneracy: scc has size 1
    if not np.any(matrix_sc):
        return np.array([[0]])
    # Degeneracy: scc has size 2
    elif np.array_equal(matrix_sc, np.array([[0, 1], [1, 0]], dtype=int)):
        return np.array([[1, -0.5], [-0.5, 1]])
    # No degeneracy
    else:
        return nx.directed_laplacian_matrix(digraph_sc)


@from_adjacency
def blsg_high(gids, connectivity):
    return spectral_gap(bls_matrix(connectivity), param='high')


@from_adjacency
def blsg_low(gids, connectivity):
    return spectral_gap(bls_matrix(connectivity), param='high')


@from_adjacency
def blsg_radius(gids, connectivity):
    return spectral_gap(bls_matrix(connectivity), param='radius')


def bls_matrix(matrix):
    #  in: tribe matrix
    # out: bauer laplacian matrix
    # non_quasi_isolated = [i for i in range(len(matrix)) if matrix[i].any()]
    # matrix_D = np.diagflat([np.count_nonzero(matrix[nqi]) for nqi in non_quasi_isolated])
    # matrix_W = np.diagflat([np.count_nonzero(np.transpose(matrix)[nqi]) for nqi in non_quasi_isolated])
    # return np.subtract(np.eye(len(non_quasi_isolated),dtype=int),np.matmul(inv(matrix_D),matrix_W))
    current_size = len(matrix)
    return np.subtract(np.eye(current_size, dtype='float64'), tps_matrix(matrix))


structural_function_list = [
    average_indegree,
    average_degree,
    average_outdegree,
    asg_low,
    asg_high,
    asg_radius,
    average_in_tribe_indegree,
    blsg_high,
    blsg_low,
    blsg_radius,
    cell_count_at_v0,
    cell_counts,
    clsg_radius,
    clsg_low,
    clsg_high,
    chief_indegree,
    chief_outdegree,
    edge_volume,
    in_edge_boundary,
    number_of_0_indegree_nodes,
    number_of_0_outdegree_nodes,
    out_edge_boundary,
    reciprocal_connections,
    tpsg_high,
    tpsg_low,
    tpsg_radius,
    tribe_size,
    tcc
]