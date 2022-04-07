import unittest
from nrc.connectome import Connectome, TribeView
from nrc.structural import (
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
)
from config import test_data_path
import numpy as np
import pandas as pd
import shutil
import quantities as qt


class GenericAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_connectome = np.array(
            [[0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]]
        )
        self.example_ndata = pd.DataFrame([1, 2, 3])
        self.example_gids = np.array([0, 1, 2, 3, 4])

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)


class TestStructuralAnalysis(GenericAnalysisTest):
    def test_single_analysis(self):
        connectome_target_path = test_data_path / "conn1"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView(transform_method_list=[], connectome_object=c)
        t.tribes()
        t.analyse([tribe_size])
        self.assertTrue(np.all(t.analysis_data['tribe_size'] == np.array([2, 2, 3, 3, 3])))

    def test_whole_analysis(self):
        connectome_target_path = test_data_path / "conn2"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView(transform_method_list=[], connectome_object=c)
        t.analyse()
        self.assertTrue(np.all(t.analysis_data['tribe_size'] == np.array([2, 2, 3, 3, 3])))
        self.assertTrue(np.all(t.analysis_data['average_indegree'] == np.array([1, 1, 1, 1, 1])))
        self.assertTrue(np.all(t.analysis_data['average_outdegree'] == np.array([1, 1, 1, 1, 1])))
        self.assertTrue(np.all(t.analysis_data['average_degree'] == 2*np.array([1, 1, 1, 1, 1])))

    def test_analysis_methods(self):
        connectome_target_path = test_data_path / "conn3"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView([], c)
        gids = c.gids[t.tribes()[0]]
        conn = c.adjacency[gids].T[gids].T
        adj = c.adjacency
        t1_args = [gids, conn, adj]
        gids = c.gids[t.tribes()[2]]
        conn = c.adjacency[gids].T[gids].T
        adj = c.adjacency
        t2_args = [gids, conn, adj]
        self.assertEqual(
            average_indegree(*t1_args),
            1
        )
        self.assertEqual(
            average_outdegree(*t1_args),
            1
        )
        self.assertEqual(
            average_degree(*t1_args),
            2
        )
        self.assertEqual(
            average_indegree(*t2_args),
            1
        )
        self.assertEqual(
            average_degree(*t2_args),
            2
        )
        self.assertEqual(
            cell_count_at_v0(*t1_args),
            [2]
        )
        self.assertEqual(
            cell_count_at_v0(*t2_args),
            [2, 1]
        )
        self.assertEqual(
            chief_indegree(*t1_args),
            1
        )
        self.assertEqual(
            chief_indegree(*t2_args),
            0
        )
        self.assertEqual(
            chief_outdegree(*t2_args),
            2
        )
        self.assertEqual(
            edge_volume(*t1_args),
            2
        )
        self.assertEqual(
            edge_volume(*t2_args),
            3
        )
        self.assertEqual(
            number_of_0_indegree_nodes(*t1_args),
            0
        )
        self.assertEqual(
            number_of_0_outdegree_nodes(*t2_args),
            1
        )
        self.assertEqual(
            in_edge_boundary(*t1_args),
            0
        )
        self.assertEqual(
            out_edge_boundary(*t2_args),
            0
        )
        self.assertEqual(
            reciprocal_connections(*t1_args),
            1
        )
        self.assertEqual(
            reciprocal_connections(*t2_args),
            0
        )
        self.assertEqual(
            average_in_tribe_indegree(*t1_args),
            average_indegree(*t1_args)
        )
        self.assertEqual(
            cell_counts(*t1_args),
            [2]
        )
        self.assertEqual(
            cell_counts(*t2_args),
            [3, 1]
        )
