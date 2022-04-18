import unittest
from nrc.connectome import Connectome, SpikeTrainsCollection, Simulation, TribeView
from config import test_data_path
import numpy as np
import pandas as pd
import shutil
import quantities as qt


class GenericConnectomeTest(unittest.TestCase):
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
        self.example_st = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [1, 2, 1, 2, 1, 2, 1, 2]]).T
        self.example_t_start = 0 * qt.s
        self.example_t_stop = 2 * qt.s
        self.example_gids = np.array([1, 2, 3, 4, 5])

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)


class TestConnectomeInstance(GenericConnectomeTest):
    def test_all_items_accessible(self):
        connectome_target_path = test_data_path/"conn1"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)
        self.assertTrue(np.all(c.adjacency == self.example_connectome))
        self.assertTrue(np.all(c.neuron_data == self.example_ndata))

    def test_uninitialized_item_is_loaded(self):
        connectome_target_path = test_data_path/"conn2"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)
        d = Connectome(connectome_target_path, None, None)
        self.assertTrue(np.all(d.adjacency == c.adjacency))
        self.assertTrue(np.all(d.neuron_data == c.neuron_data))

    def test_attribute_deletion_do_not_change_memory(self):
        connectome_target_path = test_data_path / "conn3"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)

        with self.assertRaises(PermissionError):
            del c.adjacency

    def test_instance_is_deleted_without_hd_deletion(self):
        connectome_target_path = test_data_path / "conn4"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)
        path = c.adjacency_matrix_path
        del c
        self.assertTrue(path.exists())

    def test_simulations(self):
        connectome_target_path = test_data_path / "conn5"
        self.unlink_targets.append(connectome_target_path)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(None, [example_sts] * 3)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)
        c.simulations = [s]
        c1 = Connectome(connectome_target_path)
        self.assertTrue(len(c1.simulations) == 1)

    def test_simplex_list(self):
        connectome_target_path = test_data_path / "conn6"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata)
        lists = c.simplex_list(1)
        self.assertTrue(np.all(lists[0] == np.array([[0, 1, 1],
       [1, 0, 1],
       [2, 3, 0],
       [2, 4, 0],
       [3, 4, 0]])))
        self.assertTrue(np.all(lists[1] == np.array([[2, 3, 4, 0]])))


class TestSpikeTrainInstance(GenericConnectomeTest):
    def test_all_items_accessible(self):
        target = test_data_path / "st1"
        self.unlink_targets.append(target)
        stc = SpikeTrainsCollection(target, self.example_st, self.example_t_start,
                                    self.example_t_stop, gids=self.example_gids)
        self.assertTrue(np.all(stc.spikes_array == self.example_st))
        self.assertTrue(stc.t_start == self.example_t_start)
        self.assertTrue(stc.t_stop == self.example_t_stop)

    def test_uninitialized_item_is_loaded(self):
        target = test_data_path / "st2"
        self.unlink_targets.append(target)
        stc = SpikeTrainsCollection(target, self.example_st, self.example_t_start,
                                    self.example_t_stop, gids=self.example_gids)
        stc1 = SpikeTrainsCollection(target)
        self.assertTrue(np.all(stc.spikes_array == stc1.spikes_array))

    def test_neo_spike_trains(self):
        target = test_data_path / "st3"
        self.unlink_targets.append(target)
        stc = SpikeTrainsCollection(target, self.example_st, self.example_t_start,
                                    self.example_t_stop, gids=self.example_gids)
        self.assertTrue(np.all(stc.get_neo_spike_trains()[0] == np.array([0.1, 0.3, 0.5, 0.7])))
        self.assertTrue(len(stc.get_neo_spike_trains()) == 5)

    def test_spike_counts(self):
        target = test_data_path / "st4"
        self.unlink_targets.append(target)
        stc = SpikeTrainsCollection(target, self.example_st, self.example_t_start,
                                    self.example_t_stop, gids=self.example_gids)
        self.assertTrue(np.all(np.array([4, 4, 0, 0, 0]) == stc.get_number_of_spikes()))
        self.assertTrue(np.all(np.array([2, 2, 0, 0, 0]) == stc.get_firing_rates()))
        self.assertEqual(type(stc.get_firing_rates()), qt.quantity.Quantity)

    def test_instance_from_neo_spike_trains(self):
        target1 = test_data_path / "st5"
        target2 = test_data_path / "st6"
        self.unlink_targets.append(target1)
        self.unlink_targets.append(target2)
        stc = SpikeTrainsCollection(target1, self.example_st, self.example_t_start,
                                    self.example_t_stop, gids=self.example_gids)
        stc1 = SpikeTrainsCollection.from_neo_spike_trains(stc.get_neo_spike_trains(), target2, stc.gids)
        self.assertTrue(np.all(
            np.sort(stc.spikes_array, axis=0) == np.sort(stc1.spikes_array, axis=0))
        )
        self.assertTrue(np.all(stc.t_start == stc1.t_start))
        self.assertTrue(np.all(stc.t_stop == stc1.t_stop))


class TestSimulationInstance(GenericConnectomeTest):
    def test_items_accessible(self):
        target = test_data_path / "sim1"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                  self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)

        self.assertTrue(len(s.seeds) == 3)
        self.assertTrue(np.all(s.seeds[0].spikes_array == example_sts.spikes_array))
        self.assertTrue(type(s.seeds[0]) == SpikeTrainsCollection)

    def test_uninitalized_item_is_loaded(self):
        target = test_data_path / "sim2"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                  self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)
        s1 = Simulation(target)
        self.assertTrue(np.all(s.seeds[0].spikes_array == s1.seeds[1].spikes_array))
        self.assertTrue(s.seeds[0].root_path == s1.seeds[0].root_path)

    def test_average_firing_rates(self):
        target = test_data_path / "sim3"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)
        self.assertTrue(np.all(s.average_firing_rate() == np.array([4, 4, 0, 0, 0]) / (2 * qt.s)))

    def test_average_firing_rate_profile(self):
        target = test_data_path / "sim4"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3, gids=self.example_gids)
        self.assertEqual(s.average_firing_rate_profiles(10).shape, (5, 2000))

    def test_from_firing_rate_profile(self):
        target = test_data_path / "sim5"
        target1 = test_data_path / "sim6"
        self.unlink_targets.append(target)
        self.unlink_targets.append(target1)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3, gids=self.example_gids)
        firing_rates = s.average_firing_rate_profiles(10)
        s1 = Simulation.from_firing_rate_profiles(firing_rate_profiles=firing_rates,
                                                  repetitions=5, seeds=0, root_path=target1, gids=self.example_gids)
        self.assertTrue(np.all((s.average_firing_rate_profiles(10) - s1.average_firing_rate_profiles(10)) < 10**-6))


class TestSimulationInstance(GenericConnectomeTest):
    def test_items_accessible(self):
        target = test_data_path / "sim1"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                  self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)

        self.assertTrue(len(s.seeds) == 3)
        self.assertTrue(np.all(s.seeds[0].spikes_array == example_sts.spikes_array))
        self.assertTrue(type(s.seeds[0]) == SpikeTrainsCollection)

    def test_uninitalized_item_is_loaded(self):
        target = test_data_path / "sim2"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                  self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)
        s1 = Simulation(target)
        self.assertTrue(np.all(s.seeds[0].spikes_array == s1.seeds[1].spikes_array))
        self.assertTrue(s.seeds[0].root_path == s1.seeds[0].root_path)

    def test_average_firing_rates(self):
        target = test_data_path / "sim3"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3)
        self.assertTrue(np.all(s.average_firing_rate() == np.array([4, 4, 0, 0, 0]) / (2 * qt.s)))

    def test_average_firing_rate_profile(self):
        target = test_data_path / "sim4"
        self.unlink_targets.append(target)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3, gids=self.example_gids)
        self.assertEqual(s.average_firing_rate_profiles(10).shape, (5, 2000))

    def test_from_firing_rate_profile(self):
        target = test_data_path / "sim5"
        target1 = test_data_path / "sim6"
        self.unlink_targets.append(target)
        self.unlink_targets.append(target1)
        example_sts = SpikeTrainsCollection(None, self.example_st, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target, [example_sts]*3, gids=self.example_gids)
        firing_rates = s.average_firing_rate_profiles(10)
        s1 = Simulation.from_firing_rate_profiles(firing_rate_profiles=firing_rates,
                                                  repetitions=5, seeds=0, root_path=target1, gids=self.example_gids)
        self.assertTrue(np.all((s.average_firing_rate_profiles(10) - s1.average_firing_rate_profiles(10)) < 10**-6))


class TestTribeView(GenericConnectomeTest):
    def test_tribe_view(self):
        connectome_target_path = test_data_path / "tconn1"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView([], c)
        t.tribes()

    def test_transform_add(self):
        connectome_target_path = test_data_path / "tconn2"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView([], c)
        t.tribes()
        t.add_transform(lambda x, conn: x.T, 'transpose')
        t.add_transform(lambda x, conn: x.T, 'transpose')
        t.tribes()
        self.assertTrue((connectome_target_path / 'tribes/_tribes_.pkl').exists())
        self.assertTrue((connectome_target_path / 'tribes/transpose_transpose/_tribes_.pkl').exists())

    def test_autosave(self):
        connectome_target_path = test_data_path / "tconn3"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView([], c)
        t.add_transform(lambda x, conn: x.T, 'transpose')
        t.add_transform(lambda x, conn: x.T, 'transpose')
        t1 = t.tribes()
        self.assertTrue(np.all(t.tribes() == t1))
        t.unroot()
        self.assertTrue(np.all(t1 == t.tribes(root=False)))
        t2 = TribeView(t.transform_method_list, c)
        self.assertTrue(np.all(t2.tribes() == t.tribes()))

    def test_implemented_methods(self):
        connectome_target_path = test_data_path / "tconn4"
        self.unlink_targets.append(connectome_target_path)
        c = Connectome(connectome_target_path, self.example_connectome, self.example_ndata, gids=self.example_gids)
        t = TribeView([], c)
        # threshold
        t.threshold(np.array([False, False, True, True, True]), 'test_threshold')
        self.assertTrue(np.all(t.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, True, True, True],
             [False, False, True, True, True],
             [False, False, True, True, True]]
        )))
        t.threshold(np.array([False, False, False, False, True]), 'test_threshold_2')
        self.assertTrue(np.all(t.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, True],
             [False, False, False, False, True],
             [False, False, False, False, True]]
        )))
        # second degree
        t1 = TribeView([], c)
        t1.second_degree('in')
        self.assertTrue(np.all(t1.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]])))
        t.second_degree('in')
        self.assertTrue(np.all(t.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, True, True, False],
             [False, False, True, True, False],
             [False, False, True, True, False]]
        )))
        t.second_degree('out')
        self.assertTrue(np.all(t.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, True],
             [False, False, False, False, True],
             [False, False, False, False, True]]
        )))
        t.second_degree('all')
        self.assertTrue(np.all(t.tribes() == np.array(
            [[False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, True, True, False],
             [False, False, True, True, False],
             [False, False, True, True, False]]
        )))

if __name__ == '__main__':
    unittest.main()
