import unittest
from nrc.connectome import Connectome, SpikeTrainsCollection, Simulation
from config import test_data_path
import numpy as np
import pandas as pd
import shutil
import quantities as qt


class TestConnectomeInstance(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_connectome = np.array([[1, 2], [1, 2]])
        self.example_ndata = pd.DataFrame([1, 2, 3])
        self.example_st = np.array([[1, 2, 1, 2, 1, 2, 1, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]).T
        self.example_t_start = 0 * qt.s
        self.example_t_stop = 1 * qt.s
        self.example_gids = np.array([1, 2])

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

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)


class TestSpikeTrainInstance(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_st = np.array([[1, 2, 1, 2, 1, 2, 1, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]).T
        self.example_t_start = 0 * qt.s
        self.example_t_stop = 1 * qt.s
        self.example_gids = np.array([1, 2])

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
        print(stc.get_neo_spike_trains())
        self.assertTrue(np.all(stc.get_neo_spike_trains()[0] == np.array([0.1, 0.3, 0.5, 0.7])))
        self.assertTrue(len(stc.get_neo_spike_trains()) == 2)

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)


class TestSimulationInstance(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_st = np.array([[1, 2, 1, 2, 1, 2, 1, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]).T
        self.example_t_start = 0 * qt.s
        self.example_t_stop = 1 * qt.s
        self.example_gids = np.array([1, 2])

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
        self.assertTrue(s.seeds[0].root == s1.seeds[0].root)

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
