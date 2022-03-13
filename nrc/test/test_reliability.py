import unittest
from nrc.connectome import Connectome, SpikeTrainsCollection, Simulation
from config import test_data_path
import numpy as np
import pandas as pd
import shutil
import quantities as qt


class TestConvolution(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_connectome = np.array([[1, 2], [1, 2]])
        self.example_ndata = pd.DataFrame([1, 2, 3])
        self.example_gids = np.array([1, 2, 3, 4, 5])
        self.example_extended_spike_trains = np.clip(1*np.abs(np.random.normal(1, size=500)), 0, 100)
        self.example_extended_spike_trains = np.stack(
            [self.example_extended_spike_trains, np.concatenate([i * np.ones(100) for i in range(1, 6)])]
        )
        self.example_t_stop = 100 * qt.ms
        self.example_t_start = 0 * qt.ms

    def test_convolution(self):
        example_sts = SpikeTrainsCollection(None, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        c = example_sts.convolve_with_gaussian_kernel(1, 4)
        self.assertEqual(c.shape, (5, 100))

    def test_binarization_save(self):
        target_sts = test_data_path / 'st1'
        self.unlink_targets.append(target_sts)
        example_sts = SpikeTrainsCollection(target_sts, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        a1 = example_sts.get_binned_spike_trains(time_bin=10)
        self.assertEqual(len(list(target_sts.glob("get_binned_spike_trains*"))), 1)
        self.assertEqual(a1, example_sts.get_binned_spike_trains(time_bin=10))

    def test_different_binarizations_are_ok(self):
        target_sts = test_data_path / 'st2'
        self.unlink_targets.append(target_sts)
        example_sts = SpikeTrainsCollection(target_sts, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        example_sts.get_binned_spike_trains(time_bin=10)
        example_sts.get_binned_spike_trains(time_bin=20)
        self.assertEqual(len(list(target_sts.glob("get_binned_spike_trains*"))), 2)

    def test_convolution_save(self):
        target_sts = test_data_path / 'st3'
        self.unlink_targets.append(target_sts)
        example_sts = SpikeTrainsCollection(target_sts, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        c = example_sts.convolve_with_gaussian_kernel(time_bin=1, sigma=4)
        self.assertTrue(np.all(c == example_sts.convolve_with_gaussian_kernel(time_bin=1, sigma=4)))

    def test_simulation_convolution(self):
        target_sim = test_data_path / 'sim1'
        self.unlink_targets.append(target_sim)
        example_sts = SpikeTrainsCollection(None, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target_sim, [example_sts]*3, example_sts.gids)
        c = s.convolve_with_gk(time_bin=1, sigma=4)
        c1 = s.convolve_with_gk(time_bin=1, sigma=4)
        self.assertTrue(np.all(c == c1))

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)
        pass

class TestReliability(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_connectome = np.array([[1, 2], [1, 2]])
        self.example_ndata = pd.DataFrame([1, 2, 3])
        self.example_gids = np.array([1, 2, 3, 4, 5])
        self.example_extended_spike_trains = np.clip(1*np.abs(np.random.normal(1, size=500)), 0, 100)
        self.example_extended_spike_trains = np.stack(
            [self.example_extended_spike_trains, np.concatenate([i * np.ones(100) for i in range(1, 6)])]
        ).T
        self.example_t_stop = 100 * qt.ms
        self.example_t_start = 0 * qt.ms

    def test_gk_reliability_computation(self):
        target_sim = test_data_path / 'sim1'
        self.unlink_targets.append(target_sim)
        example_sts = SpikeTrainsCollection(None, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target_sim, [example_sts]*3, example_sts.gids)
        scores = s.gk_rel_scores(time_bin=5, sigma=10)
        self.assertAlmostEqual(scores[0].value, 1)
        self.assertAlmostEqual(scores[1].value, 1)

    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)
        pass


if __name__ == '__main__':
    unittest.main()
