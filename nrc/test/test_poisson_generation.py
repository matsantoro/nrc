import unittest

import neo
import numpy as np
from nrc.connectome import Simulation, SpikeTrainsCollection
import pandas as pd
import quantities as qt
import shutil

from nrc.test.config import test_data_path


class TestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.unlink_targets = []
        self.example_connectome = np.array([[1, 2], [1, 2]])
        self.example_ndata = pd.DataFrame([1, 2, 3])
        self.example_gids = np.array([1., 2., 3., 4., 5.])
        self.example_extended_spike_trains = np.clip(100*np.abs(np.random.normal(1, size=500)), 0, 100)
        self.example_extended_spike_trains = np.stack(
            [self.example_extended_spike_trains, np.concatenate([i * np.ones(100) for i in range(1, 6)])]
        ).T
        self.example_t_stop = 100 * qt.ms
        self.example_t_start = 0 * qt.ms

    def test_poisson_generation(self):
        target_sim = test_data_path / 'poisson_sim1'
        self.unlink_targets.append(target_sim)
        example_sts = SpikeTrainsCollection(None, self.example_extended_spike_trains, self.example_t_start,
                                            self.example_t_stop, self.example_gids)
        s = Simulation(target_sim, [example_sts] * 3, example_sts.gids)
        firing_rate_profiles = s.average_firing_rate_profiles(sigma=10)
        self.assertTupleEqual(firing_rate_profiles.shape, (5, 100))
        s1 = Simulation.from_firing_rate_profiles(firing_rate_profiles,
                                                  repetitions=10,
                                                  seeds=0,
                                                  root_path=None,
                                                  gids=s.gids)
        self.assertTupleEqual(s.average_firing_rate_profiles(10).shape, s1.average_firing_rate_profiles(10).shape)


    def tearDown(self) -> None:
        for target in self.unlink_targets:
            shutil.rmtree(target, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()
