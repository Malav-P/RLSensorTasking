# run file as python -m unittest tests.test_spacenv
import sys
sys.path.append("./src")

import unittest
import numpy.matlib
import numpy as np



from SensorTasking.spacenv import SpaceEnv
from SensorTasking.metrics import Metric, distance_helper, occlude_helper
from SensorTasking.rewards import asymmetric_gaussian


class TestSpaceEnv(unittest.TestCase):

    def _set_up(self, N = 3, num_targets = 2):

        agent_orbits = numpy.matlib.repmat(np.arange(N), 3,1).T

        target_orbits = np.zeros(shape=(N,3,num_targets))

        for i in range(num_targets):
            target_orbits[:,:,i] = numpy.matlib.repmat(((i+2)*np.arange(N)), 3,1).T

        occlude_metric = Metric(occlude_helper, cutoff=99.0, params={"rB" : [1.7, 1.5, 1.1]})

        
        return agent_orbits, target_orbits, occlude_metric



    def test_constructor(self):
        
        agent_orbits, target_orbits, metric = self._set_up(N=4)

        env = SpaceEnv(agent_orbits, target_orbits, metric)

        self.assertIsInstance(env, SpaceEnv)


    def test_methods(self):

        agent_orbits, target_orbits, metric = self._set_up()

        env = SpaceEnv(agent_orbits, target_orbits, metric)


        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(obs["current action"], 1)
        self.assertTrue(not terminated)
        self.assertTrue(not truncated)
        self.assertEqual(info["tstep"], 1)

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(obs["current action"], 0)
        self.assertEqual(reward, 0)
        self.assertTrue(not terminated)
        self.assertTrue(not truncated)
        self.assertEqual(info["tstep"], 2)

        obs, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(obs["current action"], 2)
        self.assertEqual(len(env.available_actions()), 0)
        self.assertTrue(terminated)
        self.assertTrue(not truncated)
        self.assertEqual(info["tstep"], 3)


    def test_available_actions(self):

        agent_orbits, target_orbits, metric = self._set_up(N=4, num_targets=3)
        env = SpaceEnv(agent_orbits, target_orbits, metric)

        obs, reward, terminated, truncated, info = env.step(1)

        avail_actions = env.available_actions()

        self.assertEqual(len(avail_actions), 1)
        self.assertEqual(avail_actions[0], 0)

    def test_rewards(self):

        agent_orbits, target_orbits, metric = self._set_up(N=4, num_targets=3)
        env = SpaceEnv(agent_orbits, target_orbits, metric)

        env.reset()
        
        
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertAlmostEqual(reward, asymmetric_gaussian(0))
        self.assertEqual(obs["num repeat actions"], 0)

        obs, reward, terminated, truncated, info = env.step(1)
        self.assertAlmostEqual(reward, asymmetric_gaussian(1))
        self.assertEqual(obs["num repeat actions"], 1)

        obs, reward, terminated, truncated, info = env.step(0)
        self.assertAlmostEqual(reward, -1)



