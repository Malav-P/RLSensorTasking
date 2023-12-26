import unittest

import sys
sys.path.append("./src")

from SensorTasking.actor import *


class TestActor(unittest.TestCase):

    def test_constructor(self):
        input_dim = 3
        output_dim = 3

        actor = Actor(input_dim, output_dim)

        self.assertIsInstance(actor, Actor)

    def test_forward(self):
        input_dim = 3
        output_dim = 4

        actor = Actor(input_dim, output_dim)

        state = torch.tensor([1.0, 1.0, 1.0])
        action_mask = torch.tensor([False, True, True, False])

        action_probs = actor(state, action_mask=action_mask)

        self.assertAlmostEqual(action_probs[0], 0.0)
        self.assertAlmostEqual(action_probs[3], 0.0)
        self.assertAlmostEqual(action_probs.sum(), 1.0)



if __name__ == "__name__":
    unittest.main()