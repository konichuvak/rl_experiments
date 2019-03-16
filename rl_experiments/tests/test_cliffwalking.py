import unittest
from rl_experiments.envs.CliffWalking import *


class CliffWalkingTestCase(unittest.TestCase):
    def setUp(self):
        self.cliff_rewards = {(i, -1): -100 for i in range(1, 11)}
        self.cliffwalking = CliffWalking(width=12, height=4, other_rewards=self.cliff_rewards)

    def test_state_transition(self):
        state = (11, 2)
        action = np.array((1, 0))
        next_state, reward = self.cliffwalking.state_transition(state=state, action=action)
        self.assertEqual(next_state, state)

        state = (11, 2)
        action = np.array((-1, 0))
        next_state, reward = self.cliffwalking.state_transition(state=state, action=action)
        self.assertEqual(next_state, (10, 2))

    def test_q_learning(self):
        self.cliffwalking.control(n_episodes=10, algo='sarsa', verbose=True)


if __name__ == '__main__':
    unittest.main()
