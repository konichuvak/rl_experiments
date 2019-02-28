import numpy as np
from typing import Dict, List
import random
import plotly.graph_objs as go


ROOK_ACTIONS = frozenset({(0, -1), (-1, 0), (0, 1), (1, 0)})


class GridWorld(object):

    def __init__(self, width: int, height: int, actions: List[tuple] = ROOK_ACTIONS,
                 default_reward: float = -1,
                 other_rewards: Dict[tuple, float] = None):
        self.width = width
        self.height = height
        self.grid = np.ones((width, height))
        self.rewards = self._generate_rewards(default_reward, other_rewards)
        self.actions = list(map(np.array, actions))

    def _generate_rewards(self, default_reward, other_rewards):
        """
        Creates reward grid
        :param default_reward:  default reward for transitioning to a given state in grid world
        :param other_rewards:   dict with coordinates as keys and reward as values for other rewards
        :return:
        """
        rewards = self.grid * default_reward
        if other_rewards is None:
            other_rewards = {}

        for coord, r in other_rewards.items():
            rewards[coord] = r
        return rewards


class CliffWalking(GridWorld):

    def __init__(self, *args, **kwargs):
        super(CliffWalking, self).__init__(*args, **kwargs)
        self.start_state = (11, 3)
        self.goal = (0, 3)

    def state_transition(self, state, action):
        """
        :param state:   tuple of (x, y) coordinates of the agent in the grid
        :param action:  performed action
        :return:        (i, j) tuple of the next state and the reward associated with the transition
        """

        state = np.array(state)
        next_state = tuple(state + action)
        x, y = next_state
        reward = self.rewards[tuple(state)]

        # check boundary conditions
        if reward == -100:
            next_state = self.start_state
        elif x < 0 or y < 0 or x >= self.width or y >= self.height:
            next_state = tuple(state)

        return next_state, reward

    def epsilon_greedy(self, action_index, epsilon):
        if random.random() < epsilon:
            action_index = random.randint(0,len(self.actions)-1)
        return action_index

    def control(self, algo: str = 'q_learning', n_episodes: int = 100, alpha: float = 0.1, gamma: float = 1, epsilon: float = 0.1, verbose: bool = False):
        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.random.random(size=q_shape)

        per_episode_rewards = []
        for e in range(n_episodes):
            reward_sum = 0
            state = self.start_state
            while state != self.goal:
                # take action based on policy
                a = np.argmax(q_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a, epsilon)
                state_next, reward = self.state_transition(state, self.actions[a])
                q_index = tuple([a] + list(state))

                a_next = [np.argmax(q_values[:, state_next[0], state_next[1]])]
                if algo == 'sarsa':
                    a_next = self.epsilon_greedy(a_next, epsilon)
                q_index_next = tuple([a_next] + list(state_next))

                q_values[q_index] += alpha * (reward + gamma*(q_values[q_index_next]) - q_values[q_index])
                state = state_next
                reward_sum += reward
                if reward == -100:
                    break
            per_episode_rewards.append(reward_sum)
        return per_episode_rewards

    @staticmethod
    def plot_rewards(rewards: dict):
        traces = list()
        for algo, reward in rewards.items():
            traces.append(go.Scatter(
                    mode='lines',
                    y=reward,
                    name=algo,
                )
            )

        layout = dict(
                height=700,
                showlegend=True,
                xaxis=dict(
                        title='Epsiodes',
                ),
                yaxis=dict(
                        title='Sum of rewards per episode',
                )
        )
        return {'data': traces, 'layout': layout}
