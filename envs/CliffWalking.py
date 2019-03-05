import numpy as np
from typing import Dict, List
import random
import plotly.graph_objs as go
import ray

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

        # check boundary conditions
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            next_state = tuple(state)

        reward = self.rewards[tuple(next_state)]
        if reward == -100:
            next_state = self.start_state

        return next_state, reward

    def epsilon_greedy(self, action_index, epsilon):
        if random.random() < epsilon:
            action_index = random.randint(0, len(self.actions) - 1)
        return action_index

    @ray.remote
    def expected_sarsa(self, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                       epsilon: float = 0.1, verbose: bool = False):
        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            grid = init_grid

        per_episode_rewards = []
        for e in range(n_episodes):
            reward_sum = 0

            state = self.start_state
            a = np.argmax(q_values[:, state[0], state[1]])
            a = self.epsilon_greedy(a, epsilon)

            while state != self.goal:

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                # take action based on policy
                state_next, reward = self.state_transition(state, self.actions[a])
                a_next = np.argmax(q_values[:, state_next[0], state_next[1]])

                q_index = a, state[0], state[1]
                q_index_next = a_next, state_next[0], state_next[1]

                q_values_next = (1 - epsilon) * q_values[q_index_next] + \
                                sum([q_values[a, state[0], state[1]] * epsilon / len(self.actions) for a in
                                     range(len(self.actions))])
                q_values[q_index] += alpha * (reward + gamma * q_values_next - q_values[q_index])

                state, a = state_next, a_next
                reward_sum += reward

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    grid[state] = 1

            per_episode_rewards.append(reward_sum)
        return per_episode_rewards

    @ray.remote
    def sarsa(self, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
              epsilon: float = 0.1, verbose: bool = False):
        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            grid = init_grid

        per_episode_rewards = []
        for e in range(n_episodes):
            reward_sum = 0

            state = self.start_state
            a = np.argmax(q_values[:, state[0], state[1]])
            a = self.epsilon_greedy(a, epsilon)

            while state != self.goal:

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                state_next, reward = self.state_transition(state, self.actions[a])
                a_next = np.argmax(q_values[:, state_next[0], state_next[1]])
                a_next = self.epsilon_greedy(a_next, epsilon)

                q_index = a, state[0], state[1]
                q_index_next = a_next, state_next[0], state_next[1]
                q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                state, a = state_next, a_next
                reward_sum += reward

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    grid[state] = 1

            per_episode_rewards.append(reward_sum)
        return per_episode_rewards

    @ray.remote
    def q_learning(self, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                   epsilon: float = 0.1, verbose: bool = False):
        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            grid = init_grid

        per_episode_rewards = []
        for e in range(n_episodes):
            reward_sum = 0
            state = self.start_state

            while state != self.goal:

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                a = np.argmax(q_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a, epsilon)

                state_next, reward = self.state_transition(state, self.actions[a])
                a_next = np.argmax(q_values[:, state_next[0], state_next[1]])

                q_index = a, state[0], state[1]
                q_index_next = a_next, state_next[0], state_next[1]
                q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                state = state_next
                reward_sum += reward

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    grid[state] = 1

            per_episode_rewards.append(reward_sum)
        return per_episode_rewards

    @ray.remote
    def double_q_learning(self, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                          epsilon: float = 0.1, verbose: bool = False):

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)
        q2_values = q_values.copy()

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            grid = init_grid

        per_episode_rewards = []
        for e in range(n_episodes):

            reward_sum = 0
            state = self.start_state

            while state != self.goal:

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                a = np.argmax(q_values[:, state[0], state[1]])
                a2 = np.argmax(q2_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a if q_values.flatten()[a] >= q2_values.flatten()[a2] else a2, epsilon)
                q_index = a, state[0], state[1]

                # take action based on policy
                state_next, reward = self.state_transition(state, self.actions[a])

                if random.random() < 0.5:
                    a_next = np.argmax(q_values[:, state_next[0], state_next[1]])
                    q_index_next = a_next, state_next[0], state_next[1]
                    q_values[q_index] += alpha * (reward + gamma * (q2_values[q_index_next]) - q_values[q_index])
                else:
                    a_next = np.argmax(q2_values[:, state_next[0], state_next[1]])
                    q_index_next = a_next, state_next[0], state_next[1]
                    q2_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q2_values[q_index])

                state = state_next
                reward_sum += reward

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    grid[state] = 1

            per_episode_rewards.append(reward_sum)
        return per_episode_rewards

    @ray.remote
    def n_step_sarsa(self, n: int = 4, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
              epsilon: float = 0.1, verbose: bool = False):

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)

        per_episode_rewards = list()
        for _ in range(1, n_episodes + 1):

            state = self.start_state
            a = np.argmax(q_values[:, state[0], state[1]])
            a = self.epsilon_greedy(a, epsilon)

            rewards = [0]
            states = [state]
            actions = [a]

            t = -1
            T = float('inf')
            while True:
                t += 1

                if t < T:
                    state_next, reward = self.state_transition(state, self.actions[actions[t]])
                    states.append(state_next)
                    state = state_next
                    rewards.append(reward)

                    if state_next == self.goal:
                        T = t + 1
                    else:
                        a_next = np.argmax(q_values[:, state_next[0], state_next[1]])
                        a_next = self.epsilon_greedy(a_next, epsilon)
                        actions.append(a_next)

                tau = t - n + 1  # tau is the time whose state's estimate is being updated
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += pow(gamma, i - tau - 1) * rewards[i]

                    if tau + n < T:
                        G += pow(gamma, n) * q_values[actions[tau + n], states[tau + n][0], states[tau + n][1]]

                    q_index = actions[tau], states[tau][0], states[tau][1]
                    q_values[q_index] += alpha * (G - q_values[q_index])

                if tau == T - 1:
                    break

            per_episode_rewards.append(sum(rewards))

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
                range=[-200, 0],
            )
        )
        return {'data': traces, 'layout': layout}
