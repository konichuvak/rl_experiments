import numpy as np
from typing import Dict, List
import random
import plotly.graph_objs as go
import ray
from tqdm import tqdm

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

    def epsilon_greedy(self, action_index, epsilon: float = 0.1):
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

    @ray.remote
    def n_step_sarsa_off_policy(self, n: int = 4, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                     epsilon: float = 0.1, verbose: bool = False):

        behavior_policy = self.epsilon_greedy
        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.random.rand(*q_shape)

        per_episode_rewards = list()
        for _ in range(1, n_episodes + 1):

            state = self.start_state
            a = np.argmax(q_values[:, state[0], state[1]])
            a = behavior_policy(a, epsilon)

            rewards = [0]
            states = [state]
            actions = [a]
            rhos = [1]

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
                        greedy_a = np.argmax(q_values[:, state_next[0], state_next[1]])
                        a_next = behavior_policy(greedy_a, epsilon)
                        actions.append(a_next)

                        if a_next == greedy_a:
                            b = 1 - epsilon + epsilon / len(self.actions)  # prob of taking an action under b policy
                            rhos.append(1 / b)
                        else:
                            rhos.append(0)

                tau = t - n + 1  # tau is the time whose state's estimate is being updated
                if tau >= 0:

                    rho = 1
                    for i in range(tau + 1, min(tau + n - 1, T - 1) + 1):
                        if rhos[i]:
                            rho *= rhos[i]
                        else:
                            rho = 0
                            break

                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += pow(gamma, i - tau - 1) * rewards[i]

                    if tau + n < T:
                        G += pow(gamma, n) * q_values[actions[tau + n], states[tau + n][0], states[tau + n][1]]

                    q_index = actions[tau], states[tau][0], states[tau][1]
                    q_values[q_index] += alpha * rho * (G - q_values[q_index])

                if tau == T - 1:
                    break

            per_episode_rewards.append(sum(rewards))

        return per_episode_rewards

    @ray.remote
    def n_step_tree_backup(self, n: int = 4, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                     epsilon: float = 0.1, verbose: bool = False):
        """ Ridiculously slow and does not converge """

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.random.rand(*q_shape)

        behavior_policy = lambda s: random.randint(0, len(self.actions) - 1)
        target_policy = lambda s: np.argmax(q_values[:, s[0], s[1]])

        per_episode_rewards = list()
        for _ in tqdm(range(1, n_episodes + 1)):

            state = self.start_state
            a = behavior_policy(state)

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
                        a_next = behavior_policy(state_next)
                        actions.append(a_next)

                tau = t - n + 1  # tau is the time whose state's estimate is being updated
                if tau >= 0:

                    if t + 1 >= T:
                        G = rewards[T]
                    else:
                        G = rewards[t + 1] + gamma * np.max(q_values[:, state_next[0], state_next[1]])

                    for k in range(min(t, T-1), tau, -1):
                        action_taken = actions[k]
                        target_action = target_policy(states[k])

                        # take expected value over the leaf nodes, not including the action that was actually taken
                        weighted_q = 0
                        for i, a in enumerate(self.actions):
                            if i == action_taken:
                                continue
                            elif i == target_action:
                                # TODO: adapt to the case when target policy is not greedy
                                prob = 1
                            else:
                                prob = 0
                            weighted_q += prob * q_values[i, states[k][0], states[k][1]]

                        G = rewards[k] + gamma * weighted_q + gamma * (action_taken == target_action) * G

                    q_index = actions[tau], states[tau][0], states[tau][1]
                    q_values[q_index] += alpha * (G - q_values[q_index])

                if tau == T - 1:
                    break

            per_episode_rewards.append(sum(rewards))

        return per_episode_rewards

    @ray.remote
    def n_step_q_sigma(self, n: int = 4, n_episodes: int = 100, alpha: float = 0.5, gamma: float = 1,
                     epsilon: float = 0.1, verbose: bool = False):
        """ Ridiculously slow and does not converge """

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.random.rand(*q_shape)

        target_policy = self.epsilon_greedy
        behavior_policy = lambda s: random.randint(0, len(self.actions) - 1)
        prob = (epsilon / len(self.actions), 1 - epsilon + epsilon / len(self.actions))

        per_episode_rewards = list()
        for _ in tqdm(range(1, n_episodes + 1)):

            state = self.start_state
            a = behavior_policy(state)

            rewards = [0]
            states = [state]
            actions = [a]
            rhos = [1]
            sigmas = [0]

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
                        sigmas.append(not sigmas[-1])  # alternate sigma

                        a_next = behavior_policy(state_next)
                        actions.append(a_next)
                        b = 1 / len(self.actions)  # prob of taking an action under b policy

                        # prob of taking an action under pi policy
                        greedy_action = np.argmax(q_values[:, state_next[0], state_next[1]])
                        a_target = target_policy(greedy_action)
                        p = 1 - epsilon + epsilon / len(self.actions) if a_next == a_target else epsilon / len(self.actions)
                        rhos.append(p / b)

                tau = t - n + 1  # tau is the time whose state's estimate is being updated
                if tau >= 0:

                    G = 0
                    for k in range(min(t+1, T), tau, -1):

                        if k == T:
                            G = rewards[T]
                        else:
                            action_taken = actions[k]
                            greedy_action = np.argmax(q_values[:, states[k][0], states[k][1]])
                            a_target = target_policy(greedy_action)

                            weighted_q = 0
                            for i, a in enumerate(self.actions):
                                # under epsilon greedy pi
                                p = prob[i == a_target]
                                weighted_q += p * q_values[i, states[k][0], states[k][1]]

                            G = rewards[k] + gamma * weighted_q + \
                                gamma * (sigmas[k] * rhos[k] - (1 - sigmas[k]) * prob[action_taken == a_target]) * \
                                (G - q_values[action_taken, states[k][0], states[k][1]])

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

    @staticmethod
    def description():
        description = """ 
        ### Cliff Walking as per Example 6.5 of the text

        ---

        This gridworld example serves as a testbed for comparison of classical TD learning algorithms. 
        These include Sarsa, Q-learning, Expected Sarsa, Double-Q-Learning and N-step-Sarsa. 
        This is a standard undiscounted, episodic task, with start and goal states, 
        and the usual actions causing movement up, down, right, and left. 
        Reward is -1 on all transitions except those into the region marked “The Cliff”.
        Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start.
        ![rl-cliff-walking-diagram](../assets/cliff-walking-diagram.png)
        
        - n_episodes:   int = 100
        - alpha:        float = 0.5
        - gamma:        float = 1
        - epsilon:      float = 0.1
    
        ---

        """

        return description