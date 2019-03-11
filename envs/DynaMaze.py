import numpy as np
import random
import plotly.graph_objs as go
import ray
from envs.CliffWalking import GridWorld
from typing import List
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from heapdict import heapdict


class DynaMaze(GridWorld):

    def __init__(self, *args, start_state: tuple, goal: tuple, **kwargs):
        super(DynaMaze, self).__init__(*args, **kwargs)
        self.start_state = start_state
        self.goal = goal

    def state_transition(self, state, action):
        """
        :param state:   tuple of (x, y) coordinates of the agent in the grid
        :param action:  performed action
        :return:        (i, j) tuple of the next state and the reward associated with the transition
        """

        s = np.array(state)
        next_state = tuple(s + action)
        x, y = next_state

        # check boundary conditions
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            next_state = state

        # check block conditions
        if next_state in self.blocks:
            next_state = state

        # obtain transition reward
        reward = self.rewards[next_state]

        return next_state, reward

    def epsilon_greedy(self, action_index, epsilon: float = 0.2):
        if random.random() < epsilon:
            action_index = random.randint(0, len(self.actions) - 1)
        return action_index

    @staticmethod
    def randargmax(ndarray):
        """ a random tie-breaking argmax """
        return np.argmax(np.random.random(ndarray.shape) * (ndarray == ndarray.max()))

    @staticmethod
    def randmax(ndarray):
        """ a random tie-breaking max """
        return np.max(np.random.random(ndarray.shape) * (ndarray == ndarray.max()))

    @ray.remote
    def q_planning(self, planning_steps: int = 0, n_episodes: int = 50, alpha: float = 0.1, gamma: float = 0.95,
                   epsilon: float = 0.1, verbose: bool = False, seed: int = None, step_limit: int = None,
                   switch_time: int = None, new_blocks: dict = None, kappa: float = None,
                   dyna_q_plus: bool = False, dyna_q_plus_plus: bool = False,
                   ):

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)
        model = defaultdict(lambda: defaultdict(tuple))

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            for block in self.blocks:
                init_grid[block] = 2
            grid = init_grid

        episode_length = list()
        time_steps = 0
        for _ in range(n_episodes):

            if dyna_q_plus | dyna_q_plus_plus:
                tracker = defaultdict(lambda: defaultdict(int))

            state = self.start_state
            i = 1
            while state != self.goal:
                i += 1
                time_steps += 1

                a = self.randargmax(q_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a, epsilon)
                state_next, reward = self.state_transition(state, self.actions[a])

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                q_index = a, state[0], state[1]
                q_index_next = self.randargmax(q_values[:, state_next[0], state_next[1]]), state_next[0], state_next[1]
                q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                if dyna_q_plus | dyna_q_plus_plus:
                    for action in range(len(self.actions)):
                        tracker[state][action] = time_steps if action == a else 1
                        model[state][action] = (reward, state_next) if action == a else (0, state)
                else:
                    model[state][a] = (reward, state_next)

                for n in range(planning_steps):
                    s = random.choice(tuple(model.keys()))
                    if dyna_q_plus_plus:
                        # Suppose the action selected was always that for which Q(St,a) + k*sqrt(tau(St,a)) was maximal.
                        action_values = np.zeros(len(self.actions))
                        for action in range(len(self.actions)):
                            q_index, elapsed_timesteps = (action, s[0], s[1]), time_steps - tracker[s][action]
                            action_values[action] = q_values[q_index] + kappa * np.sqrt(elapsed_timesteps)
                        a = self.randargmax(action_values)
                    else:
                        a = random.choice(tuple(model[s].keys()))
                    reward, s_next = model[s][a]

                    if dyna_q_plus:
                        elapsed_timesteps = time_steps - tracker[s][a]
                        reward += kappa * np.sqrt(elapsed_timesteps)

                    q_index = a, s[0], s[1]
                    q_index_next = self.randargmax(q_values[:, s_next[0], s_next[1]]), s_next[0], s_next[1]
                    q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    for block in self.blocks:
                        grid[block] = 2
                    grid[state] = 1

                state = state_next

            episode_length.append(i)

            if switch_time and time_steps >= switch_time:
                self.blocks = new_blocks

            if step_limit and time_steps >= step_limit:
                return episode_length

        return episode_length

    # @ray.remote
    def prioritized_sweeping(self, planning_steps: int = 0, n_episodes: int = 50, alpha: float = 0.1,
                             gamma: float = 0.95,
                             epsilon: float = 0.1, theta: float = 0.00001, verbose: bool = False, seed: int = None,
                             ):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)
        model = defaultdict(lambda: defaultdict(tuple))
        p_queue = heapdict()
        predecessors = defaultdict(set)  # to track all the states leading into a given state

        if verbose:
            self.grid = self.grid.astype(int)
            grid = self.grid.copy()

        total_steps_per_episode = list()
        for e in tqdm(range(n_episodes)):
            total_steps = 0

            state = self.start_state
            while state != self.goal:

                a = self.randargmax(q_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a, epsilon)
                state_next, reward = self.state_transition(state, self.actions[a])

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)
                    print('next action:', state_next)

                model[state][a] = (reward, state_next)
                # remember state-action pairs and associated rewards that led to the next state
                predecessors[state_next].add((state, a, reward))

                q_index = a, state[0], state[1]
                q_index_next = self.randargmax(q_values[:, state_next[0], state_next[1]]), state_next[0], state_next[1]
                priority = abs(reward + gamma * (q_values[q_index_next]) - q_values[q_index])
                if priority > theta:
                    if not p_queue.get((state, a)):
                        p_queue[state, a] = 0
                    # note that python's native heapq works for min elements only
                    p_queue[state, a] = min(p_queue[state, a], priority * -1)

                for n in range(planning_steps):
                    if not p_queue:
                        break

                    s, a = p_queue.popitem()[0]
                    reward, s_next = model[s][a]

                    q_index = a, s[0], s[1]
                    q_index_next = self.randargmax(q_values[:, s_next[0], s_next[1]]), s_next[0], s_next[1]
                    q_values[q_index] += alpha * (reward + gamma * q_values[q_index_next] - q_values[q_index])

                    # deal with all the predecessors of the sample state
                    for state_prev, a_prev, reward_prev in predecessors[s]:
                        q_index = a_prev, state_prev[0], state_prev[1]
                        priority = abs(reward_prev + gamma * self.randmax(q_values[:, s[0], s[1]]) - q_values[q_index])
                        if priority > theta:
                            if not p_queue.get((state_prev, a_prev)):
                                p_queue[state_prev, a_prev] = 0
                            p_queue[state_prev, a_prev] = min(p_queue[state_prev, a_prev], priority * -1)

                    total_steps += 1

                if verbose:
                    grid = self.grid.copy()
                    grid[state] = 10

                state = state_next

            total_steps_per_episode.append(total_steps)
        return total_steps_per_episode

    @staticmethod
    def plot_learning_curves(episodes):

        traces = list()
        for n, steps_per_episode in episodes.items():
            traces.append(go.Scatter(
                mode='lines',
                y=steps_per_episode,
                name=f'{n} planning steps',
            )
            )

        layout = dict(
            height=700,
            showlegend=True,
            xaxis=dict(
                title='Epsiodes',
            ),
            yaxis=dict(
                title='Steps per episode',
            )
        )
        return {'data': traces, 'layout': layout}

    @staticmethod
    def plot_rewards(rewards):

        traces = list()
        for algo, cum_rewards in rewards.items():
            traces.append(go.Scatter(
                mode='lines',
                y=cum_rewards,
                name=algo,
            )
            )

        layout = dict(
            height=700,
            showlegend=True,
            xaxis=dict(
                title='Time steps',
            ),
            yaxis=dict(
                title='Cumulative Reward',
            )
        )
        return {'data': traces, 'layout': layout}

    @staticmethod
    def description():
        return """
        ----
        
        Consider the simple maze shown inset. 
        ### ADD MAZE FIGURE 
        In each of the 47 states there are four actions, up, down, right, and left, 
        which take the agent deterministically to the corresponding neighboring states, 
        except when movement is blocked by an obstacle or the edge of the maze, in which case the agent remains where it is. 
        Reward is zero on all transitions, except those into the goal state, on which it is +1. 
        After reaching the goal state (G), the agent returns to the start state (S) to begin a new episode. 
        This is a discounted, episodic task.
        
        ### Exercise 8.1 
        The nonplanning method looks particularly poor in Figure 8.3 because it is
        a one-step method; a method using multi-step bootstrapping would do better. Do you
        think one of the multi-step bootstrapping methods from Chapter 7 could do as well as
        the Dyna method? Explain why or why not.
        ----

        """
