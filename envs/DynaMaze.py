import numpy as np
import random
import plotly.graph_objs as go
import ray
from envs.CliffWalking import GridWorld
from collections import defaultdict
from tqdm import tqdm


class DynaMaze(GridWorld):

    def __init__(self, *args, **kwargs):
        super(DynaMaze, self).__init__(*args, **kwargs)
        self.start_state = (0, 2)
        self.goal = (self.width - 1, 0)
        self.blocks = [(2, 1), (2, 2), (2, 3), (5, 4), (7, 1), (7, 2), (7, 3)]

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

    def epsilon_greedy(self, action_index, epsilon: float = 0.1):
        if random.random() < epsilon:
            action_index = random.randint(0, len(self.actions) - 1)
        return action_index

    @staticmethod
    def randargmax(ndarray):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(ndarray.shape) * (ndarray == ndarray.max()))

    @ray.remote
    def q_planning(self, planning_steps: int = 0, n_episodes: int = 50, alpha: float = 0.1, gamma: float = 0.95,
                   epsilon: float = 0.1, verbose: bool = False, seed: int = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        q_shape = [len(self.actions)] + list(self.grid.shape)
        q_values = np.zeros(shape=q_shape)

        model = defaultdict(lambda: defaultdict(tuple))

        if verbose:
            init_grid = np.zeros((self.width, self.height), dtype=np.int64)
            grid = init_grid

        episode_length = list()
        for _ in range(n_episodes):

            state = self.start_state
            i = 1
            while state != self.goal:
                i += 1

                a = self.randargmax(q_values[:, state[0], state[1]])
                a = self.epsilon_greedy(a, epsilon)

                if verbose:
                    print(grid.T)
                    print('current state:', state)
                    print('next action:', a)

                state_next, reward = self.state_transition(state, self.actions[a])
                a_next = self.randargmax(q_values[:, state[0], state[1]])

                q_index = a, state[0], state[1]
                q_index_next = a_next, state_next[0], state_next[1]
                q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                model[state][a] = (reward, state_next)
                state = state_next

                for n in range(planning_steps):
                    s = random.choice(tuple(model.keys()))
                    a = random.choice(tuple(model[s].keys()))
                    reward, s_next = model[s][a]
                    a_next = self.randargmax(q_values[:, s_next[0], s_next[1]])

                    q_index = a, s[0], s[1]
                    q_index_next = a_next, s_next[0], s_next[1]
                    q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])

                if verbose:
                    grid = np.zeros((self.width, self.height), dtype=np.int64)
                    grid[state] = 1

            episode_length.append(i)

        return episode_length

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
        
        The grapph below shows average learning curves from an experiment in which Dyna-Q agents were applied to the maze task. 
        The initial action values were zero, the step-size parameter was alpha = 0.1, and the exploration parameter was epsilon = 0.1. 
        When selecting greedily among actions, ties were broken randomly. 
        The agents varied in the number of planning steps, n, they performed per real step. 
        For each n, the curves show the number of steps taken by the agent to reach the goal in each episode, 
        averaged over repetitions of the experiment. 
        In each repetition, the initial seed for the random number generator was held constant across algorithms. 
        Because of this, the first episode was exactly the same (about 1700 steps) for all values of n, and its data are not shown in the figure. 
        After the first episode, performance improved for all values of n, but much more rapidly for larger values. 
        Recall that the n = 0 agent is a nonplanning agent, using only direct reinforcement learning (one-step tabular Q-learning). 
        
        This was by far the slowest agent on this problem, despite the fact that the parameter values (alpha and epsilon) were optimized for it. 
        The nonplanning agent took about 25 episodes to reach (epsilon-)optimal performance, 
        whereas the n = 5 agent took about five episodes, and the n = 50 agent took only three episodes.

        ----

        """
