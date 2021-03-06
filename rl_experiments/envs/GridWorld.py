import numpy as np
import plotly.graph_objs as go
from plotly import tools
from typing import List, Tuple, Dict

ROOK_ACTIONS = frozenset({(0, -1), (-1, 0), (0, 1), (1, 0)})


class GridWorldGenerator(object):

    def __init__(self, width: int, height: int, actions: List[tuple] = ROOK_ACTIONS,
                 default_reward: float = -1,
                 other_rewards: Dict[tuple, float] = None,
                 blocks: Tuple[Tuple[int, int]] = None,
                 ):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        if blocks:
            for block in blocks:
                self.grid[block] = 2
        self.blocks = blocks
        self.default_reward = default_reward
        self.non_default_rewards = other_rewards
        self.rewards = self._generate_rewards(default_reward, other_rewards)
        self.actions = list(map(np.array, actions))

    def _generate_rewards(self, default_reward, other_rewards):
        """
        Creates reward grid
        :param default_reward:  default reward for transitioning to a given state in grid world
        :param other_rewards:   dict with coordinates as keys and reward as values for other rewards
        :return:
        """
        rewards = np.ones((self.width, self.height)) * default_reward
        if other_rewards is None:
            other_rewards = {}

        for coord, r in other_rewards.items():
            rewards[coord] = r
        return rewards

    @staticmethod
    def rescale_grid(grid, factor: int):
        scaled_grid = GridWorld(
            width=grid.width * factor,
            height=grid.height * factor,
            default_reward=grid.default_reward
            # TODO: handle non-default rewards and blocks
        )
        return scaled_grid


class GridWorld(GridWorldGenerator):
    """ Simple GridWorld as described in Sutton & Barto (2019, Example 4.1) """

    def __init__(self, *args, grid_dim: int, gamma: float, **kwargs):
        super(GridWorld, self).__init__(*args, **kwargs)
        self.grid_dim = grid_dim
        self.gamma = gamma
        self.prob = 1 / len(self.actions)

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
        if x < 0 or y < 0 or x >= self.grid_dim or y >= self.grid_dim:
            next_state = tuple(state)

        reward = self.rewards[next_state]
        return next_state, reward

    def is_terminal(self, x, y):
        return (x == 0 and y == 0) or (x == self.grid_dim - 1 and y == self.grid_dim - 1)

    def gridworld_policy_iteration(self, in_place, theta):
        """
        Iterative Policy Evaluation for estimating Vpi
        :param in_place:    whether to use the updated value function immediately overwriting the old values
        :param theta:       convergence parameter
        :return:
        """

        state_values_sequence = list()
        state_values = np.zeros((self.grid_dim, self.grid_dim))
        new_state_values = state_values.copy()
        delta = float('inf')

        while delta > theta:
            value = new_state_values if in_place else state_values
            for x in range(self.grid_dim):
                for y in range(self.grid_dim):
                    if self.is_terminal(x, y):
                        continue
                    v = 0
                    for action in self.actions:
                        next_state, reward = self.state_transition((x, y), action)
                        v += self.prob * (reward + self.gamma * value[next_state])  # bellman update
                    new_state_values[x, y] = v
            delta = np.sum(np.abs(new_state_values - state_values))
            state_values = new_state_values.copy()
            state_values_sequence.append(state_values)

        return state_values_sequence

    @staticmethod
    def plot_grid_world(state_values):
        iters = [1, 2, 3, 10, len(state_values)]
        fig = tools.make_subplots(5, 1, shared_xaxes=True, subplot_titles=[f'Iteration {i}' for i in iters],
                                  vertical_spacing=0.05)
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=10)
        layout = dict(
            height=5 * 200, width=400, title='V_k', showlegend=False
        )
        fig['layout'].update(layout)

        j = 0
        for i, sv in enumerate(state_values):
            if i + 1 not in iters:
                continue
            j += 1
            trace = go.Heatmap(
                z=np.flip(sv, axis=1),
                showlegend=False,
            )
            fig.append_trace(trace, j, 1)
        return fig

    @staticmethod
    def description():
        description = """ 
        ### Simple GridWorld as described in Example 4.1 of the text
        
        ---
        Terminal states are in the top left and bottom right corners. 
        There are four actions possible in each state: {up, down, right, left}, which deterministically cause the corresponding state transitions, 
        Actions that would take the agent off the grid in fact leave the state unchanged. 
        The reward is 1 on all transitions until the terminal state is reached. 
        In the example below, agent follows equiprobable random policy. 
        The graph shows the sequence of value functions V_k computed by iterative policy evaluation. 
        The final estimate is in fact v*, which in this case gives for each state the negation of the expected number of steps from that state until termination.
        
        ---
        
        """

        return description

