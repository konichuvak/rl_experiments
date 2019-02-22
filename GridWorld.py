import numpy as np
import plotly.graph_objs as go
from plotly import tools


class GridWorld:
    """ Simple GridWorld as described in Sutton & Barto (2019, Example 4.1) """

    def __init__(self, grid_dim, gamma):
        self.grid_dim = grid_dim
        self.gamma = gamma
        # left, up, right, down
        self.actions = list(map(np.asarray, [[0, -1], [-1, 0], [0, 1], [1, 0]]))
        self.prob = 1 / len(self.actions)

    # Grid World
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

        reward = -1
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
