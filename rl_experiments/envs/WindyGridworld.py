import numpy as np
import plotly.graph_objs as go
from plotly import tools
from tqdm import tqdm
import random

random.seed(2)


class WindyGridworld:
    """ Windy Grid World as described in Sutton & Barto (2018, Example 6.3) """

    def __init__(self, length, width, gamma, king_moves: bool = False, stochastic_wind: bool = False):
        self.length = length
        self.width = width
        self.gamma = gamma
        if king_moves:
            self.act = dict(
                zip(set(range(8)), ['up', 'left', 'down', 'right', 'up-right', 'up-left', 'down-right', 'down-left']))
            self.actions = list(map(np.asarray, [[-1, 0], [0, -1], [1, 0], [0, 1], [-1, 1], [-1, -1], [1, 1], [1, -1]]))
        else:
            self.act = dict(zip([0, 1, 2, 3], ['up', 'left', 'down', 'right', ]))
            self.actions = list(map(np.asarray, [[-1, 0], [0, -1], [1, 0], [0, 1]]))

        self.stochastic_wind = stochastic_wind
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.start = (3, 0)
        self.goal = (3, 7)

    def state_transition(self, state, action):
        """
        :param state:   tuple of (y, x) coordinates of the agent in the grid
        :param action:  performed action
        :return:        (i, j) tuple of the next state and the reward associated with the transition
        """
        state = np.array(state)
        y, x = tuple(map(int, state + action))
        y = max(0, y - self.wind_strength[state[1]])
        if self.stochastic_wind and self.wind_strength[state[1]]:
            y += random.randint(-1, 1)
        next_state = (y, x)  # add some wind

        # check boundary conditions
        if y < 0 or x < 0 or y >= self.length or x >= self.width:
            next_state = tuple(state)

        reward = -1
        return next_state, reward

    def sarsa(self, n_episodes: int = 100, alpha: float = 0.5, epsilon: float = 0.1, verbose: bool = False):
        action_values = np.zeros((self.length, self.width, len(self.actions)))
        policy = np.random.randint(0, len(self.actions), (self.length, self.width), dtype=np.int64)
        possible_actions = set(range(len(self.actions)))

        def take_action(s):
            greedy_action = policy[s]
            choices = tuple(possible_actions - {greedy_action})
            if random.random() < epsilon:
                a = choices[random.randint(0, len(self.actions) - 2)]
            else:
                a = greedy_action
            return a

        timestamps = list()
        num_moves = list()
        ts = -1

        if verbose:
            init_grid = np.zeros((self.length, self.width), dtype=np.int64)
            grid = init_grid

        for episode in tqdm(range(1, n_episodes + 1)):

            state, action = self.start, policy[self.start]
            sa = state[0], state[1], action
            moves = 0
            prev_policy = policy.copy()
            while state != self.goal:

                if verbose:
                    print(grid)
                    print('next action:', self.act[action])
                    print('next state:', state)

                moves += 1
                ts += 1
                next_state, reward = self.state_transition(state, self.actions[action])
                next_action = take_action(next_state)
                next_sa = next_state[0], next_state[1], next_action

                action_values[sa] += alpha * (reward + self.gamma * action_values[next_sa] - action_values[sa])
                sa, state, action = next_sa, next_state, next_action

                policy[state] = np.argmax(action_values[state])

                if verbose:
                    grid = np.zeros((self.length, self.width), dtype=np.int64)
                    grid[state] = 1

            if np.array_equal(prev_policy, policy) and episode > 10:
                break  # policy converged

            timestamps.append(ts)
            num_moves.append(moves)

        return action_values, timestamps, num_moves

    @staticmethod
    def plot_learning_rate(timestamps, title):
        trace = go.Scatter(
            mode='lines',
            x=timestamps,
            name='Learning Rate',
        )

        layout = dict(
            height=700,
            title=title,
            showlegend=True,
            xaxis=dict(
                title='Timestamps',
            ),
            yaxis=dict(
                title='Episodes',
            )
        )
        return {'data': [trace], 'layout': layout}

    @staticmethod
    def description():
        description = """ 
        ### Windy Gridworld as per Example 6.5 of the text

        ---

        Windy Gridworld is a standard gridworld, with start and goal states, but with one difference: 
        there is a crosswind running upward through the middle of the grid. 
        The actions are the standard four —- up, down, right, left —- 
        but in the middle region the resultant next states are shifted upward by a “wind”, 
        the strength of which varies from column to column. 
        The strength of the wind is given below each column, in number of cells shifted upward. 
        For example, if you are one cell to the right of the goal, then the action left takes you to the cell just above the goal. 
        This is an undiscounted 100 episodic task, with constant rewards of -1 until the goal state is reached. 
        The graph shows the results of applying epsilon-greedy Sarsa to this task, with the initial values Q(s,a) = 0 for all s, a. 
        The increasing slope of the graph shows that the goal was reached more quickly over time. 
        Note that Monte Carlo methods cannot easily be used here because termination is not guaranteed for all policies. 
        If a policy was ever found that caused the agent to stay in the same state, then the next episode would never end. 
        Online learning methods such as Sarsa do not have this problem because they quickly learn during the episode that such policies are poor, and switch to something else.
        
        Exercise 6.9: Windy Gridworld with King’s Moves (programming) 
        
        Re-solve the windy gridworld assuming eight possible actions, including the diagonal moves, rather than the usual four. 
        How much better can you do with the extra actions? Can you do even better by including a ninth action that causes no movement at all other than that caused by the wind?
        
        Exercise 6.10: Stochastic Wind (programming) 
        
        Re-solve the windy gridworld task with King’s moves, 
        assuming that the effffct of the wind, if there is any, is stochastic, sometimes varying by 1 from the mean values given for each column. 
        That is, a third of the time you move exactly according to these values, as in the previous exercise, 
        but also a third of the time you move one cell above that, 
        and another third of the time you move one cell below that. 
        For example, if you are one cell to the right of the goal and you move left, 
        then one-third of the time you move one cell above the goal, 
        one-third of the time you move two cells above the goal, and one-third of the time you move to the goal.

        ---

        """

        return description


if __name__ == "__main__":
    wg = WindyGridworld(length=7, width=10, gamma=1)
    action_values, timestamps, moves = wg.sarsa(n_episodes=170)
