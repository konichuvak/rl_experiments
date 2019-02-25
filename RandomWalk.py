import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm
import random

random.seed(1)


class RandomWalk:

    def __init__(self):
        self.current_state = 3
        self.gamma = 1

    def walk(self):
        action = -1 if random.random() < 0.5 else 1
        self.current_state += action

    def generate_episode(self):
        self.current_state = 3
        state_reward = list()
        while True:
            if self.current_state == 0:
                state_reward.append((self.current_state, 0))
                return state_reward
            elif self.current_state == 6:
                state_reward.append((self.current_state, 1))
                return state_reward
            else:
                state_reward.append((self.current_state, 0))
                self.walk()

    def mc_prediction(self, n_episodes: int = 100, first_visit: bool = True, alpha: float = 0.1):

        state_values_list = list()
        state_values = dict(zip(range(7), [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]))

        for _ in tqdm(range(1, n_episodes + 1)):
            state_values_list.append(state_values.copy())
            episode = self.generate_episode()
            g = 0
            if first_visit:
                episode = dict(episode[::-1])
                for state, reward in episode.items():
                    g = self.gamma * g + reward
                    v = state_values[state]
                    state_values[state] = v + alpha * (g - v)
            else:
                raise Exception('all visit MC is not implemented')

        return state_values_list

    def td_prediction(self, n_episodes: int = 100, alpha: float = 0.1):

        state_values_list = list()
        state_values = dict(zip(range(7), [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]))
        terminal_states = [0, 6]

        for _ in tqdm(range(1, n_episodes + 1)):
            state_values_list.append(state_values.copy())
            self.current_state = 3
            reward = 0
            while self.current_state not in terminal_states:
                s0 = self.current_state
                v = state_values[s0]
                self.walk()
                s1 = self.current_state
                state_values[s0] += alpha * (reward + self.gamma * state_values[s1] - v)
        return state_values_list

    @staticmethod
    def plot_state_values(values):
        true_values = np.array(list(range(1, 6))) / 6
        iters = [0, 1, 10, 100]
        traces = list()
        for i, value in enumerate(values):
            if i not in iters:
                continue
            estimated_values = list(values[i].values())[1:-1]
            traces.append(
                    go.Scatter(
                            mode='lines',
                            y=estimated_values,
                            x=['A', 'B', 'C', 'D', 'E'],
                            name=f'iteration {i}'
                    )
            )
        traces.append(
                go.Scatter(
                        mode='lines',
                        y=true_values,
                        x=['A', 'B', 'C', 'D', 'E'],
                        name=f'true values'
                )
        )

        layout = dict(
                height=600,
                title='Value Estimation',
                showlegend=True
        )
        return {'data': traces, 'layout': layout}

    def plot_rmse(self, values):
        true_values = np.array(list(range(1, 6))) / 6

        traces = list()
        error = values.copy()
        for algo in ['TD', 'MC']:
            for alpha in error[algo].keys():
                rmse = list()
                for state_values in values[algo][alpha]:
                    rmse.append(np.sqrt(np.sum(np.power(state_values - true_values, 2)) / true_values.size))

                traces.append(
                        go.Scatter(
                                mode='lines',
                                y=rmse,
                                name=f'{algo}_alpha={alpha}',
                                marker=dict(color='crimson' if algo == 'MC' else 'skyblue')
                        )
                )
        layout = dict(
                height=600,
                title='Empirical RMSE averaged over states',
                showlegend=True,
                xaxis=dict(title='Walks / Episodes', titlefont=dict(size=13)),
                yaxis=dict(title='Error', titlefont=dict(size=13)),
        )
        return {'data': traces, 'layout': layout}
