import random

import numpy as np
import plotly.graph_objs as go
import ray
from tqdm import tqdm


class ValueFunction:
    
    def __init__(self, walk_length, group_size):
        assert walk_length > group_size
        self.group_size = group_size
        self.agg_values = np.zeros(walk_length // group_size)
    
    def value(self, state):
        """ approximates the value of the state"""
        group_index = (state - 1) // self.group_size
        return self.agg_values[group_index]
    
    def update(self, delta, state):
        """ updates weights for a group of states """
        group_index = (state - 1) // self.group_size
        try:
            self.agg_values[group_index] += delta
        except IndexError:
            print(state)
            exit(1)


class RandomWalk:
    
    def __init__(self, length: int, termination_reward: tuple = (0, 1), state_aggregation: int = 0):
        self.length = length
        self.init_state = self.state = (length + 1) // 2
        self.gamma = 1
        self.true_values = np.array(list(range(1, length + 1)), dtype=np.float64) / (length + 1)
        self.terminal_states = (0, length + 1)
        self.termination_reward = termination_reward
        self.state_aggregation = state_aggregation
    
    def walk(self):
        action = -1 if random.random() < 0.5 else 1
        if self.state_aggregation:
            step = action * np.random.randint(1, self.state_aggregation + 1)
            self.state = max(min(self.state + step, self.length + 1), 0)  # ensure we are not out of bounds
        else:
            self.state += action

        return self.state
    
    def generate_episode(self):
        self.state = self.init_state
        state_reward = list()
        while True:
            if self.state == self.terminal_states[0]:
                state_reward.append((self.state, self.termination_reward[0]))
                return state_reward
            elif self.state == self.terminal_states[1]:
                state_reward.append((self.state, self.termination_reward[1]))
                return state_reward
            else:
                state_reward.append((self.state, 0))
                self.walk()
    
    @ray.remote
    def mc_prediction(self, n_episodes: int = 100, first_visit: bool = True, alpha: float = 0.1):
        
        state_values_list = list()
        state_values = dict(zip(
            range(self.length + 2),
            [self.termination_reward[0]] + [0.5 for _ in range(self.length)] + [self.termination_reward[1]]
        ))
        
        for _ in range(1, n_episodes + 1):
            state_values_list.append(np.array(list(state_values.values())[1:-1]))
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
    
    @ray.remote
    def td_prediction(self, n_episodes: int = 100, alpha: float = 0.1):
        
        state_values_list = list()
        state_values = dict(zip(
            range(self.length + 2),
            [self.termination_reward[0]] + [0.5 for _ in range(self.length)] + [self.termination_reward[1]]
        ))
        
        for _ in range(1, n_episodes + 1):
            state_values_list.append(np.array(list(state_values.values())[1:-1]))
            self.state = self.init_state
            reward = 0
            while self.state not in self.terminal_states:
                s0 = self.state
                v = state_values[s0]
                s1 = self.walk()
                state_values[s0] += alpha * (reward + self.gamma * state_values[s1] - v)
        return state_values_list
    
    @ray.remote
    def n_step_td_prediction(self, n: int, n_episodes: int = 10, alpha: float = 0.1):
        """
        returns an array of length n_episodes of state values (np.arrays of shape (lenght of the random walk, )
        """

        self.termination_reward = (-1, 1)
        self.true_values = np.arange(-self.length + 1, self.length + 1, 2) / (self.length + 1.)

        state_values_list = list()
        state_values = dict(zip(
            range(self.length + 2),
            [self.termination_reward[0]] + [0 for _ in range(self.length)] + [self.termination_reward[1]]
        ))

        for _ in range(1, n_episodes + 1):
            T = float('inf')

            self.state = self.init_state
            rewards = [0]
            states = [self.state]
            t = -1

            while True:
                t += 1
                if t < T:
                    s1 = self.walk()
                    states.append(s1)
                    if s1 in self.terminal_states:
                        rewards.append(self.termination_reward[self.terminal_states.index(s1)])
                        T = t + 1
                    else:
                        rewards.append(0)

                tau = t - n + 1  # tau is the time whose state's estimate is being updated
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += pow(self.gamma, i - tau - 1) * rewards[i]

                    if tau + n < T:
                        G += pow(self.gamma, n) * state_values[states[tau + n]]

                    state_values[states[tau]] += alpha * (G - state_values[states[tau]])

                if tau == T - 1:
                    break

            state_values_list.append(np.array(list(state_values.values())[1:-1]))

        return np.asarray(state_values_list)
    
    def gradient_mc(self, value_function: ValueFunction, n_episodes: int, alpha: float = 2e-5, first_visit=True):
        """ generate episodes of the random walk updating value function according to gradient Monte Carlo algorithm """
        
        state_visitation = np.zeros(self.length + 2)
        
        for _ in tqdm(range(1, n_episodes + 1)):
            episode = self.generate_episode()
            g = 0
            if first_visit:
                episode = dict(episode[::-1])  # gets rif of the duplicate states in the trajectory
                for state, reward in episode.items():
                    state_visitation[state] += 1
                    g = self.gamma * g + reward
                    if state in self.terminal_states:
                        continue
                    v = value_function.value(state)
                    delta = alpha * (g - v)
                    value_function.update(delta, state)
            else:
                raise Exception('all visit MC is not implemented')
        
        return state_visitation
    
    @ray.remote
    def batch_updates(self, algo: str = 'TD', n_episodes: int = 100, alpha: float = 0.001):
    
        state_values = np.array([0] + [0.5 for _ in range(self.length)] + [1])
    
        rmse = np.zeros(n_episodes)
        episodes = list()
    
        for i in range(n_episodes):
            episode = self.generate_episode()
            episodes.append(episode)

            while True:
                increments = np.zeros(self.length + 2)
                for episode in episodes:
                    if algo == 'MC':
                        episode = dict(episode[::-1])  # remove duplicates and reverse the rollout
                        g = 0
                        for state, reward in episode.items():
                            g = self.gamma * g + reward
                            increments[state] += g - state_values[state]
                    else:
                        for j in range(len(episode) - 1):
                            current_state, reward = episode[j]
                            next_state = episode[j + 1][0]
                            increments[current_state] += reward + state_values[next_state] - state_values[current_state]
                increments *= alpha
                if np.sum(np.abs(increments)) < 1e-3:
                    break  # small increments imply convergence of the value function
                state_values += increments

            rmse[i] = np.sqrt(np.sum(np.power(state_values[1:-1] - self.true_values, 2)) / self.true_values.size)
    
        return rmse
    
    def plot_rmse(self, errors, x_axis: list):
        
        traces = list()
        
        if tuple(errors.keys()) == ('MC', 'TD'):
            for n, alphas in errors.items():
                for alpha, rmse_per_episode in alphas.items():
                    traces.append(
                        go.Scatter(
                            mode='lines',
                            y=rmse_per_episode,
                            name=f'{n}_alpha={alpha}',
                            marker=dict(color='crimson' if n == 'MC' else 'skyblue')
                        )
                    )
        
        else:
            for step, rmse_per_alpha in errors.items():
                traces.append(
                    go.Scatter(
                        mode='lines',
                        y=rmse_per_alpha,
                        x=x_axis,
                        name=f'n={step}',
                    )
                )
        
        layout = dict(
            height=700,
            title='Empirical RMSE averaged over states',
            showlegend=True,
            xaxis=dict(title='Alphas' if x_axis != ('MC', 'TD') else 'Walks / Episodes', titlefont=dict(size=13)),
            yaxis=dict(title='Error', titlefont=dict(size=13)),
        )
        return {'data': traces, 'layout': layout}
    
    def plot_state_values_fa(self, state_values, state_visitation):
        traces = list()
        
        traces.append(
            go.Scatter(
                mode='lines',
                y=state_values,
                name=f'Approximate MC values'
            )
        )
        traces.append(
            go.Scatter(
                mode='lines',
                y=np.arange(-self.length + 1, self.length + 1, 2) / (self.length + 1.),
                name=f'True values'
            )
        )
        traces.append(
            go.Histogram(
                x=state_visitation,
                marker=dict(
                    color='gainsboro'
                ),
                name='State visitation distribution'
            )
        )
        
        layout = dict(
            height=600,
            title='Value Estimation',
            showlegend=True
        )
        return {'data': traces, 'layout': layout}
    
    def plot_state_values(self, values, iters: tuple = (0, 1, 10, 100)):
        traces = list()

        for i, value in enumerate(values):
            if i not in iters:
                continue
            traces.append(
                go.Scatter(
                    mode='lines',
                    y=value,
                    x=['A', 'B', 'C', 'D', 'E'],
                    name=f'iteration {i}'
                )
            )
        traces.append(
            go.Scatter(
                mode='lines',
                y=self.true_values,
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
    
    @staticmethod
    def plot_batch_rmse(errors):
        traces = list()
        for algo, rmse in errors.items():
            traces.append(
                go.Scatter(
                    mode='lines',
                    y=rmse,
                    name=f'{algo}',
                    marker=dict(color='crimson' if algo == 'MC' else 'skyblue')
                )
            )
        layout = dict(
            height=600,
            title='Batch Training',
            showlegend=True,
            xaxis=dict(title='Walks / Episodes', titlefont=dict(size=13)),
            yaxis=dict(title='RMSE averaged over states', titlefont=dict(size=13)),
        )
        return {'data': traces, 'layout': layout}
    
    @staticmethod
    def description():
        markdown_text = """
        
        ### Random Walk Environment as per Examples 6.2 and 7.1 in text
        
        ---
        
        Experiment is formulated as a Markov Reward Process as there is no need to distinguish the dynamics due to the environment from those due to the agent.
        In this MRP, all episodes start in the center state, then proceed either left or right by one state on each step, with equal probability.         
        Because this task is undiscounted, the true value of each state is the probability of terminating on the right if starting from that state. 
        Episodes terminate either on the extreme left or the extreme right.
        
        In the TD(0) vs MC case we compare performance of two classic algorithms by measuring Root Mean Squared Error 
        between the true state values of the random walk and the ones estimated by the agent. 
        In this case we set the reward to 0 for all states except for the rightmost state.
        State value estimates are averaged over the number of states of the process (*Walk Length*), then averaged over the number of experiments.
        
        We then generalize these two algorithms via n-step TD method. 
        This time, the rewards are -1 on the left and 1 on the right with. All states are initialized with value 0.
        Results are shown for n-step TD methods with a range of values for n and step-size alpha. 
        The performance measure was kept the same except averaging occurs across over first 10 episodes of the run in addition to averaging over states and experiments.   
        
        ---
           
        """
        return markdown_text


if __name__ == "__main__":
    rw = RandomWalk()
    ep = rw.generate_episode()
