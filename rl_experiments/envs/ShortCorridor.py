from math import log

import numpy as np
import plotly.graph_objs as go
import ray

from envs.GridWorld import GridWorldGenerator


class ShortCorridor(GridWorldGenerator):
    
    def __init__(self, *args, **kwargs):
        super(ShortCorridor, self).__init__(*args, **kwargs)
        self.start_state = 0
        self.goal = 3
    
    @staticmethod
    def state_transition(state, action):
        """
        :param state:   x coordinate of the agent in the grid
        :param action:  performed action (-1 is left, 1 is right)
        :return:        x coordinate of the next state and the reward associated with the transition
        """
        reward = -1
        if state == 0 or state == 2:
            next_state = max(0, state + action)
        elif state == 1:
            next_state = state - action
        else:
            next_state, reward = state, 0
        return next_state, reward
    
    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    
    def generate_episode(self, policy):
        state, reward, reward_sum = 0, 0, 0
        states, actions, rewards = [], [], []

        while True:
            states.append(state)
            action = np.random.choice(self.actions, p=policy)
            actions.append(action)
            state, reward = self.state_transition(state, action)
            rewards.append(reward)
            reward_sum += reward
            if reward == 0:
                break
        
        return states, actions, rewards, reward_sum

    @ray.remote
    def reinforce(self, n_episodes: int = 1000, gamma: float = 1., alpha: float = 0.1, epsilon: float = 0.1):
        """ REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for pi* """
        theta = np.array([1.47, -1.47])
        x = np.array([[0, 1], [1, 0]])  # right / left
        exploration = epsilon / len(self.actions)
        
        total_rewards = list()
        for _ in range(n_episodes):
            policy = self.softmax(theta.dot(x))
            
            # redistribute probabilities to ensure exploration
            min_prob = np.argmin(policy)
            if policy[min_prob] < exploration:
                policy[:] = 1 - exploration
                policy[min_prob] = exploration
            
            states, actions, rewards, reward_sum = self.generate_episode(policy)
            total_rewards.append(reward_sum)
            for t, (state, action) in enumerate(zip(states, actions)):
                # calculate return on the trajectory
                G = 0
                for k in range(t + 1, len(states)):
                    G += gamma ** (k - t - 1) * rewards[k]
                # G = sum(rewards[t + 1:])  # when no discount is applied
                
                # update policy parameters
                h = theta.dot(x)
                pi = self.softmax(h)
                # Gradient of log of softmax function. See the link for derivation details:
                # https://math.stackexchange.com/questions/2013050/log-of-softmax-function-derivative
                grad_log_pi = x[int(action == -1)] - np.dot(x, pi)
                theta += alpha * (gamma ** t) * G * grad_log_pi
        
        return total_rewards
    
    @staticmethod
    def plot_rewards(rewards):
        traces = list()
        for alpha, reward in rewards.items():
            traces.append(go.Scatter(
                mode='lines',
                y=reward,
                name=f'2^{log(alpha, 2)}',
            ))
        
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
