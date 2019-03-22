import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm

from rl_experiments.envs.GridWorld import GridWorldGenerator


class ShortCorridor(GridWorldGenerator):
    
    def __init__(self, *args, **kwargs):
        super(ShortCorridor, self).__init__(*args, **kwargs)
        self.start_state = 0
        self.goal = 3
    
    @staticmethod
    def state_transition(state, action):
        """
        :param state:   x coordinate of the agent in the grid
        :param action:  performed action
        :return:        x coordinate of the next state and the reward associated with the transition
        """
        reward = -1
        if state == 0 or state == 2:
            next_state = max(0, state + action)
        elif state == 1:
            next_state = state - action
        elif state == 3:
            next_state, reward = state, 0
        return next_state, reward
    
    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
    
    def generate_episode(self, policy):
        state, reward, reward_sum = 0, 0, 0
        states, actions, rewards = [], [], []
        
        while state != self.goal:
            states.append(state)
            action = np.random.choice(self.actions, size=1, p=policy)
            actions.append(action)
            state, reward = self.state_transition(state, action)
            rewards.append(reward)
            reward_sum += reward
        
        return states, actions, rewards, reward_sum
    
    def reinforce(self, n_episodes: int = 1000, gamma: float = 1., alpha: float = 0.1, epsilon: float = 0.1):
        """ REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for pi*"""
        # self.theta = np.zeros(2)
        theta = np.array([1.47, -1.47])
        x = np.array([[1, 0], [0, 1]])  # right / left 2x2
        exploration = epsilon / len(self.actions)
        
        total_rewards = list()
        for _ in tqdm(range(n_episodes)):
            policy = self.softmax(theta.dot(x))
            
            # redistribute probabilities to ensure exploration
            min_prob = np.argmin(policy)
            if policy[min_prob] < exploration:
                policy[:] = 1 - exploration
                policy[min_prob] = exploration
            
            states, actions, rewards, reward_sum = self.generate_episode(policy)
            total_rewards.append(reward_sum)
            T = len(states)
            for t, (state, action) in enumerate(zip(states, actions)):
                # calculate return on the trajectory
                # G = 0
                # for k in range(t+1, T+1):
                #     G += gamma ** (k-t-1) * rewards[k]
                G = sum(rewards[t + 1:])  # when no discount is applied
                
                # update policy parameters
                h = theta.dot(x)
                pi = self.softmax(h)
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
                name=alpha,
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
                # range=[-200, 0],
            )
        )
        return {'data': traces, 'layout': layout}
