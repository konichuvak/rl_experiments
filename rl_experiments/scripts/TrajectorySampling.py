import random
from collections import deque

import numpy as np
import plotly.graph_objs as go
import ray

from rl_experiments.utils import randargmax


class TrajectorySampling:

    def __init__(self, n_states: int, b: int):
        self.n_states = n_states
        self.b = b
        self.start_state, self.terminal_state = 0, 'X'
        self.actions = (0, 1)
        self.transition = np.random.randint(n_states, size=(n_states, len(self.actions), b))
        self.rewards = np.random.randn(n_states, len(self.actions), b)  # rewards for each transition are N(0, 1)
        self.state_action_space = [(j, i) for j in range(n_states) for i in range(len(self.actions))]

    def state_transition(self, state, action):
        if random.random() < 0.1:
            return self.terminal_state, 0
        else:
            state_next = np.random.randint(self.b)  # all of b states are equally likely
        return self.transition[state, action, state_next], self.rewards[state, action, state_next]

    def policy_evaluation(self, q):
        # Evaluate the value of the start state for the greedy policy
        # derived from @q under the MDP @task
        # use Monte Carlo method to estimate the state value

        rollouts = 1000
        returns = list()
        for _ in range(rollouts):
            state, rewards = self.start_state, 0
            while state != self.terminal_state:
                action = randargmax(q[state])
                state, reward = self.state_transition(state, action)
                rewards += reward
            returns.append(rewards)
        return np.mean(returns)

    @ray.remote
    def uniform(self, step_limit: int):
        """
            iterative policy evaluation for estimating v_pi with one-step tabular expected updates
            cycles through all state-actions uniformly, updating each in place
            using expected updates over all successor states

        :param step_limit:  max number of iterations
        :return:
        """
        state_values = {0: 0}
        q = np.zeros((self.n_states, len(self.actions)))
        full_sweep = len(self.state_action_space)  # policy evaluation is performed after a full sweep
        sa = deque(self.state_action_space * full_sweep)
        for step in range(1, step_limit + 1):
            state, action = sa.popleft()
            next_states = self.transition[state, action]
            q[state, action] = 0.9 * np.mean(
                self.rewards[state, action] + np.max(q[next_states], axis=1))

            if step % (full_sweep / 10) == 0:
                v_pi = self.policy_evaluation(q)
                state_values[step] = v_pi

        return state_values

    # perform expected update from an on-policy distribution of the MDP @task
    @ray.remote
    def on_policy(self, step_limit: int, epsilon: float = 0.1):
        """
            iterative policy evaluation for estimating v_pi with one-step tabular expected updates
            simulates episodes starting from the same start state, updating each state-action pair
            along the trajectory under epsilon-greedy policy

        :param step_limit:    max number of iterations
        :param epsilon:       exploration parameter
        :return:
        """
        state_values = {0: 0}
        q = np.zeros((self.n_states, len(self.actions)))

        def behavior_policy(state):
            return np.random.choice(self.actions) if np.random.rand() < epsilon else randargmax(q[state])

        state = self.start_state
        for step in range(1, step_limit + 1):

            action = behavior_policy(state)
            next_state, reward = self.state_transition(state, action)

            q[state, action] = 0.9 * np.mean(
                self.rewards[state, action] + np.max(q[self.transition[state, action]], axis=1))

            if next_state == 'X':
                next_state = self.start_state
            state = next_state

            if step % (len(self.state_action_space) / 10) == 0:
                v_pi = self.policy_evaluation(q)
                state_values[step] = v_pi

        return state_values

    @staticmethod
    def plot(values: dict):

        traces = list()
        colors = ['darkgreen', 'skyblue', 'crimson']
        for method, branching in values.items():
            for i, (b, trace) in enumerate(branching.items()):
                traces.append(go.Scatter(
                    mode='lines',
                    y=list(trace.values()),
                    x=list(trace.keys()),
                    marker=dict(color=colors[i], symbol='x-open-dot'),
                    name=f'{method} with branching factor {b}',
                ))
        layout = dict(
            height=700,
            showlegend=True,
            xaxis=dict(title='Computation time, in expected updates', ),
            yaxis=dict(title='Value of start state under greedy policy', )
        )
        fig = {'data': traces, 'layout': layout}
        return fig

    @staticmethod
    def description():
        return """
        ### Reproducing Figure 8.8:
        
        ---
        
        Focusing on the on-policy distribution could be beneficial because it causes vast,
        uninteresting parts of the space to be ignored, or it could be detrimental because it causes
        the same old parts of the space to be updated over and over. We conducted a small
        experiment to assess the effect empirically. To isolate the effect of the update distribution,
        we used entirely one-step expected tabular updates, as defined by (8.1). In the uniform
        case, we cycled through all state–action pairs, updating each in place, and in the on-policy
        case we simulated episodes, all starting in the same state, updating each state–action pair
        that occurred under the current epsilon-greedy policy (epsilon=0.1). The tasks were undiscounted
        episodic tasks, generated randomly as follows. From each of the |S| states, two actions
        were possible, each of which resulted in one of b next states, all equally likely, with a
        different random selection of b states for each state–action pair. The branching factor, b,
        was the same for all state–action pairs. In addition, on all transitions there was a 0.1
        probability of transition to the terminal state, ending the episode. The expected reward
        on each transition was selected from a Gaussian distribution with mean 0 and variance 1.
        At any point in the planning process one can stop and exhaustively compute
        v_pi(s0), the true value of the start state under the greedy policy, pi, given the current
        action-value function Q, as an indication of how well the agent would do on
        a new episode on which it acted greedily (all the while assuming the model is correct).
        
        The upper part of the figure to the right shows results averaged over 200 sample tasks with 1000 states and
        branching factors of 1, 3, and 10. The quality of the policies found is plotted as a function of the 
        number of expected updates completed. In all cases, sampling according to the on-policy distribution
        resulted in faster planning initially and retarded planning in the long run. The effect was stronger, 
        and the initial period of faster planning was longer, at smaller branching factors. In other experiments,
        we found that these effects also became stronger as the number of states increased. For example, the lower
        part of the figure shows results for a branching factor of 1 for tasks with 10,000 states. 
        In this case the advantage of on-policy focusing is large and long-lasting.
        
        ---
        
        """


if __name__ == '__main__':
    ts = TrajectorySampling(n_states=1000, b=1)
    step_limit = 20000
    uniform = ts.uniform(step_limit)
    on_policy = ts.on_policy(step_limit)
