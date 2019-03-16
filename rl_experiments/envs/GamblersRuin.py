import numpy as np
import plotly.graph_objs as go


class GamblersRuin:
    """ Gambler's Problem as described in Sutton & Barto (2019, Example 4.3) """

    def __init__(self, goal=100, p_heads=0.4):
        self.goal = goal
        self.states = list(range(1, goal))
        self.reward = 0
        self.p_heads = p_heads
        self.terminal_state = 0

    def expected_update(self, state, action, state_values):
        """
        :param state:           current gambler's equity
        :param action:          stake that that the gambler made
        :param state_values:    current state-value matrix
        :return:                updated value of the state
        """
        return self.p_heads * state_values[state + action] + (1 - self.p_heads) * state_values[state - action]

    def value_iteration(self, state_values: np.ndarray, in_place: bool):
        """
        Iterative Policy Evaluation using Value Iteration for estimating V by acting greedy

        :param state_values:    state-value matrix V_hat
        :param in_place:        whether to use the updated value function immediately overwriting the old values
        :return:                updated state-value matrix V_hat after full DP sweeps
        """
        new_state_values = state_values.copy()
        theta = 1e-9
        delta = float('inf')
        state_values_seq = [state_values]

        while delta > theta:
            value_matrix = new_state_values if in_place else state_values
            for state in self.states:
                # choose the action that maximizes value of the state
                action_values = list()
                for action in range(min(state, self.goal - state) + 1):
                    action_values.append(self.expected_update(state, action, value_matrix))
                new_state_values[state] = max(action_values)

            delta = np.sum(np.abs(new_state_values - state_values))
            print(f'State-value delta: {delta}')
            state_values = new_state_values.copy()
            state_values_seq.append(state_values)

        return state_values_seq

    def compute_optimal_policy(self, state_values: np.ndarray):
        """
        Computes optimal policy pi* that is greedy wrt the current value function v

        :param state_values:    optimal state-value matrix V*
        :return:                optimal policy Pi*
        """
        opt_policy = np.zeros(state_values.shape, dtype=np.int)
        for state in self.states:
            # choose the policy to be the action that maximizes the expected return for this state
            action_values = list()
            for action in range(1, min(state, self.goal - state) + 1):
                v = self.expected_update(state, action, state_values)
                action_values.append(v)
            opt_policy[state] = np.argmax(np.round(action_values, 5)) + 1

        return opt_policy

    def policy_iteration(self, in_place: bool) -> (np.ndarray, np.ndarray):

        state_values = np.zeros(self.goal + 1)
        state_values[self.goal] = 1.0

        state_values_seq = self.value_iteration(state_values, in_place)
        policy = self.compute_optimal_policy(state_values_seq[-1])

        return state_values_seq, policy

    @staticmethod
    def plot_value_iterations(state_values_seq):
        iters = [1, 2, 3, int(len(state_values_seq) / 2), len(state_values_seq)]
        traces = list()
        for i, sv in enumerate(state_values_seq):
            if i + 1 not in iters:
                continue
            traces.append(
                    go.Scatter(
                            mode='lines',
                            y=sv,
                            name=f'sweep {i + 1}'
                    )
            )

        layout = dict(
                height=600,
                title='Value Iterations',
                showlegend=True
        )
        return {'data': traces, 'layout': layout}

    @staticmethod
    def plot_optimal_policy(policy):
        traces = [
            go.Scatter(
                    mode='lines',
                    y=policy,
                    showlegend=False
            )
        ]
        layout = dict(
                height=600,
                title='Optimal Policy'
        )
        return {'data': traces, 'layout': layout}

    @staticmethod
    def description():
        description = """ 
        ### Gambler's Problem as described in Example 4.3 of the text

        ---
        A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. 
        If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. 
        The game ends when the gambler wins by reaching his **Goal**, or loses by running out of money. 
        On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. 
        This problem can be formulated as an undiscounted, episodic, finite MDP. 
        The state is the gambler’s capital and the actions are stakes. 
        The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1. 
        The state-value function then gives the probability of winning from each state. 
        A policy is a mapping from levels of capital to stakes. 
        The optimal policy maximizes the probability of reaching the goal. 
        If **Probabiliy of heads** is known, then the entire problem is known and it can be solved, for instance, by value iteration. 
        The graph below shows the change in the value function over successive sweeps of value iteration, and the final policy found. 
        This policy is optimal, but not unique. 
        In fact, there is a whole family of optimal policies, all corresponding to ties for the argmax action selection with respect to the optimal value function. 
        Can you guess what the entire family looks like?

        ---

        """

        return description
