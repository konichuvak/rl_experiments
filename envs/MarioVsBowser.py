import numpy as np
import plotly.graph_objs as go
from plotly import tools
from scipy.stats import binom
from math import ceil


class MarioVsBowser:

    def __init__(self, mario_hp=30, bowser_hp=100, shrooms=3, mario_dmg=5, p_slash=0.8, slash_n_p=(5, 0.4),
                 fire_n_p=(10, 0.7), in_place=True):
        self.states = np.zeros((mario_hp, bowser_hp, shrooms))

        self.in_place = in_place
        self.v_pi = np.zeros((mario_hp + 1, bowser_hp + 1, shrooms + 1))

        self.mario_dmg = mario_dmg

        self.p_slash = p_slash
        max_slash_dmg = slash_n_p[0]
        self.slash_dmg = np.arange(max_slash_dmg + 1, dtype=np.int64)
        self.slash_pmf = binom.pmf(np.arange(max_slash_dmg + 1), *slash_n_p)

        self.p_fire = 1 - p_slash
        max_fire_dmg = fire_n_p[0]
        self.fire_dmg = np.arange(max_fire_dmg + 1, dtype=np.int64)
        self.fire_pmf = binom.pmf(np.arange(max_fire_dmg + 1), *fire_n_p)

    @staticmethod
    def marios_action(shrooms: int, marios_hp: int = None, policy: bool = False):
        """

        :param policy:      whether to follow policy or not
        :param shrooms:     number of mushrooms available
        :param marios_hp:   current Mario's hit points
        :return:            a single action taken according to the policy or
                            a set of actions available
        """
        if policy:
            if shrooms > 0 and marios_hp <= 5:
                return 1
            return 0
        else:
            if shrooms > 0:
                return {0, 1}
            return {0}

    def expected_update(self, state, value_matrix):
        """ performs bellman update for a single state """
        value = 0
        hp = np.maximum(state[0] - self.slash_dmg, np.zeros(self.slash_dmg.shape, dtype=np.int64))
        value += self.p_slash * self.slash_pmf.dot(value_matrix[[hp], state[1], state[2]].T)[0]

        hp = np.maximum(state[0] - self.fire_dmg, np.zeros(self.fire_dmg.shape, dtype=np.int64))
        value += self.p_fire * self.fire_pmf.dot(value_matrix[[hp], state[1], state[2]].T)[0]
        return value

    def policy_evaluation(self):
        """
            Consider the following policy: whenever Mario’s HP becomes smaller or equal to 5,
            Mario consumes a mushroom if any is left.
        """
        policy_state_values, policy = self.v_pi.copy(), self.v_pi.copy()
        new_state_values = policy_state_values.copy()
        theta = 1e-4
        delta = float('inf')
        while delta > theta:
            value_matrix = new_state_values if self.in_place else policy_state_values
            for m in range(policy_state_values.shape[2]):
                for y in range(1, policy_state_values.shape[1]):
                    for x in range(1, policy_state_values.shape[0]):

                        # mario's action
                        if not self.marios_action(policy=True, marios_hp=x, shrooms=m):
                            policy[x, y, m] = 0
                            next_state = [x, max(0, y - self.mario_dmg), m]
                            if next_state[1] == 0:
                                new_state_values[x, y, m] = 1
                                continue
                        else:
                            policy[x, y, m] = 1
                            next_state = [x + ceil(0.5 * (30 - x)), y, m - 1]

                        # bowser's action
                        new_state_values[x, y, m] = self.expected_update(next_state, value_matrix)

            delta = np.sum(np.abs(new_state_values - policy_state_values))
            print(f'State-value delta: {delta}')
            policy_state_values = new_state_values.copy()

        return policy_state_values, policy

    def value_iteration(self):

        state_values, optimal_policy = self.v_pi.copy(), self.v_pi.copy()
        new_state_values = state_values.copy()
        theta = 1e-9
        delta = float('inf')

        while delta > theta:
            value_matrix = new_state_values if self.in_place else state_values
            for m in range(state_values.shape[2]):
                for y in range(1, state_values.shape[1]):
                    for x in range(1, state_values.shape[0]):
                        # choose the action that maximizes value of the state
                        terminal = False
                        action_values = list()  # attack or eat shroom
                        for eat_shroom in self.marios_action(m):

                            # marios's action
                            if not eat_shroom:
                                next_state = [x, max(0, y - 5), m]
                                if next_state[1] == 0:
                                    new_state_values[x, y, m] = 1
                                    optimal_policy[x, y, m] = 0
                                    terminal = True
                                    break
                            else:
                                next_state = [x + ceil(0.5 * (30 - x)), y, m - 1]

                            # bowser's action
                            action_values.append(self.expected_update(next_state, value_matrix))

                        if terminal:
                            continue

                        new_state_values[x, y, m] = max(action_values)
                        optimal_policy[x, y, m] = np.argmax(action_values)

            delta = np.sum(np.abs(new_state_values - state_values))
            print(f'State-value delta: {delta}')
            state_values = new_state_values.copy()

        return state_values, optimal_policy

    def plot_policies(self, state_values, policies):

        fig = tools.make_subplots(
            rows=4, cols=4,
            # vertical_spacing=0.05, ['State Values under Policy', 'Optimal State Values', 'Policy', 'Optimal Policy']
            subplot_titles=[f'\U0001f344 = {i}' for i in range(4)] * 4,
            specs=[
                # [{'is_3d': True}]*4, [{'is_3d': True}]*4,
                [{'is_3d': False}] * 4, [{'is_3d': False}] * 4,
                [{'is_3d': False}] * 4, [{'is_3d': False}] * 4,
            ]
        )
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=12)
        layout = dict(
            height=1000,
            title='Mario VS Bowser MDP', showlegend=False
        )
        fig['layout'].update(layout)

        for i, matrix in enumerate(state_values + policies):
            for j in range(matrix.shape[-1]):
                # print(matrix[:, :, j])
                trace = go.Heatmap(
                    z=matrix[:, :, j],
                    showscale=False,
                    colorscale='Viridis'
                )
                fig.append_trace(trace, i + 1, j + 1)

        fig['layout']['yaxis1'].update(dict(title='Policy State Value Function'))
        fig['layout']['yaxis5'].update(dict(title='Optimal State Value Function'))
        fig['layout']['yaxis9'].update(dict(title='Heuristic Policy'))
        fig['layout']['yaxis13'].update(dict(title='Optimal Policy'))

        return fig

    @staticmethod
    def description():
        return """
        A solution to MDP framework designed by Frédéric Godin, Ph.D., FSA, ACIA for a 2019 Reinforcement Learning class.

        ---
        
        You want to help an Italian plummer with a stunning mustache called Mario to fight a mighty evil dragon
        called Bowser and save the Princess Peach. You want to design a fighting strategy for Mario which maximizes
        his chances of winning.
    
        Rules of the battle are the following:
        Mario initially has 30 health points (HP) whereas Bowser starts the fight with 100 HP.
        Mario starts the fight with 3 mushrooms in his possession.
    
        The first contender to fall at or under 0 HP loses the fight.
        The fight consists in a sequence of rounds which continue until one of the opponents loses.
        On each round, Mario start by performing an action, followed from an action by Bowser.
    
        Everytime Mario performs an action he has two choices. He can either attack Bowser or eat a mushroom.
        When Mario attacks, Bowser loses 5 HP.
        When Mario eats a Mushroom, he loses a mushroom and recovers half of his missing HP,
        i.e. he recovers 0.5(30-x) where x is the current HP of Mario.
        If Mario has no mushroom left, the only possible action for him is to attack Bowser.
    
        When Bowser performs an action, he either slashes Mario with his claws with probability 80%
        or blows fire at Mario with probability 20%.
        When Bowser slashes, Mario loses a number of HP distributed according to a Binomial(n = 5; p = 0.4).
        When Bowser blows fire, Mario loses a number of HP distributed according to a Binomial(n = 10; p = 0.7).
        
        ---
        
        """