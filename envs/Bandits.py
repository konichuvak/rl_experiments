import numpy as np
import plotly.graph_objs as go
from plotly import tools
from random import random, sample
from utils import cache


class Bandits:
    """ Selected exercises from Chapter 2 of Sutton & Barto (2019) """

    ####################################################################################################################
    # ALGORITHMS

    def kArmedTestbed(self, k, nplays, epsilon):
        """
        :param k:           the number of armed bandits in the experiment
        :param nplays:      the number of plays in a given run
        :param epsilon:     probability of selecting a random action
        """

        actions = list(range(1, k + 1))
        action_values = {a: 0 for a in actions}  # (v = current value-estimate, n = # of times action was sampled)
        samples = [0] * k

        muvec = np.random.standard_normal(k)  # list of expected rewards
        actionvec = list()  # list of actions taken
        rewardvec = list()  # list of realized rewards

        for t in range(nplays):
            if random() < epsilon:
                a = sample(actions, 1)[0]
            else:
                a = max(action_values, key=action_values.get)
            actionvec.append(a)
            samples[a - 1] += 1
            r = np.random.normal(muvec[a - 1], 1)
            rewardvec.append(r)
            # use equal weighting to update the value estimates
            action_values[a] = action_values[a] + (r - action_values[a]) / samples[a - 1]
        return muvec, rewardvec, actionvec

    def NonStationaryTestbed(self, k, nplays, epsilon, seed=1, alpha=None):
        """
        :param k:           the number of armed bandits in the experiment
        :param nplays:      the number of plays in a given run
        :param epsilon:     probability of selecting a random action
        :param seed:        random seed used to ensure reproducibility of simulation results
        :param alpha:       parameter that determines exponential weightings

        """
        np.random.seed(seed)
        z = np.random.standard_normal((nplays, k))
        actions = list(range(1, k + 1))
        action_values = {a: 0 for a in actions}

        mumat = np.zeros((nplays, k))
        actionvec = list()  # list of actions taken
        rewardvec = list()  # list of realized rewards
        qmat = np.zeros((nplays, k))

        if alpha:
            for t in range(nplays):
                if random() < epsilon:
                    a = sample(actions, 1)[0]
                else:
                    a = max(action_values, key=action_values.get)
                actionvec.append(a)
                r = mumat[t - 1] + 0.01 * z[t - 1]
                mumat[t] = r
                rewardvec.append(r)
                action_values[a] = action_values[a] + alpha * (r[a - 1] - action_values[a])
                qmat[t] = list(action_values.values())
            return mumat, rewardvec, actionvec, qmat

        else:
            samples = [0] * k
            for t in range(1, nplays):
                if random() < epsilon:
                    a = sample(actions, 1)[0]
                else:
                    a = max(action_values, key=action_values.get)
                actionvec.append(a)
                samples[a - 1] += 1
                r = mumat[t - 1] + 0.01 * z[t - 1]
                mumat[t] = r
                rewardvec.append(r[a - 1])
                # use equal weighting to update the value estimates
                action_values[a] = action_values[a] + (r[a - 1] - action_values[a]) / samples[a - 1]
                qmat[t] = list(action_values.values())
            return mumat, rewardvec, actionvec, qmat

    def action_preference(self, nplays, k, alpha=0.02, seed=1):
        np.random.seed(seed)

        mumat = np.random.normal([-0.5, 0, 0.5], [1, 1, 1], (nplays, k))
        h_mat = np.zeros((nplays + 1, k))
        actions = list(range(1, k + 1))
        rolling_rewards = [0, 0]

        for t in range(nplays):
            prob = np.exp(h_mat[t])
            p = prob.copy()
            for a in actions:
                prob[a - 1] = p[a - 1] / sum(p)
            try:
                assert sum(prob) == 1.0
            except AssertionError:
                pass
            a = np.random.choice(actions, 1, p=prob)[0] - 1
            r = mumat[t][a]
            if not rolling_rewards[1]:
                rolling_rewards = [r, 1]
            else:
                rolling_rewards[1] += 1
                rolling_rewards[0] = (rolling_rewards[0] / (rolling_rewards[1] - 1) + r) / rolling_rewards[1]

            for act in range(k):
                if a == act:
                    h_mat[t + 1][a] = h_mat[t][a] + alpha * (r - rolling_rewards[0]) * (1 - prob[a])
                else:
                    h_mat[t + 1][act] = h_mat[t][act] - alpha * (r - rolling_rewards[0]) * prob[a]
        return h_mat

    ####################################################################################################################
    # PLOTTING

    @staticmethod
    def generate_plot(nplays, rewards, optimality):
        x_axis = list(range(nplays))

        subplots = ['Average reward', '% Optimal action']  # these have their own yaxis

        # Main Layout
        n = len(subplots)
        fig = tools.make_subplots(
            rows=n,
            cols=1,
            subplot_titles=subplots,
            vertical_spacing=0.1,
        )

        fig['layout'].update(
            height=800,
            # width=1600,
            showlegend=True,
            title='k-armed bandit',
            # titlefont={"size": 25},
            # margin={'l': 100, 't': 0, 'r': 100},
            # hovermode='closest',
        )

        colors = {
            0.0 : 'darkgreen',
            0.1 : 'midnightblue',
            0.01: 'crimson',
        }

        for epsilon, average in rewards.items():
            trace = go.Scatter(
                x=x_axis,
                y=average,
                mode='lines',
                name=f'epsilon={epsilon}',
                marker=dict(
                    color=colors[epsilon],
                )
            )
            fig.append_trace(trace, 1, 1)

        for epsilon, percent_optimal in optimality.items():
            trace = go.Scatter(
                x=x_axis,
                y=percent_optimal,
                mode='lines',
                name=f'epsilon={epsilon}',
                marker=dict(
                    color=colors[epsilon],
                )
            )
            fig.append_trace(trace, 2, 1)

        return fig

    @staticmethod
    def plot_non_stationary(j, nplays, expected_rewards, qs):
        x_axis = list(range(nplays))

        layout = dict(
            height=800,
            # width=1600,
            showlegend=True,
            title=f'Arm number {j}',
            titlefont={"size": 18},
        )

        traces = list()
        for i, q in enumerate(qs):
            if not q.any():
                continue
            traces.append(go.Scatter(
                x=x_axis,
                y=q,
                mode='lines',
                name='Uniform' if i == 0 else 'Exponential',
                marker=dict(
                    color='midnightblue' if i == 0 else 'crimson',
                )
            ))
        traces.append(go.Scatter(
            x=x_axis,
            y=expected_rewards[0] if expected_rewards[0].any() else expected_rewards[1],
            mode='lines',
            name='Expected rewards',
            marker=dict(
                color='darkgreen',
            )
        ))

        return {'data': traces, 'layout': layout}

    @staticmethod
    def plot_gradient_bandit(j, nplays, h_mat):
        x_axis = list(range(nplays))
        colors = {
            1: 'darkgreen',
            2: 'midnightblue',
            3: 'crimson',
        }
        mu = [-0.5, 0, 0.5]
        layout = dict(
            height=500,
            # width=1600,
            showlegend=True,
            title=f'Arm number {j}, mu = {mu[j - 1]}',
            titlefont={"size": 18},
        )
        trace = go.Scatter(
            x=x_axis,
            y=h_mat,
            mode='lines',
            marker=dict(
                color=colors[j],
            ),
            name=f'mu = {mu[j - 1]}'
        )

        return {'data': [trace], 'layout': layout}
