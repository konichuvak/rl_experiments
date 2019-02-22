import numpy as np
import pandas as pd
from Utilities import NoPrint
import plotly.graph_objs as go
from plotly import tools
from tqdm import tqdm
import random
np.random.seed(1)
random.seed(1)


class Blackjack:
    # TODO: implement every visit MC

    def __init__(self, gamma=1):
        self.gamma = gamma
        self.cards = np.concatenate((np.arange(1, 11), np.repeat(10, 3)))  # 2-10, J=10, Q=10, K=10, A=1

        self.offset = 12  # index 0 of a matrix would indicate value of a player summing upto 12
        self.players_policy = np.array([np.vstack((np.ones((8, 10)), np.zeros((2, 10))))]*2, dtype=np.int64)
        self.dealer_policy = np.concatenate((np.ones(5, dtype=np.int64), np.zeros(5, dtype=np.int64)))

    def get_card(self):
        return np.random.choice(self.cards)

    def generate_episode(self, init_state=None, init_action=None):
        """ Generate an episode following given policy """
        rollout = {
            'states': list(),
            'actions': list(),
            'rewards': list()
        }
        player, useable_ace_player = 0, 0
        dealer_card1, dealer, useable_ace_dealer = 0, 0, 0

        if not init_state:
            while player < 12:
                card = self.get_card()
                player += card
                if not useable_ace_player and card == 1 and player <= 10:
                    useable_ace_player = 1
                    player += 10

            for i in range(2):
                card = self.get_card()
                if i == 0:
                    dealer_card1 = card
                dealer += card
                if not useable_ace_dealer and card == 1:
                    useable_ace_dealer = 1
                    dealer += 10

        else:
            player, dealer_card1, useable_ace_player = init_state
            # the only way to have 21 is to have 1 usable ace
            if player == 21:
                useable_ace_player = 1

            dealer = dealer_card1
            if dealer_card1 == 1:
                useable_ace_dealer = 1
                dealer += 10
            card = self.get_card()
            dealer += card
            if not useable_ace_dealer and card == 1:
                useable_ace_dealer = 1
                dealer += 10

        state = [useable_ace_player, player, dealer_card1]
        print(f'Initial State: {state}, dealers sum is {dealer}')

        # rollout an episode
        while player <= 21:
            print(state)

            rollout['states'].append(tuple(state.copy()))
            if init_action is not None:
                action = init_action
                init_action = None
            else:
                action = self.players_policy[state[0], state[1] - self.offset, dealer_card1-1]
            rollout['actions'].append(action)

            if action:
                card = self.get_card()
                print(f"Player hits and gets {card}")
                player += card
                state[1] = player
                if player == 21:
                    rollout['states'].append(tuple(state.copy()))
                    rollout['actions'].append(0)
                    rollout['rewards'].append(0)
                    break
                elif player > 21:
                    if useable_ace_player:
                        state[1] = player = player - 11
                        state[0] = useable_ace_player = useable_ace_player - 1
                        print(f'Player uses ace to transition to {state}')
                        rollout['rewards'].append(0)
                        continue
                    else:
                        print('Player is over')
                        rollout['rewards'].append(-1)
                        return rollout
                else:
                    rollout['rewards'].append(0)
            else:
                print("Player sticks")
                break

        # player hit 21 or sticked
        # run the dealer's rollout

        while dealer < 21:
            if dealer <= 11:
                action = 1
            else:
                action = self.dealer_policy[dealer - self.offset]
            if action:
                card = self.get_card()
                print(f"Dealer hits and gets {card}")
                dealer += card
                if dealer == 21:
                    print('Dealer is even')
                    if player == 21:
                        rollout['rewards'].append(0)
                    else:
                        rollout['rewards'].append(-1)
                    return rollout
                elif dealer > 21:
                    if useable_ace_dealer:
                        dealer -= 11
                        useable_ace_dealer -= 1
                        print(f'Dealer uses ace to transition to {player, dealer}')
                        continue
                    else:
                        print('Dealer is over')
                        rollout['rewards'].append(1)
                        return rollout
                else:
                    continue
            else:
                print("Dealer sticks")
                break

        # awaiting reward resolution as both sticked
        print(f'Resolving {state}, dealers sum is {dealer}')
        if player > dealer:
            rollout['rewards'].append(1)
        elif player == dealer:
            rollout['rewards'].append(0)
        else:
            rollout['rewards'].append(-1)

        return rollout

    def mc_prediction(self, n_episodes, first_visit: bool = True) -> np.ndarray:
        """
        :param n_episodes:      number of episodes to sample
        :param policy:          numpy matrix of actions corresponding to each state
        :param first_visit:     whether to use first visit to the state only when calculating sample mean
        :return:                estimated state values associated with the policy
        """
        state_values = np.zeros((2, 10, 10))
        self.n_samples_returns = state_values.copy()

        for _ in tqdm(range(1, n_episodes+1)):
            with NoPrint():
                episode = self.generate_episode()
            g = 0
            if first_visit:
                states = episode['states']
                if len(states) != len(list(map(set, states))):
                    episode = pd.DataFrame.from_dict(episode)
                    episode = episode.drop_duplicates(subset=['states'], keep='first')
                    states = reversed(list(episode['states']))
                    rewards = reversed(list(episode['rewards']))
                else:
                    rewards = reversed(episode['rewards'])

                for s in reversed(states):
                    state = (s[0], s[1] - self.offset, s[2]-1)
                    g = self.gamma * g + rewards.__next__()
                    n = self.n_samples_returns[state] = self.n_samples_returns[state] + 1
                    v = state_values[state]
                    state_values[state] = v + (g - v)/n
            else:
                raise Exception('all visit MC is not implemented')

        return state_values

    def monte_carlo_es(self, n_episodes: int, first_visit: bool = True):
        q_values = np.zeros((2, 2, 10, 10))  # (useable ace, action_taken, player's 12-21, dealer's ace-10, )
        n_visits = q_values.copy()

        # i = dict(zip(range(1, 11), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        # j = dict(zip(range(1, 11), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        for _ in tqdm(range(n_episodes)):
            # exploring starts
            init_action = random.randint(0, 1)
            init_state = [random.randint(12, 21), random.randint(1, 10), random.randint(0, 1)]
            with NoPrint():
                episode = self.generate_episode(init_state=init_state, init_action=init_action)

            # TODO: debug the edge case for when in [0, 21, x]
            # for x in range(1, 11):
            #     if [0, 21, x] in episode['states']:
            #         print(episode)
            # if episode['states'][-1][0] == 0 and episode['states'][-1][1] == 21:
            #     # print('found last')
            #     # print(episode)
            #     i[episode['states'][-1][2]] += 1
            #     if episode['rewards'][-1] == 1:
            #         j[episode['states'][-1][2]] += 1
            state_actions = episode['state_actions'] = list(zip(episode['states'], episode['actions']))
            del episode['states']
            del episode['actions']

            g = 0
            if first_visit:

                if len(state_actions) != len(list(map(set, state_actions))):
                    episode = pd.DataFrame.from_dict(episode)
                    episode = episode.drop_duplicates(subset=['state_actions'], keep='first')
                    state_actions = episode['state_actions']
                    rewards = reversed(list(episode['rewards']))
                else:
                    rewards = reversed(episode['rewards'])

                for s, action in reversed(state_actions):
                    state = (s[0], s[1] - self.offset, s[2]-1)
                    g = self.gamma * g + rewards.__next__()

                    n = n_visits[action][state] = n_visits[action][state] + 1
                    q = q_values[action][state]
                    q_values[action][state] = q + (g - q) / n

                    greedy_acton = self.players_policy[state]
                    q_value = -float('inf')
                    for a in (0, 1):
                        action_value = q_values[a][state]
                        if action_value > q_value:
                            q_value = action_value
                            greedy_acton = a
                    self.players_policy[state] = greedy_acton

            else:
                raise Exception('all visit MC is not implemented')

        return q_values, self.players_policy, n_visits

    def monte_carlo_epsilon_greedy(self, n_episodes: int, epsilon: float = 0.1, first_visit: bool = True):
        q_values = np.zeros((2, 2, 10, 10))
        n_visits = q_values.copy()

        for _ in tqdm(range(n_episodes)):
            with NoPrint():
                episode = self.generate_episode()

            state_actions = episode['state_actions'] = list(zip(episode['states'], episode['actions']))
            del episode['states']
            del episode['actions']

            g = 0
            if first_visit:

                if len(state_actions) != len(list(map(set, state_actions))):
                    episode = pd.DataFrame.from_dict(episode)
                    episode = episode.drop_duplicates(subset=['state_actions'], keep='first')
                    state_actions = episode['state_actions']
                    rewards = reversed(list(episode['rewards']))
                else:
                    rewards = reversed(episode['rewards'])

                for s, action in reversed(state_actions):
                    state = (s[0], s[1] - self.offset, s[2]-1)
                    g = self.gamma * g + rewards.__next__()
                    n = n_visits[action][state] = n_visits[action][state] + 1
                    q = q_values[action][state]
                    q_values[action][state] = q + (g - q) / n

                    greedy_acton = self.players_policy[state]
                    q_value = -float('inf')
                    for a in (0, 1):
                        action_value = q_values[a][state]
                        if action_value > q_value:
                            q_value = action_value
                            greedy_acton = a
                    if random.random() < epsilon:
                        self.players_policy[state] = random.randint(0, 1)
                    else:
                        self.players_policy[state] = greedy_acton

            else:
                raise Exception('all visit MC is not implemented')

        return q_values, self.players_policy, n_visits

    @staticmethod
    def plot_value_function(state_values, n_episodes=None):
        fig = tools.make_subplots(2, 1,
                                  subplot_titles=['No Usable Ace', 'With Usable Ace'],
                                  vertical_spacing=0.2
                                  )
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=13)
        layout = dict(
            height=800,
            # width=400,
            title=f'After {n_episodes} episodes' if n_episodes else 'Optimal Value Function',
            showlegend=False,
        )
        fig['layout'].update(layout)

        for i, sv in enumerate(state_values):
            trace = go.Heatmap(
                z=sv,
                showlegend=False,
                showscale=False,
                colorscale='Viridis',
                x=list(range(1, 11)),
                y=list(range(12, 22))
            )
            fig.append_trace(trace, i+1, 1)

        fig['layout']['yaxis1'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['yaxis2'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['xaxis1'].update(dict(title='dealer showing', titlefont=dict(size=12)))
        fig['layout']['xaxis2'].update(dict(title='dealer showing', titlefont=dict(size=12)))

        return fig

    @staticmethod
    def plot_policy(policy):
        fig = tools.make_subplots(2, 1,
                                  subplot_titles=['No Usable Ace', 'With Usable Ace'],
                                  vertical_spacing=0.2
                                  )
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=13)
        layout = dict(
            height=800,
            title=f'Optimal Policy',
            showlegend=False,
        )
        fig['layout'].update(layout)

        for i in range(2):
            heatmap = go.Heatmap(
                z=policy[i, :, :],
                showlegend=False,
                showscale=False,
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                x=list(range(1, 11)),
                y=list(range(12, 22))
            )
            fig.append_trace(heatmap, i+1, 1)

        fig['layout']['yaxis1'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['yaxis2'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['xaxis1'].update(dict(title='dealer showing', titlefont=dict(size=12)))
        fig['layout']['xaxis2'].update(dict(title='dealer showing', titlefont=dict(size=12)))

        return fig

    @staticmethod
    def plot_n_visits(n_visits, action):
        fig = tools.make_subplots(2, 1,
                                  subplot_titles=['No Usable Ace', 'With Usable Ace'],
                                  vertical_spacing=0.2
                                  )
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=13)
        layout = dict(
            height=800,
            title=f'Sampled States with action {action} taken',
            showlegend=False,
        )
        fig['layout'].update(layout)

        for i in range(2):
            heatmap = go.Heatmap(
                z=n_visits[i, :, :],
                showlegend=False,
                showscale=False,
                colorscale='Viridis',
                zmin=0,
                x=list(range(1, 11)),
                y=list(range(12, 22))
            )
            fig.append_trace(heatmap, i + 1, 1)

        fig['layout']['yaxis1'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['yaxis2'].update(dict(title='players sum', titlefont=dict(size=12)))
        fig['layout']['xaxis1'].update(dict(title='dealer showing', titlefont=dict(size=12)))
        fig['layout']['xaxis2'].update(dict(title='dealer showing', titlefont=dict(size=12)))

        return fig


if __name__ == "__main__":
    bj = Blackjack()
    # sv = bj.mc_prediction(1000)
    # fig = bj.plot_value_function(sv)
    av, policy, n_visits = bj.monte_carlo_es(1000)
    sv = np.max(av, axis=0)
    p = np.argmax(av, axis=0)

    fig1 = bj.plot_value_function(sv)
    fig2 = bj.plot_policy(policy)





