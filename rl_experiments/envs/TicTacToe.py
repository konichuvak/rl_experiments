import numpy as np
import random

random.seed(1)
from random import random, randint
from tqdm import tqdm
import plotly.graph_objs as go
from plotly import tools


class Board:

    def __init__(self, state=None):
        self.states = []
        self.board = np.zeros((3, 3))
        if isinstance(state, np.ndarray):
            self.board = state

    def __bytes__(self):
        return self.board.copy().tostring()

    def get_valid_moves(self) -> list:
        return np.dstack(np.where(self.board == 0))[0]

    def move(self, coord: tuple, piece: int):
        self.board[coord] = piece

    def epsilon_greedy_move(self, policy: np.ndarray, piece: int, epsilon: float = 0.1):
        valid = self.get_valid_moves()
        if valid.size == 0:
            raise Exception()
        if random() < epsilon:
            coord = tuple(valid[randint(0, len(valid) - 1)])
        else:
            # take the action that maximizes action_value function for the next state
            coord = policy[self.board.tostring()]
        self.move(coord, piece)
        return coord

    def random_move(self, player: int) -> tuple:
        valid = self.get_valid_moves()
        if valid.size == 0:
            raise Exception()

        coord = tuple(valid[randint(0, len(valid) - 1)])
        self.move(coord, player)
        return coord

    def has_won(self, player: int) -> int:
        """ Scans rows, columns, and diagonals for a win for """
        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board[i, :] == player).all()
            cols_win = (self.board[:, i] == player).all()

            if rows_win or cols_win:
                return 1

        diag1_win = (np.diag(self.board) == player).all()
        diag2_win = (np.diag(np.fliplr(self.board)) == player).all()

        if diag1_win or diag2_win:
            # Checks both diagonals for match
            return 1

        return False

    def is_full(self) -> bool:
        return self.get_valid_moves().size == 0

    def is_draw(self) -> bool:
        return self.is_full() and not self.has_won()

    def random_board(self):
        """
        there are maximum of 4 pieces of any player on the board to make it valid
        and the number of 1's should always match the number of -1's
        """
        # TODO:
        self.board = np.zeros((3, 3))
        moves = randint(1, 4)
        for i in range(moves):
            print(self.board)
            if i < 2:

                self.random_move(1)
                self.random_move(-1)
            else:
                # ensure no winning boards occur during generation
                b = self.board.copy()
                for player in (1, -1):
                    actions = list()
                    coord = self.random_move(player)
                    actions.append(coord)

                    # does not work if any of the actions lead to a win the current boart
                    if self.has_won(player):
                        # try another move
                        self.board = b
                        new_move = self.random_move(player)

                    b = self.board.copy()


class TicTacToe:

    def play(self, policy=None, init_state=None):
        """
        :param policy:      
        :return:            result of the game from the perspective of player 1
        """
        board = Board()

        if init_state:
            # exploring starts ensures that all the possible actions are taken from any possible state
            player = 1
            board.board = init_state
            action = board.random_move(player)
            rollout = [(init_state, action)]
        else:
            rollout = {'states': list(), 'actions': list()}
            player = -1

        while not board.has_won(player):
            if board.is_full():
                player = 0
                break
            player *= -1
            if player == 1:
                current_board = bytes(board)
                if policy:
                    if policy.get(current_board):
                        # if the policy is defined, use it
                        action = policy[current_board]
                        board.move(action, player)
                    else:
                        # if the policy does not specify action at a given state, then take an action at random
                        action = board.random_move(player)
                    if init_state:
                        rollout.append((current_board, action))
                    else:
                        rollout['states'].append(current_board)
                        rollout['actions'].append(action)
                else:
                    action = board.random_move(player)
                    rollout['states'].append(current_board)
                    rollout['actions'].append(action)
            else:
                board.random_move(player)  # second player follows a random policy

        return rollout, player

    def monte_carlo_prediction(self, n_iters: int):
        state_values = dict()
        n_visits = state_values.copy()

        # generate trajectories following a random policy
        gamma = 0.9
        for _ in range(n_iters):
            episode, reward = self.play()

            g = reward
            states = episode['states']
            for state in reversed(states):
                g *= gamma

                # first visit Monte Carlo
                states = states[:-1]
                if state in states:
                    continue

                if state in state_values:
                    n = n_visits[state] = n_visits[state] + 1
                    v = state_values[state]
                    state_values[state] = v + (g - v) / n
                else:
                    state_values[state] = g
                    n_visits[state] = 1

        return state_values

    def monte_carlo_es(self, n_iters: int):
        q_values = dict()
        n_visits = q_values.copy()
        policy = {np.zeros((3, 3)).tostring(): (1, 1)}
        possible_actions = list(zip([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2] * 3))

        gamma = 0.9
        for _ in tqdm(range(n_iters)):

            np.random.randint(-1, 2, 9)
            # TODO: generate random boards
            # init_state = [randint(12, 21), randint(1, 10), randint(0, 1)]
            episode, reward = self.play(policy)

            g = 0
            rewards = [0] * (len(episode) - 1) + [reward]
            for state, action in reversed(episode):
                g = gamma * g + rewards.pop()

                # first visit Monte Carlo (irrelevant in tic-tac-toe due to abscence of recurrent states)
                # episode = episode[:-1]
                # if (state, action) in episode:
                #     continue

                if state in q_values:
                    n = n_visits[state][action] = n_visits[state][action] + 1
                    q = q_values[state][action]
                    q_values[state][action] = q + (g - q) / n

                    # unexplored states should be preferred over states with negative rewards
                    # although the unexplored might not be available, so check for availability first
                    board = np.fromstring(state).reshape((3, 3))
                    valid_moves = np.dstack(np.where(board == 0))[0]

                    greedy_acton = policy[state]
                    q_value = -float('inf')
                    for move in valid_moves:
                        # b = board.copy()
                        move = tuple(move)
                        action_value = q_values[state][move]  # if have not been in (s, a) yet, init to 0
                        # track max q_value for all possible actions in the current state
                        if action_value > q_value:
                            q_value = action_value
                            greedy_acton = move
                    policy[state] = greedy_acton
                    # policy[state] = max(q_values[state], key=q_values[state].get)

                else:
                    n_visits[state] = np.zeros((3, 3))
                    q_values[state] = np.zeros((3, 3))
                    n_visits[state][action] = 1
                    q_values[state][action] = g

                    board = np.fromstring(state).reshape((3, 3))
                    valid_moves = np.dstack(np.where(board == 0))[0].astype(np.int64)
                    ix = randint(0, valid_moves.shape[0] - 1)
                    policy[state] = tuple(valid_moves[ix])

        return q_values, policy, n_visits

    def plot_boards(self, states, q_values, policy, samples):
        fig = tools.make_subplots(len(states), 4,
                                  # vertical_spacing=0.05,
                                  subplot_titles=['States', 'Q Values', 'Optimal Policy', 'Number of Samples'])
        layout = dict(
            height=len(states) * 300,
            # width=400,
            title='Optimal Tic-Tac-Toe', showlegend=False
        )
        fig['layout'].update(layout)

        for i, state in enumerate(states):
            s = go.Heatmap(
                z=np.fromstring(state).reshape((3, 3)),
                showlegend=False,
                colorscale='Viridis',
                showscale=False,
                zmin=-1,
                zmax=1,
            )
            qq = go.Heatmap(
                z=q_values[state],
                showlegend=False,
                showscale=False,
                zmin=-1,
                zmax=1,
            )
            heatmap = np.zeros((3, 3))
            heatmap[policy[state]] = 1
            pi = go.Heatmap(
                z=heatmap,
                showlegend=False,
                showscale=False,
                zmin=0,
                zmax=1,
            )
            sample = go.Heatmap(
                z=samples[state],
                showlegend=False,
                showscale=False,
                zmin=0,
            )
            fig.append_trace(s, i + 1, 1)
            fig.append_trace(qq, i + 1, 2)
            fig.append_trace(pi, i + 1, 3)
            fig.append_trace(sample, i + 1, 4)

        return fig


if __name__ == "__main__":
    ttt = TicTacToe()
    b = Board()
    for i in range(10):
        b.random_board()
        print(b.board)
        print()
    # q, pi, n = ttt.monte_carlo_es(10000)
