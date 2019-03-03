import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from style import *
from app import app

import numpy as np
from tqdm import tqdm

from envs.GridWorld import GridWorld
# from CarRental import CarRental
from envs.GamblersRuin import GamblersRuin
# from MarioVsBowser import MarioVsBowser
from envs.Blackjack import Blackjack
from envs.TicTacToe import TicTacToe
from envs.RandomWalk import RandomWalk
from envs.WindyGridworld import WindyGridworld
from envs.CliffWalking import CliffWalking

layout = html.Div([
    html.Div(
        children=[

            ################
            # HEADER
            ################

            html.Div(
                id='header',
                children=[
                    html.Div(
                        id='title_div',
                        children=[
                            html.H2(
                                "RL Experiments",
                                style={
                                    # 'border': '2px solid',
                                    'font-family'  : 'Avenir',
                                    'verticalAlign': 'middle',
                                    'display'      : 'flex',
                                }
                            ),
                        ],
                        style={
                            'font-family'  : 'Avenir',
                            "font-size"    : "120%",
                            # "width"        : "80%",
                            'display'      : 'table-cell',
                            'verticalAlign': 'middle',
                            # 'position': 'relative',
                            # 'textAlign': 'center',

                        },
                        className='two columns',
                    ),
                    html.Div(
                        id='tabs_div',
                        children=[
                            dcc.Tabs(
                                id='section',
                                style=container_style,
                                value='Random Walk',
                                children=[
                                    dcc.Tab(
                                        label='Grid World',
                                        value='Grid World',
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    # dcc.Tab(
                                    #     label='Car Rental',
                                    #     value='Car Rental',
                                    #     style=tab_style,
                                    #     selected_style=selected_style
                                    # ),
                                    dcc.Tab(
                                        label="Gambler's Ruin",
                                        value="Gambler's Ruin",
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    # dcc.Tab(
                                    #     label="Mario VS Bowser",
                                    #     value="Mario VS Bowser",
                                    #     style=tab_style,
                                    #     selected_style=selected_style
                                    # ),
                                    dcc.Tab(
                                        label="Blackjack",
                                        value="Blackjack",
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    dcc.Tab(
                                        label='Tic Tac Toe',
                                        value='Tic Tac Toe',
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    dcc.Tab(
                                        label='Random Walk',
                                        value='Random Walk',
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    dcc.Tab(
                                        label='Windy Gridworld',
                                        value='Windy Gridworld',
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),
                                    dcc.Tab(
                                        label='Cliff Walking',
                                        value='Cliff Walking',
                                        style=tab_style,
                                        selected_style=selected_style
                                    ),

                                ],
                                content_style={
                                    # 'borderLeft'  : '1px solid #d6d6d6',
                                    # 'borderRight' : '1px solid #d6d6d6',
                                    # 'borderBottom': '1px solid #d6d6d6',
                                    # 'padding'     : '0px'
                                },
                                parent_style={
                                    # 'maxWidth': '5000px',
                                    # 'margin'  : '0'
                                }
                            ),
                        ],
                        className='eight columns',
                    ),
                    html.Div(
                        id='button_div',
                        children=[
                            html.Button('Stop', id='button'),
                        ],
                        className='one column',
                    ),
                ],
                style={
                    # 'backgroundColor': DECREASING_COLOR,
                    'textAlign'    : 'center',
                    'fontFamily'   : 'Avenir',
                    'border-bottom': '2px #104b7d solid',
                    'display'      : 'flex',
                    'align-items'  : 'center',
                },
                className='row'
            ),

            ################
            # BODY
            ################

            html.Div(
                id='static_components',
                children=[

                    #################
                    # DP
                    html.Div(
                        id='gamma_div',
                        children=[
                            html.Label('Gamma:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='gamma',
                                placeholder='Decay constant',
                                type='number',
                                value=1,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='grid_size_div',
                        children=[
                            html.Label('Square grid dimensions :', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='grid_size',
                                placeholder='NxN grid',
                                type='number',
                                value=4,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='in_place_div',
                        children=[
                            html.Label('In place value updates:', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='in_place',
                                options=[{'label': s, 'value': s} for s in ['True', 'False']],
                                value='True'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='prob_heads_div',
                        children=[
                            html.Label('Probability of heads:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='prob_heads',
                                placeholder='0.4',
                                type='number',
                                step=0.1,
                                value=0.4,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    ####################################################################################
                    html.Div(
                        id='goal_div',
                        children=[
                            html.Label('EV target:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='goal',
                                placeholder='100$',
                                type='number',
                                step=10,
                                value=100,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='max_cars_div',
                        children=[
                            html.Label('Maximum cars at location:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='max_cars',
                                placeholder='20',
                                type='number',
                                step=5,
                                value=20,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='max_move_cars_div',
                        children=[
                            html.Label('Maximum cars to move overnight:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='max_move_cars',
                                placeholder='5',
                                type='number',
                                step=1,
                                value=5,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='rental_rate_div',
                        children=[
                            html.Label('Rental rates at locations 1 and 2:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='rental_rate',
                                placeholder='3, 4',
                                type='text',
                                value='3, 4',
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='return_rate_div',
                        children=[
                            html.Label('Return rates at locations 1 and 2:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='return_rate',
                                placeholder='3, 2',
                                type='text',
                                value='3, 2',
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='rental_credit_div',
                        children=[
                            html.Label('Reward for renting a car:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='rental_credit',
                                placeholder='10$',
                                type='number',
                                step=2,
                                value=10,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='move_car_cost_div',
                        children=[
                            html.Label('Cost of moving a car:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='move_car_cost',
                                placeholder='2$',
                                type='number',
                                step=1,
                                value=2,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    ####################################################################################
                    html.Div(
                        id='task_div',
                        children=[
                            html.Label('Task:', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='task',
                                options=[{'label': s, 'value': s} for s in ['Evaluation', 'Control']],
                                value='Control'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='exploration_div',
                        children=[
                            html.Label('Exploration Method:', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='exploration',
                                options=[{'label': s, 'value': s} for s in
                                         ['Exploring Starts', 'Epsilon Greedy']],
                                value='Epsilon Greedy'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='n_iter_div',
                        children=[
                            html.Label('Number of MC samples:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='n_iter',
                                placeholder='10000',
                                type='number',
                                step=10000,
                                value=10000,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='behavior_div',
                        children=[
                            html.Label('Behavioral policy', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='behavior',
                                options=[{'label': s, 'value': s} for s in ['Random', 'Epsilon greedy']],
                                value='True'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='policy_div',
                        children=[
                            html.Label('Off Policy', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='off_policy',
                                options=[{'label': s, 'value': s} for s in ['True', 'False']],
                                value='True'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='features_div',
                        children=[
                            html.Label('Features', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='feature',
                                options=[{'label': s, 'value': s} for s in
                                         ['Simple', 'Stochastic Wind', 'King Moves']],
                                value='Simple'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),

                    html.Div(
                        id='comparison_div',
                        children=[
                            html.Label('Compare algorithms', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='comparison',
                                options=[{'label': s, 'value': s} for s in
                                         ['TD vs MC', 'n-steps']],
                                value='TD vs MC'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                ],
                className='row'
            ),

            html.Div(
                id='dynamic_components',
                children=[
                    html.Div(
                        id='rl-display-values',
                        children=[
                            dcc.Loading(id='rl-display-results', type='cube')
                            # type='graph', className='pv6')
                        ],
                        style={
                            'textAlign' : 'center',
                            'fontFamily': 'Avenir',
                        }
                    ),
                ],
                className='row',
            ),
            html.Br(),
            dcc.Link('To Bandits', href='/bandits'),
            html.Br(),
            dcc.Link('To HOME', href='/'),
        ],
        style={
            'width'       : '100%',
            'fontFamily'  : 'Avenir',
            'margin-left' : 'auto',
            'margin-right': 'auto',
            'boxShadow'   : '0px 0px 5px 5px rgba(204,204,204,0.4)'
        }
    )
])


@app.callback(
    Output('features_div', 'style'),
    [Input('section', 'value')],
)
def task_div(section):
    if section in ["Windy Gridworld"]:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('task_div', 'style'),
    [Input('section', 'value')],
)
def task_div(section):
    if section == "Blackjack":
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('policy_div', 'style'),
    [Input('section', 'value')],
)
def task_div(section):
    if section == "Blackjack":
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# @app.callback(
#     Output('behavior_div', 'style'),
#     [Input('off_policy', 'value'),
#      Input('section', 'value')],
# )
# def behavior_div(off_policy, section):
#     if section == "Blackjack":
#         if eval(off_policy):
#             return {'display': 'block'}
#     return {'display': 'none'}


@app.callback(
    Output('exploration_div', 'style'),
    [
        Input('section', 'value'),
        Input('task_div', 'style'),
    ],
)
def exploration_div(section, control):
    if section == "Blackjack":
        if control == {'display': 'block'}:
            return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('comparison_div', 'style'),
    [Input('section', 'value')],
)
def comparison_div(section):
    if section in ["Random Walk"]:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('n_iter_div', 'style'),
    [Input('section', 'value')],
)
def n_iter_div(section):
    if section in ["Blackjack", "Tic Tac Toe"]:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('goal_div', 'style'),
    [Input('section', 'value')],
)
def goal_div(section):
    if section == "Gambler's Ruin":
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('prob_heads_div', 'style'),
    [Input('section', 'value')],
)
def prob_heads_div(section):
    if section == "Gambler's Ruin":
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('in_place_div', 'style'),
    [Input('section', 'value')],
)
def in_place_div(section):
    if section in ['Grid World', 'Car Rental', "Gambler's Ruin", 'Mario VS Bowser']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('grid_size_div', 'style'),
    [Input('section', 'value')],
)
def grid_size_div(section):
    if section in ['Grid World']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('gamma_div', 'style'),
    [Input('section', 'value')],
)
def gamma_div(section):
    if section in ['Grid World', 'Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('max_cars_div', 'style'),
    [Input('section', 'value')],
)
def max_cars_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('max_move_cars_div', 'style'),
    [Input('section', 'value')],
)
def max_move_cars_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('rental_rate_div', 'style'),
    [Input('section', 'value')],
)
def rental_rate_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('return_rate_div', 'style'),
    [Input('section', 'value')],
)
def return_rate_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('rental_credit_div', 'style'),
    [Input('section', 'value')],
)
def rental_credit_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('move_car_cost_div', 'style'),
    [Input('section', 'value')],
)
def move_car_cost_div(section):
    if section in ['Car Rental']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


#####################################################


@app.callback(
    Output('gamma', 'value'),
    [Input('section', 'value')],
)
def gamma_value(section):
    if section in ['Car Rental']:
        return 0.9
    else:
        return 1


@app.callback(
    Output('button', 'children'),
    [Input('button', 'n_clicks')],
    [State('button', 'children')],
)
def disable_enable_button(clicked, state):
    print('BUTTON CLICK', clicked, state)
    if state == 'Run experiment':
        return 'Stop'
    elif state == 'Stop':
        return 'Run experiment'


@app.callback(
    Output('rl-display-results', 'children'),
    [
        Input('button', 'n_clicks')
    ],
    [
        State('button', 'children'),
        State('section', 'value'),

        # Shared across DP
        State('in_place', 'value'),

        # Gridworld DP
        State('grid_size', 'value'),
        State('gamma', 'value'),

        # Car Rentals

        # Gambler's Ruin
        State('prob_heads', 'value'),
        State('goal', 'value'),

        # Blackjack
        State('task', 'value'),
        State('exploration', 'value'),
        State('n_iter', 'value'),
        State('off_policy', 'value'),
        State('behavior', 'value'),

        # Windy Gridworld
        State('feature', 'value'),

        # Random Walk
        State('comparison', 'value')

    ],
)
def RL(clicks, button_state, section,
       in_place,
       grid_size, gamma,
       prob_heads, goal,
       task, exploration, n_iter,
       off_policy, behavior,
       feature,
       comparison
       ):
    print(clicks, button_state, section,
          in_place,
          grid_size, gamma,
          prob_heads, goal,
          off_policy, behavior
          )

    if not clicks:
        return
    if button_state == 'Stop':
        return

    if section == 'Grid World':

        gw = GridWorld(grid_dim=grid_size, gamma=gamma)
        sv = gw.gridworld_policy_iteration(in_place=bool(in_place), theta=1e-4)
        fig = gw.plot_grid_world(sv)
        return html.Div(
            dcc.Graph(
                id='results',
                figure=fig
            ),
        )

    elif section == 'Car Rental':
        cr = CarRental()
        state_values, policies = cr.policy_iteration(in_place=bool(in_place))
        fig = cr.plot_policies(state_values[-1], policies[:-1])
        return html.Div(
            dcc.Graph(
                id='values',
                figure=fig
            ),
        )

    elif section == "Gambler's Ruin":
        gr = GamblersRuin(goal=goal, p_heads=prob_heads)
        state_values_seq, policy = gr.policy_iteration(bool(in_place))

        return [
            html.Div(
                dcc.Graph(
                    id='values',
                    figure=gr.plot_value_iterations(state_values_seq)
                ),
                className=f'six columns',
            ),
            html.Div(
                dcc.Graph(
                    id='policy',
                    figure=gr.plot_optimal_policy(policy)
                ),
                className=f'six columns',
            )
        ]

    elif section == "Mario VS Bowser":
        mb = MarioVsBowser()
        sv, p = mb.policy_evaluation()
        sv_star, p_star = mb.value_iteration()
        fig = mb.plot_policies([sv, sv_star], [p, p_star])
        return html.Div(
            dcc.Graph(
                id='values',
                figure=fig
            ),
        )

    elif section == "Blackjack":
        bj = Blackjack()

        if task == 'Evaluation':

            if bool(off_policy):
                return [
                    html.Div(
                        dcc.Graph(
                            id='value_estimate',
                            figure=bj.plot_value_function(bj.mc_prediction(10000), 10000)
                        ),
                        className=f'six columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='better_value_estimate',
                            figure=bj.plot_value_function(bj.mc_prediction(500000), 500000)
                        ),
                        className=f'six columns',
                    )
                ]

            # elif task == 'Off-Policy Learning':
            #     true_value = -0.27726
            #     simulations = 100
            #     ordinary_msr = np.zeros(n_iter)
            #     weighted_msr = np.zeros(n_iter)
            #     for _ in tqdm(range(simulations)):
            #         ordinary_value, weighted_value = bj.monte_carlo_off_policy(n_iter)
            #         ordinary_msr += np.power(ordinary_value - true_value, 2)
            #         weighted_msr += np.power(weighted_value - true_value, 2)
            #     ordinary_msr /= simulations
            #     weighted_msr /= simulations
            #
            #     return [
            #         html.Div(
            #             dcc.Graph(
            #                 id='learning_curves',
            #                 figure=bj.plot_learning_curves(ordinary_msr, weighted_msr)
            #             ),
            #         )
            #     ]

            else:
                q = bj.monte_carlo_off_policy_evalualtion(n_iter)
                sv = np.max(q, axis=0)
                return [
                    html.Div(
                        dcc.Graph(
                            id='value_estimate',
                            figure=bj.plot_value_function(sv)
                        ),
                        className=f'six columns',
                    ),
                ]

        elif task == 'Control':
            if off_policy == 'True':
                q, policy = bj.monte_carlo_off_policy_control(n_iter)
                sv = np.max(q, axis=0)
                return [
                    html.Div(
                        dcc.Graph(
                            id='optimal_value_function',
                            figure=bj.plot_value_function(sv)
                        ),
                        className=f'six columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='optimal_policy',
                            figure=bj.plot_policy(policy)
                        ),
                        className=f'six columns',
                    ),
                ]

            else:

                if exploration == 'Exploring Starts':
                    av, policy, n_visits = bj.monte_carlo_es(n_iter)
                elif exploration == 'Epsilon Greedy':
                    av, policy, n_visits = bj.monte_carlo_epsilon_greedy(n_iter)
                sv = np.max(av, axis=0)

                return [
                    html.Div(
                        dcc.Graph(
                            id='optimal_value_function',
                            figure=bj.plot_value_function(sv)
                        ),
                        className=f'three columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='optimal_policy',
                            figure=bj.plot_policy(policy)
                        ),
                        className=f'three columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='samples0',
                            figure=bj.plot_n_visits(n_visits[0], 0)
                        ),
                        className=f'three columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='samples1',
                            figure=bj.plot_n_visits(n_visits[1], 1)
                        ),
                        className=f'three columns',
                    )
                ]

    elif section == 'Tic Tac Toe':
        ttt = TicTacToe()
        # if exploration == 'Exploring Starts':
        q, pi, n = ttt.monte_carlo_es(n_iter)
        # elif exploration == 'Epsilon Greedy':
        #     q, pi, n = ttt.monte_carlo_epsilon_greedy(n_iter)

        states = sorted([(np.sum(np.abs(av)), state) for state, av in q.items()], reverse=True)[:20]
        states = [s[1] for s in states]

        fig = ttt.plot_boards(states, q, pi, n)
        return html.Div(
            dcc.Graph(
                id='values',
                figure=fig
            ),
        )

    elif section == 'Random Walk':

        if comparison == 'TD vs MC':

            length = 5
            n_iter = 100
            simulations = 100

            rw = RandomWalk(length)
            mc_values = rw.mc_prediction(n_iter)
            td_values = rw.td_prediction(n_iter)
            fig1 = rw.plot_state_values(mc_values)
            fig2 = rw.plot_state_values(td_values)

            # rmse comparison
            alphas = {
                'TD': [0.15, 0.1, 0.05],
                'MC': [0.01, 0.02, 0.03, 0.04],
            }
            values = {
                'MC': dict(),
                'TD': dict()
            }

            for n in ('TD', 'MC'):
                for alpha in alphas[n]:
                    values[n][alpha] = np.zeros((simulations, n_iter, length))
                    for i in range(simulations):
                        v = getattr(rw, f'{n.lower()}_prediction')(n_episodes=n_iter, alpha=alpha)
                        values[n][alpha][i] = np.array([np.array(list(episode.values())[1:-1]) for episode in v])

                    values[n][alpha] = np.mean(values[n][alpha], axis=0)

            fig3 = rw.plot_rmse(values, tuple(values.keys()))

            # batch updates
            errors = {
                'MC': np.zeros((simulations, n_iter)),
                'TD': np.zeros((simulations, n_iter)),
            }

            for algo in errors:
                for i in tqdm(range(simulations)):
                    errors[algo][i] = rw.batch_updates(algo=algo)
                errors[algo] = np.mean(errors[algo], axis=0)

            fig4 = rw.plot_batch_rmse(errors)

            return [
                html.Div(
                    dcc.Graph(
                        id='values_mc',
                        figure=fig1,
                    ),
                    className=f'three columns',
                ),
                html.Div(
                    dcc.Graph(
                        id='values_td',
                        figure=fig2,
                    ),
                    className=f'three columns',
                ),
                html.Div(
                    dcc.Graph(
                        id='rmse',
                        figure=fig3,
                    ),
                    className=f'three columns',
                ),
                html.Div(
                    dcc.Graph(
                        id='rmse_batch',
                        figure=fig4,
                    ),
                    className=f'three columns',
                ),

            ]

        elif comparison == 'n-steps':
            length = 19
            n_iter = 10
            simulations = 100

            rw = RandomWalk(length)

            alphas = [alpha / 10 for alpha in range(11)]
            steps = [2 ** p for p in range(10)]
            values = dict(zip(steps, [dict() for _ in range(len(steps))]))

            for n in steps:
                for alpha in alphas:
                    print(n, alpha)
                    values[n][alpha] = np.zeros((simulations, n_iter, length))
                    values[n][alpha] = np.zeros((simulations, n_iter, length))
                    for i in range(simulations):
                        state_values = rw.n_step_td_prediction(n=n, n_episodes=n_iter, alpha=alpha, seed=i)
                        sv = np.array([np.array(list(episode.values())[1:-1]) for episode in state_values])
                        values[n][alpha][i] = sv

            fig3 = rw.plot_rmse(values, tuple(values.keys()))

            return [
                html.Div(
                    dcc.Graph(
                        id='values',
                        figure=fig3,
                    ),
                    className=f'three columns',
                ),
            ]

    elif section == "Windy Gridworld":

        wg = WindyGridworld(length=7, width=10, gamma=1, king_moves=False, stochastic_wind=False)
        action_values, timestamps, moves = wg.sarsa(n_episodes=170)
        fig1 = wg.plot_learning_rate(timestamps, title="Rook Moves")

        wg = WindyGridworld(length=7, width=10, gamma=1, king_moves=True, stochastic_wind=False)
        action_values, timestamps, moves = wg.sarsa(n_episodes=170)
        fig2 = wg.plot_learning_rate(timestamps, title="King Moves")

        wg = WindyGridworld(length=7, width=10, gamma=1, king_moves=True, stochastic_wind=True)
        action_values, timestamps, moves = wg.sarsa(n_episodes=170)
        fig3 = wg.plot_learning_rate(timestamps, title='Stochastic Wind')
        return [
            html.Div(
                dcc.Graph(
                    id='learning_rate',
                    figure=fig1,
                ),
                className=f'four columns',

            ),
            html.Div(
                dcc.Graph(
                    id='learning_rate_king_moves',
                    figure=fig2,
                ),
                className=f'four columns',
            ),
            html.Div(
                dcc.Graph(
                    id='learning_rate_stochastic_wind',
                    figure=fig3,
                ),
                className=f'four columns',
            ),
        ]

    elif section == "Cliff Walking":
        cliff_rewards = {(i, -1): -100 for i in range(1, 11)}
        cw = CliffWalking(width=12, height=4, other_rewards=cliff_rewards)

        simulations = 100
        n_episodes = 500

        algos = ['sarsa', 'q_learning', 'expected_sarsa', 'double_q_learning']
        rewards = dict(zip(algos, [np.zeros((simulations, n_episodes)) for _ in range(len(algos))]))

        for i in tqdm(range(simulations)):
            for algo in algos:
                rewards[algo][i] = getattr(cw, algo)(n_episodes=n_episodes, verbose=False)

        for algo, sims in rewards.items():
            rewards[algo] = np.mean(sims, axis=0)
        fig1 = cw.plot_rewards(rewards)

        # alphas = [alpha / 10 for alpha in range(1, 11)]
        # per_episode_rewards = {
        #     'asymptotic': dict(zip(alphas, [np.zeros((50000, 100000)) for _ in range(len(alphas))])),
        #     'interim'   : dict(zip(alphas, [np.zeros((10, 100)) for _ in range(len(alphas))])),
        # }
        # for cv in ['asymptotic', 'interim']:
        #     for alpha in tqdm(alphas):
        #         for algo in ['sarsa', 'q_learninig', 'expected_sarsa']:
        #             if cv == 'asymptotic':
        #                 for i in range(50000):
        #                     per_episode_rewards[cv][alpha][i] = cw.control(n_episodes=100000, algo=algo, alpha=alpha)
        #             else:
        #                 for i in range(10):
        #                     per_episode_rewards[cv][alpha][i] = cw.control(n_episodes=100, algo=algo, alpha=alpha)
        #         per_episode_rewards[alpha] = np.mean(per_episode_rewards[alpha], axis=0)
        #
        # fig2 = cw.compare_performance(per_episode_rewards)

        return [
            html.Div(
                dcc.Graph(
                    id='rewards',
                    figure=fig1,
                ),
                className=f'six columns',

            ),
            # html.Div(
            #         dcc.Graph(
            #                 id='comparisons',
            #                 figure=fig2,
            #         ),
            #         className=f'six columns',
            #
            # ),
        ]