import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app

from textwrap import dedent
from assets.style import *

from envs.GridWorld import GridWorld
# from CarRental import CarRental
from envs.GamblersRuin import GamblersRuin
# from MarioVsBowser import MarioVsBowser
from envs.Blackjack import Blackjack
from envs.TicTacToe import TicTacToe
from envs.RandomWalk import RandomWalk
from envs.WindyGridworld import WindyGridworld
from envs.CliffWalking import CliffWalking
from envs.DynaMaze import DynaMaze

import importlib
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import ray

ray.init(ignore_reinit_error=True)

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
                                    [
                                        html.Div(
                                                id='title_div',
                                                children=[
                                                    html.H2("RL Experiments"),
                                                ],
                                                className='two columns title',
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
                                                                        selected_style=selected_style,
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
                                                                # dcc.Tab(
                                                                #         label='Tic Tac Toe',
                                                                #         value='Tic Tac Toe',
                                                                #         style=tab_style,
                                                                #         selected_style=selected_style
                                                                # ),
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
                                                                dcc.Tab(
                                                                        label='Dyna Maze',
                                                                        value='Dyna Maze',
                                                                        style=tab_style,
                                                                        selected_style=selected_style
                                                                ),

                                                            ],
                                                    ),
                                                ],
                                                className='eight columns',
                                        ),
                                        html.Div(
                                                id='button_div',
                                                children=[
                                                    html.Button('Stop', id='button'),
                                                ],
                                                className='one column custom_button',
                                        ),
                                    ],
                                    className='row'
                            ),
                            html.Div(
                                    id='description_div',
                                    children=[
                                        html.Div(
                                                id='description',
                                                children=[dcc.Markdown([dedent(RandomWalk.description())])],
                                                style={'margin-right': '5%', 'margin-left': '5%'}
                                        ),
                                    ],
                                    className='row description'
                            ),
                        ],
                        className='header'
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
                                        html.Label('Goal:', style={'textAlign': 'center'}),
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
                                    id='simulation_div',
                                    children=[
                                        html.Label('Simulations:', style={'textAlign': 'center'}),
                                        dcc.Input(
                                                id='simulation',
                                                type='number',
                                                step=10,
                                                placeholder='100',
                                                value=100
                                        )
                                    ],
                                    style={'display': 'none'},
                                    className='one column',
                            ),

                            html.Div(
                                    id='task_div',
                                    children=[
                                        html.Label('Task:', style={'textAlign': 'center'}),
                                        dcc.Dropdown(
                                                id='task',
                                                options=[{'label': s, 'value': s} for s in ['Evaluation', 'Control']],
                                                value='Evaluation'
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
                                        html.Label('Number of episodes:', style={'textAlign': 'center'}),
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
                                                options=[{'label': s, 'value': s} for s in
                                                         ['Random', 'Epsilon greedy']],
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
                                    id='feature_div',
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
                                    id='walk_length_div',
                                    children=[
                                        html.Label('Walk Length', style={'textAlign': 'center'}),
                                        dcc.Input(
                                                id='walk_length',
                                                placeholder='10000',
                                                type='number',
                                                step=5,
                                                value=5,
                                                style={'width': '100%', 'textAlign': 'center'}
                                        )
                                    ],
                                    style={'display': 'none'},
                                    className='one columns',
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
            }
    )
])


display = ({'display': 'none'}, {'display': 'block'})
output_ids = sorted([
    'behavior_div', 'comparison_div', 'walk_length_div', 'feature_div', 'exploration_div', 'simulation_div', 'task_div',
    'policy_div', 'n_iter_div', 'prob_heads_div', 'goal_div', 'grid_size_div', 'gamma_div', 'max_cars_div',
    'max_move_cars_div', 'rental_rate_div', 'rental_credit_div', 'move_car_cost_div',

])
active_outputs = OrderedDict(zip(output_ids, (display[0] for _ in range(len(output_ids)))))
outputs_components = [Output(output_id, 'style') for output_id in output_ids]


@app.callback(
        outputs_components,
        [
            Input('section', 'value'),
            Input('task', 'value'),
            Input('off_policy', 'value'),

        ],
        [
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
            State('exploration', 'value'),
            State('n_iter', 'value'),
            State('behavior', 'value'),

            # Windy Gridworld
            State('feature', 'value'),

            # Random Walk
            State('comparison', 'value'),
            State('walk_length', 'value')
        ]
)
def show_hide(section, task, off_policy,
              in_place,
              grid_size, gamma,
              prob_heads, goal,
              exploration, n_iter,
              behavior,
              feature,
              comparison, walk_length):

    print(section)
    show = set()
    if section in ["Random Walk"]:
        show = {'comparison', 'walk_length', 'n_iter', 'simulation'}

    elif section in ["Windy Gridworld"]:
        show = {'feature'}

    elif section in ['Blackjack']:
        show = {'task', 'n_iter', 'policy'}
        if task == 'Evaluation':
            if off_policy == 'True':
                show |= {'behavior', 'simulation'}
        elif task == 'Control':
            if off_policy == 'True':
                show |= {'behavior'}
            else:
                show |= {'exploration'}

    elif section in {'Tic Tac Toe'}:
        show = {'n_iter'}

    elif section in {"Gambler's Ruin"}:
        show = {'prob_heads', 'goal'}

    elif section in {"Grid World"}:
        show = {'grid_size', 'gamma'}

    elif section in {"Car Rental"}:
        pass

    elif section in {"Cliff Walking"}:
        show = {'simulation', 'n_iter'}

    elif section in {"Dyna Maze"}:
        show = {'simulation', 'n_iter'}

    show = {f'{component}_div' for component in show}

    activations = active_outputs.copy()
    for comp in show:
        activations[comp] = display[1]

    return list(activations.values())


@app.callback(
        Output('in_place_div', 'style'),
        [Input('section', 'value')],
)
def in_place_div(section):
    if section in ['Grid World', 'Car Rental', "Gambler's Ruin", 'Mario VS Bowser']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

#####################################################
# DEFAULT VALUES
# TODO: refactor as in show/hide


@app.callback(
        Output('n_iter', 'value'),
        [
            Input('task', 'value'),
            Input('off_policy', 'value'),
            Input('comparison', 'value'),
            Input('section', 'value')
        ],
)
def n_iter(task, off_policy, comparison, section):
    if section == 'Blackjack':
        if task == 'Evaluation':
            if off_policy == 'True':
                return 1000
    elif section == 'Random Walk':
        if comparison == 'TD vs MC':
            return 100
        elif comparison == 'n-steps':
            return 10
    elif section == 'Cliff Walking':
        return 500
    elif section == 'Dyna Maze':
        return 50

    return 10000


@app.callback(
        Output('simulation', 'value'),
        [
            Input('task', 'value'),
            Input('off_policy', 'value'),
            Input('comparison', 'value'),
            Input('section', 'value')
        ],
)
def simulation(task, off_policy, comparison, section):
    print('sim')
    print(section)
    if section == 'Blackjack':
        if task == 'Evaluation':
            if off_policy == 'True':
                return 1000
    elif section == 'Random Walk':
        if comparison == 'TD vs MC':
            return 100
        elif comparison == 'n-steps':
            return 100
    elif section == 'Cliff Walking':
        return 100
    elif section == 'Dyna Maze':
        return 30

    return 100


@app.callback(
        Output('walk_length', 'value'),
        [Input('comparison', 'value')],
)
def walk_length(comparison):
    return 5 if comparison in ["TD vs MC"] else 19


@app.callback(
        Output('gamma', 'value'),
        [Input('section', 'value')],
)
def gamma_value(section):
    if section in ['Car Rental']:
        return 0.9
    else:
        return 1

#####################################################


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
        Output('description', 'children'),
        [Input('section', 'value')]
)
def description(section):
    classname = section.replace("\'", '').replace(' ', '')
    class_ = getattr(importlib.import_module(f"envs.{classname}"), classname)
    description = " "
    try:
        description = getattr(class_, 'description')()
    except Exception as e:
        print(e)
    return dcc.Markdown(dedent(description))


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
            State('simulation', 'value'),

            # Windy Gridworld
            State('feature', 'value'),

            # Random Walk
            State('comparison', 'value'),
            State('walk_length', 'value')

        ],
)
def RL(clicks, button_state, section,
       in_place,
       grid_size, gamma,
       prob_heads, goal,
       task, exploration, n_iter,
       off_policy, behavior, simulations,
       feature,
       comparison, walk_length
       ):
    print(clicks, button_state, section,
          in_place,
          grid_size, gamma,
          prob_heads, goal,
          off_policy, behavior
          )

    if not clicks:
        raise PreventUpdate
    if button_state == 'Stop':
        raise PreventUpdate

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

            if off_policy == 'False':
                # TODO: add online plotting of the value function

                description = """
                In the **On-Policy** Evaluation **Task** (Example 5.1) we consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits.
                To find the state-value function for this policy by a Monte Carlo approach, one simulates many blackjack games using the policy and averages the returns following each state. 
                In this way, we obtained the estimates of the state-value function shown in the graph. 
                The estimates for states with a usable ace are less certain and less regular because these states are less common. 
                Try various values for number of episodes to observe the convergence for yourself.
                """
                return [
                    html.Div(
                            id='graph-description',
                            children=dcc.Markdown(dedent(description)),
                            className='six columns'
                    ),
                    html.Div(
                            dcc.Graph(
                                    id='value_estimate',
                                    figure=bj.plot_value_function(bj.mc_prediction(n_iter), n_iter)
                            ),
                            className=f'six columns',
                    ),
                    # html.Div(
                    #         dcc.Graph(
                    #                 id='better_value_estimate',
                    #                 figure=bj.plot_value_function(bj.mc_prediction(500000), 500000)
                    #         ),
                    #         className=f'six columns',
                    # )
                ]

            else:
                true_value = -0.27726
                ordinary_msr = np.zeros(n_iter)
                weighted_msr = np.zeros(n_iter)
                for _ in tqdm(range(simulations)):
                    ordinary_value, weighted_value = bj.monte_carlo_off_policy(n_iter)
                    ordinary_msr += np.power(ordinary_value - true_value, 2)
                    weighted_msr += np.power(weighted_value - true_value, 2)
                ordinary_msr /= simulations
                weighted_msr /= simulations

                description = """
                In contrast, in the **Off-Policy** Evaluation **Task** (Example 5.4) we apply both ordinary and weighted importance-sampling methods to estimate the value of a single blackjack state from on-policy data. 
                Recall that one of the advantages of Monte Carlo methods is that they can be used to evaluate a single state without forming estimates for any other states. 
                In this example, we evaluated the state in which the dealer is showing a deuce, the sum of the player’s cards is 13, and the player has a usable ace 
                (that is, the player holds an ace and a deuce, or equivalently three aces). 
                The data was generated by starting in this state then choosing to hit or stick at random with equal probability (the behavior policy). 
                The target policy was to stick only on a sum of 20 or 21, as in Example 5.1. 
                The value of this state under the target policy is approximately  0.27726 
                (this was determined by separately generating one-hundred million episodes using the target policy and averaging their returns). 
                Both off-policy methods closely approximated this value after 1000 off-policy episodes using the random policy. 
                To make sure they did this reliably, we performed 100 independent runs, each starting from estimates of zero and learning for **Number of episodes**. 
                The graph shows the resultant learning curves—the squared error of the estimates of each method as a function of number of episodes, averaged over the 100 runs. 
                The error approaches zero for both algorithms, but the weighted importance-sampling method has much lower error at the beginning, as is typical in practice.
                """
                fig = bj.plot_learning_curves(ordinary_msr, weighted_msr)
                return [
                    html.Div(
                            id='graph-description',
                            children=dcc.Markdown(dedent(description)),
                            className='six columns'
                    ),
                    html.Div(
                            dcc.Graph(
                                    id='learning_curves',
                                    figure=fig
                            ),
                            className='six columns'
                    )
                ]

            # TODO: add param to choose this incremental evaluation
            # else:
            #     q = bj.monte_carlo_off_policy_evalualtion(n_iter)
            #     sv = np.max(q, axis=0)
            #     return [
            #         html.Div(
            #             dcc.Graph(
            #                 id='value_estimate',
            #                 figure=bj.plot_value_function(sv)
            #             ),
            #             className=f'six columns',
            #         ),
            #     ]

        elif task == 'Control':

            if off_policy == 'True':
                description = """
                The following graph shows an off-policy Monte Carlo control method, based on GPI and weighted importance sampling, for estimating Pi* and q*. 
                The target policy Pi~Pi* is the greedy policy with respect to Q, which is an estimate of q*. 
                The behavior policy b can be anything, but in order to assure convergence of Pi to the optimal policy, 
                an infinite number of returns must be obtained for each pair of state and action. 
                This can be assured by choosing b to be epsilon-soft. 
                The policy Pi converges to optimal at all encountered states even though actions are selected according to a di↵erent soft policy b, 
                which may change between or even within episodes.
                """
                # TODO: apply this algorithm to racetrack problem instead

                q, policy = bj.monte_carlo_off_policy_control(n_iter)
                sv = np.max(q, axis=0)
                return [
                    html.Div(
                            id='graph-description',
                            children=dcc.Markdown(dedent(description)),
                            className='six columns'
                    ),
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
                ]

            else:

                if exploration == 'Exploring Starts':
                    av, policy, n_visits = bj.monte_carlo_es(n_iter)
                elif exploration == 'Epsilon Greedy':
                    av, policy, n_visits = bj.monte_carlo_epsilon_greedy(n_iter)
                sv = np.max(av, axis=0)

                description = """
                It is straightforward to apply Monte Carlo ES to blackjack. 
                Because the episodes are all simulated games, it is easy to arrange for exploring starts that include all possibilities. 
                In this case one simply picks the dealer’s cards, the player’s sum, and whether or not the player has a usable ace, all at random with equal probability. 
                As the initial policy we use the policy evaluated in the previous blackjack example, that which sticks only on 20 or 21. 
                The initial action-value function can be zero for all state–action pairs. Figure 5.2 shows the optimal policy for blackjack found by Monte Carlo ES. 
                This policy is the same as the “basic” strategy of Thorp (1966) with the sole exception of the leftmost notch in the policy for a usable ace, which is not present in Thorp’s strategy. 
                We are uncertain of the reason for this discrepancy, but confident that what is shown here is indeed the optimal policy for the version of blackjack we have described.
                
                Without the assumption of exploring starts, however, we cannot simply improve the policy by making it greedy with respect to the current value function, 
                because that would prevent further exploration of nongreedy actions. 
                Fortunately, GPI does not require that the policy be taken all the way to a greedy policy, 
                only that it be moved toward a greedy policy. 
                In our on-policy method we will move it only to an epsilon-greedy policy.
                
                """
                return [
                    html.Div(
                            id='graph-description',
                            children=dcc.Markdown(dedent(description)),
                            className='six columns'
                    ),
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

            length = walk_length
            sims = simulations

            rw = RandomWalk(length)
            mc_values = ray.get(rw.mc_prediction.remote(rw, n_iter))
            td_values = ray.get(rw.td_prediction.remote(rw, n_iter))
            fig1 = rw.plot_state_values(mc_values)
            fig2 = rw.plot_state_values(td_values)

            # rmse comparison
            alphas = {
                'TD': [0.15, 0.1, 0.05],
                'MC': [0.01, 0.02, 0.03, 0.04],
            }

            true_values = np.arange(-length + 1, length + 1, 2) / (length + 1.)
            errors = {
                'MC': dict(),
                'TD': dict()
            }

            for n in ('TD', 'MC'):
                for alpha in tqdm(alphas[n]):
                    state_values = [getattr(rw, f'{n.lower()}_prediction').remote(rw, n_iter, alpha) for _ in
                                    range(sims)]
                    state_values = np.asarray(ray.get(state_values))
                    rmse = np.sum(np.sqrt(np.sum(np.power(state_values - true_values, 2), axis=2)), axis=0)
                    rmse /= sims * np.sqrt(length)
                    errors[n][alpha] = rmse

            fig3 = rw.plot_rmse(errors, list(range(n_iter)))

            # batch updates
            errors = {
                'MC': np.zeros((sims, n_iter)),
                'TD': np.zeros((sims, n_iter)),
            }

            for algo in errors:
                errors[algo] = np.asarray(ray.get([rw.batch_updates.remote(rw, algo) for _ in tqdm(range(sims))]))
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
            """
            Exercise 7.3 Why do you think a larger random walk task (19 states instead of 5) was
            used in the examples of this chapter? Would a smaller walk have shifted the advantage
            to a different value of n? How about the change in left-side outcome from 0 to −1 made
            in the larger walk? Do you think that made any difference in the best value of n?
            """

            length = walk_length
            episodes = n_iter
            sims = simulations

            rw = RandomWalk(length)

            alphas = [alpha / 10 for alpha in range(0, 11)]
            steps = [2 ** p for p in range(10)]

            true_values = np.arange(-length + 1, length + 1, 2) / (length + 1.)
            errors = dict(zip(steps, [alphas.copy() for _ in range(len(steps))]))

            for n_step in tqdm(steps):
                for i, alpha in enumerate(alphas):
                    state_values = [rw.n_step_td_prediction.remote(rw, n_step, episodes, alpha) for _ in range(sims)]
                    state_values = np.asarray(ray.get(state_values))
                    rmse = np.sum(np.sqrt(np.sum(np.power(state_values - true_values, 2), axis=2)))
                    rmse /= sims * n_iter * np.sqrt(length)
                    errors[n_step][i] = rmse

            fig3 = rw.plot_rmse(errors, alphas)

            return [
                html.Div(
                        dcc.Graph(
                                id='values',
                                figure=fig3,
                        ),
                        className=f'six columns',
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

        algos = ['sarsa', 'q_learning', 'expected_sarsa', 'double_q_learning', 'n_step_sarsa', 'n_step_sarsa_off_policy'] #  'n_step_q_sigma', 'n_step_tree_backup'
        rewards = dict(zip(algos, [np.zeros((simulations, n_iter)) for _ in range(len(algos))]))

        for algo in tqdm(algos):
            res = [getattr(cw, algo).remote(cw, n_episodes=n_iter, verbose=False) for _ in range(simulations)]
            for i, sim in enumerate(ray.get(res)):
                rewards[algo][i] = sim

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


