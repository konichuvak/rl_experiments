import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from textwrap import dedent
from rl_experiments.assets.style import *

from rl_experiments.envs.GridWorld import GridWorld
# from CarRental import CarRental
from rl_experiments.envs.GamblersRuin import GamblersRuin
from rl_experiments.envs.MarioVsBowser import MarioVsBowser
from rl_experiments.envs.Blackjack import Blackjack
from rl_experiments.envs.TicTacToe import TicTacToe
from rl_experiments.envs.RandomWalk import RandomWalk
from rl_experiments.envs.WindyGridworld import WindyGridworld
from rl_experiments.envs.CliffWalking import CliffWalking
from rl_experiments.envs.DynaMaze import DynaMaze

from rl_experiments.scripts.ExpectedVsSampleUpdates import ExpectedVsSampleUpdates

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
                                            dcc.Tab(
                                                label="Mario Vs Bowser",
                                                value="Mario Vs Bowser",
                                                style=tab_style,
                                                selected_style=selected_style
                                            ),
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
                                                dcc.Tab(
                                                    label='Expected Vs Sample Updates',
                                                    value='Expected Vs Sample Updates',
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
                    html.Div(
                        id='maze_type_div',
                        children=[
                            html.Label('Maze type', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='maze_type',
                                options=[{'label': s, 'value': s} for s in
                                         ['Dyna Maze', 'Blocking Maze', 'Shortcut Maze', 'Prioritized Sweeping']],
                                value='Dyna Maze'
                            )
                        ],
                        style={'display': 'none'},
                        className='two columns',
                    ),
                    html.Div(
                        id='planning_steps_div',
                        children=[
                            html.Label('Planning Steps', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='planning_steps',
                                options=[{'label': s, 'value': s} for s in [0, 10, 50]],
                                value=10
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='switch_time_div',
                        children=[
                            html.Label('Maze Switching Time-step', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='switch_time',
                                placeholder='1000',
                                type='number',
                                step=100,
                                value=1000,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='step_limit_div',
                        children=[
                            html.Label('Last Time-step', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='step_limit',
                                placeholder='3000',
                                type='number',
                                step=100,
                                value=3000,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='step_size_div',
                        children=[
                            html.Label('Step Size (alpha)', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='step_size',
                                placeholder='1',
                                type='number',
                                step=0.1,
                                value=1,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='time_weight_div',
                        children=[
                            html.Label('Time Weight (kappa)', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='time_weight',
                                placeholder='0.0001',
                                type='number',
                                step=0.001,
                                value=0.0001,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='simulation_div',
                        children=[
                            html.Label('Simulations:', style={'textAlign': 'center'}),
                            dcc.Input(
                                id='simulation',
                                placeholder='100',
                                type='number',
                                step=10,
                                value=100,
                                style={'width': '100%', 'textAlign': 'center'}
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
                    ),
                    html.Div(
                        id='distribution_div',
                        children=[
                            html.Label('Successor Distribution:', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='distribution',
                                options=[{'label': s, 'value': s} for s in ['exponential', 'uniform', 'normal']],
                                value='uniform'
                            )
                        ],
                        style={'display': 'none'},
                        className='one column',
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
                        },
                    ),
                ],
                className='row',
            ),
            html.Br(),
            dcc.Link('To Bandits', href='/bandits'),
            html.Br(),
            dcc.Link('To HOME', href='/'),
        ],
        className='page',
    )
])

display = ({'display': 'none'}, {'display': 'block'})
output_ids = sorted({
    'behavior_div', 'comparison_div', 'walk_length_div', 'feature_div', 'exploration_div', 'simulation_div', 'task_div',
    'policy_div', 'n_iter_div', 'prob_heads_div', 'goal_div', 'grid_size_div',
    'gamma_div', 'max_cars_div', 'max_move_cars_div', 'rental_rate_div', 'rental_credit_div', 'move_car_cost_div',
    'maze_type_div', 'switch_time_div', 'step_limit_div', 'step_size_div', 'time_weight_div', 'planning_steps_div',
    'distribution_div'
})
active_outputs = OrderedDict(zip(output_ids, (display[0] for _ in range(len(output_ids)))))
outputs_components = [Output(output_id, 'style') for output_id in output_ids]


@app.callback(
    outputs_components,
    [
        Input('section', 'value'),
        Input('task', 'value'),
        Input('off_policy', 'value'),
        Input('maze_type', 'value'),
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
        State('walk_length', 'value'),

        State('distribution', 'value'),

    ]
)
def show_hide(section, task, off_policy, maze_type,
              in_place,
              grid_size, gamma,
              prob_heads, goal,
              exploration, n_iter, behavior,
              feature,
              comparison, walk_length,
              distribution
              ):
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
        if maze_type == 'Dyna Maze':
            show = {'simulation', 'n_iter', 'maze_type'}
        elif maze_type == 'Blocking Maze':
            show = {'simulation', 'maze_type', 'switch_time', 'step_limit', 'step_size', 'time_weight',
                    'planning_steps'}
        elif maze_type == 'Shortcut Maze':
            show = {'simulation', 'maze_type', 'switch_time', 'step_limit', 'step_size', 'time_weight',
                    'planning_steps'}
        elif maze_type == 'Prioritized Sweeping':
            show = {'simulation', 'maze_type', 'step_size', 'planning_steps'}

    elif section in {'Expected Vs Sample Updates'}:
        show = {'simulation', 'distribution'}

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
    if section in ['Grid World', 'Car Rental', "Gambler's Ruin", 'Mario Vs Bowser']:
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
        Input('maze_type', 'value'),
        Input('section', 'value'),
    ],
)
def simulation(task, off_policy, comparison, maze_type, section):
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
        if maze_type == 'Dyna Maze':
            return 30
        elif maze_type == 'Blocking Maze':
            return 20
        elif maze_type == 'Shortcut Maze':
            return 5

    return 100


@app.callback(
    [
        Output('switch_time', 'value'),
        Output('step_limit', 'value'),
    ],
    [
        Input('maze_type', 'value')
    ],
)
def switching_maze(maze_type):
    if maze_type == 'Blocking Maze':
        return [1000, 3000]
    elif maze_type == 'Shortcut Maze':
        return [3000, 6000]
    else:
        return [1000, 3000]


@app.callback(
    Output('planning_steps', 'value'),
    [Input('maze_type', 'value')],
)
def planning_steps(maze_type):
    if maze_type == 'Blocking Maze':
        return 10
    elif maze_type == 'Shortcut Maze':
        return 50
    # elif maze_type == 'Dyna Maze':
    #     return [0, 10, 50]
    else:
        return 10


@app.callback(
    Output('time_weight', 'value'),
    [Input('maze_type', 'value')],
)
def time_weight(maze_type):
    return 0.0001 if maze_type == 'Blocking Maze' else 0.001


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
    try:
        class_ = getattr(importlib.import_module(f"envs.{classname}"), classname)
    except ImportError:
        class_ = getattr(importlib.import_module(f"scripts.{classname}"), classname)

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
        State('walk_length', 'value'),

        # Maze Type
        State('maze_type', 'value'),
        State('step_size', 'value'),
        State('step_limit', 'value'),
        State('time_weight', 'value'),
        State('switch_time', 'value'),
        State('planning_steps', 'value'),

        State('distribution', 'value'),
    ],
)
def RL(clicks, button_state, section,
       in_place,
       grid_size, gamma,
       prob_heads, goal,
       task, exploration, n_iter,
       off_policy, behavior, simulations,
       feature,
       comparison, walk_length,
       maze_type, step_size, step_limit, time_weight, switch_time, planning_steps,
       distribution,
       ):
    print(clicks, button_state, section,
          in_place,
          grid_size, gamma,
          prob_heads, goal,
          off_policy, behavior,
          maze_type, planning_steps, step_size, step_limit, time_weight, switch_time
          )

    if not clicks:
        raise PreventUpdate
    if button_state == 'Stop':
        raise PreventUpdate

    if section == 'Grid World':

        gw = GridWorld(width=grid_size, height=grid_size, grid_dim=grid_size, gamma=gamma, default_reward=-1)
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

    elif section == "Mario Vs Bowser":
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
                sv, n_samples = bj.mc_prediction(n_iter)
                return [
                    html.Div(
                        dcc.Graph(
                            id='value_estimate',
                            figure=bj.plot_value_function(sv, n_iter)
                        ),
                        className=f'six columns',
                    ),
                    html.Div(
                        dcc.Graph(
                            id='visits',
                            figure=bj.plot_n_visits(n_samples)
                        ),
                        className=f'six columns',
                    ),
                    html.Div(
                        id='graph-description',
                        children=dcc.Markdown(dedent(description)),
                        className='one row'
                    ),
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

        algos = ['sarsa', 'q_learning', 'expected_sarsa', 'double_q_learning', 'n_step_sarsa',
                 'n_step_sarsa_off_policy']  # 'n_step_q_sigma', 'n_step_tree_backup'
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

    elif section == 'Dyna Maze':

        if maze_type == 'Dyna Maze':
            """
            ### Dyna Maze

            The graph below shows average learning curves from an experiment in which Dyna-Q agents were applied to the maze task.
            The initial action values were zero, the step-size parameter was alpha = 0.1, and the exploration parameter was epsilon = 0.1.
            When selecting greedily among actions, ties were broken randomly.
            The agents varied in the number of planning steps, n, they performed per real step.
            For each n, the curves show the number of steps taken by the agent to reach the goal in each episode,
            averaged over repetitions of the experiment.
            In each repetition, the initial seed for the random number generator was held constant across algorithms.
            Because of this, the first episode was exactly the same (about 1700 steps) for all values of n, and its data are not shown in the figure.
            After the first episode, performance improved for all values of n, but much more rapidly for larger values.
            Recall that the n = 0 agent is a nonplanning agent, using only direct reinforcement learning (one-step tabular Q-learning).

            This was by far the slowest agent on this problem, despite the fact that the parameter values (alpha and epsilon) were optimized for it.
            The nonplanning agent took about 25 episodes to reach (epsilon-)optimal performance,
            whereas the n = 5 agent took about five episodes, and the n = 50 agent took only three episodes.
            """
            dm = DynaMaze(width=9, height=6, default_reward=0, other_rewards={(8, 0): 1},
                          start_state=(0, 2), goal=(8, 0))
            dm.blocks = [(2, 1), (2, 2), (2, 3), (5, 4), (7, 1), (7, 2), (7, 3)]

            episode_length = dict()
            for i, n in tqdm(enumerate((0, 5, 50))):
                episodes = ray.get([
                    dm.q_planning.remote(dm, planning_steps=planning_steps, n_episodes=n_iter, seed=n,
                                         ) for _ in range(simulations)])
                episode_length[n] = np.mean(np.asarray(episodes), axis=0)

            fig = dm.plot_learning_curves(episode_length)
            return [
                html.Div(
                    dcc.Graph(
                        id='steps_per_episode',
                        figure=fig,
                        className='six columns'
                    ),
                ),
            ]

        elif maze_type == 'Blocking Maze':

            new_blocks = [(i, 3) for i in range(1, 9)]

            cumulative_rewards = {'dyna_q': list(), 'dyna_q_plus': list(), 'dyna_q_plus_plus': list()}

            for algo in tqdm(cumulative_rewards.keys()):
                dyna_q_plus = algo == 'dyna_q_plus'
                dyna_q_plus_plus = algo == 'dyna_q_plus_plus'

                dm = DynaMaze(width=9, height=6, default_reward=0, other_rewards={(8, 0): 1},
                              start_state=(3, 5), goal=(8, 0), blocks=[(i, 3) for i in range(8)])

                episode_length = ray.get([dm.q_planning.remote(
                    dm, planning_steps=planning_steps, n_episodes=100000000, step_limit=step_limit,
                    switch_time=switch_time, new_blocks=new_blocks, alpha=step_size, seed=seed, kappa=time_weight,
                    dyna_q_plus=dyna_q_plus, dyna_q_plus_plus=dyna_q_plus_plus
                ) for seed in range(simulations)])

                per_step_rewards = list()
                for i in range(simulations):
                    rewards = [0]
                    for j in range(step_limit):
                        if j == episode_length[i][0]:
                            rewards.append(rewards[-1] + 1)
                            episode_length[i][1] += episode_length[i][0]
                            episode_length[i].pop(0)
                        else:
                            rewards.append(rewards[-1])
                    per_step_rewards.append(rewards)

                cumulative_rewards[algo] = np.mean(np.asarray(per_step_rewards), axis=0)

            fig = dm.plot_rewards(cumulative_rewards)

            description = """
            ### Example 8.2: Blocking Maze

            A maze example illustrating this relatively minor
            kind of modeling error and recovery from it is shown in Figure 8.4. Initially, there is a
            short path from start to goal, to the right of the barrier, as shown in the upper left of the
            figure. After 1000 time steps, the short path is “blocked,” and a longer path is opened up
            along the left-hand side of the barrier, as shown in upper right of the figure. The graph
            shows average cumulative reward for a Dyna-Q agent and an enhanced Dyna-Q+ agent
            to be described shortly. The first part of the graph shows that both Dyna agents found
            the short path within 1000 steps. When the environment changed, the graphs become
            flat, indicating a period during which the agents obtained no reward because they were
            wandering around behind the barrier. After a while, however, they were able to find the
            new opening and the new optimal behavior.
            Greater difficulties arise when the environment changes to become better than it was
            before, and yet the formerly correct policy does not reveal the improvement. In these
            cases the modeling error may not be detected for a long time, if ever.
            """
            return [
                html.Div(
                    dcc.Graph(
                        id='cumulative-rewards',
                        figure=fig,
                        className='six columns'
                    ),
                ),
            ]

        elif maze_type == 'Shortcut Maze':
            """
            ### Exercise 8.4 Shortcut Maze

            The exploration bonus described above actually changes the estimated values of states and actions.
            Is this necessary? Suppose the bonus k * sqrt(tau) was used not in updates, but solely in action selection.
            That is, suppose the action selected was always that for which Q(St, a) + k * sqrt(tau(St, a)) was maximal.
            Carry out a gridworld experiment that tests and illustrates the strengths and weaknesses of this
            alternate approach.
            """

            new_blocks = [(i, 3) for i in range(1, 8)]

            cumulative_rewards = {'dyna_q': list(), 'dyna_q_plus': list(), 'dyna_q_plus_plus': list()}

            for algo in tqdm(cumulative_rewards.keys()):
                dyna_q_plus = algo == 'dyna_q_plus'
                dyna_q_plus_plus = algo == 'dyna_q_plus_plus'

                dm = DynaMaze(width=9, height=6, default_reward=0, other_rewards={(8, 0): 1},
                              start_state=(3, 5), goal=(8, 0), blocks=[(i, 3) for i in range(1, 9)])

                episode_length = ray.get([dm.q_planning.remote(
                    dm, planning_steps=planning_steps, n_episodes=100000000, step_limit=step_limit,
                    switch_time=switch_time, new_blocks=new_blocks, alpha=step_size, seed=seed, kappa=time_weight,
                    dyna_q_plus=dyna_q_plus, dyna_q_plus_plus=dyna_q_plus_plus,
                ) for seed in range(simulations)])

                per_step_rewards = list()
                for i in range(simulations):
                    rewards = [0]
                    for j in range(step_limit):
                        if j == episode_length[i][0]:
                            rewards.append(rewards[-1] + 1)
                            episode_length[i][1] += episode_length[i][0]
                            episode_length[i].pop(0)
                        else:
                            rewards.append(rewards[-1])
                    per_step_rewards.append(rewards)

                cumulative_rewards[algo] = np.mean(np.asarray(per_step_rewards), axis=0)

            fig = dm.plot_rewards(cumulative_rewards)

            description = """
            ### Example 8.3: Shortcut Maze

            ---

            The problem caused by this kind of
            environmental change is illustrated
            by the maze example shown in Figure
            8.5. Initially, the optimal path is
            to go around the left side of the barrier
            (upper left). After 3000 steps,
            however, a shorter path is opened up
            along the right side, without disturbing
            the longer path (upper right).
            The graph shows that the regular
            Dyna-Q agent never switched to the
            shortcut. In fact, it never realized
            that it existed. Its model said that
            there was no shortcut, so the more it
            planned, the less likely it was to step
            to the right and discover it. Even
            with an "-greedy policy, it is very
            unlikely that an agent will take so
            many exploratory actions as to discover
            the shortcut.

            ---

            """
            return [
                html.Div(
                    dcc.Graph(
                        id='cumulative-rewards',
                        figure=fig,
                        className='six columns'
                    ),
                ),
            ]

        elif maze_type == 'Prioritized Sweeping':

            from rl_experiments.envs.maze_gen import recursive_backtracker

            updates_until_optimal = {'dyna_q': dict(), 'prioritized_sweeping': dict()}

            for algo in tqdm(updates_until_optimal.keys()):

                if algo == 'dyna_q':
                    continue

                for dim in range(100, 6001, 50):

                    # generate a maze
                    grid = recursive_backtracker(width=dim, height=dim, seed=dim).astype(int)
                    blocks = [tuple(block) for block in np.argwhere(grid == 1)]
                    free_space = np.argwhere(grid == 0)

                    # randomly determine start and goal state for now (use heuristic and distance measure)
                    start_state = goal_state = tuple(free_space[np.random.randint(0, free_space.shape[0] - 1)])
                    while goal_state == start_state:
                        goal_state = tuple(free_space[np.random.randint(0, free_space.shape[0] - 1)])

                    other_rewards = {block: -10 for block in blocks}
                    other_rewards[goal_state] = dim ** 2
                    dm = DynaMaze(width=dim, height=dim, default_reward=-1, other_rewards=other_rewards,
                                  start_state=start_state, goal=goal_state, blocks=blocks)
                    dm.grid[start_state] = -5
                    dm.grid[goal_state] = 5

                    # solve the maze
                    if algo == 'prioritized_sweeping':
                        theta = 0.0001

                        # num_steps = ray.get([dm.prioritized_sweeping.remote(
                        #     dm, planning_steps=planning_steps, n_episodes=10, alpha=step_size, theta=theta,
                        #     seed=seed, verbose=True
                        # ) for seed in range(simulations)])
                        num_steps = [dm.prioritized_sweeping(
                            planning_steps=planning_steps, n_episodes=10, alpha=step_size, gamma=0.95, theta=theta,
                            seed=seed, verbose=True,
                        ) for seed in range(simulations)]

                    elif algo == 'dyna_q':
                        planning_steps = 5
                        step_size = 0.5

                        num_steps = ray.get([dm.q_planning.remote(
                            dm, planning_steps=planning_steps, n_episodes=100000000, alpha=step_size, seed=seed,
                        ) for seed in range(simulations)])

                    updates_until_optimal[algo][dim] = np.mean(np.asarray(num_steps))
                    print(updates_until_optimal)

            print(updates_until_optimal)
            fig = dm.plot_convergence_speed(updates_until_optimal)
            return [
                html.Div(
                    dcc.Graph(
                        id='prioritized-sweeping',
                        figure=fig,
                        className='six columns'
                    ),
                ),
            ]

    elif section == 'Expected Vs Sample Updates':

            es = ExpectedVsSampleUpdates()

            simulations = 100
            error = dict()
            for b in tqdm((2, 10, 100, 1000, 10000)):
                rmse = np.asarray(ray.get([es.q_updates.remote(b, distribution) for _ in range(simulations)]))
                error[b] = np.mean(rmse, axis=0)

            return [
                html.Div(
                    dcc.Graph(
                        id='expected-vs-sample-updates',
                        figure=es.plot_rmse(error),
                        className='six columns'
                    ),
                ),
            ]

