import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import os
import numpy as np
import pandas as pd
import itertools
import datetime
import time
import _pickle
from tqdm import tqdm
from Utilities import threaded, cache
from Bandits import Bandits
from GridWorld import GridWorld
# from CarRental import CarRental
from GamblersRuin import GamblersRuin
# from MarioVsBowser import MarioVsBowser
from Blackjack import Blackjack
from TicTacToe import TicTacToe
from RandomWalk import RandomWalk
from WindyGridworld import WindyGridworld
from CliffWalking import CliffWalking

external_stylesheets = ["https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

####################################################################################################################
# SCHEMA
all_options = {
    'Bandits': {'Stationary Bandits', 'Non-Stationary Bandits', 'Action Preference'},
    'DP'     : {'GridWorld', 'CarRental', 'GamblersRuin', 'MarioVsBowser'},
    'RL'     : {'Blackjack', 'TicTacToe'},
    'FA'     : {}
}

# Options for dropdowns and selectors
epsilon = [0.5, 0.1, 0.01, 0]
num_bandits = list(range(10, 101, 10))
steps = list(range(1000, 5001, 1000))
simulations = [1000] + list(range(2000, 10001, 2000))
weightings = ['Exponential', 'Uniform', 'Both']
alphas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

####################################################################################################################
# DESIGN

card_style = {
    "box-shadow": "0 4px 5px 0 rgba(0,0,0,0.14), 0 1px 10px 0 rgba(0,0,0,0.12), 0 2px 4px -1px rgba(0,0,0,0.3)"
}
BLUES = ["rgb(210, 218, 255)", "rgb(86, 117, 255)", "rgb(8, 31, 139)", "rgb(105, 125, 215)", "rgb(84, 107, 208)",
         "rgb(210, 210, 210)", "rgb(102, 103, 107)", "rgb(19, 23, 37)", ]
tab_style = {
    'width'     : '50%',
    # 'border': 'none',
    # 'borderLeft' : 'thin lightgrey solid',
    # 'borderRight': 'thin lightgrey solid',
    'borderTop' : '2px white solid',
    # 'boxShadow': 'inset 0px -1px 0px 0px lightgrey',
    # 'background': "rgb(210, 218, 255)",
    'fontFamily': 'Avenir',
    'fontSize'  : '1.5vw',
    'textAlign' : 'center',
    # 'fontColor': 'white'

}
selected_style = {
    # 'boxShadow': 'none',
    # 'boxShadow'  : 'inset 0px -1px 0px 0px lightgrey',
    'borderLeft' : 'thin lightgrey solid',
    'borderRight': 'thin lightgrey solid',
    'borderTop'  : '3px #104b7d solid',
    'background' : 'white',
    'fontFamily' : 'Avenir',
    'fontSize'   : '1.5vw',
    'textAlign'  : 'center',
}
container_style = {
    'width'        : '100%',
    'verticalAlign': 'middle',
    'display'      : 'inlineBlock',
    # 'boxShadow': 'inset 0px -1px 0px 0px lightgrey',
    # 'background': 'rgb(0, 0, 0)',
    # 'alignItems': 'center',
    # 'padding'   : '40px 100px',
    'padding'      : '20px ',

}

####################################################################################################################
app.layout = html.Div(
        html.Div([

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
                                                        label='Stationary Bandits',
                                                        value='Stationary Bandits',
                                                        style=tab_style,
                                                        selected_style=selected_style
                                                ),
                                                dcc.Tab(
                                                        label='Non-Stationary Bandits',
                                                        value='Non-Stationary Bandits',
                                                        style=tab_style,
                                                        selected_style=selected_style
                                                ),
                                                dcc.Tab(
                                                        label='Action Preference',
                                                        value='Action Preference',
                                                        style=tab_style,
                                                        selected_style=selected_style
                                                ),
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

            html.Div(
                    id='static_components',
                    children=[
                        ##########################
                        # BANDITS
                        html.Div(
                                id='bandits_div',
                                children=[
                                    html.Label('Bandits:', style={'textAlign': 'center'}),
                                    dcc.Dropdown(
                                            id='bandits',
                                            options=[{'label': k, 'value': k} for k in num_bandits],
                                            value=num_bandits[0]
                                    )
                                ],
                                style={'display': 'block'},
                                className='one column',
                        ),
                        html.Div(
                                id='epsilon_div',
                                children=[
                                    html.Label('Epsilon:', style={'textAlign': 'center'}),
                                    dcc.Dropdown(
                                            id='epsilon',
                                            options=[{'label': e, 'value': e} for e in epsilon],
                                            multi=True,
                                            value=epsilon[1]
                                    )
                                ],
                                style={'display': 'block'},
                                className='three columns',
                        ),
                        html.Div(
                                id='steps_div',
                                children=[
                                    html.Label('Steps:', style={'textAlign': 'center'}),
                                    dcc.Dropdown(
                                            id='steps',
                                            options=[{'label': s, 'value': s} for s in steps],
                                            value=steps[0]
                                    )
                                ],
                                style={'display': 'block'},
                                className='one column',
                        ),
                        html.Div(
                                id='simulations_div',
                                children=[
                                    html.Label('Simulations:', style={'textAlign': 'center'}),
                                    dcc.Dropdown(
                                            id='simulation',
                                            options=[{'label': s, 'value': s} for s in simulations],
                                            value=simulations[1]
                                    )
                                ],
                                style={'display': 'block'},
                                className='one column',
                        ),
                        html.Div(
                                id='weighting_div',
                                children=[
                                    html.Label('Weighting Method:', style={'textAlign': 'center'}),
                                    dcc.Dropdown(
                                            id='weighting',
                                            options=[{'label': s, 'value': s} for s in weightings],
                                            value=weightings[0]
                                    )
                                ],
                                style={'display': 'none'},
                                className='two columns',
                        ),
                        html.Div(
                                id='alpha_div',
                                children=[
                                    html.Label('Alpha:', style={'textAlign': 'center'}),
                                    # dcc.Dropdown(
                                    #     id='alpha',
                                    #     options=[{'label': s, 'value': s} for s in alphas],
                                    #     value=alphas[0]
                                    # )
                                    dcc.Input(
                                            id='alpha',
                                            placeholder='Weighting constant',
                                            type='number',
                                            value=0.025,
                                            style={'width': '100%', 'textAlign': 'center'}
                                    )
                                ],
                                style={'display': 'none'},
                                className='two columns',
                        ),

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
                    ],
                    className='row'
            ),

            html.Div(
                    id='dynamic_components',
                    children=[
                        html.Div(
                                id='display-values',
                                children=[
                                    dcc.Loading(id='display-results', type='cube')
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
        ],
                style={
                    'width'       : '100%',
                    'fontFamily'  : 'Avenir',
                    'margin-left' : 'auto',
                    'margin-right': 'auto',
                    'boxShadow'   : '0px 0px 5px 5px rgba(204,204,204,0.4)'
                }
        )
        # type='cube',ff
        # fullscreen=True
)

####################################################################################################################
# EXTERNAL CSS / JS

# app.css.config.serve_locally = True
# app.scripts.config.serve_locally = True
# app.config['suppress_callback_exceptions'] = True

# Append an externally hosted JS code

# Append an externally hosted CSS stylesheet
app.css.append_css({
    'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})


####################################################################################################################
# CALLBACKS

#####################################################
# STATIC CONTROLS SHOW/HIDE

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
        Output('n_iter_div', 'style'),
        [Input('section', 'value')],
)
def n_iter_div(section):
    if section in ["Blackjack", "Tic Tac Toe", 'Random Walk']:
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
        Output('simulations_div', 'style'),
        [Input('section', 'value')],
)
def simulations_div(section):
    if section in ['Stationary Bandits']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
        Output('epsilon_div', 'style'),
        [Input('section', 'value')],
)
def eplsions_div(section):
    if section in ['Stationary Bandits', 'Non-Stationary Bandits']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
        Output('bandits_div', 'style'),
        [Input('section', 'value')],
)
def bandits_div(section):
    if section in ['Stationary Bandits', 'Non-Stationary Bandits', 'Action Preference']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
        Output('steps_div', 'style'),
        [Input('section', 'value')],
)
def steps_div(section):
    if section in ['Stationary Bandits', 'Non-Stationary Bandits', 'Action Preference']:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
        Output('weighting_div', 'style'),
        [Input('section', 'value')],
)
def weighting_div(section):
    if section == 'Non-Stationary Bandits':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
        Output('alpha_div', 'style'),
        [
            Input('weighting_div', 'style'),
            Input('weighting', 'value')
        ],
        [
            State('section', 'value')
        ]
)
def alpha_div(style, weighting, section):
    if section in ['Non-Stationary Bandits']:
        if style == {'display': 'block'}:
            if weighting == 'Uniform':
                return {'display': 'none'}
            elif weighting in ['Exponential', 'Both']:
                return {'display': 'block'}
        elif style == {'display': 'none'}:
            return {'display': 'none'}
    elif section in ['Action Preference']:
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
        Output('epsilon', 'value'),
        [Input('section', 'value')],
)
def epsilon_value(section):
    if section in ['Stationary Bandits']:
        return epsilon[1:]
    else:
        return epsilon[1]


@app.callback(
        Output('epsilon_div', 'className'),
        [Input('section', 'value')],
)
def epsilon_field_width(section):
    if section in ['Stationary Bandits']:
        return 'three columns'
    else:
        return 'one column'


@app.callback(
        Output('epsilon', 'multi'),
        [Input('section', 'value')],
)
def bandits_value(section):
    if section == 'Stationary Bandits':
        return True
    else:
        return False


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
        Output('bandits', 'options'),
        [Input('section', 'value')],
)
def bandits_options(section):
    if section == 'Stationary Bandits':
        b = list(range(10, 101, 10))
        return [{'label': k, 'value': k} for k in b]
    elif section in ['Non-Stationary Bandits']:
        b = list(range(1, 11))
        return [{'label': k, 'value': k} for k in b]
    elif section in ['Action Preference']:
        b = [3]
        return [{'label': k, 'value': k} for k in b]
    else:
        return []


@app.callback(
        Output('bandits', 'value'),
        [Input('bandits', 'options')],
        [State('section', 'value')]
)
def bandits_value(options, section):
    if section == 'Stationary Bandits':
        return options[0]['value']
    elif section == 'Non-Stationary Bandits':
        return options[1]['value']
    elif section == 'Action Preference':
        return 3


@app.callback(
        Output('alpha', 'value'),
        [
            Input('section', 'value')
        ]
)
def weighting_value(section):
    if section in ['Non-Stationary Bandits']:
        return 0.025
    elif section in ['Action Preference']:
        return 0.02
    else:
        return 0.025


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
        Output('display-results', 'children'),
        [
            Input('button', 'n_clicks')
        ],
        [
            State('button', 'children'),
            State('section', 'value'),

            # Bandits
            State('simulation', 'value'),
            State('steps', 'value'),
            State('bandits', 'value'),
            State('epsilon', 'value'),
            State('weighting', 'value'),
            State('alpha', 'value'),

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

        ],
)
def gen_argstring(clicks, button_state, section,
                  simulations, steps, bandits, epsilons, weighting, alpha,
                  in_place,
                  grid_size, gamma,
                  prob_heads, goal,
                  task, exploration, n_iter,
                  off_policy, behavior,
                  feature
                  ):
    print(clicks, button_state, section,
          simulations, steps, bandits, epsilons, weighting, alpha,
          in_place,
          grid_size, gamma,
          prob_heads, goal,
          off_policy, behavior
          )

    if not clicks:
        return
    if button_state == 'Stop':
        return

    M = simulations
    k = bandits
    nplays = steps

    # BANDITS
    if section == 'Stationary Bandits':
        bandits = Bandits()
        if isinstance(epsilons, float):
            epsilons = [epsilons]

        avg_reward = {e: list() for e in epsilons}
        optimality_ratio = {e: list() for e in epsilons}

        for e in tqdm(epsilons):
            expected_rewards, observed_rewards, actions = np.zeros((M, k)), np.zeros((M, nplays)), np.zeros((M, nplays))
            for i in tqdm(range(M)):
                expected_rewards[i], observed_rewards[i], actions[i] = bandits.kArmedTestbed(k, nplays, e)

            avg_reward[e] = np.average(observed_rewards, axis=0)  # compute average rewards
            opt = np.argmax(expected_rewards, axis=1).reshape(M, 1) + np.ones((M, 1))  # take argmax over all states
            act = np.ma.masked_values(actions, opt).mask  # filter the optimal actions
            optimality_ratio[e] = np.average(act, axis=0)

        fig = bandits.generate_plot(steps, avg_reward, optimality_ratio)
        return html.Div(
                dcc.Graph(
                        id='results',
                        figure=fig
                ),
        )

    elif section == 'Non-Stationary Bandits':
        bandits = Bandits()
        expected_rewards_uni, observed_rewards_uni, actions_uni, qmat_uni = \
            np.zeros((1, k)), None, None, np.zeros((1, k))
        expected_rewards_exp, observed_rewards_exp, actions_exp, qmat_exp = \
            np.zeros((1, k)), None, None, np.zeros((1, k))

        if weighting == 'Both':
            expected_rewards_uni, observed_rewards_uni, actions_uni, qmat_uni = \
                bandits.NonStationaryTestbed(k, nplays, epsilons, 1)
            expected_rewards_exp, observed_rewards_exp, actions_exp, qmat_exp = \
                bandits.NonStationaryTestbed(k, nplays, epsilons, 1, alpha)
        elif weighting == 'Exponential':
            expected_rewards_exp, observed_rewards_exp, actions_exp, qmat_exp = \
                bandits.NonStationaryTestbed(k, nplays, epsilons, 2, alpha)
        elif weighting == 'Uniform':
            expected_rewards_uni, observed_rewards_uni, actions_uni, qmat_uni = \
                bandits.NonStationaryTestbed(k, nplays, epsilons, 3)

        children = list()
        for j in range(k):
            fig = bandits.plot_non_stationary(j + 1, nplays, [expected_rewards_uni[:, j], expected_rewards_exp[:, j]],
                                              [qmat_uni[:, j], qmat_exp[:, j]])
            children.append(
                    dcc.Graph(
                            id=f'results_{j}',
                            figure=fig
                    ),
            )

        return html.Div(
                children=children
        )

    elif section == 'Action Preference':
        bandits = Bandits()
        h_mat = bandits.action_preference(int(nplays), k=k)
        children = list()
        for j in range(k):
            fig = bandits.plot_gradient_bandit(j + 1, nplays, h_mat[:, j])
            children.append(
                    dcc.Graph(
                            id=f'results_{j}',
                            figure=fig
                    ),
            )
        return html.Div(
                children=children
        )

    elif section == 'Grid World':

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
        rw = RandomWalk()
        n_iter = 100  # N_EPISODES

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
        for alpha in list(alphas.values())[0]:
            values['MC'][alpha] = np.zeros((100, 100, 5))
            values['TD'][alpha] = np.zeros((100, 100, 5))
            for i in range(100):
                mc_values = rw.mc_prediction(n_episodes=n_iter, alpha=alpha)
                mc_values = [np.array(list(episode.values())[1:-1]) for episode in mc_values]
                values['MC'][alpha][i] = np.array(mc_values)

                td_values = rw.td_prediction(n_episodes=n_iter, alpha=alpha)
                td_values = [np.array(list(episode.values())[1:-1]) for episode in td_values]
                values['TD'][alpha][i] = np.array(td_values)
            values['MC'][alpha] = np.mean(values['MC'][alpha], axis=0)
            values['TD'][alpha] = np.mean(values['TD'][alpha], axis=0)

        fig3 = rw.plot_rmse(values)

        # batch updates
        errors = {
            'MC': np.zeros((100, n_iter)),
            'TD': np.zeros((100, n_iter)),
        }

        for i in tqdm(range(100)):
            for algo in errors:
                errors[algo][i] = rw.batch_updates(algo=algo)

        for algo in errors:
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

    elif section == "Windy Gridworld":

        # king_moves = True if feature == 'King Moves' else False
        # stochastic_wind = True if feature == 'Stochastic Wind' else False
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

        algos = ['sarsa', 'q_learning']
        rewards = dict(zip(algos, [np.zeros((simulations, n_episodes)), np.zeros((simulations, n_episodes))]))

        for i in tqdm(range(simulations)):
            for algo in algos:
                rewards[algo][i] = cw.control(algo=algo, n_episodes=n_episodes, verbose=False)

        for algo, sims in rewards.items():
            rewards[algo] = np.mean(sims, axis=0)
        fig = cw.plot_rewards(rewards)

        return [
            html.Div(
                    dcc.Graph(
                            id='rewards',
                            figure=fig,
                    ),
                    className=f'four columns',

            ),
        ]


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=1337, debug=True)
