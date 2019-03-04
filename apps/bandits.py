import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from envs.Bandits import Bandits
from style import *
import numpy as np
from tqdm import tqdm
import ray

ray.init(ignore_reinit_error=True)

# Options for dropdowns and selectors
epsilon = [0.5, 0.1, 0.01, 0]
num_bandits = list(range(10, 101, 10))
steps = list(range(1000, 5001, 1000))
simulations = [1000] + list(range(2000, 10001, 2000))
weightings = ['Exponential', 'Uniform', 'Both']
alphas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

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
                                value='Action Preference',
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
                                placeholder='Step Size',
                                type='number',
                                value=0.025,
                                style={'width': '100%', 'textAlign': 'center'}
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
            dcc.Link('To RL', href='/rl'),
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


#####################################################
# STATIC CONTROLS SHOW/HIDE

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


###############################################################

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
    ],
)
def BANDITS(clicks, button_state, section,
            simulations, steps, bandits, epsilons, weighting, alpha,
            ):
    print(clicks, button_state, section,
          simulations, steps, bandits, epsilons, weighting, alpha,
          )

    if not clicks:
        return
    if button_state == 'Stop':
        return

    M = simulations
    k = bandits
    nplays = steps

    if section == 'Stationary Bandits':

        bandits = Bandits()
        if isinstance(epsilons, float):
            epsilons = [epsilons]

        avg_reward = {e: list() for e in epsilons}
        optimality_ratio = {e: list() for e in epsilons}

        for e in tqdm(epsilons):
            res = np.array(ray.get([bandits.kArmedTestbed.remote(k, nplays, e) for _ in range(M)]))
            expected_rewards, observed_rewards, actions = map(np.stack, [res[:, i] for i in range(3)])

            avg_reward[e] = np.average(observed_rewards, axis=0)  # compute average rewards
            opt = np.argmax(expected_rewards, axis=1).reshape(M, 1) + np.ones((M, 1))  # take argmax over all states
            act = np.ma.masked_values(actions, opt).mask  # filter the optimal actions
            optimality_ratio[e] = np.average(act, axis=0)

        fig = bandits.generate_plot(steps, rewards=avg_reward, optimality=optimality_ratio)
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
            fig = bandits.plot_non_stationary(j + 1, nplays,
                                              expected_rewards=[expected_rewards_uni[:, j], expected_rewards_exp[:, j]],
                                              qs=[qmat_uni[:, j], qmat_exp[:, j]])
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
            fig = bandits.plot_gradient_bandit(j + 1, nplays, h_mat=h_mat[:, j])
            children.append(
                dcc.Graph(
                    id=f'results_{j}',
                    figure=fig
                ),
            )
        return html.Div(
            children=children
        )
