import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from rl_experiments.app import db
import plotly.graph_objs as go
from rl_experiments.assets.style import *
import numpy as np
import pickle

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
                                    html.H2("Training Board"),
                                ],
                                className='two columns title',
                            ),
                        ],
                        className='row'
                    ),
                    # html.Div(
                    #     id='description_div',
                    #     children=[
                    #         html.Div(
                    #             id='description',
                    #             children=[dcc.Markdown([dedent(RandomWalk.description())])],
                    #             style={'margin-right': '5%', 'margin-left': '5%'}
                    #         ),
                    #     ],
                    #     className='row description'
                    # ),
                ],
                className='header'
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id='q-values',
                    ),
                ],
                className='four columns',
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id='grid',
                    ),
                    dcc.Interval(id='grid_interval', interval=1000 * 3, n_intervals=0),
                ],
                className='four columns',
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id='priority',
                    ),
                ],
                className='four columns',
            ),
            html.Br(),
            dcc.Link('To RL', href='/rl'),
            html.Br(),
            dcc.Link('To Bandits', href='/bandits'),
            html.Br(),
            dcc.Link('To HOME', href='/'),
        ],
        className='page',
    )
])


@app.callback(
    [
        Output('grid', 'figure'),
        Output('q-values', 'figure'),
        Output('priority', 'figure'),
    ],
    [
        Input('grid_interval', 'n_intervals')
    ]
)
def train(n_intervals):
    q_values = pickle.loads(db.hget('Prioritized Sweeping', 'Q-Values'))
    graph1 = [
        go.Heatmap(
            z=np.mean(q_values, axis=0),
            showscale=False,
        )
    ]
    layout1 = dict(
        height=700,
        width=700,
        title='Q Values'
    )
    fig1 = {'data': graph1, 'layout': layout1}

    grid = pickle.loads(db.hget('Prioritized Sweeping', 'Grid'))
    graph2 = [
        go.Heatmap(
            z=grid,
            showscale=False,
            colorscale='Viridis',
        )
    ]
    layout2 = dict(
        height=700,
        width=700,
        title='Grid'
    )
    fig2 = {'data': graph2, 'layout': layout2}

    p_queue = pickle.loads(db.hget('Prioritized Sweeping', 'Priority'))
    priority = np.zeros((q_values.shape[0], q_values.shape[1], q_values.shape[1]))
    for state, action in p_queue:
        priority[action, state[0], state[1]] = p_queue[state, action] * (-1)

    graph3 = [
        go.Heatmap(
            z=np.mean(priority, axis=0),
            showscale=False,
            colorscale='Viridis',
        )
    ]
    layout3 = dict(
        height=700,
        width=700,
        title='Priority'
    )
    fig3 = {'data': graph3, 'layout': layout3}

    return [fig1, fig2, fig3]
