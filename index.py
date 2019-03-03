import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import bandits, rl

# TODO: add layout for tabular vs FA


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.Br(),
    dcc.Link('bandits', href='/bandits'),
    html.Br(),
    dcc.Link('rl', href='/rl'),
])


# Update the index
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/rl':
        return rl.layout
    elif pathname == '/bandits':
        return bandits.layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=1337, debug=True)