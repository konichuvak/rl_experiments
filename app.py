import dash

external_stylesheets = ["https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

####################################################################################################################
# SCHEMA
all_options = {
    'Bandits': {'Stationary Bandits', 'Non-Stationary Bandits', 'Action Preference'},
    'DP'     : {'GridWorld', 'CarRental', 'GamblersRuin', 'MarioVsBowser'},
    'RL'     : {'Blackjack', 'TicTacToe'},
    'FA'     : {}
}
