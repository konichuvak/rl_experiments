import dash
import redis

external_stylesheets = ["https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

db = redis.StrictRedis(port=6379, password='IueksS7Ubh8G3DCwVzrTd8rAVOwq3M5x')
