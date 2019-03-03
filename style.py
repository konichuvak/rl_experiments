from app import app

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
# EXTERNAL CSS / JS

# app.css.config.serve_locally = True
# app.scripts.config.serve_locally = True
# app.config['suppress_callback_exceptions'] = True

# Append an externally hosted JS code

# Append an externally hosted CSS stylesheet
app.css.append_css({
    'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})
