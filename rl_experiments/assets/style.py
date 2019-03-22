from app import app

card_style = {
    "box-shadow": "0 4px 5px 0 rgba(0,0,0,0.14), 0 1px 10px 0 rgba(0,0,0,0.12), 0 2px 4px -1px rgba(0,0,0,0.3)"
}
BLUES = ["rgb(210, 218, 255)", "rgb(86, 117, 255)", "rgb(8, 31, 139)", "rgb(105, 125, 215)", "rgb(84, 107, 208)",
         "rgb(210, 210, 210)", "rgb(102, 103, 107)", "rgb(19, 23, 37)", ]

gradients = ['rgb(115, 132, 212)', 'rgb(169, 120, 219)', 'rgb(211, 107, 218)', 'rgb(237, 84, 199)',
    'rgb(244, 70, 157)', 'rgb(240, 90, 127)', 'rgb(238, 117, 124)', 'rgb(230, 193, 119)']

tab_style = {
    'borderLeft' : 'thin lightgrey solid',
    'borderRight': 'thin lightgrey solid',
    'borderTop' : '2px white solid',
    'boxShadow': 'inset 0px -1px 0px 0px lightgrey',
    'fontSize'  : '0.7vw',
    'color': 'black',

}
selected_style = {
    'borderLeft' : 'thin lightgrey solid',
    'borderRight': 'thin lightgrey solid',
    'background-image': f"linear-gradient(to top left, {','.join(gradients[:4])})",
    'color': 'white',
    'fontSize'   : '0.7vw',
}
container_style = {
    # 'width'        : '100%',
    'verticalAlign': 'middle',
    # 'display'      : 'inlineBlock',
    # 'boxShadow': 'inset 0px -1px 0px 0px lightgrey',
    'alignItems': 'center',
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
