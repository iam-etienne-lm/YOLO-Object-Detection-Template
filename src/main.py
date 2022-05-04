"""
main python file for the app
it is designed as a single file for faster debugging and if possible avoiding lifecycle issues
"""


# dependencies
import os
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# internal modules
import config
from components import navbar
from views import index, doc, interface, onnx, training_notebook


# exposing the instances
# Exposing the Flask Server to enable configuring it for logging in
#server = Flask(__name__)
app = Dash(__name__, title="app training", external_stylesheets=[dbc.themes.CYBORG, config.fontawesome])

server = app.server

###############################################################################
#                                MAIN                                         #
###############################################################################


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.layout,
    html.Div(id='page-content')
])


@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return index.layout
    if pathname == '/doc':
        return doc.layout
    elif pathname == '/interface':
        return interface.layout
    if pathname == '/onnx':
        return onnx.layout
    elif pathname == '/training_notebook':
        return training_notebook.layout

    else:
        return '404'


if __name__ == "__main__":
    app.run_server(debug=config.debug, host=config.host, port=config.port)
