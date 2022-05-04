""" Creates the navbar component displayed everywhere """
# global imports
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# local imports
import config


layout = dbc.Nav(
    className="nav nav-pills",
    children=[
        # logo/home
        # dbc.NavItem(html.Img(src=app.get_asset_url("logo.PNG"), height="40px")),
        # about
        dbc.NavItem(
            html.Div(
                [
                    dbc.NavLink("About", id="about-popover", active=False),
                    dbc.Popover(
                        id="about",
                        is_open=False,
                        target="about-popover",
                        children=[
                            dbc.PopoverHeader("How it works"),
                            
                        ],
                    ),
                ]
            )
        ),
        # links
        dbc.DropdownMenu(
            label="Links",
            nav=True,
            children=[
                dbc.DropdownMenuItem(
                    [html.I(className="fa fa-linkedin"), "  Contacts"],
                    href=config.contacts,
                    target="_blank",
                ),
                dbc.DropdownMenuItem(
                    [html.I(className="fa fa-github"), "  Code"],
                    href=config.code,
                    target="_blank",
                ),
                dbc.DropdownMenuItem(
                    [html.I(className="fa fa-medium"), "  Tutorial"],
                    href=config.tutorial,
                    target="_blank",
                ),
            ],
        ),
        html.Button(
                children='doc',
                n_clicks=0,
                type='submit',
                id='log-out-button1',
                className='btn btn-primary',
                hidden=False
            ),
        dcc.Location(id="doc", refresh=True),


                html.Button(
                children='index',
                n_clicks=0,
                type='submit',
                id='log-out-button2',
                className='btn btn-primary',
                hidden=False
            ),
        dcc.Location(id="index", refresh=True),


                html.Button(
                children='interface',
                n_clicks=0,
                type='submit',
                id='log-out-button3',
                className='btn btn-primary',
                hidden=False
            ),
        dcc.Location(id="interface", refresh=True),


                html.Button(
                children='onnx',
                n_clicks=0,
                type='submit',
                id='log-out-button4',
                className='btn btn-primary',
                hidden=False
            ),
        dcc.Location(id="onnx", refresh=True),


                html.Button(
                children='training_notebook',
                n_clicks=0,
                type='submit',
                id='log-out-button5',
                className='btn btn-primary',
                hidden=False
            ),
        dcc.Location(id="training_notebook", refresh=True)
    ],
)


# Python functions for about navitem-popover
@callback(
    output=Output("about", "is_open"),
    inputs=[Input("about-popover", "n_clicks")],
    state=[State("about", "is_open")],
)
def about_popover(n, is_open):
    """function for popup"""
    if n:
        return not is_open
    return is_open


@callback(
    output=Output("about-popover", "active"),
    inputs=[Input("about-popover", "n_clicks")],
    state=[State("about-popover", "active")],
)
def about_active(n, active):
    """function for popup"""
    if n:
        return not active
    return active
