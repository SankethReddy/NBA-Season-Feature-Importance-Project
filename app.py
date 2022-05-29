# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from joblib import load

def getSeason(year):
    return str(year) + '-' + str(year+1)

server = Flask(__name__)
app = dash.Dash(__name__, server = server, url_base_pathname='/NBAAnalytics/FeatureImportances/')
dfAll = pd.read_csv('2008-2018 NBA Team Data.csv')
dfAll['Season'] = [getSeason(y) for y in dfAll['Year']]
season_array = np.sort(dfAll['Season'].unique())
model_lst = ['Random Forest Regressor']

app.layout = html.Div([
    html.Div([
        html.H1('How Important Each Metric Was in Determining Winning % From 2008-2018')
        ], style = {'text-align': 'center', 'color': 'blue', 'font-family': 'verdana'}),
    html.Div([
        html.Label('Pick a Season'),
        dcc.Dropdown(
            id='season_dropdown',
            options = [{'label': i, 'value': i} for i in season_array],
            value = season_array[0]
            )
        ],style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
    html.Div([
        html.Label('Pick a Regression Model'),
        dcc.Dropdown(
            id='regression_dropdown',
            options=[{'label': i, 'value': i} for i in model_lst],
            value = model_lst[0]
            )
        ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
    html.Div(
        dcc.Graph(id='graph', figure=go.Figure())
        )    
    ])

@app.callback(
    Output('graph', 'figure'),
    [Input('season_dropdown', 'value'),
     Input('regression_dropdown', 'value')]
    )
def update_graph(season, regressor):
    feature_columns = ['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%',
                   'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD']
    model_string = season + ' ' + regressor + '.joblib'
    model = load(filename=model_string)
    df_feature_importances = pd.DataFrame(model.feature_importances_, 
                                      index = feature_columns, 
                                      columns = ['Importance']).sort_values('Importance', ascending = False).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar({
        'x': df_feature_importances['index'].values,
        'y': df_feature_importances['Importance'].values,
        'text': df_feature_importances['index'].values
        }
        ))
    fig.update_layout(xaxis_title = 'Metric', yaxis_title = 'Metric Importances',
                        title = {
                                'text': 'Metric Importances for the ' + season + ' NBA Season with the ' + regressor + ' Model',
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'x': 0.5,
                                'y': 0.9
                                })
    return fig

if __name__ == '__main__':
    app.run_server()