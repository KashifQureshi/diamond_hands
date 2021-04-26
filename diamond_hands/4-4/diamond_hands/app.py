import base64
import textwrap
import time
from datetime import date
from io import BytesIO
from itertools import chain

import dash
import dash_core_components as dcc
import dash_dangerously_set_inner_html as dngr
import dash_html_components as html
import dash_table
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from flask import Flask, render_template
from flask_caching import Cache
from plotly.subplots import make_subplots
from sklearn import linear_model, tree, neighbors
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

from diamond_hands.helper import *
from diamond_hands.queries import *

app = Flask(__name__)

TIMEOUT = 60
dash_app_home = dash.Dash(__name__, server=app, url_base_pathname='/home/')
dash_app_home.title = 'Diamond Hands 💎🙏'
dash_app_analytics = dash.Dash(__name__, server=app, url_base_pathname='/analytics/')
dash_app_analytics.title = 'Diamond Hands 💎🙏'
dash_app_data = dash.Dash(__name__, server=app, url_base_pathname='/data/')
dash_app_data.title = 'Diamond Hands 💎🙏'

cache_data = Cache(dash_app_data.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

cache_home = Cache(dash_app_home.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

cache_analytics = Cache(dash_app_analytics.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})


@cache_analytics.memoize(timeout=TIMEOUT)
@cache_home.memoize(timeout=TIMEOUT)
@cache_data.memoize(timeout=TIMEOUT)
def get_submission_normalized(company, entity):
    submissions_data_home = get_submissions_from_date(company, entity, True, '2021-01-01', '2021-03-31')
    df_submission = pd.json_normalize(submissions_data_home)
    return df_submission


@cache_analytics.memoize(timeout=TIMEOUT)
@cache_home.memoize(timeout=TIMEOUT)
@cache_data.memoize(timeout=TIMEOUT)
def get_stock_normalized(company, start_date, end_date):
    stock_data_home = get_stock_data_by_date_range(company, start_date, end_date)
    df_stock = pd.json_normalize(stock_data_home)
    return df_stock


def customwrap(s, width=30):
    return "<br>".join(textwrap.wrap(s, width=width))


######################################################################################################
#####################################    DASH - Analytics Tab    #####################################
######################################################################################################

attributes = {"score.value": "Total Submission Score", "comments.value": "Total Comments",
              "doc_count": "Total Submissions", "negative.value": "Negative Sentiment Score (Avg)",
              "positive.value": "Positive Sentiment Score (Avg)"}

models = {'Linear Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}


@dash_app_analytics.callback(Output("loading-output", "children"),
                             Input("entity-attribute", "value"))
def loading(value):
    time.sleep(0)
    return value


@dash_app_analytics.callback(Output("loading-output-2", "children"),
                             Input("entity-dropdown", "value"))
def loading_2(value):
    time.sleep(0)
    return value


# setting layout for analytics tab
dash_app_analytics.layout = html.Div([
    # div for static header
    dngr.DangerouslySetInnerHTML(
        dash_header.replace('$home', 'inactive').replace('$data', 'inactive').replace('$analytics', 'active').replace(
            '$aboutus', 'inactive')),
    html.Div([
        html.Div([
            html.Div([
                html.P("Select Period:"),
                dcc.DatePickerRange(
                    id='ml_daterange',
                    start_date_placeholder_text="Start Period",
                    end_date_placeholder_text="End Period",
                    start_date=date(2021, 1, 1),
                    end_date=datetime.now(),
                )
            ], style={'margin-top': '10px'}),
            html.Div([
                html.P("Select Model:"),
                dcc.Dropdown(
                    id='model-name',
                    options=[{'label': x, 'value': x}
                             for x in models],
                    value='Linear Regression',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'color': '#000'}
                )], style={'margin-top': '10px'}),
            html.Div([
                html.P("Company:"),
                dcc.Dropdown(
                    id='company-attribute',
                    options=[{'label': c.upper(), 'value': c} for c in ["gme", "amc"]],
                    value='gme',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'color': '#000'}
                )], style={'margin-top': '10px'}),
            html.Div([
                html.P("Stock Attribute:"),
                dcc.Dropdown(
                    id='stock-attribute',
                    options=[{'label': c.capitalize(), 'value': c} for c in ["close", "volume"]],
                    value='close',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'color': '#000'}
                )], style={'margin-top': '10px'}),
            html.Div([
                html.P("Entities:"),
                dcc.Dropdown(
                    id='entity-attribute',
                    options=[{'label': c.capitalize(), 'value': c} for c in
                             ["All", "Elon Musk", "DeepFuckingValue", "Robinhood", "Melvin Capital"]],
                    value='All',
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block', 'color': '#000'}
                )], style={'margin-top': '10px'}),
            html.Div([
                html.P("Reddit Attribute:"),
                dcc.Dropdown(
                    id='reddit-attribute',
                    options=[{'label': attributes[c], 'value': c} for c in attributes],
                    value=['score.value', 'comments.value'],
                    clearable=False,
                    multi=True,
                    style={'width': '200px', 'height': '200px', 'display': 'inline-block', 'color': '#000'}
                )], style={'margin-top': '10px'}),

        ], style={'position': 'relative', 'float': 'left'}),
        html.Div([dcc.Loading(
            id="loading",
            children=[
                html.Div([
                    dcc.Graph(id="graph", style={'margin-top': '20px'})
                ], style={'position': 'relative', 'float': 'left', 'margin-top': '20px', 'margin-left': '10px',
                          'width': '1050px'}),
                html.Div([
                    html.H3("Correlation Matrix")
                ], style={'width': '200px', 'margin-left': '100px', 'margin-top': '100px', 'position': 'relative',
                          'float': 'left'}),
                html.Div([dash_table.DataTable(
                    id='corr',
                    columns=[{"name": i, "id": i} for i in
                             ['Attributes', "Close Price", "Trade Volume", "Submissions", "Scores", "Comments"]])
                ], style={'width': '400px', 'display': 'inline-block', 'margin-left': '40px', 'margin-top': '20px'})
            ], style={'position': 'relative', 'float': 'left', 'margin-top': '20px'}),
        ])
    ], style={'width': '100%', 'position': 'relative', 'float': 'left', 'display': 'inline-block',
              'margin-left': '50px',
              'margin-top': '125px'}),
    # div for graph and its filters
    html.Div([
        html.Div([
            # filter for companies
            html.Div([
                html.P(["Companies: "]),
                dcc.Dropdown(id="company-dropdown", clearable=False, value='gme',
                             options=[{'label': c, 'value': c} for c in ["gme", "amc"]],
                             style={'width': '150px', 'display': 'inline-block', 'color': '#000'})],
                style={'position': 'relative', 'float': 'left'}),
            # filter for entities
            html.Div([
                html.P(["Entities: "]),
                dcc.Dropdown(id="entity-dropdown", clearable=False, value='All',
                             options=[{'label': c, 'value': c} for c in
                                      ["All", "Elon Musk", "DeepFuckingValue", "Robinhood",
                                       "Melvin Capital"]],
                             style={'width': '150px', 'display': 'inline-block', 'color': '#000'})],
                style={'position': 'relative', 'float': 'left',
                       'margin-left': '20px'}),
            # filter for attributes
            html.Div([
                html.P(["Attribute: "], style={'margin-left': '40px'}),
                dcc.Dropdown(id="attribute-dropdown", clearable=False,
                             value='score.value',
                             options=[{'label': attributes[c], 'value': c} for c in attributes],
                             style={'width': '200px', 'display': 'inline-block', 'color': '#000'})],
                style={'position': 'relative', 'float': 'left',
                       'margin-left': '20px'}),
        ], style={'margin-top': '20px', 'margin-left': '100px'}),
        html.Div([dcc.Loading(
            id="loading-2",
            children=[
                # graph stock vs submission data
                html.Div([dcc.Graph(id='multi-axis-chart',
                                    style={'position': 'relative', 'float': 'left', 'width': '800px'}
                                    )], style={'width': '800px', 'position': 'relative', 'float': 'left',
                                               'display': 'inline-block',
                                               'margin-left': '100px'}),
                # div for wordcloud image
                html.Div([html.Center(html.Img(id="image_wc"))],
                         style={'width': '400px', 'position': 'relative', 'float': 'left',
                                'margin-left': '50px', 'margin-bottom': '50px'}),
            ])
        ], style={'margin-top': '150px'})
    ], style={'position': 'relative', 'float': 'left',
              'margin-top': '150px', 'width': '100%'})
])


def get_column(data, column):
    data[column] = data[column].fillna(data[column].mean())
    return data[column]


# ML models
@dash_app_analytics.callback(
    Output("graph", "figure"),
    [Input('model-name', "value")],
    [Input('company-attribute', "value")],
    [Input('stock-attribute', "value")],
    [Input('reddit-attribute', "value")],
    [Input('ml_daterange', "start_date")],
    [Input('ml_daterange', "end_date")],
    [Input('loading', "value")],
    [State('entity-attribute', "value")], )
def train_and_display(name, company, stock_attr, reddit_attr, start_date, end_date, loading, entity):
    df_submission = get_submission_normalized(company, entity)
    df_stock = get_stock_normalized(company, start_date, end_date)
    df = df_stock.join(df_submission, lsuffix='date', rsuffix='key_string')
    X = get_column(df, reddit_attr)
    Y = get_column(df, stock_attr)

    train_idx, test_idx = train_test_split(df.index, test_size=.25, random_state=0)
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'
    X_train = df.loc[train_idx, reddit_attr]
    y_train = df.loc[train_idx, stock_attr]
    X_train = np.nan_to_num(X_train)
    y_train = y_train.fillna(0)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
    model = models[name]()
    model.fit(X_train, y_train)
    df['prediction'] = model.predict(X)

    # x_range = np.linspace(X.min(), X.max(), 100)
    # y_range = model.predict(x_range.reshape(-1, 1))
    fig = px.scatter(
        df, x=stock_attr, y='prediction',
        marginal_x='histogram', marginal_y='histogram',
        color='split', trendline='ols'
    )
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=Y.min(), y0=Y.min(),
        x1=Y.max(), y1=Y.max()
    )
    fig.update_layout(
        title_text=customwrap(
            "<b> Finding Correlation between Reddit with Stock market with Different Prediction Models during 2021 - Q1 </b>",
            75)
    )
    return fig


@dash_app_analytics.callback(
    Output('corr', 'data'),
    [Input('company-attribute', "value"),
     Input('entity-attribute', "value")],
    [Input('ml_daterange', "start_date")],
    [Input('ml_daterange', "end_date")]
)
def update_corr_table(company, entity, start_date, end_date):
    # calling helper function, since it is same for previous table
    df_submission = get_submission_normalized(company, entity)
    df_stock = get_stock_normalized(company, start_date, end_date)
    df = df_stock.join(df_submission, lsuffix='date', rsuffix='key_string')
    df['negative.value'].fillna(df['negative.value'].mean())
    cols = ['close', 'volume', 'doc_count', 'score.value', 'comments.value']
    df = df[cols]
    df_corr = df.corr()
    df_corr[cols] = df_corr[cols].round(5)
    # print(df_corr)
    df_corr.insert(loc=0, column='Attributes', value=["Close Price", "Trade Volume", "Submissions",
                                                      "Scores", "Comments"])
    df_corr = df_corr.rename(
        columns={"close": "Close Price", "volume": "Trade Volume", "doc_count": "Submissions",
                 "score.value": "Scores", "comments.value": "Comments"})
    # print(df_corr.head())
    return df_corr.to_dict('records')


# Callback to update graph with id = 'multi-axis-chart'
@dash_app_analytics.callback(
    Output('multi-axis-chart', 'figure'),
    [Input("company-dropdown", "value")],
    [Input("attribute-dropdown", "value")],
    [Input("loading-2", "value")],
    [State("entity-dropdown", "value")],
)
def update_figure(company, attribute, loading, entity):
    pd.set_option('display.max_columns', None)
    df_submission = get_submission_normalized(company, entity)
    df_stock = get_stock_normalized(company, '2021-01-01', '2021-03-31')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df_submission["key_as_string"], y=df_submission[attribute],
                   name=str(company).upper() + " reddit " + customwrap(attributes[attribute], 15)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Ohlc(x=df_stock['date'], open=df_stock['open'], high=df_stock['high'], low=df_stock['low'],
                close=df_stock['close'], name=str(company).upper() + ' OHCL Trace'),
        # go.Scatter(x=df_stock["date"], y=df_stock["close"], name=company + " stock price"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text=customwrap(
            "<b>Can we find a relationship between Reddit submissions, influencers, and stock price?</b>", 75)
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Reddit Submission Attribute</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Stock Price</b>", secondary_y=True)
    return fig


@cache_home.memoize(timeout=TIMEOUT)
def get_submission_df(company, entity):
    return get_submissions_from_date(company, entity, True, '2021-01-01', '2021-03-31')


# Callback to update wordcloud image with id = 'image_wc'
@dash_app_analytics.callback(
    Output('image_wc', 'src'),
    [Input('image_wc', 'id'),
     Input("company-dropdown", "value"), Input("entity-dropdown", "value"), ]
)
def save_word_cloud(b, company, entity):
    img = BytesIO()
    data = get_submission_df(company, entity)
    entites = [e['key'] for e in list(chain.from_iterable([d['keyphrases']['buckets'] for d in data]))]

    wordcloud = WordCloud(
        background_color='white',
        width=400,
        height=400
    ).generate(' '.join(entites))
    wordcloud.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


#######################################################################################################
########################################    DASH - Data Tab    ########################################
#######################################################################################################


# getting data in dataframes
@cache_data.memoize(timeout=TIMEOUT)
def get_sub_table():
    return submission_data_df()


@cache_data.memoize(timeout=TIMEOUT)
def get_stock_table():
    return stock_data_df()


# setting layout for data tab
dash_app_data.layout = html.Div([
    # div for static header
    dngr.DangerouslySetInnerHTML(
        dash_header.replace('$home', 'inactive').replace('$data', 'active').replace('$analytics', 'inactive').replace(
            '$aboutus', 'inactive')),
    # H1 heading
    html.Div([html.Br(), html.H1('Reddit Submission Data')], style={'margin-top': '150px', 'margin-left': '150px'}),
    # div for table from dataframe for submission data
    html.Div([dash_table.DataTable(id='table-sorting-filtering-sub',
                                   columns=[{'name': i, 'id': i, 'deletable': True} for i in
                                            sorted(get_sub_table().columns)],
                                   page_current=0, page_size=PAGE_SIZE, page_action='custom', filter_action='custom',
                                   filter_query='', sort_action='custom', sort_mode='multi', sort_by=[],
                                   style_header={'backgroundColor': 'rgb(245, 230, 220)', 'color': 'rgb(54, 57, 63)'},
                                   style_cell={'backgroundColor': 'rgb(245, 230, 220)', 'color': 'rgb(54, 57, 63)'})],
             style={'width': '80%', 'display': 'inline-block', 'margin-left': '100px'}),
    # H1 heading
    html.Div([html.Br(), html.H1('Stock Data')], style={'margin-left': '150px'}),
    # div for table from dataframe for stock data
    html.Div([dash_table.DataTable(id='table-sorting-filtering-stock',
                                   columns=[{'name': i, 'id': i, 'deletable': True} for i in
                                            sorted(get_stock_table().columns)],
                                   page_current=0, page_size=PAGE_SIZE, page_action='custom', filter_action='custom',
                                   filter_query='', sort_action='custom', sort_mode='multi', sort_by=[],
                                   style_header={'backgroundColor': 'rgb(245, 230, 220)', 'color': 'rgb(54, 57, 63)'},
                                   style_cell={'backgroundColor': 'rgb(245, 230, 220)', 'color': 'rgb(54, 57, 63)'})],
             style={'width': '80%', 'display': 'inline-block', 'margin-left': '100px'})
])


# Callback to update submission table id = 'table-sorting-filtering-sub', based on filter,sort,etc
@dash_app_data.callback(
    Output('table-sorting-filtering-sub', 'data'),
    Input('table-sorting-filtering-sub', "page_current"),
    Input('table-sorting-filtering-sub', "page_size"),
    Input('table-sorting-filtering-sub', 'sort_by'),
    Input('table-sorting-filtering-sub', 'filter_query'))
def update_table(page_current, page_size, sort_by, filter):
    # calling helper function, since it is same for next table
    return update_filtered_table(page_current, page_size, sort_by, filter, get_sub_table())


# Callback to update submission table id = 'table-sorting-filtering-sub', based on filter,sort,etc
@dash_app_data.callback(
    Output('table-sorting-filtering-stock', 'data'),
    Input('table-sorting-filtering-stock', "page_current"),
    Input('table-sorting-filtering-stock', "page_size"),
    Input('table-sorting-filtering-stock', 'sort_by'),
    Input('table-sorting-filtering-stock', 'filter_query'))
def update_table(page_current, page_size, sort_by, filter):
    # calling helper function, since it is same for previous table
    return update_filtered_table(page_current, page_size, sort_by, filter, get_stock_table())


#######################################################################################################
########################################    DASH - HOME Tab    ########################################
#######################################################################################################

@dash_app_home.callback(Output("loading-output-3", "children"),
                        [Input("ticker-dropdown", "value")])
def loading_3(value):
    print(value)
    time.sleep(0)
    return value


dash_app_home.layout = html.Div([
    dngr.DangerouslySetInnerHTML(
        dash_header.replace('$home', 'active').replace('$data', 'inactive').replace('$analytics', 'inactive').replace(
            '$aboutus', 'inactive')),
    # sentiment chart 
    html.Div([
        # filter for action
        html.Div([
            html.P(["Action:  "]),
            dcc.Dropdown(id="action-dropdown", clearable=False, value='sell',
                         options=[{'label': c.capitalize(), 'value': c} for c in ["buy", "sell"]],
                         style={'width': '150px', 'display': 'inline-block', 'color': '#000'})
        ], style={'position': 'relative', 'float': 'left', 'margin-top': '20px', 'margin-left': '20px'}
        ),
        # filter for Ticker
        html.Div([
            html.P(["Ticker:  "]),
            dcc.Dropdown(id="ticker-dropdown", clearable=False,
                         value='gme', options=[{'label': c.upper(), 'value': c} for c in
                                               ["gme", "amc"]],
                         style={'width': '150px', 'display': 'inline-block', 'color': '#000'})
        ], style={'position': 'relative', 'float': 'left', 'margin-top': '20px', 'margin-left': '20px'}
        ),
    ], style={'margin-top': '120px', 'margin-left': '200px'}),
    # negative & positive sentiments datewise
    html.Div([dcc.Loading(
        id="loading-3",
        children=[
            html.Div([
                dcc.Graph(id='sentiment-line-chart')])
        ])
    ], style={'margin-left': '100px', 'width': '90%', 'display': 'inline-block', 'position': 'relative',
              'float': 'left', 'margin-top': '20px'}),
    # sunburst chart
    html.Div([
        dcc.Graph(id='sunburst'), ],
        style={'display': 'inline-block', 'margin-top': '100px', 'margin-left': '50px', 'position': 'relative',
               'float': 'left'}),
])


@cache_home.memoize(timeout=TIMEOUT)
def get_sunburst_data():
    sentiments = ['Negative', 'Positive', 'Neutral']
    companies = ['gme', 'amc', 'appl']
    df = pd.DataFrame()
    q = ''
    for company in companies:
        for sentiment in sentiments:
            q += json.dumps({"index": "submissions"}) + '\n'
            q += json.dumps(get_submissions_from_sunburst(company, sentiment)) + '\n'

    responses = requests.post(
        url=host + '/submissions/_msearch', data=q, headers={'Content-Type': 'application/json'},
        auth=HTTPBasicAuth(user, passw)
    )
    count = len(sentiments)
    comp = 0
    for response in json.loads(responses.content)['responses']:
        data = [x['_source'] for x in response['hits']['hits']]
        x = pd.json_normalize(data)
        x['company'] = companies[comp]
        df = df.append(x)
        count -= 1
        if count == 0:
            comp += 1
            count = len(sentiments)
    df = df.rename(
        columns={"num_comments": "comments", "analysis.sentiment.sentiment": "sentiment", "score": "submission_score"})
    df = df.fillna(0)
    df['sentiment_score'] = df[['analysis.sentiment.scores.Negative', 'analysis.sentiment.scores.Positive',
                                'analysis.sentiment.scores.Neutral']].max(axis=1)
    df = df.drop(
        columns=['analysis.sentiment.scores.Negative', 'analysis.sentiment.scores.Positive',
                 'analysis.sentiment.scores.Neutral'])
    df['title'] = df['title'].apply(customwrap)
    return df


@dash_app_home.callback(
    Output('sunburst', 'figure'),
    [Input("sunburst", "value")]
)
def update_sunburst(attribute):
    df = get_sunburst_data()
    fig = px.sunburst(df, path=['company', 'sentiment', 'title'], values='sentiment_score')
    fig.update_layout(width=500, height=500, paper_bgcolor='rgb(54, 57, 63)', plot_bgcolor='rgb(54, 57, 63)')
    fig.update_layout(title_text=customwrap("<b>Top submissions based on sentiments for various stocks<b>", 20),
                      title_font_color='rgb(245, 230, 220)')
    return fig


@dash_app_home.callback(
    Output('sentiment-line-chart', 'figure'),
    [Input("action-dropdown", "value")],
    [Input("loading-3", "value")],
    [Input("ticker-dropdown", "value")]
)
def update_sentiment_line_chart(action, loading, ticker):
    pd.set_option('display.max_columns', None)
    df_submission = get_submission_normalized(action + ' ' + ticker, 'all')
    df_stock = get_stock_normalized(ticker, '2021-01-01', '2021-03-31')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df_submission["key_as_string"], y=df_submission['negative.value'],
                   name=ticker + " " + action + " NEGATIVE"), secondary_y=True, )
    fig.add_trace(
        go.Scatter(x=df_submission["key_as_string"], y=df_submission['positive.value'],
                   name=ticker + " " + action + " POSITIVE"), secondary_y=True, )
    fig.add_trace(
        go.Scatter(x=df_submission["key_as_string"], y=df_submission['neutral.value'],
                   name=ticker + " " + action + " NEUTRAL"), secondary_y=True, )
    fig.add_trace(
        go.Scatter(x=df_submission["key_as_string"],
                   y=abs(df_submission['positive.value'] - df_submission['negative.value']),
                   name=ticker + " " + action + " DIFF"), secondary_y=True, )
    fig.add_trace(
        go.Bar(x=df_stock["date"], y=df_stock["volume"], name=ticker + " trade volume"), secondary_y=False,
    )
    fig.update_layout(xaxis=dict(rangeslider=dict(
        visible=True
    )),
        title_text="<b>How do sentiments for Buying / Selling affect Trade Volume?</b>",
        title_font_color='rgb(245, 230, 220)', paper_bgcolor='rgb(54, 57, 63)', plot_bgcolor='rgb(54, 57, 63)',
        legend_font_color='rgb(245, 230, 220)')
    # Set x-axis title
    fig.update_xaxes(title_text="Date", title_font_color='rgb(245, 230, 220)', color='rgb(245, 230, 220)')
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Sentiment Scores</b>", title_font_color='rgb(245, 230, 220)',
                     color='rgb(245, 230, 220)', secondary_y=True)
    fig.update_yaxes(title_text="<b>Stock Trade Volume</b>", title_font_color='rgb(245, 230, 220)',
                     color='rgb(245, 230, 220)', secondary_y=False)
    return fig


######################################################################################################
#######################################    Flask App Routes    #######################################
######################################################################################################


@app.route("/analytics")
def dash_analytics():
    return dash_app_analytics.index()


@app.route("/data")
def dash_data():
    return dash_app_data.index()


@app.route("/")
@app.route("/home")
def home():
    return dash_app_home.index()


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


if __name__ == '__main__':
    app.run(debug=True, port=5051)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    # serve(app, port=9000, host='localhost')
