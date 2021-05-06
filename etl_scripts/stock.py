from pandas_datareader import data
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import hashlib
import datetime
import etl_scripts.config_properties as conf


epoch = datetime.datetime.utcfromtimestamp(0)
stocks = conf.stocks
start = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
end = datetime.datetime.now()
host = conf.es_host
user = conf.es_user
passw = conf.es_password


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


for stock in stocks:
    stock_data = data.DataReader(stock, 'yahoo', start, end)
    print(stock_data)
    stock_data['symbol'] = stock
    stock_data['exchange'] = 'NYSE'
    stock_data['date'] = stock_data.index.values
    stock_data = stock_data.rename(
        columns={"High": "high", "Low": "low", "Open": "open", "Close": "close", "Volume": "volume",
                 "Adj Close": "adj_close"})

    data_es = ''
    for s_data in stock_data.to_dict(orient='records'):
        s_data['date'] = s_data['date'].to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S")
        data_es += json.dumps({"index": {"_index": "stocks", "_id": hashlib.md5(
            (s_data['symbol'] + str(s_data['date'])).encode()).hexdigest()}}) + '\n' + json.dumps(s_data) + '\n'

    response = requests.post(
        url=host + '/_bulk',
        data=data_es,
        headers={'Content-Type': 'application/json'}, auth=HTTPBasicAuth(user, passw)
    )
    print(response.content)

# reference :
# https://towardsdatascience.com/how-to-get-market-data-from-the-nyse-in-less-than-3-lines-python-41791212709c
