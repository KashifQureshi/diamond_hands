import json
import time

import requests
from requests.auth import HTTPBasicAuth
import datetime
import etl_scripts.config_properties as conf

host = conf.es_host
user = conf.es_user
passw = conf.es_password
fields_to_extract = ','.join(conf.fields)

def fetch_last_record_date():
    query = {
        "_source": "created_utc_epoch",
        "query": {
            "match_all": {}
        },
        "size": 1,
        "sort": {
            "created_utc_epoch": {
                "order": "desc"
            }
        }
    }
    response = requests.post(
        url=host + '/submissions/_search',
        json=query, auth=HTTPBasicAuth(user, passw)
    )
    hits = [x['_source'] for x in json.loads(response.content)['hits']['hits']]
    if len(hits) == 0:
        return '2021-01-01'
    else:
        return hits[0]['created_utc_epoch']


if __name__ == '__main__':
    pages = int(conf.batch_size / 100)
    print('total batches:', pages)
    list_of_items = []
    data_es = ''
    print('Started Extracting data...')
    for page in range(1, pages + 1):
        print('Processing batch:', page, 'of', pages)
        if len(list_of_items) == 0:
            last_date = fetch_last_record_date()
        else:
            last_date = max([x['created_utc'] for x in list_of_items])
        for subreddit in conf.subreddits:
            extract_api = conf.extract_api_url.replace('SUBREDDIT', subreddit).replace('START_DATE', str(last_date)) \
                .replace('FIELDS', fields_to_extract)
            response = str(requests.get(extract_api).content.decode('utf8'))
            list_of_items.extend(json.loads(response)['data'])
            time.sleep(0.5)
    print('Finished Extracting data...')
    print('Creating Bulk Request Body...')
    for data in list_of_items:
        # print(data['created_utc'])
        data['analyzed'] = False
        data['created_utc_epoch'] = data['created_utc']
        data['retrieved_on_epoch'] = data['retrieved_on']
        data['retrieved_on'] = datetime.datetime.fromtimestamp(data['retrieved_on']).strftime("%Y-%m-%dT%H:%M:%S")
        data['created_utc'] = datetime.datetime.fromtimestamp(data['created_utc']).strftime("%Y-%m-%dT%H:%M:%S")
        data_es += json.dumps({"index": {"_index": "submissions", "_id": data['id']}}) + '\n' + json.dumps(
            data) + '\n'
    print('Started loading data...')
    response = requests.post(
        url=host + '/submissions/_bulk?refresh',
        data=data_es,
        headers={'Content-Type': 'application/json'}, auth=HTTPBasicAuth(user, passw)
    )
    print('Finished loading data...')
    print(response.content[:100])
