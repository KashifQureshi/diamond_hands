import json

import boto3
import requests
from requests.auth import HTTPBasicAuth

import etl_scripts.config_properties as conf

host = conf.es_host
user = conf.es_user
passw = conf.es_password
aws_access_key_id = conf.aws_access_key_id
aws_secret_access_key = conf.aws_secret_access_key

comprehend = boto3.client(service_name='comprehend', region_name='us-east-2', use_ssl=True,
                          aws_access_key_id=aws_access_key_id,  # conf.aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)  # conf.aws_secret_access_key)


def analyze_sentiment(comprehend, batch):
    text_list = [item["title"] for item in batch]
    results = comprehend.batch_detect_sentiment(TextList=text_list, LanguageCode='en')
    results = results["ResultList"]
    for result in results:
        index = result["Index"]
        sentiment = result["Sentiment"]
        scores = result["SentimentScore"]
        batch[index].update({"analysis": {}})
        batch[index]['analysis']['sentiment'] = {'sentiment': sentiment, "scores": scores}
    return batch


def analyze_entity(comprehend, batch):
    text_list = [item["title"] for item in batch]
    results = comprehend.batch_detect_entities(TextList=text_list, LanguageCode='en')
    results = results["ResultList"]
    for result in results:
        index = result["Index"]
        entities = result["Entities"]
        batch[index]['analysis']['entites'] = entities
    return batch


def analyze_keyphrase(comprehend, batch):
    text_list = [item["title"] for item in batch]
    results = comprehend.batch_detect_key_phrases(TextList=text_list, LanguageCode='en')
    results = results["ResultList"]
    for result in results:
        index = result["Index"]
        key_phrases = result["KeyPhrases"]
        batch[index]['analysis']['keyphrases'] = key_phrases
    return batch


def get_non_analyzed_data():
    # get non-analyzed data from es
    entities = ["elon musk", "robinhood", "deepfuckingvalue", "melvin capital"]
    query_get_not_analyzed_data = {
        "query": {
            "bool": {
                "must": [
                    {
                        "script": {
                            "script": {
                                "source": "doc['analyzed'].value == false",
                                "lang": "painless"
                            }
                        }
                    },
                    {
                        "bool": {
                            "should":
                                [
                                    {
                                        "bool": {
                                            "must": [
                                                {
                                                    "match": {
                                                        "title": {
                                                            "query": token,
                                                            "fuzziness": 1
                                                        }
                                                    }
                                                } for token in y.split(' ')
                                            ]
                                        }
                                    }
                                    for y in entities
                                ]
                        }
                    }
                ]
            }
        },
        "size": 2500
    }
    print(query_get_not_analyzed_data)
    response = requests.post(
        url=host + '/submissions/_search',
        json=query_get_not_analyzed_data, auth=HTTPBasicAuth(user, passw)
    )
    all_data = [x['_source'] for x in json.loads(response.content)['hits']['hits']]
    return all_data


if __name__ == '__main__':
    print('Fetching data to Analyze...')
    data_to_analyze = get_non_analyzed_data()
    batch_size = 25
    batches = [data_to_analyze[i * batch_size:(i + 1) * batch_size] for i in
               range((len(data_to_analyze) + batch_size - 1) // batch_size)]
    print('Total Batches:', len(batches))
    batch_no = 1
    analyzed_data = []
    for batch in batches:
        print('Processing batch:', batch_no, ' of', len(batches))
        # add sentiment analysis attributes to the data.
        print('Analyzing from comprehend...')
        batch = analyze_sentiment(comprehend, batch)
        batch = analyze_entity(comprehend, batch)
        batch = analyze_keyphrase(comprehend, batch)
        analyzed_data.extend(batch)
        batch_no += 1

    # load data back to elasticsearch
    print('Pushing data back to Elasticsearch..')
    data_es = ''
    for data in analyzed_data:
        data['analyzed'] = True
        data_es += json.dumps({"index": {"_index": "submissions", "_id": data['id']}}) + '\n' + json.dumps(
            data) + '\n'
    response = requests.post(
        url=host + '/_bulk',
        data=data_es,
        headers={'Content-Type': 'application/json'}, auth=HTTPBasicAuth(user, passw)
    )
    print(response.content)
