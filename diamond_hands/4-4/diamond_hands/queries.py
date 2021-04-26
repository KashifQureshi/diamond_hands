import json
from datetime import datetime

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from diamond_hands.config import *


def get_submissions_from_sunburst(symbol, sentiment):
    query_submission_data = {
        "_source": ["num_comments", "subreddit", "title", "score", "analysis.sentiment.sentiment",
                    "analysis.sentiment.scores." + sentiment,
                    "analysis.keyphrases.Text", "analysis.keyphrases.Score"],
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "analysis.entites.Text": symbol
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "match": {
                            "analysis.sentiment.sentiment": sentiment
                        }
                    }
                ]
            }
        },
        "size": 5,
        "sort": {
            "_script": {
                "type": "number",
                "script": {
                    "lang": "painless",
                    "source": "doc['analysis.sentiment.scores." + sentiment + "'].value * (1 + doc['score'].value) + (1 + doc['num_comments'].value)"
                },
                "order": "desc"
            }
        }
    }
    return query_submission_data


def get_submissions_from_date(symbol, entity, analyzed, startDate, endDate):
    query_submission_data = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "bool": {
                                        "must": [
                                            {
                                                "match": {
                                                    y: {
                                                        "query": token,
                                                        "fuzziness": 1
                                                    }
                                                }
                                            } for token in symbol.split(' ')
                                        ]
                                    }
                                }
                                for y in ["title", "selftext"]
                            ]
                        }
                    },
                    {
                        "match": {
                            "analyzed": analyzed
                        }
                    },
                    {
                        "bool": {
                            "should": [
                                {
                                    "bool": {
                                        "must": [
                                            {
                                                "match": {
                                                    y: {
                                                        "query": token,
                                                        "fuzziness": 1
                                                    }
                                                }
                                            } for token in entity.split(' ')
                                        ]
                                    }
                                }
                                for y in ["title", "selftext"]
                            ]
                        }
                    }
                ]
            }
        },
        "size": 0,
        "aggs": {
            "stats_per_day": {
                "date_histogram": {
                    "field": "created_utc",
                    "format": "yyyy-MM-dd",
                    "calendar_interval": "1d",
                    "min_doc_count": 1
                },
                "aggs": {
                    "entites": {
                        "terms": {
                            "field": "analysis.entites.Text.keyword"
                        }
                    },
                    "keyphrases": {
                        "terms": {
                            "field": "analysis.keyphrases.Text.keyword"
                        }
                    },
                    "comments": {
                        "sum": {
                            "field": "num_comments"
                        }
                    },
                    "score": {
                        "sum": {
                            "field": "score"
                        }
                    },
                    "negative": {
                        "avg": {
                            "field": "analysis.sentiment.scores.Negative"
                        }
                    },
                    "positive": {
                        "avg": {
                            "field": "analysis.sentiment.scores.Positive"
                        }
                    },
                    "neutral": {
                        "avg": {
                            "field": "analysis.sentiment.scores.Neutral"
                        }
                    },
                    "mixed": {
                        "avg": {
                            "field": "analysis.sentiment.scores.Mixed"
                        }
                    }
                }
            }
        }
    }
    if entity.lower() == 'all':
        del query_submission_data['query']['bool']['must'][2]
    # print(json.dumps(query_submission_data))
    response = requests.post(
        url=host + '/submissions/_search',
        json=query_submission_data, auth=HTTPBasicAuth(user, passw)
    )

    data = json.loads(response.content)['aggregations']['stats_per_day']['buckets']
    return data


def get_stock_data_by_date_range(symbol, start_date, end_date):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "symbol": symbol
                        }
                    },
                    {
                        "range": {
                            "date": {
                                "gte": start_date,
                                "lte": end_date
                            }
                        }
                    }
                ]
            }

        },
        "size": 100,
        "sort": {
            "date": {
                "order": "asc"
            }
        }
    }
    response = requests.post(
        url=host + '/stocks/_search',
        json=query, auth=HTTPBasicAuth(user, passw)
    )

    stock_data = [x['_source'] for x in json.loads(response.content)['hits']['hits']]
    return stock_data


def stock_data_df():
    df_stock = pd.json_normalize(get_stock_data_by_date_range('gme amc', '2021-01-01', '2021-03-31'))
    df_stock[['open', 'high', 'low', 'close', 'adj_close']] = df_stock[
        ['open', 'high', 'low', 'close', 'adj_close']].round(5)
    return df_stock


def submission_data_df():
    df_sub = pd.DataFrame()
    df_gme = pd.json_normalize(get_submissions_from_date('gme', 'all', True, '2021-01-01', '2021-03-31'))
    df_gme['Company'] = 'GME'
    df_amc = pd.json_normalize(get_submissions_from_date('amc', 'all', True, '2021-01-01', '2021-03-31'))
    df_amc['Company'] = 'AMC'
    df_sub = df_sub.append(df_gme)
    df_sub = df_sub.append(df_amc)
    df_sub = df_sub.drop(
        columns=['entites.doc_count_error_upper_bound', 'entites.sum_other_doc_count', 'entites.buckets', 'mixed.value',
                 'keyphrases.doc_count_error_upper_bound', 'keyphrases.sum_other_doc_count', 'key',
                 'keyphrases.buckets'])
    df_sub = df_sub.rename(
        columns={"key_as_string": "Date", "doc_count": "Total Submissions", "score.value": "Submission Scores",
                 "negative.value": "Negative Score", "positive.value": "Positive Score",
                 "neutral.value": "Neutral Score",
                 "comments.value": "Submission Comments"})
    df_sub[['Negative Score', 'Negative Score', 'Neutral Score', 'Positive Score']] = df_sub[
        ['Negative Score', 'Negative Score', 'Neutral Score', 'Positive Score']].round(5)
    return df_sub
