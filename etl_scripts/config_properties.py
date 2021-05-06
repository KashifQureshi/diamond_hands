# API info : https://github.com/pushshift/api
aws_access_key_id = 'TEST'
aws_secret_access_key = 'TEST'
es_host = 'http://YOUR_IP:9200'
es_user = 'YOUR_USER'
es_password = 'YOUR_PASS'
extract_api_url = 'https://api.pushshift.io/reddit/search/submission/?subreddit=SUBREDDIT&after=START_DATE&size=100&fields=FIELDS&sort=asc&sort_type=created_utc'
subreddits = ['wallstreetbets']
fields = ['id', 'title', 'subreddit', 'selftext', 'created_utc', 'retrieved_on', 'num_comments', 'num_crossposts',
          'score']
batch_size = 5000
stocks = ['GME', 'AMC']
