"# diamond_hands" 
etl_scripts:
    reddit_extract.py: This is used to extract data from pushshift api for reddit.
                       API DOCUMENTATION: https://github.com/pushshift/api
    reddit_enrich.py: This is used to enrich the reddit data with sentiments, entities and keyphrases (using AWS Comprehend)
    stock.py: Extracts stock data from yahoo finance using pandas_datareader library

diamond_hands: 
    Web app (Flask + Plotly + Dash)
