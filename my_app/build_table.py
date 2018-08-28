# Imports
import time
import pickle
import requests
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
import pandas as pd
from build_model import FraudClassifier, FakeClassifier
from bs4 import BeautifulSoup
import random_forest

starttime=time.time()

# Import pickle from data
model = pickle.load(open("data/model.pkl", "rb" ))


# Create MongoDB instance
client = MongoClient('mongodb://localhost:27017/')
db = client['fraud_case_study']

# Create collection called "events"
events = db['events']


# Write function to pull "live data" from API
def pull_data(sequence_number=0):
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    response = requests.post(url, json={'api_key': api_key,
                                    'sequence_number': sequence_number})
    raw_data = response.json()
    return raw_data

# Write function
def data_feed():
    sequence_number = 1
    while True:
        print('Pulling data...')
        raw_data = pull_data(sequence_number)
        data = raw_data['data']
        #print(data)
        sequence_number = raw_data['_next_sequence_number']
        print(f'Got {len(data)} rows of data')
        if data:
            df = pd.DataFrame(data)
            print(df.columns)
            print('Predicting...')
            df['probability_fraud'] = model.predict_proba(df)[:,1] 
            print('Inserting into MongoDB...')
            events.insert_many(df.to_dict(orient='records'))
        print('Sleeping for 60 seconds...')
        time.sleep(60)

def build_table():
    data = list(events.find())
    df = pd.DataFrame(data)
    df['event_created'] = df['event_created'].apply(lambda x : time.strftime("%a, %b %d, %Y", time.localtime(x)))
    df['user_created'] = df['user_created'].apply(lambda x : time.strftime("%a, %b %d, %Y", time.localtime(x)))
    df['See More'] = df['_id'].apply(make_link)
    df['Priority Level'] = pd.cut(df['probability_fraud'], [0.0,0.3,0.6,0.999], 3, labels=['Low', 'Medium', 'High'], include_lowest=True)
    df = df[['object_id','name','event_created', 'probability_fraud', 'Priority Level', 'See More']]
    df.columns = ['ID Number', 'Name', 'Event Created', 'Probability of Fraud', 'Priority Level', 'See More']
    df['Probability of Fraud'] = df['Probability of Fraud'].round(2)
    with pd.option_context('display.max_colwidth', -1):
        raw_table = df.to_html(classes='table table-striped table-hover', index=False, escape=False)
    return style_table(raw_table)


def style_table(raw_table):
    old_html = '<table border="1" class="dataframe'
    new_html = '<table border="1px solid white" style="background-color: #708090" class="dataframe sortable' 
    table = raw_table.replace(old_html, new_html)
    return table

def make_link(object_id):
    _id = str(object_id)
    link_url = f"/details/{_id}"
    link = f'<a class="details_link" href="{link_url}">details</a>'
    return link

def strip_html_tags(html):
    return BeautifulSoup(html, 'html.parser').text


def get_details_for_one_event(_id):
    data = events.find({'_id': ObjectId(_id)})
    df = pd.DataFrame.from_records(data)
    df['description'] = df['description'].apply(strip_html_tags)
    with pd.option_context('display.max_colwidth', -1):
        return df.T.to_html()

if __name__ == '__main__':
    data_feed()

