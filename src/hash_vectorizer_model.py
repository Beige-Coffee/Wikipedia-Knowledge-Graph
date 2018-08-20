import pandas as pd
import numpy as np
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestRegressor
import nltk
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import feature_engineering as feature
from sklearn.metrics import mean_squared_error


def import_data_from_tsv(tsv_file):
    """ tsv because Wikipedia data comes in this format"""
    data = pd.read_csv(tsv_file, sep='\t', header=None)
    return pd.DataFrame(data=list(data[0].apply(literal_eval)))


def extract_labels(dataframe):
    """ returns a numpy array of labels converted into integers ranging from 1-5 (inclusive)"""
    classes = {"stub": 0, "start": 1, "c": 2, "b": 3, "ga": 4, "fa": 5} 
    dataframe["label"] = dataframe['label'].map(classes)
    return dataframe["label"].values


def transform_data(dataframe):
    dataframe['cleaned_text'] = dataframe['text'].apply(feature_engineering.clean_wiki_markup)
    dataframe.dropna(inplace=True)
    dataframe = dataframe.loc[:, ['cleaned_text']]
    return dataframe


def hash_vectorize_data(dataframe):
    hash_vectorizer = HashingVectorizer(n_features=5000)
    hash_vectorizer.fit(dataframe['cleaned_text'])
    X_transformed = hash_vectorizer.transform(dataframe['cleaned_text'])
    hash_df = pd.DataFrame(X_transformed.todense())
    y = dataframe.label.values


rf = RandomForestRegressor(n_estimators=10, random_state=910)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
mean_squared_error(y_test, predictions)