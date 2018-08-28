import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestRegressor
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_engineering import feature_engineering
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
    dataframe['num_web_citations'] = dataframe['text'].apply(feature_engineering.find_num_web_citations)
    dataframe['num_book_citations'] = dataframe['text'].apply(feature_engineering.find_num_book_citations)
    dataframe['num_news_citations'] = dataframe['text'].apply(feature_engineering.find_num_news_citations)
    dataframe['num_quotes'] = dataframe['text'].apply(feature_engineering.find_num_quotes)
    dataframe['num_h3_headers'] = dataframe['text'].apply(feature_engineering.find_num_h3_headers)
    dataframe['num_internal_links'] = dataframe['text'].apply(feature_engineering.find_num_internal_links)
    dataframe['num_h2_headers'] = dataframe['text'].apply(feature_engineering.find_num_h2_headers)
    dataframe['has_infobox'] = dataframe['text'].str.contains('{{Infobox').astype(int)
    dataframe['num_categories'] = dataframe['text'].apply(feature_engineering.find_num_categories)
    dataframe['num_images'] = dataframe['text'].apply(feature_engineering.find_num_images)
    dataframe['num_ISBN'] = dataframe['text'].apply(feature_engineering.find_num_ISBN)
    dataframe['num_references'] = dataframe['text'].apply(feature_engineering.find_num_references)
    dataframe['article_length'] = dataframe['text'].apply(feature_engineering.find_article_length)
    dataframe['num_difficult_words'] = dataframe['cleaned_text'].apply(feature_engineering.find_num_difficult_words)
    dataframe['dale_chall_readability_score'] = dataframe['cleaned_text'].apply(feature_engineering.find_dale_chall_readability_score)
    dataframe['readability_index'] = dataframe['cleaned_text'].apply(feature_engineering.find_automated_readability_index)
    dataframe['linsear_write_formula'] = dataframe['cleaned_text'].apply(feature_engineering.find_linsear_write_formula)
    dataframe['gunning_fog_index'] = dataframe['cleaned_text'].apply(feature_engineering.find_gunning_fog_index)
    dataframe['smog_index'] = dataframe['cleaned_text'].apply(feature_engineering.find_smog_index)
    dataframe['syllable_count'] = dataframe['cleaned_text'].apply(feature_engineering.find_syllable_count)
    dataframe['lexicon_count'] = dataframe['cleaned_text'].apply(feature_engineering.find_lexicon_count)
    dataframe['sentence_count'] = dataframe['cleaned_text'].apply(feature_engineering.find_sentence_count)
    dataframe['num_footnotes'] = dataframe['text'].apply(feature_engineering.find_num_footnotes)
    dataframe['num_note_tags'] = dataframe['text'].apply(feature_engineering.find_num_note_tags)
    dataframe['num_underlines'] = dataframe['text'].apply(feature_engineering.find_num_underlines)
    dataframe['num_journal_citations'] = dataframe['text'].apply(feature_engineering.find_num_journal_citations)
    dataframe['num_about_links'] = dataframe['text'].apply(feature_engineering.find_num_about_links)
    dataframe['num_wikitables'] = dataframe['text'].apply(feature_engineering.find_num_wikitables)
    dataframe.dropna(inplace=True)
    dataframe = dataframe.loc[:, ['label', 'has_infobox','num_categories','num_images','num_ISBN','num_references','article_length',
                'num_difficult_words','dale_chall_readability_score','readability_index','linsear_write_formula',
                'gunning_fog_index', 'num_web_citations','num_book_citations','num_news_citations',
                'num_quotes','num_h3_headers','num_internal_links', 'num_h2_headers', 'syllable_count',
                'lexicon_count', 'sentence_count','num_footnotes', 'num_note_tags', 'num_underlines', 'num_journal_citations',
                'num_about_links', 'num_wikitables', 'smog_index']]
    return dataframe

rf_model = pickle.load(open('/random_forest_model.sav', 'rb'))

class WikiArticleRegressor():

    def __init__(self, model):
        """ Instantiate a WikiArticleRegressor Class. 

        Parameters
        ----------
        none:

        Returns
        -------
        WikiArticleClassifier (class): A WikiArticleRegressor Object
        """
        self.model = model

    def fit(self, X, y):
        """ Fit . 

        Parameters
        ----------
        none:

        Returns
        -------
        WikiArticleClassifier (class): A WikiArticleRegressor Object
        """
        transformed_data = transform_data(X).values
        self.model.fit(transformed_data,y)

    def predict(self, X):
        transformed_data = transform_data(X).values
        return self.model.predict(transformed_data)
