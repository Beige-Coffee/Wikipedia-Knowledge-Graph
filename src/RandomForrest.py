import pandas as pd
import numpy as np
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestClassifier

def clean_wiki_markup(raw_article):
    semi_cleaned_article = wikicorpus.filter_wiki(raw_article)
    cleaned_article = semi_cleaned_article.replace("\n", "").replace("\'", "").replace("()", "").replace("=", "").replace("|alt","").replace("\xa0","")
    return cleaned_article

def find_num_categories(raw_article):
    return raw_article.count("[[Category:")

def find_num_images(raw_article):
    return raw_article.count("[[Image:")

def find_num_ISBN(raw_article):
    return raw_article.count("ISBN")

def find_num_references(raw_article):
    return raw_article.count("</ref>")

def find_article_length(cleaned_article):
    return len(article)

def find_num_difficult_words(cleaned_article):
    return textstat.difficult_words(cleaned_article)

def find_dale_chall_readability_score(cleaned_article):
    return textstat.dale_chall_readability_score(cleaned_article)

def find_automated_readability_index(cleaned_article):
    return textstat.automated_readability_index(cleaned_article)

def find_linsear_write_formula(cleaned_article):
    return textstat.linsear_write_formula(cleaned_article)

def find_gunning_fog_index(cleaned_article):
    return textstat.gunning_fog(cleaned_article)

def transform_dataframe(raw_dataframe):
    raw_dataframe['has_infobox'] = raw_dataframe['text'].str.contains('{{Infobox').astype(int)
    raw_dataframe['num_categories'] = raw_dataframe['text'].apply(find_num_categories)
    raw_dataframe['num_images'] = raw_dataframe['text'].apply(find_num_images)
    raw_dataframe['num_references'] = raw_dataframe['text'].apply(find_num_references)
    raw_dataframe['cleaned_text'] = raw_dataframe['text'].apply(clean_wiki_markup)
    raw_dataframe['article_length'] = raw_dataframe['cleaned_text'].apply(find_article_length)
    raw_dataframe['num_difficult_words'] = raw_dataframe['cleaned_text'].apply(find_num_difficult_words)
    raw_dataframe['readability_index'] = raw_dataframe['cleaned_text'].apply(find_automated_readability_index)
    raw_dataframe['dale_chall_readability_score'] = raw_dataframe['cleaned_text'].apply(find_dale_chall_readability_score)
    raw_dataframe['linsear_write_formula'] = raw_dataframe['cleaned_text'].apply(find_linsear_write_formula)
    raw_dataframe['gunning_fog_index'] = raw_dataframe['cleaned_text'].apply(find_gunning_fog_index)
    
    ])

def get_y(dataframe):
    classes = {"stub": 0, "start": 1, "c": 2, "b": 3, "ga": 4, "fa": 5} 
    dataframe["label"] = dataframe['label'].map(classes)
    return dataframe["label"]

def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    data : A dataframe 
    """
    data = pd.read_csv(file, sep='\t', header=None)
    data = pd.DataFrame(data=list(data[0].apply(literal_eval)))
    return data

class FraudModel():

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators = 1000)

    def fit(self, X,y):
        cleaned_data = make_clean_df(X).values.T
        self.rf.fit(cleaned_data,y)

    def predict(self, X):
        cleaned_real_time = make_clean_df(X).values.T
        return self.rf.predict_proba(cleaned_real_time)

