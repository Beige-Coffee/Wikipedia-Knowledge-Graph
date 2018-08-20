import pandas as pd
import numpy as np
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Feature_Engineering.feature_engineering as feature


def import_data_from_tsv(tsv_file):
    """ tsv because Wikipedia data comes in this format"""
    data = pd.read_csv(csv_file, sep='\t', header=None)
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
    dataframe.dropna(inplace=True)
    dataframe = dataframe.loc[:, ['has_infobox','num_categories','num_images','num_ISBN','num_references','article_length',
                 'num_difficult_words','dale_chall_readability_score','readability_index','linsear_write_formula',
                 'gunning_fog_index', 'num_web_citations','num_book_citations','num_news_citations',
                'num_quotes','num_h3_headers','num_internal_links', 'num_h2_headers', 'syllable_count',
                'lexicon_count', 'sentence_count']]
    return dataframe

class WikiArticleClassifier():

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators = 1000)

    def fit(self, X, y):
        transformed_data = transform_data(X).values
        self.rf.fit(transformed_data,y)

    def predict(self, X):
        transformed_data = transform_data(X).values
        return self.rf.predict(transformed_data)
    
    
    def predict_proba(self, X):
        transformed_data = transform_data(X).values
        return self.rf.predict_proba(transformed_data)






vectorizer = TfidfVectorizer()
vectorizer.fit(data['cleaned_text'])
X_transformed = vectorizer.transform(data['cleaned_text'])
tfidf_df = pd.DataFrame(X_transformed.todense())
df1 = data.loc[:, ['has_infobox','num_categories','num_images','num_ISBN','num_references','article_length',
                 'num_difficult_words','dale_chall_readability_score','readability_index','linsear_write_formula',
                 'gunning_fog_index', 'smog_index']]
X = pd.concat([df1, tfidf_df], axis=1)
y = data.label.values

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.20, random_state=910)

clf = RandomForestClassifier(n_estimators=1000, random_state=910)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
score = accuracy_score(y_test, predictions)

print(score)

