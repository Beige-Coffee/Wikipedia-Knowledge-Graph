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
import wiki_article_regressor as war



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
        """ Fit model with training data. 

        Parameters
        ----------
        X (np.array): Numpy array or sparse matrix of shape [n_samples,n_features]
                    Training data

        y : numpy array of shape [n_samples, n_targets]
                    Target values

        Returns
        -------
        WikiArticleClassifier (class): Returns an instance of self
        """
        transformed_labled_data = war.transform_data(X).values
        self.model.fit(transformed_labled_data,y)

    def predict(self, X):
        """ Fit model with training data. 

        Parameters
        ----------
        X (np.array): Numpy array or sparse matrix of shape [n_samples,n_features]
                    Samples.

        Returns
        -------
        WikiArticleClassifier (class): Predicted values
        """
        transformed_data = war.get_features(war.transform_data(X)).values
        return self.model.predict(transformed_data)
