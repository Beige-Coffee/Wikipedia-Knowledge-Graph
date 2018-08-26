import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



class FakeClassifier():
    """Classifier that produces completely random predictions"""

    def __init__(self, n_classes=2, *args, **kargs):
        self._n_classes=n_classes 
    
    def fit(self, X, y):
        """Memorize number of target calsses"""
        self._n_classes = len(np.unique(y))
    
    def predict(self, X):
        np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        return np.array(
            [self._random_proba() for _ in range(len(X))]
            )
    
    def _random_proba(self, n_classes=2):
        """return a probability for each class - sum to 1"""
        probas = np.random.random(size=self._n_classes)
        return probas / sum(probas) # probabilities sum to 1





class FraudClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._classifier = MultinomialNB()

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        # Code to fit the model.
        self._vectorizer.fit(X)
        self._classifier.fit(self._vectorizer.transform(X), y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        #return self._classifier.predict_proba(self._vectorizer.transform(X))
        return (np.random.random(), 2)

    def predict(self, X):
        """Make predictions on new data."""
        return self._classifier.predict(self._vectorizer.transform(X))

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        return self._classifier.score(self._vectorizer.transform(X), y)


def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    data = pd.read_json(filename)
    X = data['description']
    data['fraud'] = data.acct_type.apply(lambda x: 1 if 'fraud' in x else 0)
    y = data['fraud']
    return X, y

def main():
    X, y = get_data("data/data.json")
    tc = FakeClassifier()
    tc.fit(X, y)
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(tc, f)

if __name__ == '__main__':
    main()
