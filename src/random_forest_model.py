import pandas as pd
import numpy as np
import json
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/wiki_transformed_data.csv')
data.dropna(axis=0, inplace=True)


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