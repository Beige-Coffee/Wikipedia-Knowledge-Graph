import pandas as pd
import numpy as np
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

file = '../data/enwiki.observations.text_wp10.30k.tsv'
raw_data = pd.read_csv(file, sep='\t', header=None)

data = pd.DataFrame(data=list(raw_data[0].apply(literal_eval)))
data = data[data['text'] != ""]
data = data[data['text'].str.contains("#redirect") == False]
data = data[data['text'].str.contains("may refer to:\n\n*") == False]
data = data[data['text'].str.contains("can refer to:\n") == False]
data = data[data['text'].str.contains("could refer to:\n") == False]
data = data[data['text'].str.contains("#REDIRECT") == False]
data = data[data['text'].str.contains("== Matches ==\n:") == False]
data = data[data['text'].str.contains("{{underconstruction") == False]

classes = {"stub": 0, "start": 1, "c": 2, "b": 3, "ga": 4, "fa": 5} 
data["label"] = data['label'].map(classes)

data = data[:10000].copy()

def clean_wiki_markup(raw_article):
    semi_cleaned_article = wikicorpus.filter_wiki(raw_article)
    cleaned_article = semi_cleaned_article.replace("\n", "").replace("\'", "").replace("()", "").replace("=", "").replace("|alt","").replace("\xa0","")
    return cleaned_article


data['cleaned_text'] = data['text'].apply(clean_wiki_markup)
data.dropna(inplace=True)

data = data.loc[:, ['cleaned_text']]
y = data.label.values


hash_vectorizer = HashingVectorizer(n_features=5000)

hash_vectorizer.fit(dataframe['cleaned_text'])

X_transformed = hash_vectorizer.transform(dataframe['cleaned_text'])

hash_df = pd.DataFrame(X_transformed.todense())

y = dataframe.label.values


X_train, X_test, y_train, y_test = train_test_split(hash_df.values, y, test_size=0.20, random_state=910)


RF = RandomForestRegressor(n_estimators=500, random_state=910, )

RF.fit(X_train, y_train)

predictions = RF.predict(X_test)

print(mean_squared_error(y_test, predictions))