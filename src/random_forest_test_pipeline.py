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
    return len(cleaned_article)

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

def find_syllable_count(cleaned_article):
    return textstat.syllable_count(cleaned_article)

def find_lexicon_count(cleaned_article):
    return textstat.lexicon_count(cleaned_article, removepunct=True)

def find_sentence_count(cleaned_article):
    return textstat.sentence_count(cleaned_article)

def find_smog_index(cleaned_article):
    return textstat.smog_index(cleaned_article)

def find_num_web_citations(raw_article):
    return raw_article.count("{{cite web")

def find_num_book_citations(raw_article):
    return raw_article.count("{{cite book")

def find_num_news_citations(raw_article):
    return raw_article.count("{{cite news")

def find_num_quotes(raw_article):
    return raw_article.count("quote=")

def find_num_h3_headers(raw_article):
    return raw_article.count("\n===")

def find_num_internal_links(raw_article):
    return (raw_article.count("[[") // 2)

def find_num_h2_headers(raw_article):
    return (raw_article.count("\n==") - find_num_h3_headers(raw_article))

data['num_web_citations'] = data['text'].apply(find_num_web_citations)
data['num_book_citations'] = data['text'].apply(find_num_book_citations)
data['num_news_citations'] = data['text'].apply(find_num_news_citations)
data['num_quotes'] = data['text'].apply(find_num_quotes)
data['num_h3_headers'] = data['text'].apply(find_num_h3_headers)
data['num_internal_links'] = data['text'].apply(find_num_internal_links)
data['num_h2_headers'] = data['text'].apply(find_num_h2_headers)
data['cleaned_text'] = data['text'].apply(clean_wiki_markup)
data['has_infobox'] = data['text'].str.contains('{{Infobox').astype(int)
data['num_categories'] = data['text'].apply(find_num_categories)
data['num_images'] = data['text'].apply(find_num_images)
data['num_ISBN'] = data['text'].apply(find_num_ISBN)
data['num_references'] = data['text'].apply(find_num_references)
data['article_length'] = data['text'].apply(find_article_length)
data['num_difficult_words'] = data['cleaned_text'].apply(find_num_difficult_words)
data['dale_chall_readability_score'] = data['cleaned_text'].apply(find_dale_chall_readability_score)
data['readability_index'] = data['cleaned_text'].apply(find_automated_readability_index)
data['linsear_write_formula'] = data['cleaned_text'].apply(find_linsear_write_formula)
data['gunning_fog_index'] = data['cleaned_text'].apply(find_gunning_fog_index)
data['smog_index'] = data['cleaned_text'].apply(find_smog_index)
data['syllable_count'] = data['cleaned_text'].apply(find_syllable_count)
data['lexicon_count'] = data['cleaned_text'].apply(find_lexicon_count)
data['sentence_count'] = data['cleaned_text'].apply(find_sentence_count)

data.dropna(inplace=True)

df1 = data.loc[:, ['has_infobox','num_categories','num_images','num_ISBN','num_references','article_length',
                 'num_difficult_words','dale_chall_readability_score','readability_index','linsear_write_formula',
                 'gunning_fog_index', 'num_web_citations','num_book_citations','num_news_citations',
                'num_quotes','num_h3_headers','num_internal_links', 'num_h2_headers', 'syllable_count',
                 'lexicon_count', 'sentence_count']]
y = data.label.values

X_train, X_test, y_train, y_test = train_test_split(df1.values, y, test_size=0.20, random_state=910)

RF = RandomForestRegressor(n_estimators=2000, random_state=910, )

RF.fit(X_train, y_train)

predictions = RF.predict(X_test)

print(mean_squared_error(y_test, predictions))