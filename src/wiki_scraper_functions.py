import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import lxml.etree
import urllib
import urllib.request
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import time
from feature_engineering import feature_engineering

def browse_to_category(browser, category):
    """ Uses a Selenium browser to get Wikipedia Category page.

        Parameters
        ----------
        browser (Selenium Webdriver): Web scraping browser
        category (str): A specific category to send the browser to (E.g., Machine Learning)

        Returns
        -------
        None : Does not return an object
    """
    url = 'https://en.wikipedia.org/wiki/Category:' + category
    browser.get(url)


def get_category_title(browser):
    """ Uses a Selenium browser to retrieve the Wikipedia Category describing a page.

        Parameters
        ----------
        browser (Selenium Webdriver): Web scraping browser

        Returns
        -------
        title (str): Wikipedia category
    """
    headline = browser.find_elements_by_css_selector('h1.firstHeading')
    title = [text.text for text in headline]
    return title[0].partition('Category:')[2]


def get_sub_categories(browser):
    """ Uses a Selenium browser to retrieve a list of Wikipedia sub-categories on a page.

        Parameters
        ----------
        browser (Selenium Webdriver): Web scraping browser

        Returns
        -------
        sub-categories (list): A list of Wikipedia sub-categories
    """
    subs = browser.find_elements_by_css_selector('a.CategoryTreeLabel')
    return [category.text for category in subs]


def click_to_sub_category_page(browser, page):
    """ Uses a Selenium browser to click to a Wikipedia sub-category page.
        Note: Must be hierarchically above the sub-category on the site map. 

        Parameters
        ----------
        page (str): Sub-category to be clicked

        Returns
        -------
        None : Does not return an object
    """
    browser.find_element_by_link_text(page).click()


def filter_pages(pages):
    """ Filters out erroneous text from the list of pages scraped under a sub-category . 

        Parameters
        ----------
        pages (list): A list of Wikipedia pages under a sub-category

        Returns
        -------
        pages (list): A filtered list of Wikipedia pages under a sub-category
    """
    for page in pages:
        if len(page) < 2:
            pages.remove(page)
    return pages


def get_wiki_xml(title):
    """  Fetches the raw markup text for a specified Wikipedia page. 

        Parameters
        ----------
        title (str): Title of Wikipedia page to be fetched

        Returns
        -------
        revs (str): Raw Wikipedia markup text for page
    """
    title = title
    params = { "format":"xml", "action":"query", "prop":"revisions", "rvprop":"timestamp|user|comment|content" }
    params["titles"] = "API|%s" % urllib.parse.quote(title.encode("utf8"))
    qs = "&".join("%s=%s" % (k, v)  for k, v in params.items())
    url = "http://en.wikipedia.org/w/api.php?%s" % qs
    tree = lxml.etree.parse(urllib.request.urlopen(url))
    revs = tree.xpath('//rev')
    return (revs[-1].text)

def get_wiki_category_quality(category):
    """  Scrape the sub-categories and subsequent pages for a Wikipedia category. 

        Parameters
        ----------
        category (str): Wikipedia category to be scraped

        Returns
        -------
        generator (dict): dict of sub-category, page title, and raw Wikipedia text for each article
    """
    category = category
    browser = Chrome()
    browse_to_category(browser, category)
    subs = get_sub_categories(browser)
    for sub_cat in subs:
        browse_to_category(browser, category)
        time.sleep(2)
        click_to_sub_category_page(browser, sub_cat)
        title = get_category_title(browser)
        try:
            w = browser.find_element_by_class_name('mw-category')
            pages = filter_pages(w.text.split('\n'))
            for page in pages:
                post = {'category': title,
                    'page': page,
                    'text' : get_wiki_xml(page)}
                yield post
        except NoSuchElementException:
            w = browser.find_elements_by_class_name('mw-content-ltr')
            pages = filter_pages(w[0].text.split('\n'))[2:]
            for page in pages:
                post = {'category': title,
                    'page': page,
                    'text' : get_wiki_xml(page)}
                yield post


def predict_category_quality(pages_df, model): 
    """ Returns a dataframe with sub-categories, page titles, and predicted article quality.

        Parameters
        ----------
        dataframe (Pandas DataFrame): DataFrame with sub-categories, page titles, and raw Wikipedia text
        model (predictive model object): Model trained to predict the quality of Wikipedia articles

        Returns
        -------
        dataframe (Pandas DataFrame): DataFrame with sub-categories, page titles, and predicted article quality
    """              
    data = pd.DataFrame(list(pages_df))
    data = data[data['text'] != ""]
    data = data[data['text'].str.contains("#redirect") == False]
    data = data[data['text'].str.contains("may refer to:\n\n*") == False]
    data = data[data['text'].str.contains("can refer to:\n") == False]
    data = data[data['text'].str.contains("could refer to:\n") == False]
    data = data[data['text'].str.contains("#REDIRECT") == False]
    data = data[data['text'].str.contains("== Matches ==\n:") == False]
    data = data[data['text'].str.contains("{{underconstruction") == False]
    data['cleaned_text'] = data['text'].apply(feature_engineering.clean_wiki_markup)
    data['num_web_citations'] = data['text'].apply(feature_engineering.find_num_web_citations)
    data['num_book_citations'] = data['text'].apply(feature_engineering.find_num_book_citations)
    data['num_news_citations'] = data['text'].apply(feature_engineering.find_num_news_citations)
    data['num_quotes'] = data['text'].apply(feature_engineering.find_num_quotes)
    data['num_h3_headers'] = data['text'].apply(feature_engineering.find_num_h3_headers)
    data['num_internal_links'] = data['text'].apply(feature_engineering.find_num_internal_links)
    data['num_h2_headers'] = data['text'].apply(feature_engineering.find_num_h2_headers)
    data['has_infobox'] = data['text'].str.contains('{{Infobox').astype(int)
    data['num_categories'] = data['text'].apply(feature_engineering.find_num_categories)
    data['num_images'] = data['text'].apply(feature_engineering.find_num_images)
    data['num_ISBN'] = data['text'].apply(feature_engineering.find_num_ISBN)
    data['num_references'] = data['text'].apply(feature_engineering.find_num_references)
    data['article_length'] = data['text'].apply(feature_engineering.find_article_length)
    data['num_difficult_words'] = data['cleaned_text'].apply(feature_engineering.find_num_difficult_words)
    data['dale_chall_readability_score'] = data['cleaned_text'].apply(feature_engineering.find_dale_chall_readability_score)
    data['readability_index'] = data['cleaned_text'].apply(feature_engineering.find_automated_readability_index)
    data['linsear_write_formula'] = data['cleaned_text'].apply(feature_engineering.find_linsear_write_formula)
    data['gunning_fog_index'] = data['cleaned_text'].apply(feature_engineering.find_gunning_fog_index)
    data['smog_index'] = data['cleaned_text'].apply(feature_engineering.find_smog_index)
    data['syllable_count'] = data['cleaned_text'].apply(feature_engineering.find_syllable_count)
    data['lexicon_count'] = data['cleaned_text'].apply(feature_engineering.find_lexicon_count)
    data['sentence_count'] = data['cleaned_text'].apply(feature_engineering.find_sentence_count)
    data['num_footnotes'] = data['text'].apply(feature_engineering.find_num_footnotes)
    data['num_note_tags'] = data['text'].apply(feature_engineering.find_num_note_tags)
    data['num_underlines'] = data['text'].apply(feature_engineering.find_num_underlines)
    data['num_journal_citations'] = data['text'].apply(feature_engineering.find_num_journal_citations)
    data['num_about_links'] = data['text'].apply(feature_engineering.find_num_about_links)
    data['num_wikitables'] = data['text'].apply(feature_engineering.find_num_wikitables)
    data.dropna(inplace=True)
    X = data.loc[:, ['has_infobox','num_categories','num_images','num_ISBN','num_references','article_length',
                'num_difficult_words','dale_chall_readability_score','readability_index','linsear_write_formula',
                'gunning_fog_index', 'num_web_citations','num_book_citations','num_news_citations',
                'num_quotes','num_h3_headers','num_internal_links', 'num_h2_headers', 'syllable_count',
                'lexicon_count', 'sentence_count','num_footnotes', 'num_note_tags', 'num_underlines', 'num_journal_citations',
                'num_about_links', 'num_wikitables', 'smog_index']].values
    predictions = model.predict(X)
    data['predicted_quality'] = predictions
    predicted_df = data.loc[:, ['category', 'page', 'predicted_quality']]
    return predicted_df


def group_pages_by__sub_category(dataframe):
    """ Returns a dataframe with mean article scores, grouped by sub-categories.

        Parameters
        ----------
        dataframe (Pandas DataFrame): DataFrame with sub-categories, page titles, and raw Wikipedia text

        Returns
        -------
        dataframe (Pandas DataFrame): DataFrame with sub-categories and mean predicted article quality
    """  
    return dataframe.groupby(by='category').mean()