import pandas as pd
import numpy as np
from ast import literal_eval
from textstat.textstat import textstat
from gensim.corpora import wikicorpus

def clean_wiki_markup(raw_article):
    """ Removes Wikipedia markup from text and return cleaned text.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        cleaned_article (str): Cleaned Wikipedia text
    """
    semi_cleaned_article = wikicorpus.filter_wiki(raw_article)
    cleaned_article = semi_cleaned_article.replace("\n", "").replace("\'", "").replace("()", "").replace("=", "").replace("|alt","").replace("\xa0","")
    return cleaned_article


def find_num_categories(raw_article):
    """ Finds the estimated number of categories listed at the bottom of a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of categories listed in text
    """
    return raw_article.count("[[Category:")


def find_num_images(raw_article):
    """ Finds the estimated number of images in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of images present in text
    """
    return raw_article.count("[[Image:")


def find_num_ISBN(raw_article):
    """ Finds the estimated number of ISBN's listed in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of ISBN's listed in text
    """
    return raw_article.count("ISBN")


def find_num_references(raw_article):
    """ Finds the estimated number of references listed in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of references listed in text
    """
    return raw_article.count("</ref>")


def find_article_length(cleaned_article):
    """ Finds the article length (in characters) of a Wikipedia article.

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (int): Article length (in characters)
    """
    return len(cleaned_article)


def find_num_difficult_words(cleaned_article):
    """ Finds the number of difficult words in a Wikipedia article. Words are considered difficult if they do not 
        appear in a list of the 3,000 most common words that a 4th grader can understand.

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (int): Number of 'difficult' words
    """
    return textstat.difficult_words(cleaned_article)


def find_dale_chall_readability_score(cleaned_article):
    """ Uses the New Dale-Chall Formula to find a score that represents the grade-level of reading that characterizes the text.
        Scores can be interpreted as:

                Score              Level of Understanding
            ____________________________________________________
            4.9 or lower	|   average 4th-grade student or lower
            5.0–5.9	average |   5th or 6th-grade student
            6.0–6.9	average |   7th or 8th-grade student
            7.0–7.9	average |   9th or 10th-grade student
            8.0–8.9	average |   11th or 12th-grade student
            9.0–9.9	average |   13th to 15th-grade (college) student

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number describing the text's Dale-Chall score
    """
    return textstat.dale_chall_readability_score(cleaned_article)


def find_automated_readability_index(cleaned_article):
    """ Uses the Automated Readability Index to calculate a score that approximates the grade level needed
          to comprehend the text. 

            For example: If the score is 8, then the grade-level needed to comprehend the text is 8th. 

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number describing the Automated Readability Index 
    """
    return textstat.automated_readability_index(cleaned_article)


def find_linsear_write_formula(cleaned_article):
    """ Uses the Linsear Write Formula to calculate a score that approximates the grade level needed
          to comprehend the text. 

            For example: If the score is 8, then the grade-level needed to comprehend the text is 8th. 

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number describing the Linsear Write score 
    """
    return textstat.linsear_write_formula(cleaned_article)


def find_gunning_fog_index(cleaned_article):
    """ Uses the Gunning Gog Index to calculate a score that approximates the grade level needed
          to comprehend the text. 

            For example: If the score is 8, then the grade-level needed to comprehend the text is 8th. 

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number describing the Gunning Gog Index 
    """
    return textstat.gunning_fog(cleaned_article)


def find_smog_index(cleaned_article):
    """ Uses the SMOG index to calculate a score that approximates the grade level needed to comprehend the text. 

        For example: If the score is 8, then the grade-level needed to comprehend the text is 8th. 

        Texts of fewer than 30 sentences are statistically invalid, because the SMOG formula was normed on 30-sentence samples. 
        
        textstat requires at least 3 sentences for a result.

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number describing the Smog Index
    """
    return textstat.smog_index(cleaned_article)

def find_num_web_citations(raw_article):
    """ Finds the estimated number of web citations within a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of web citations
    """
    return raw_article.count("{{cite web")


def find_num_book_citations(raw_article):
    """ Finds the estimated number of book citations within a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of book citations
    """
    return raw_article.count("{{cite book")


def find_num_news_citations(raw_article):
    """ Finds the estimated number of news citations within a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of news citations
    """
    return raw_article.count("{{cite news")


def find_num_quotes(raw_article):
    """ Finds the estimated number of quotes mentioned in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of quotes in Wikipedia article
    """
    return raw_article.count("quote=")


def find_num_h3_headers(raw_article):
    """ Finds the estimated number of h3 headers in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of h3 headers in Wikipedia article
    """
    return raw_article.count("\n===")


def find_num_internal_links(raw_article):
    """ Finds the estimated number of internal links in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of internal links in Wikipedia article
    """
    return raw_article.count("[[")


def find_num_h2_headers(raw_article):
    """ Finds the estimated number of h2 headers in a Wikipedia article.

        Parameters
        ----------
        raw_article (str): Wikipedia markup text

        Returns
        -------
        (int): Number of h2 headers in Wikipedia article
    """
    return (raw_article.count("\n==") - find_num_h3_headers(raw_article))


def find_syllable_count(cleaned_article):
    """ Returns the number of syllables present in text.

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (float): Number syllables present in text
    """
    return textstat.syllable_count(cleaned_article)


def find_lexicon_count(cleaned_article):
    """ Returns the number of words in text. 
        Optional removepunct arugment specifies whether or not to remove punctuation symbols while counting lexicons. 
        Default value for removepunct is True. This removes  punctuation before counting lexicon items.

        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (int): Number lexicon items in text
    """ 
    return textstat.lexicon_count(cleaned_article, removepunct=True)


def find_sentence_count(cleaned_article):
    """ Returns the number of sentences in text. 
        Parameters
        ----------
        cleaned_article (str): Cleaned Wikipedia text

        Returns
        -------
        (int): Number sentences in text
    """ 
    return textstat.sentence_count(cleaned_article)

def find_num_footnotes(raw_article):
    """ Finds the estimated number of footnotes a Wikipedia article.

    Parameters
    ----------
    raw_article (str): Wikipedia markup text

    Returns
    -------
    (int): Number of estimated footnotes in Wikipedia article
    """
    return raw_article.count("{{")