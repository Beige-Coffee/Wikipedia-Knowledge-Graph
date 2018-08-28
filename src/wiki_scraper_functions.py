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
import pymongo
from pymongo import MongoClient
import pickle


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


def cick_to_sub_category_page(page):
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