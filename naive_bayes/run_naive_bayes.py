import numpy as np
from pymongo import MongoClient
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import NaiveBayes


def load_data():
    client = MongoClient()
    db = client.nyt_dump
    coll = db.articles

    articles = coll.find({'$or': [{'section_name':'Sports'},
                                  {'section_name': 'Fashion & Style'}]})

    article_text_labels =  [(' '.join(article['content']), article['section_name'])
                     for article in articles]
    tokenizer = RegexpTokenizer(r'\w+')
    article_text = [tokenizer.tokenize(x[0].lower()) for x in article_text_labels]
    sections = [x[1] for x in article_text_labels]

    return article_text, sections
