# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 22:42:04 2025

@author: Aries
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

def clean_text(text):
    # 1. Remove headers — keep only text after the first blank line
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        text = parts[1]  # everything after headers
    else:
        text = parts[0]  # fallback

    # 2. Remove HTML tags if present
    text = re.sub(r"<.*?>", " ", text)

    # 3. Lowercase the text
    text = text.lower()

    # 4. Remove anything that is not a letter or whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def preprocess_text(text):
    stop_words = set(stopwords.words('English'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    cleaned = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens
        if word not in stop_words and word.isalpha() and len(word) >= 3
    ]
    return ' '.join(cleaned)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.apply(lambda text: preprocess_text(clean_text(text)))