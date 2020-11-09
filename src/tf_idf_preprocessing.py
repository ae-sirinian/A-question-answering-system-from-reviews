#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:10:24 2020

@author: aram
"""
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
import numpy as np

def _convert_lower_case(text):
    return np.char.lower(text)

def _remove_stop_words(text):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(text))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def _remove_punctuation(text):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], ' ')
        text = np.char.replace(text, "  ", " ")
    text = np.char.replace(text, ',', '')
    return text

def _remove_apostrophe(text):
    return np.char.replace(text, "'", "")

def _stemming(text):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(text))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def _convert_numbers(text):
    tokens = word_tokenize(str(text))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def _preprocess_text_for_tf_idf(text):
    text = _convert_lower_case(text)
    text = _remove_punctuation(text)
    text = _remove_apostrophe(text)
    text = _remove_stop_words(text)
    text = _convert_numbers(text)
    text = _stemming(text)
    text = _remove_punctuation(text)
    text = _convert_numbers(text)
    text = _stemming(text)
    text = _remove_punctuation(text)
    text = _remove_stop_words(text)
    return text.strip()

def _preprocess_dataframe_for_tf_idf(dataframe):
    preprocessed_dataframe = []
    
    for index, row in dataframe.iterrows():
        text = row["text"]
        name = row["name"]
        preprocessed_dataframe.append(
                [name, word_tokenize(str(_preprocess_text_for_tf_idf(text)))] )
    return preprocessed_dataframe
