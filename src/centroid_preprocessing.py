#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:19:11 2020

@author: aram
"""
import string
import re
import datetime
import numpy as np
from textblob import Word

def _has_numbers(input_string):
    return any(char.isdigit() for char in input_string)

def _has_letters(input_string):
    return any(char.isalpha() for char in input_string)

#(0
def _filter_do_nothing(word):
    result = str(word)
    
    return result

#(1
def _filter_lower_case(word):
    result = str(np.char.lower(word))
    
    return result

#(2
def _filter_correct_spelling(word):
    result = str(Word(word).correct())
    
    return result

#(3
def _filter_lower_case_and_correct_spelling(word):
    result = str(Word(_filter_correct_spelling(word)).correct())
    
    return result

#(4
def _filter_replace_punctuation_with_something(word, replace_with = "_"):
    firstString = string.punctuation
    secondString = replace_with * len(firstString)
    thirdString = ""
    translation = word.maketrans(firstString, secondString, thirdString)
    result = str(translation)
    
    return result

#(5
def _filter_lower_case_and_replace_punctuation_with_underscore(word):
    word = _filter_correct_spelling(word)
    firstString = string.punctuation
    secondString = "_" * len(firstString)
    thirdString = ""
    translation = word.maketrans(firstString, secondString, thirdString)
    result = str(translation)
    
    return result

#(6
def _filter_lemmatize(word):
    result = str(Word(word).lemmatize())
    
    return result

#(7
def _filter_correct_spelling_and_lemmatize(word):
    result = str(Word(word).correct().lemmatize())
    
    return result

#(8
def _filter_lower_case_and_lemmatize(word):
    result = str(Word(_filter_correct_spelling(word)).lemmatize())
    
    return result

#(9
def _filter_lower_case_and_correct_spelling_and_lemmatize(word):
    result = str(Word(_filter_correct_spelling(word)).correct().lemmatize())
    
    return result

#(10
def _filter_phone_numbers(text):
    pattern = re.compile(r'\d{3}[-.]\d{3}[-.]\d{4}')
    matches = pattern.finditer(text)
    matches_lst = [i.group(0) for i in matches]
    result = str(text)
    
    if len(matches_lst) != 0:
        return "number"
    return result

#(11
def _filter_emails(text):
    pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    matches = pattern.finditer(text)
    matches_lst = [i.group(0) for i in matches]
    result = str(text)
    
    if len(matches_lst) != 0:
        return "email"
    return result

#(12
def _filter_dates(text):
    try:
        datetime.datetime.strptime(text,"%m/%d/%y")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%d/%m/%y")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%m.%d.%y")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%d.%m.%y")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%d/%m")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%d.%m")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%m/%y")
        return "date"
    except:
        pass
    try:
        datetime.datetime.strptime(text,"%m.%y")
        return "date"
    except:
        pass
    
    return str(text)

#(13
def _filter_url(text):
    pattern = re.compile(r'(https?://)?(www\.)?(\w+)(\.\w+)')
    matches = pattern.finditer(text)
    matches_lst = [i.group(0) for i in matches]
    result = str(text)
    
    if len(matches_lst) != 0:
        return "URL"
    return result

#(14
def _filter_time(word):
    if ( (word[-2:].lower() == "am" and
          _has_numbers(word[:-2]) and
          (not _has_letters(word[:-2]))) 
    or (word[-5:].lower() == "amish" and
        _has_numbers(word[:-5]) and
        (not _has_letters(word[:-5]))) 
    or (word[-1:].lower() == "a" and
        _has_numbers(word[:-1]) and
        (not _has_letters(word[:-1])))
    ):
        return "am"
    
    if ( (word[-2:].lower() == "pm" and
          _has_numbers(word[:-2]) and
          (not _has_letters(word[:-2]))) 
    or (word[-5:].lower() == "pmish" and
        _has_numbers(word[:-5]) and
        (not _has_letters(word[:-5])))
    or (word[-1:].lower() == "p" and
        _has_numbers(word[:-1]) and
        (not _has_letters(word[:-1])))
    ):
        return "pm"
    
    if ( (word[-4:].lower() == "mins" and
          _has_numbers(word[:-4]) and
          (not _has_letters(word[:-4])))
    or (word[-5:].lower() == "pmish" and
        _has_numbers(word[:-5]) and
        (not _has_letters(word[:-5])))
    ):
        return "mins"
    
    if (word[-2:].lower() == "hr" and
        _has_numbers(word[:-2]) and
        (not _has_letters(word[:-2]))
    ):
        if ( _has_numbers(word[:-2]) and (not _has_letters(word[:-2])) ):
            return "hour"
    
    return str(word)

def filter_word(i, word):
    switcher = {
        0: _filter_do_nothing,
        1: _filter_lower_case,
        2: _filter_correct_spelling,
        3: _filter_lower_case_and_correct_spelling,
        4: _filter_replace_punctuation_with_something,
        5: _filter_lower_case_and_replace_punctuation_with_underscore,
        6: _filter_lemmatize,
        7: _filter_correct_spelling_and_lemmatize,
        8: _filter_lower_case_and_lemmatize,
        9: _filter_lower_case_and_correct_spelling_and_lemmatize,
        10: _filter_phone_numbers,
        11: _filter_emails,
        12: _filter_dates,
        13: _filter_url,
        14: _filter_time
    }
    
    func = switcher.get(i, lambda: "Invalid filter number option!")
    new_words = func(word)
    return new_words