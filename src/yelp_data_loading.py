#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:13:09 2020

@author: aram
"""
import pandas as pd
import sys
from config import TIPS_FROM_YELP_FILE_PATH
from config import REVIEWS_FROM_YELP_FILE_PATH

def load_yelp_data(load_reviews = True, load_tips = True,
                   reviews_file_path = REVIEWS_FROM_YELP_FILE_PATH, 
                   tips_file_path = TIPS_FROM_YELP_FILE_PATH):
    reviews_df = pd.DataFrame()
    tips_df = pd.DataFrame()

    if(load_reviews == True):
        try:
            print("Loading Yelp's reviews...")
            reviews_df = pd.read_json(reviews_file_path, lines=True)
            print("Yelp's reviews loaded successfully!")
        except ValueError:
            print("Error while loading Yelp's data:", 
                  " File with Yelp's reviews can't be found in ", 
                  reviews_file_path, " file path!")
            sys.exit()
    
    if(load_tips == True):
        try:
            print("Loading Yelp's tips...")
            tips_df = pd.read_json(tips_file_path, lines=True)
            print("Yelp's tips loaded successfully!")
        except ValueError:
            print("Error while loading Yelp's data:",
                  " File with Yelp's tips can't be found in ", 
                  tips_file_path, " file path!")
            sys.exit()
    
    if(load_reviews == True and load_tips == True):
        tips_df = tips_df.drop(columns=['pos_sent', 'neg_sent', 'neu_sent'])
        reviews_df = reviews_df.drop(columns=['pos_sent', 'neg_sent', 'neu_sent'])
        return tips_df, reviews_df
    elif(load_reviews == True and load_tips == False):
        reviews_df = reviews_df.drop(columns=['pos_sent', 'neg_sent', 'neu_sent'])
        return reviews_df
    elif(load_reviews == False and load_tips == True):
        tips_df = tips_df.drop(columns=['pos_sent', 'neg_sent', 'neu_sent'])
        return tips_df
    else:
        return    