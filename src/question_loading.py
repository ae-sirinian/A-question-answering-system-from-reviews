#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:56:16 2020

@author: aram
"""
import pandas as pd
import config

def load_questions_from_yelp(
        file_path = config.QUESTIONS_WITH_ANSWERS_FROM_YELP_FILE_PATH):
    config.questions_from_yelp = []
    
    answers_only_from_first_question_pages = pd.read_csv(file_path)
    answers_only_from_first_question_pages_groups = answers_only_from_first_question_pages.groupby(['question', 'business_name'])
    
    for name, group in answers_only_from_first_question_pages_groups:
        question = name[0]
        business = name[1]
        
        if " - Temporarily Closed" in business:
            business = business.replace(" - Temporarily Closed", "")
        
        if "-Temporarily Closed" in business:
            business = business.replace("-Temporarily Closed", "")
        
        config.questions_from_yelp.append([question, business])