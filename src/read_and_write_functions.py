#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:28:54 2020

@author: aram
"""
import pickle
import pandas as pd
import config
import re
from os import path
import sys

def write_tf_idf_or_id_to_pkl_file(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def read_tf_idf_or_id_from_pkl_file(file_path):
    tf_idf_or_id = None
    
    if path.isfile(file_path) == False:
        temporarily_close_substing = "_Temporarily_Closed"
        if temporarily_close_substing in file_path:
            file_path = file_path.replace(temporarily_close_substing, "")
            if path.isfile(file_path) == False:
                print("File " + file_path + " does not exist")
                return tf_idf_or_id
        else:
            print("File " + file_path + " does not exist")
            return tf_idf_or_id
    
    with open(file_path, 'rb') as f:
        tf_idf_or_id = pickle.load(f)
    
    return tf_idf_or_id

def _clear_file_contents(file_path):
    f = open(file_path, "w+")
    f.close()

def write_dataframe_to_csv_file(file_path, dataframe):
    _clear_file_contents(file_path)
    
    with open(file_path, 'a') as f:
        dataframe.to_csv(f)

def read_dataframe_from_csv_file(file_path):
    try:    
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("File " + file_path + " cannot be found.")
        print("Please compute it and try again!")
        sys.exit()

def _parse_business_name_for_file_path(name):
    return re.sub('[^0-9a-zA-Z]+', '_', name)

def _compute_tf_idf_file_path_for_piece_of_dataframe(parsed_name, source_name):
    file_path = ""
    
    source_name = source_name.lower()
    if ((source_name == "tips") or (source_name == "1")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews") or (source_name == "2")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "tips_with_bidaf") or (source_name == "3")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews_with_bidaf") or (source_name == "4")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "tips_with_bert") or (source_name == "5")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews_with_bert") or (source_name == "6")):
        file_path = config.TF_IDF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    else:
        return None
    
    return file_path

def _compute_df_file_path_for_piece_of_dataframe(parsed_name, source_name):
    file_path = ""
    
    source_name = source_name.lower()
    if ((source_name == "tips") or (source_name == "1")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews") or (source_name == "2")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "tips_with_bidaf") or (source_name == "3")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews_with_bidaf") or (source_name == "4")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "tips_with_bert") or (source_name == "5")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    elif ((source_name == "reviews_with_bert") or (source_name == "6")):
        file_path = config.DF_RAM_FRIENDLY_MODEL_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_DIR_PATH + parsed_name + ".pkl"
    else:
        return None
    
    return file_path

def get_file_path_from_business_name_for_tf_idf(business, source_name):
    parse_business_name = _parse_business_name_for_file_path(business)
    file_path = _compute_tf_idf_file_path_for_piece_of_dataframe(
            parse_business_name, source_name)
    
    return file_path

def get_file_path_from_business_name_for_df(business, source_name):
    parse_business_name = _parse_business_name_for_file_path(business)
    file_path = _compute_df_file_path_for_piece_of_dataframe(
            parse_business_name, source_name)
    
    return file_path

def get_tf_idf_from_business_name(business, source_name):
    file_path = get_file_path_from_business_name_for_tf_idf(business, source_name)
    tf_idf = read_tf_idf_or_id_from_pkl_file(file_path)
    
    return tf_idf

def get_df_from_business_name(business, source_name):
    file_path = get_file_path_from_business_name_for_df(business, source_name)
    DF = read_tf_idf_or_id_from_pkl_file(file_path)
    
    return DF

def load_yelp_data(file_path):
    source_df = pd.DataFrame()

    try:
        print("Loading Yelp's data...")
        source_df = pd.read_json(file_path, lines=True)
        print("Yelp's data loaded successfully!")
    except ValueError:
        print("Error while loading Yelp's data:", 
              " File with Yelp's data can't be found in ", 
              file_path, " file path!")
        sys.exit()
    
    return source_df