#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:23:41 2020

@author: aram
"""
import numpy as np
from time import process_time
from textblob import TextBlob
from centroid_preprocessing import filter_word
from config import NUM_OF_FEATURES
from config import NUM_OF_FILTERS
from config import MAX_NUM_OF_FILTERS
from multiprocessing import Pool
from read_and_write_functions import write_dataframe_to_csv_file
import config
import pandas as pd
from progress_bar import printProgressBar

def _compute_all_centroids_without_preprocessing(source_df):
    index2word_set = set(config.wv.index2word)
    centroid_list = []
    progress_bar_size = len(source_df)
    
    z = 0
    printProgressBar(z, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    for idx, row in source_df.iterrows():
        n_words = 0
        feature_vec = np.zeros((NUM_OF_FEATURES, ), dtype='float32')
        zen = TextBlob(row.text)
        
        for word in zen.words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, config.wv[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
            centroid_list.append(feature_vec)
        else:
            centroid_list.append(None)
        
        z += 1
        printProgressBar(z, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    
    if "centroid" in source_df.columns:
        source_df.drop(columns=["centroid"])
    source_df["centroid"] = centroid_list
    
    return source_df

def _compute_all_centroids_with_preprocessing(source_df):
    index2word_set = set(config.wv.index2word)
    centroid_list = []
    progress_bar_size = len(source_df)
    
    z = 0
    printProgressBar(z, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    for idx, row in source_df.iterrows():
        n_words = 0
        feature_vec = np.zeros((NUM_OF_FEATURES, ), dtype='float32')
        zen = TextBlob(row.text)
        
        for word in zen.words:
            for i in range(NUM_OF_FILTERS):
                new_word = filter_word(i, word)
                if new_word in index2word_set:
                    n_words += 1
                    feature_vec = np.add(feature_vec, config.wv[new_word])
                    break
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
            centroid_list.append(feature_vec)
        else:
            centroid_list.append(None)
        
        z += 1
        printProgressBar(z, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    
    if "centroid" in source_df.columns:
        source_df.drop(columns=["centroid"])
    source_df["centroid"] = centroid_list
    
    return source_df

def _parallelize_dataframe(df, func, n_cores = 2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

def compute_all_centroids_and_save_result_in_a_file(
        source_df, file_path, with_preprocessing = False):    
    print("Computing centroids with preprocessing...")
    t_start = process_time()
    if with_preprocessing:
        source_df = _parallelize_dataframe(
                source_df,
                _compute_all_centroids_with_preprocessing)
    else:
        source_df = _parallelize_dataframe(
                source_df,
                _compute_all_centroids_without_preprocessing)
    t_stop = process_time()
    print("Centroids with preprocessing computed successfully!")
    print("Elapsed time ", t_stop-t_start, "sec")
    
    print("Saving computed centroids...")
    t_start = process_time()
    write_dataframe_to_csv_file(file_path, source_df)
    t_stop = process_time()
    print("Computed centroids saved successfully!")
    print("Elapsed time ", t_stop-t_start, "sec")
    
    return source_df

def compute_centroid_from_sentence(sentence):
    index2word_set = set(config.wv.index2word)
    zen = TextBlob(sentence)
    feature_vec = np.zeros((NUM_OF_FEATURES, ), dtype='float32')
    n_words = 0
    
    for word in zen.words:
        for i in range(MAX_NUM_OF_FILTERS):
            new_word = filter_word(i, word)
            if new_word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, config.wv[new_word])
                break
    
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    
    return feature_vec