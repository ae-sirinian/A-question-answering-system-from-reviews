#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:23:41 2020

@author: aram
"""
import pandas as pd
import numpy as np
from multiprocessing import Pool
from read_and_write_functions import write_dataframe_to_csv_file
from read_and_write_functions import get_tf_idf_from_business_name
from read_and_write_functions import get_df_from_business_name
from tf_idf_preprocessing import _preprocess_text_for_tf_idf
from centroid_preprocessing import filter_word
import config
from textblob import TextBlob
from progress_bar import printProgressBar
from collections import Counter
import sys

tf_idf = {}
compute_all_iwcs_with_tf_idf_preprocessing_flag = True
compute_all_iwcs_with_word2vec_preprocessing_flag = True
number_of_cores = 1
source_name_for_iwcs = None

def _get_document_frequency_from_df(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def _compute_tf_idf_for_query(words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf, DF, corpus_length):
    tokens = []
    tf_idf = {}
    N = corpus_length
    
    for list_of_words in words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf:
        word_possibly_with_tf_idf_preprocessing_from_sentence = list_of_words[0]
        tokens.append(word_possibly_with_tf_idf_preprocessing_from_sentence)
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = _get_document_frequency_from_df(token, DF)
        idf = np.log((N+1)/(df+1))
        
        tf_idf[token] = tf*idf
    
    return tf_idf

def compute_iwcs_from_sentence(source_df, sentence, business, source_name,
                               with_tf_idf_preprocessing = True,
                               with_word2vec_preprocessing = True):
    #TF-IDF inverted index have to be computed
    
    #If tf-idf was computed with preprocessing then with_tf_idf_preprocessing = True
    #If tf-idf was computed without preprocessing then with_tf_idf_preprocessing = False
    #If word2vec was computed with preprocessing then with_word2vec_preprocessing = True
    #If word2vec was computed without preprocessing then with_word2vec_preprocessing = False
    zen = TextBlob(sentence)
    words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf = []
    iwcs_score = np.zeros((config.NUM_OF_FEATURES, ), dtype='float32')
    tf_idf = None
    DF = None
    
    if with_tf_idf_preprocessing:
        for word in zen.words:
            preprocessed_text_for_tf_idf = str(_preprocess_text_for_tf_idf(word))
            if (preprocessed_text_for_tf_idf != ""):
                new_item = [preprocessed_text_for_tf_idf, str(word)]
                words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf.append(
                        new_item)
    else:
        words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf = [
                [str(word), str(word)] for word in zen.words if word != ""]
    
    #May be need to first loop over words with WV and the TF-IDF
    try:
        position_in_document = source_df[source_df["text"] == sentence].index[0]
        
        tf_idf = get_tf_idf_from_business_name(business, source_name)
        
        if tf_idf == None:
            return iwcs_score
    except IndexError:
        DF = get_df_from_business_name(business, source_name)
        
        if DF == None:
            return iwcs_score
    
    
    if (DF == None and tf_idf != None):
        for list_of_words in words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf:
            word_possibly_with_tf_idf_preprocessing_from_sentence = list_of_words[0]
            word_unpreprocessing_with_tf_idf = list_of_words[1]
            if word_possibly_with_tf_idf_preprocessing_from_sentence in tf_idf:
                token_tf_idfs_and_positions = tf_idf[word_possibly_with_tf_idf_preprocessing_from_sentence]
                for token_tf_idf_and_position in token_tf_idfs_and_positions:
                    token_position = token_tf_idf_and_position[0]
                    token_tf_idf = token_tf_idf_and_position[1]
                    if (token_position != position_in_document):
                        continue
                    if with_word2vec_preprocessing:
                        for i in range(config.NUM_OF_FILTERS):
                            new_word_with_word2vec_preprocessing = filter_word(i, word_unpreprocessing_with_tf_idf)
                            if new_word_with_word2vec_preprocessing in config.index2word_set:
                                iwcs_score = np.add(iwcs_score,
                                                    np.multiply(token_tf_idf,
                                                                config.wv[new_word_with_word2vec_preprocessing]))
                                break
                    else:
                        if word_unpreprocessing_with_tf_idf in config.index2word_set:
                            iwcs_score = np.add(iwcs_score, np.multiply(token_tf_idf, config.wv[word_unpreprocessing_with_tf_idf]))
                    break
    elif (DF != None and tf_idf == None):
        #TODO a version of the above without the TF-IDF loading but with TF-IDF computation by using the DF allready computed
        corpus_length = None
        try:
            corpus_length = len(source_df.groupby("name").get_group(business))#TODO check for names with -Temporary closed
        except:
            print("Error. Business not found: ", business)
            sys.exit()
        tf_idf = _compute_tf_idf_for_query(
                words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf,
                DF, corpus_length)
        
        for list_of_words in words_preprocessed_with_tf_idf_if_with_tf_idf_preprocessing_true_and_words_unpreprocessing_with_tf_idf:
            word_possibly_with_tf_idf_preprocessing_from_sentence = list_of_words[0]
            word_unpreprocessing_with_tf_idf = list_of_words[1]
            if word_possibly_with_tf_idf_preprocessing_from_sentence in tf_idf:
                token_tf_idf = tf_idf[word_possibly_with_tf_idf_preprocessing_from_sentence]
                if with_word2vec_preprocessing:
                    for i in range(config.NUM_OF_FILTERS):
                        new_word_with_word2vec_preprocessing = filter_word(
                                i, word_unpreprocessing_with_tf_idf)
                        if new_word_with_word2vec_preprocessing in config.index2word_set:
                            iwcs_score = np.add(
                                    iwcs_score, np.multiply(token_tf_idf,
                                                            config.wv[new_word_with_word2vec_preprocessing]))
                            break
                else:
                    if word_unpreprocessing_with_tf_idf in config.index2word_set:
                        iwcs_score = np.add(iwcs_score, np.multiply(token_tf_idf, config.wv[word_unpreprocessing_with_tf_idf]))
    
    #print(iwcs_score)
    return iwcs_score

def _compute_all_iwcs(source_df):
    global compute_all_iwcs_with_tf_idf_preprocessing_flag
    global compute_all_iwcs_with_word2vec_preprocessing_flag
    global source_name_for_iwcs
    
    iwcs_score_list = []
    progress_bar_size = len(source_df)
    
    i = 0
    printProgressBar(i, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    for idx, row in source_df.iterrows():
        sentence = row.text
        business = row['name']
        
        iwcs_score = compute_iwcs_from_sentence(source_df, sentence, business, 
                                                source_name_for_iwcs, 
                                                with_tf_idf_preprocessing = compute_all_iwcs_with_tf_idf_preprocessing_flag, 
                                                with_word2vec_preprocessing = compute_all_iwcs_with_word2vec_preprocessing_flag)
        iwcs_score_list.append(iwcs_score)
        i += 1
        printProgressBar(i, progress_bar_size, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    
    if "iwcs" in source_df.columns:
        source_df.drop(columns=["iwcs"])
    source_df["iwcs"] = iwcs_score_list
    
    return source_df

def _parallelize_dataframe(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

def compute_all_iwcs_and_save_result_in_a_file(source_df, source_name,
                                               file_path,
                                               with_tf_idf_preprocessing,
                                               with_word2vec_preprocessing):
    global compute_all_iwcs_with_tf_idf_preprocessing_flag
    global compute_all_iwcs_with_word2vec_preprocessing_flag
    global number_of_cores
    global source_name_for_iwcs
    
    source_name_for_iwcs = source_name
    source_with_iwcs = pd.DataFrame()
    compute_all_iwcs_with_tf_idf_preprocessing_flag = with_tf_idf_preprocessing
    compute_all_iwcs_with_word2vec_preprocessing_flag = with_word2vec_preprocessing
    
    print("Initializing IWCS model...")
    print("    Calculating IWCS...")
    source_with_iwcs = _parallelize_dataframe(source_df, _compute_all_iwcs, n_cores = number_of_cores)
    '''
    elif ((source_name == "yelp") or (source_name == "3")):
        #TODO make it compute for Yelp's answers
        pass
    '''
    print("    IWCS calculated successfully!")
    
    print("    Saving IWCS dictionary")
    write_dataframe_to_csv_file(file_path, source_with_iwcs)
    print("    IWCS dictionary saved successfully!")
    print("IWCS model initialized successfully!")
    
    return source_with_iwcs
