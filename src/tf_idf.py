#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:28:54 2020

@author: aram
"""
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from tf_idf_preprocessing import _preprocess_dataframe_for_tf_idf
from tf_idf_preprocessing import _preprocess_text_for_tf_idf
from read_and_write_functions import write_tf_idf_or_id_to_pkl_file
from read_and_write_functions import read_tf_idf_or_id_from_pkl_file
from read_and_write_functions import get_file_path_from_business_name_for_tf_idf
from read_and_write_functions import get_file_path_from_business_name_for_df
import pandas as pd
from answering_with_bidaf import answer_with_bidaf
from answering_with_bert import answer_with_bert
import sys
from collections import Counter
import numpy as np
from progress_bar import printProgressBar


def _calculate_df_for_all_words_in_document(preprocessed_list, dataframe_piece):
    word_indexes = {}
    DF = {}
    N = len(preprocessed_list)
    dataframe = dataframe_piece[2]
    
    for i in range(N):
        tokens = preprocessed_list[i][1]
        #Position in document not realy needed here
        original_text = dataframe.iloc[i]["text"]
        position_in_document = dataframe[dataframe["text"] == original_text].index[0]
        
        for w in tokens:
            try:
                word_indexes[w].add(position_in_document)
            except:
                word_indexes[w] = {position_in_document}
    
    for i in word_indexes:
        DF[i] = len(word_indexes[i])
    return DF

def _get_document_frequency_from_df(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def _calculate_tf_idf_from_dataframe(preprocessed_list, DF, dataframe_piece):
    tf_idf = {}
    N = len(preprocessed_list)
    dataframe = dataframe_piece[2]
    
    for i in range(N):
        tokens = preprocessed_list[i][1]
        counter = Counter(tokens)
        words_count = len(tokens)
        original_text = dataframe.iloc[i]["text"]
        position_in_document = dataframe[dataframe["text"] == original_text].index[0]
        
        for token in np.unique(tokens):
            tf = counter[token]/words_count
            df = _get_document_frequency_from_df(token, DF)
            idf = np.log((N+1)/(df+1))
            
            try:
                tf_idf[token].add((position_in_document, tf*idf))
            except:
                tf_idf[token] = {(position_in_document, tf*idf)}
        
    return tf_idf

def _break_dataframe_to_compute_tf_idf_model(dataframe, source_name):
    list_of_dataframe_pieces = []
    dataframe_grouped = dataframe.groupby("name")
    
    source_name = source_name.lower()
    for name, group in dataframe_grouped:
        file_path = get_file_path_from_business_name_for_tf_idf(name, source_name)
        if file_path == None:
            break
        file_path2 = get_file_path_from_business_name_for_df(name, source_name)
        if file_path2 == None:##########if file_path == None
            break
        list_of_dataframe_pieces.append([name, file_path, group, file_path2])
    
    return list_of_dataframe_pieces

def compute_tf_idf_model(dataframe, source_name):
    print("Initializing TF-IDF model...")
    source_name = source_name.lower()
    if ((source_name != "tips") and (source_name != "1") and (source_name != "reviews") and (source_name != "2") and (source_name != "tips_with_bidaf") and (source_name != "3") and (source_name != "reviews_with_bidaf") and (source_name != "4") and (source_name != "tips_with_bert") and (source_name != "5") and (source_name != "reviews_with_bert") and (source_name != "6")):
        print("Error: In compute_tf_idf_model_ram_friendly source_name is not valid!")
        sys.exit()
    
    list_of_dataframe_pieces = _break_dataframe_to_compute_tf_idf_model(
            dataframe, source_name)
    
    number_of_dataframe_pieces = len(list_of_dataframe_pieces)
    i = 0
    printProgressBar(i, number_of_dataframe_pieces, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    for dataframe_piece in list_of_dataframe_pieces:
        #name = dataframe_piece[0]
        file_path_for_tf_idf = dataframe_piece[1]
        file_path_for_idf = dataframe_piece[3]
        dataframe_piece_group = dataframe_piece[2]
        preprocessed_list = _preprocess_dataframe_for_tf_idf(dataframe_piece_group)
        DF = _calculate_df_for_all_words_in_document(preprocessed_list, dataframe_piece)
        tf_idf = _calculate_tf_idf_from_dataframe(preprocessed_list, DF, dataframe_piece)
        write_tf_idf_or_id_to_pkl_file(tf_idf, file_path_for_tf_idf)
        write_tf_idf_or_id_to_pkl_file(DF, file_path_for_idf)
        
        i += 1
        printProgressBar(i, number_of_dataframe_pieces, prefix = 'Progress:', suffix = 'Complete', autosize = True)
    
    print("TF-IDF model initialized successfully!")

def matching_score(k, query, business, source_df, source_name,
                   with_bidaf = False, with_bert = False,
                   bidaf_threshold = 0):
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    
    preprocessed_query = _preprocess_text_for_tf_idf(query)
    file_path = get_file_path_from_business_name_for_tf_idf(business, source_name)
    
    scores = {}
    document_text = []
    score = []
    tf_idf = read_tf_idf_or_id_from_pkl_file(file_path)
    tokens = word_tokenize(str(preprocessed_query))
    
    if tf_idf == None:
        return pd.DataFrame()
    
    for token in tokens:
        if token in tf_idf:
            token_tf_idfs_and_positions = tf_idf[token]
            for token_tf_idf_and_position in token_tf_idfs_and_positions:
                token_position = token_tf_idf_and_position[0]
                token_tf_idf = token_tf_idf_and_position[1]
                try:
                    scores[token_position] += token_tf_idf
                except:
                    scores[token_position] = token_tf_idf
    
    scores = sorted(scores.items(),
                    key=lambda x: x[1], reverse=True)
    
    for answer in scores[:k]:
        document_id = (int(answer[0]))
        temp_text = source_df.loc[document_id, "text"]
        if with_bidaf:
            temp_text = answer_with_bidaf(temp_text, query)
        if with_bert:
            temp_text = answer_with_bert(temp_text, query)
        document_text.append(temp_text)
        score.append(float(answer[1]))
    
    raw_data = {
        'answer':document_text,
        'similarity':score
    }
    answers = pd.DataFrame(raw_data, columns = ['answer', 'similarity'])
    
    return answers
