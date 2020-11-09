#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:28:36 2020

@author: aram
"""
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from computing_centroids import compute_centroid_from_sentence
from answering_with_bidaf import answer_with_bidaf
from answering_with_bert import answer_with_bert
from compute_iwcs import compute_iwcs_from_sentence


def _is_key_in_group(key, group):
    if key in group.groups.keys():
        return True
    else:
        return False

def _compute_cosine_similarity_of_queries_with_answers(queries_wv, list_of_answer_wv, list_of_answer_text):
    text_list = []
    sim_list = []
    
    for text, centroid in zip(list_of_answer_text, list_of_answer_wv):
        sim = 1 - cosine(queries_wv, centroid)
        text_list.append(text)
        sim_list.append(sim)
    
    return text_list, sim_list

def _answer_query_with_centroids(
        source_df, query, business, with_bidaf, with_bert,  num_of_answers,
        rearranging_after_bidaf = True,
        rearranging_after_bert = True):
	
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    #t_start0 = process_time()################################
    #t_start = process_time()################################
    text_list = []
    sim_list = []
    query_centroid = compute_centroid_from_sentence(query)
    source_df_group = source_df.groupby("name")
    source_centroids = []
    #t_stop = process_time()###################################
    #print("Elapsed time area 1:", t_stop-t_start, "sec")######################
    #=========================================================================
    #t_start = process_time()################################
    if "centroid" not in source_df.columns:
        print("Error: Can't answer query without centroids!")
        sys.exit()
    if not(_is_key_in_group(business, source_df_group)):
        return
    
    specific_group = source_df_group.get_group(business)
    specific_group = specific_group[specific_group["centroid"].notna()]
    """
    df = _parallelize_dataframe(specific_group,
                                _compute_cosine_similarity_of_queries_with_source_with_centroids,
                                n_cores = number_of_cores)
    """
    
    for x in specific_group["centroid"]:
        source_centroids.append(np.fromstring(x[1:-1], sep=' '))
        
    text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
            query_centroid, source_centroids,
            specific_group["text"].values.tolist())
    
    df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
    df = df.sort_values(by=['similarity'], ascending = False)
    df = df.iloc[:num_of_answers]
    #t_stop = process_time()###################################
    #print("Elapsed time area 2:", t_stop-t_start, "sec")######################
    #=========================================================================
    #t_start = process_time()################################
    #TODO bidaf_threshold not needed
    
    if with_bidaf:
        if (rearranging_after_bidaf == True):
            text_list = df["answer"].tolist()
            
            new_answers_text = [answer_with_bidaf(passage, query) for passage in text_list]
            new_answers_wv = [compute_centroid_from_sentence(passage) for passage in new_answers_text]
            
            text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
                    query_centroid, new_answers_wv, new_answers_text)
            
            df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
            df = df.sort_values(by=['similarity'], ascending = False)
            df = df.iloc[:num_of_answers]
        else:
            text_list = df["answer"].tolist()
            sim_list =  df["similarity"].tolist()
            new_answers_text = [answer_with_bidaf(passage, query) for passage in text_list]
            df = pd.DataFrame({'answer': new_answers_text, 'similarity': sim_list})
    
    if with_bert:
        if (rearranging_after_bert == True):
            text_list = df["answer"].tolist()
            
            new_answers_text = [answer_with_bert(passage, query) for passage in text_list]
            new_answers_wv = [compute_centroid_from_sentence(passage) for passage in new_answers_text]
            
            text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
                    query_centroid, new_answers_wv, new_answers_text)
            
            df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
            df = df.sort_values(by=['similarity'], ascending = False)
            df = df.iloc[:num_of_answers]
        else:
            text_list = df["answer"].tolist()
            sim_list =  df["similarity"].tolist()
            new_answers_text = [answer_with_bert(passage, query) for passage in text_list]
            df = pd.DataFrame({'answer': new_answers_text, 'similarity': sim_list})
    #t_stop = process_time()###################################
    #print("Elapsed time area 3:", t_stop-t_start, "sec")######################
    #t_stop0 = process_time()###################################
    #print("Elapsed time area total:", t_stop0-t_start0, "sec")######################
    
    return df

def _answer_query_with_iwcs(
        source_df, query, business, source_name, with_bidaf, with_bert,
        num_of_answers, with_tf_idf_preprocessing, with_word2vec_preprocessing,
        rearranging_after_bidaf = True,
        rearranging_after_bert = True):
	
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    #start_time = process_time()######################################################
    text_list = []
    sim_list = []
    
    query_iwcs = compute_iwcs_from_sentence(source_df, query, business, source_name,
                                            with_tf_idf_preprocessing = with_tf_idf_preprocessing,
                                            with_word2vec_preprocessing = with_word2vec_preprocessing)
    
    #end_time = process_time()#######################################################
    #computation_time_in_sec = end_time - start_time################################
    #print("Query's IWCS computation time in (sec) \t: ", computation_time_in_sec)##########################
    #==================================================================================================
    #start_time = process_time()######################################################
    source_df_group = source_df.groupby("name")
    source_iwcs = []
    
    if "iwcs" not in source_df.columns:
        print("Error: Can't answer query without iwcs!")
        sys.exit()
    if not(_is_key_in_group(business, source_df_group)):
        return
    
    specific_group = source_df_group.get_group(business)
    specific_group = specific_group[specific_group["iwcs"].notna()]
    """
    df = _parallelize_dataframe(specific_group,
                                _compute_cosine_similarity_of_queries_with_iwcs,
                                n_cores = number_of_cores)
    """
    for x in specific_group["iwcs"]:
        source_iwcs.append(np.fromstring(x[1:-1], sep=' '))
    
    text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
            query_iwcs, source_iwcs,
            specific_group["text"].values.tolist())
    
    df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
    df = df.sort_values(by=['similarity'], ascending = False)
    df = df.iloc[:num_of_answers]
    #TODO bidaf_threshold not needed
    
    #end_time = process_time()#######################################################
    #computation_time_in_sec = end_time - start_time################################
    #print("Other staff computation time in (sec) \t: ", computation_time_in_sec)##########################
    #====================================================================================================
    #start_time = process_time()######################################################
    if with_bidaf:
        if (rearranging_after_bidaf == True):
            text_list = df["answer"].tolist()
            
            new_answers_text = [answer_with_bidaf(passage, query) for passage in text_list]
            new_answers_wv = [compute_iwcs_from_sentence(source_df,
                                                         passage,
                                                         business,
                                                         source_name,
                                                         with_tf_idf_preprocessing = with_tf_idf_preprocessing,
                                                         with_word2vec_preprocessing = with_word2vec_preprocessing) for passage in new_answers_text]
            
            text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
                    query_iwcs, new_answers_wv, new_answers_text)
            
            df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
            df = df.sort_values(by=['similarity'], ascending = False)
            df = df.iloc[:num_of_answers]
        else:
            text_list = df["answer"].tolist()
            sim_list =  df["similarity"].tolist()
            new_answers_text = [answer_with_bidaf(passage, query) for passage in text_list]
            df = pd.DataFrame({'answer': new_answers_text, 'similarity': sim_list})
    
    if with_bert:
        if (rearranging_after_bert == True):
            text_list = df["answer"].tolist()
            
            new_answers_text = [answer_with_bert(passage, query) for passage in text_list]
            new_answers_wv = [compute_iwcs_from_sentence(source_df,
                                                         passage,
                                                         business,
                                                         source_name,
                                                         with_tf_idf_preprocessing = with_tf_idf_preprocessing,
                                                         with_word2vec_preprocessing = with_word2vec_preprocessing) for passage in new_answers_text]
            
            text_list, sim_list = _compute_cosine_similarity_of_queries_with_answers(
                    query_iwcs, new_answers_wv, new_answers_text)
            
            df = pd.DataFrame({'answer': text_list, 'similarity': sim_list})
            df = df.sort_values(by=['similarity'], ascending = False)
            df = df.iloc[:num_of_answers]
        else:
            text_list = df["answer"].tolist()
            sim_list =  df["similarity"].tolist()
            new_answers_text = [answer_with_bert(passage, query) for passage in text_list]
            df = pd.DataFrame({'answer': new_answers_text, 'similarity': sim_list})
    #end_time = process_time()#######################################################
    #computation_time_in_sec = end_time - start_time################################
    #print("BiDAF's computation time in (sec) \t: ", computation_time_in_sec)##########################
    return df
