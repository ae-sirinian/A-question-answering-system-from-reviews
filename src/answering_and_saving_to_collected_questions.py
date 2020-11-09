#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:23:41 2020

@author: aram
"""
from query_answering import _answer_query_with_centroids
from query_answering import _answer_query_with_iwcs
from tf_idf import matching_score
from read_and_write_functions import write_dataframe_to_csv_file
from progress_bar import printProgressBar
import pandas as pd
import config
import time


def _compute_all_answers_from_list_of_questions_with_tf_idf(source_df,
                                                           source_name,
                                                           questions_list,
                                                           with_bidaf = False,
                                                           with_bert = False):
    question_list = []
    business_list = []
    execution_time_in_sec_list = []
    answer_list = []
    similarity_list = []
    number_of_questions = len(questions_list)
    
    printProgressBar(0, number_of_questions, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for i, question in enumerate(questions_list):
        num_of_answers = config.NUM_OF_ANSWERS_PER_QUESTION
        query = question[0]
        business = question[1]
        
        start_time = time.time()
        answ = matching_score(num_of_answers, query, business, source_df,
                              source_name, with_bidaf = with_bidaf,
                              with_bert = with_bert)
        end_time = time.time()
        execution_time_in_sec = end_time - start_time
        
        if (answ is not None) and (len(answ) != 0):
            length = 0
            if (len(answ) < num_of_answers):
                length = len(answ)
            else:
                length = num_of_answers
            question_list += [query for x in range(length)]
            business_list += [business for x in range(length)]
            execution_time_in_sec_list += [execution_time_in_sec for x in range(length)]
            answer_list += answ['answer'].tolist()
            similarity_list += answ['similarity'].tolist()
        
        printProgressBar(i + 1, number_of_questions, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)
    
    raw_data = {
        'question':question_list,
        'business':business_list,
        'answer':answer_list,
        'similarity':similarity_list,
        'execution_time_in_sec':execution_time_in_sec_list
    }
    questions_with_answers = pd.DataFrame(raw_data, columns = [
            'question', 'business', 'answer', 'similarity',
            'execution_time_in_sec'])
    
    return questions_with_answers

def compute_all_answers_from_yelp_and_write_to_file_with_tf_idf(tips_df,
                                                                reviews_df,
                                                                with_bidaf = False,
                                                                with_bert = False):
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    
    if with_bidaf:
        if len(tips_df) != 0:
            source_name = "tips_with_bidaf"
            print("Computing answers with tf idf from tips with BiDAF...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    tips_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf, with_bert = with_bert)
            print("Answers with tf idf from tips with BiDAF computed successfully!")
            
            print("Saving answers with tf idf from tips with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with tf idf from tips with BiDAF saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bidaf"
            print("Computing answers with tf idf from reviews with BiDAF...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf, with_bert = with_bert)
            print("Answers with tf idf from reviews with BiDAF computed successfully!")
            
            print("Saving answers with tf idf from reviews with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with tf idf from reviews with BiDAF saved successfully!")
    
    if with_bert:
        if len(tips_df) != 0:
            source_name = "tips_with_bert"
            print("Computing answers with tf idf from tips with BERT...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    tips_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert)
            print("Answers with tf idf from tips with BERT computed successfully!")
            
            print("Saving answers with tf idf from tips with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BERT_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with tf idf from tips with BERT saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bert"
            print("Computing answers with tf idf from reviews with BERT...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert)
            print("Answers with tf idf from reviews with BERT computed successfully!")
            
            print("Saving answers with tf idf from reviews with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BERT_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with tf idf from reviews with BERT saved successfully!")
    
    if (not with_bidaf) and (not with_bert):
        if len(tips_df) != 0:
            source_name = "tips"
            print("Computing answers with tf idf from tips...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    tips_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf, with_bert = with_bert)
            print("Answers with tf idf from tips computed successfully!")
            
            print("Saving answers with tf idf from tips...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_TF_IDF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with tf idf from tips saved successfully!")
        
        if len(reviews_df) != 0:
            source_name = "reviews"
            print("Computing answers with tf idf from reviews...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_tf_idf(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_bidaf = with_bidaf, with_bert = with_bert)
            print("Answers with tf idf from reviews computed successfully!")
            
            print("Saving answers with tf idf from reviews...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_TF_IDF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with tf idf from reviews saved successfully!")

def _compute_all_answers_from_list_of_questions_with_word2vec_centroid(source_df,
                                                                      questions_list,
                                                                      with_bidaf = False,
                                                                      with_bert = False,
                                                                      rearranging_after_bidaf = True,
                                                                      rearranging_after_bert = True):
    question_list = []
    business_list = []
    answer_list = []
    similarity_list = []
    execution_time_in_sec_list = []
    number_of_questions = len(questions_list)
    
    printProgressBar(0, number_of_questions, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for i, question in enumerate(questions_list):
        #TODO what if answ NONE?
        num_of_answers = config.NUM_OF_ANSWERS_PER_QUESTION
        query = question[0]
        business = question[1]
        
        start_time = time.time()
        answ = _answer_query_with_centroids(source_df, query, business,
                                            with_bidaf,
                                            with_bert,
                                            num_of_answers,
                                            rearranging_after_bidaf = rearranging_after_bidaf,
                                            rearranging_after_bert = rearranging_after_bert)
        end_time = time.time()
        execution_time_in_sec = end_time - start_time
        
        if answ is not None:
            length = 0
            if (len(answ) < num_of_answers):
                length = len(answ)
            else:
                length = num_of_answers
            question_list += [query for x in range(length)]
            business_list += [business for x in range(length)]
            execution_time_in_sec_list += [execution_time_in_sec for x in range(length)]
            answer_list += answ['answer'].tolist()
            similarity_list += answ['similarity'].tolist()
        
        printProgressBar(i + 1, number_of_questions, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)
    
    raw_data = {
        'question':question_list,
        'business':business_list,
        'answer':answer_list,
        'similarity':similarity_list,
        'execution_time_in_sec':execution_time_in_sec_list
    }
    
    questions_with_answers = pd.DataFrame(raw_data, columns = [
            'question', 'business', 'answer', 'similarity',
            'execution_time_in_sec'])
    
    return questions_with_answers

def compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(tips_df,
                                                                           reviews_df,
                                                                           with_bidaf = False,
                                                                           with_bert = False,
                                                                           rearranging_after_bidaf = True,
                                                                           rearranging_after_bert = True):
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    
    if with_bidaf and rearranging_after_bidaf:
        if len(tips_df) != 0:
            print("Computing answers with word2vec from tips with BiDAF...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    tips_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BiDAF computed successfully!")
            
            print("Saving answers with word2vec from tips with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BiDAF saved successfully!")
        if len(reviews_df) != 0:
            print("Computing answers with word2vec from reviews with BiDAF...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    reviews_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BiDAF computed successfully!")
            
            print("Saving answers with word2vec from reviews with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BiDAF saved successfully!")
    
    if with_bidaf and (not rearranging_after_bidaf):
        if len(tips_df) != 0:
            print("Computing answers with word2vec from tips with BiDAF without rearranging after BiDAF...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    tips_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BiDAF without rearranging after BiDAF computed successfully!")
            
            print("Saving answers with word2vec from tips with BiDAF without rearranging after BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BiDAF without rearranging after BiDAF saved successfully!")
        if len(reviews_df) != 0:
            print("Computing answers with word2vec from reviews with BiDAF without rearranging after BiDAF...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    reviews_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BiDAF without rearranging after BiDAF computed successfully!")
            
            print("Saving answers with word2vec from reviews with BiDAF without rearranging after BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BiDAF without rearranging after BiDAF saved successfully!")
    
    if with_bert and rearranging_after_bert:
        if len(tips_df) != 0:
            print("Computing answers with word2vec from tips with BERT...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    tips_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BERT computed successfully!")
            
            print("Saving answers with word2vec from tips with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BERT saved successfully!")
        if len(reviews_df) != 0:
            print("Computing answers with word2vec from reviews with BERT...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    reviews_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BERT computed successfully!")
            
            print("Saving answers with word2vec from reviews with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BERT saved successfully!")
    
    if with_bert and (not rearranging_after_bert):
        if len(tips_df) != 0:
            print("Computing answers with word2vec from tips with BERT without rearranging after BERT...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    tips_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BERT without rearranging after BERT computed successfully!")
            
            print("Saving answers with word2vec from tips with BERT without rearranging after BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BERT without rearranging after BERT saved successfully!")
        if len(reviews_df) != 0:
            print("Computing answers with word2vec from reviews with BERT without rearranging after BERT...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    reviews_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BERT without rearranging after BERT computed successfully!")
            
            print("Saving answers with word2vec from reviews with BERT without rearranging after BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BERT without rearranging after BERT saved successfully!")
    
    if (not with_bidaf) and (not with_bert):
        if len(tips_df) != 0:
            print("Computing answers with word2vec from tips...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    tips_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips computed successfully!")
            
            print("Saving answers with word2vec from tips...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_WV_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips saved successfully!")
        if len(reviews_df) != 0:
            print("Computing answers with word2vec from reviews...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_centroid(
                    reviews_df, config.questions_from_yelp,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews computed successfully!")
            
            print("Saving answers with word2vec from reviews...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_WV_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews saved successfully!")

def _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(source_df,
                                                                  source_name,
                                                                  questions_list,
                                                                  with_tf_idf_preprocessing,
                                                                  with_word2vec_preprocessing,
                                                                  with_bidaf = False,
                                                                  with_bert = False,
                                                                  rearranging_after_bidaf = True,
                                                                  rearranging_after_bert = True):
    question_list = []
    business_list = []
    answer_list = []
    similarity_list = []
    execution_time_in_sec_list = []
    number_of_questions = len(questions_list)
    
    printProgressBar(0, number_of_questions, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for i, question in enumerate(questions_list):
        num_of_answers = config.NUM_OF_ANSWERS_PER_QUESTION
        query = question[0]
        business = question[1]
        
        start_time = time.time()
        answ = _answer_query_with_iwcs(source_df, query, business, source_name,
                                       with_bidaf, with_bert, num_of_answers,
                                       with_tf_idf_preprocessing,
                                       with_word2vec_preprocessing,
                                       #bidaf_threshold = bidaf_threshold,
                                       rearranging_after_bidaf = rearranging_after_bidaf,
                                       rearranging_after_bert = rearranging_after_bert)
        end_time = time.time()
        execution_time_in_sec = end_time - start_time
        
        if answ is not None:
            length = 0
            if (len(answ) < num_of_answers):
                length = len(answ)
            else:
                length = num_of_answers
            question_list += [query for x in range(length)]
            business_list += [business for x in range(length)]
            execution_time_in_sec_list += [execution_time_in_sec for x in range(length)]
            answer_list += answ['answer'].tolist()
            similarity_list += answ['similarity'].tolist()
        
        printProgressBar(i + 1, number_of_questions, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)
    
    raw_data = {
        'question':question_list,
        'business':business_list,
        'answer':answer_list,
        'similarity':similarity_list,
        'execution_time_in_sec':execution_time_in_sec_list
    }
    
    questions_with_answers = pd.DataFrame(raw_data, columns = [
            'question', 'business', 'answer', 'similarity',
            'execution_time_in_sec'])
    
    return questions_with_answers

def compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(tips_df,
                                                                       reviews_df,
                                                                       with_tf_idf_preprocessing,
                                                                       with_word2vec_preprocessing,
                                                                       with_bidaf = False,
                                                                       with_bert = False,
                                                                       rearranging_after_bidaf = True,
                                                                       rearranging_after_bert = True):
    if (with_bidaf == True) and (with_bert == True):
        raise ValueError("Error: Can't answer query with centroids BERT and with BiDAF")
    
    if with_bidaf and rearranging_after_bidaf:
        if len(tips_df) != 0:
            source_name = "tips_with_bidaf"
            print("Computing answers with word2vec from tips with BiDAF...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    tips_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BiDAF computed successfully!")
            
            print("Saving answers with word2vec from tips with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BiDAF saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bidaf"
            print("Computing answers with word2vec from reviews with BiDAF...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BiDAF computed successfully!")
            
            print("Saving answers with word2vec from reviews with BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BiDAF saved successfully!")
    
    if with_bidaf and (not rearranging_after_bidaf):
        if len(tips_df) != 0:
            source_name = "tips_with_bidaf"
            print("Computing answers with word2vec from tips with BiDAF without rearranging after BiDAF...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    tips_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BiDAF without rearranging after BiDAF computed successfully!")
            
            print("Saving answers with word2vec from tips with BiDAF without rearranging after BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BiDAF without rearranging after BiDAF saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bidaf"
            print("Computing answers with word2vec from reviews with BiDAF without rearranging after BiDAF...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BiDAF without rearranging after BiDAF computed successfully!")
            
            print("Saving answers with word2vec from reviews with BiDAF without rearranging after BiDAF...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BiDAF without rearranging after BiDAF saved successfully!")
    
    if with_bert and rearranging_after_bert:
        if len(tips_df) != 0:
            source_name = "tips_with_bert"
            print("Computing answers with word2vec from tips with BERT...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    tips_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BERT computed successfully!")
            
            print("Saving answers with word2vec from tips with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BERT saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bert"
            print("Computing answers with word2vec from reviews with BERT...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BERT computed successfully!")
            
            print("Saving answers with word2vec from reviews with BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BERT saved successfully!")
    
    if with_bert and (not rearranging_after_bert):
        if len(tips_df) != 0:
            source_name = "tips_with_bert"
            print("Computing answers with word2vec from tips with BERT without rearranging after BERT...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    tips_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips with BERT without rearranging after BERT computed successfully!")
            
            print("Saving answers with word2vec from tips with BERT without rearranging after BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips with BERT without rearranging after BERT saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews_with_bert"
            print("Computing answers with word2vec from reviews with BERT without rearranging after BERT...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews with BERT without rearranging after BERT computed successfully!")
            
            print("Saving answers with word2vec from reviews with BERT without rearranging after BERT...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews with BERT without rearranging after BERT saved successfully!")
    
    if (not with_bidaf) and (not with_bert):
        if len(tips_df) != 0:
            source_name = "tips"
            print("Computing answers with word2vec from tips...")
            questions_with_answers_from_tips = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    tips_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from tips computed successfully!")
            
            print("Saving answers with word2vec from tips...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_TIPS_WITH_IWCS_FILE_PATH,
                    questions_with_answers_from_tips)
            print("Answers with word2vec from tips saved successfully!")
        if len(reviews_df) != 0:
            source_name = "reviews"
            print("Computing answers with word2vec from reviews...")
            questions_with_answers_from_reviews = _compute_all_answers_from_list_of_questions_with_word2vec_iwcs(
                    reviews_df, source_name, config.questions_from_yelp,
                    with_tf_idf_preprocessing, with_word2vec_preprocessing,
                    with_bidaf = with_bidaf,
                    with_bert = with_bert,
                    rearranging_after_bidaf = rearranging_after_bidaf,
                    rearranging_after_bert = rearranging_after_bert)
            print("Answers with word2vec from reviews computed successfully!")
            
            print("Saving answers with word2vec from reviews...")
            write_dataframe_to_csv_file(
                    config.ANSWERS_FROM_REVIEWS_WITH_IWCS_FILE_PATH,
                    questions_with_answers_from_reviews)
            print("Answers with word2vec from reviews saved successfully!")