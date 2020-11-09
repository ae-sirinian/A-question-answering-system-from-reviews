#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:57:20 2020

@author: aram
"""

from computing_centroids import compute_all_centroids_and_save_result_in_a_file
from compute_iwcs import compute_all_iwcs_and_save_result_in_a_file
from tf_idf import compute_tf_idf_model
from word_vectors_creating import create_word_vectors_with_word2vec
from allennlp.predictors.predictor import Predictor
from question_loading import load_questions_from_yelp
from read_and_write_functions import load_yelp_data
import pandas as pd
import config
import sys

tips_df = None
reviews_df = None
model_code_chosen = None
user_wants_to_use_preprocessing = True
with_tf_idf_preprocessing_compute_iwcs = True
with_word2vec_preprocessing_compute_iwcs = True

WV_CENTROID_WITH_TIPS_CODE_NUMBER = 1
WV_CENTROID_WITH_REVIEWS_CODE_NUMBER = 2
WV_CENTROID_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER = 3
WV_CENTROID_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER = 4
IWCS_WITH_TIPS_CODE_NUMBER = 5
IWCS_WITH_REVIEWS_CODE_NUMBER = 6
IWCS_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER = 7
IWCS_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER = 8
TF_IDF_WITH_TIPS_CODE_NUMBER = 9
TF_IDF_WITH_REVIEWS_CODE_NUMBER = 10
TF_IDF_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER = 11
TF_IDF_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER = 12

def print_all_available_models():
    print("All available models are:")
    print(str(WV_CENTROID_WITH_TIPS_CODE_NUMBER) + ") \t" + "WCS+Without_BiDAF with Yelp's Tips")
    print(str(WV_CENTROID_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WCS+Without_BiDAF with Yelp's Reviews")
    #print(str(WV_CENTROID_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "WV_Centroid+BiDAF with Tips or WV_Centroid+BERT with Tips (with or without rearranging after BiDAF or BERT)")
    print(str(WV_CENTROID_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "WCS+With_BiDAF with Yelp's Tips (with or without rearranging after BiDAF)")
    #print(str(WV_CENTROID_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WV_Centroid+BiDAF with Reviews or WV_Centroid+BERT with Reviews (with or without rearranging after BiDAF or BERT)")
    print(str(WV_CENTROID_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WCS+With_BiDAF with Yelp's Reviews (with or without rearranging after BiDAF)")
    print(str(IWCS_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Tips")
    print(str(IWCS_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Reviews")
    #print(str(IWCS_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+BiDAF with Tips or IWCS+BERT with Tips (with or without rearranging after BiDAF or BERT)")
    print(str(IWCS_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Tips (with or without rearranging after BiDAF)")
    #print(str(IWCS_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+BiDAF with Reviews or IWCS+BERT with Reviews(with or without rearranging after BiDAF or BERT)")
    print(str(IWCS_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Reviews (with or without rearranging after BiDAF)")
    print(str(TF_IDF_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Tips")
    print(str(TF_IDF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Reviews")
    #print(str(TF_IDF_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+BiDAF with Tips or TF-IDF+BERT with Tips")
    print(str(TF_IDF_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Tips")
    #print(str(TF_IDF_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+BiDAF with Reviews or TF-IDF+BERT with Reviews")
    print(str(TF_IDF_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Reviews")
    

def _init_model(model_code): 
    global tips_df
    global reviews_df
    
    tips_df = pd.DataFrame()
    reviews_df = pd.DataFrame()
    config.wv = create_word_vectors_with_word2vec()
    config.index2word_set = set(config.wv.index2word)
    config.predictor = Predictor.from_path(config.BIDAF_MODEL_FILE_PATH)
    load_questions_from_yelp()
    
    
    models_with_tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long = [
            WV_CENTROID_WITH_TIPS_CODE_NUMBER,
            IWCS_WITH_TIPS_CODE_NUMBER,
            TF_IDF_WITH_TIPS_CODE_NUMBER]
    
    models_with_reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long = [
            WV_CENTROID_WITH_REVIEWS_CODE_NUMBER,
            IWCS_WITH_REVIEWS_CODE_NUMBER,
            TF_IDF_WITH_REVIEWS_CODE_NUMBER]
    
    models_with_tips_with_rows_removed_if_text_length_too_short = [
            WV_CENTROID_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER,
            IWCS_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER,
            TF_IDF_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER]
    
    models_with_reviews_with_rows_removed_if_text_length_too_short = [
            WV_CENTROID_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER,
            IWCS_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER,
            TF_IDF_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER]
    
    if model_code in models_with_tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long:
        tips_df = load_yelp_data(config.TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH)
    
    if model_code in models_with_reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long:
        reviews_df = load_yelp_data(config.REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH)
    
    if model_code in models_with_tips_with_rows_removed_if_text_length_too_short:
        tips_df = load_yelp_data(config.TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH)
    
    if model_code in models_with_reviews_with_rows_removed_if_text_length_too_short:
        reviews_df = load_yelp_data(config.REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH)
    

def _compute_files_needed_for_model(model_code):
    global tips_df
    global reviews_df
    global user_wants_to_use_preprocessing
    global with_tf_idf_preprocessing_compute_iwcs
    global with_word2vec_preprocessing_compute_iwcs
    
    if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER):
        compute_tf_idf_model(tips_df, "tips")
    
    if (model_code == TF_IDF_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER):
        compute_tf_idf_model(tips_df, "tips_with_bidaf")
    
    if (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
        compute_tf_idf_model(reviews_df, "reviews")
    
    if (model_code == TF_IDF_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER):
        compute_tf_idf_model(reviews_df, "reviews_with_bidaf")
    
    if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER):
        compute_all_centroids_and_save_result_in_a_file(with_preprocessing = user_wants_to_use_preprocessing,
                                                        source_df = tips_df,
                                                        file_path = config.FILE_PATH_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_CENTROIDS_WITH_PREPROCESSING)
    
    if (model_code == WV_CENTROID_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER):
        compute_all_centroids_and_save_result_in_a_file(with_preprocessing = user_wants_to_use_preprocessing,
                                                        source_df = tips_df,
                                                        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING)
    
    if (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
        compute_all_centroids_and_save_result_in_a_file(with_preprocessing = user_wants_to_use_preprocessing,
                                                        source_df = reviews_df,
                                                        file_path = config.FILE_PATH_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_CENTROIDS_WITH_PREPROCESSING)
    
    if (model_code == WV_CENTROID_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER):
        compute_all_centroids_and_save_result_in_a_file(with_preprocessing = user_wants_to_use_preprocessing,
                                                        source_df = reviews_df,
                                                        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING)
    
    if (model_code == IWCS_WITH_TIPS_CODE_NUMBER):
        source_name = "tips"
        file_path = config.FILE_PATH_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_IWCS
        compute_all_iwcs_and_save_result_in_a_file(tips_df,
                                                   source_name,
                                                   file_path,
                                                   with_tf_idf_preprocessing_compute_iwcs,
                                                   with_word2vec_preprocessing_compute_iwcs)
    
    if (model_code == IWCS_BIDAF_OR_BERT_WITH_TIPS_CODE_NUMBER):
        source_name = "tips_with_bidaf"
        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        compute_all_iwcs_and_save_result_in_a_file(tips_df,
                                                   source_name,
                                                   file_path,
                                                   with_tf_idf_preprocessing_compute_iwcs,
                                                   with_word2vec_preprocessing_compute_iwcs)
    
    if (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
        source_name = "reviews"
        file_path = config.FILE_PATH_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_IWCS
        compute_all_iwcs_and_save_result_in_a_file(reviews_df,
                                                   source_name,
                                                   file_path,
                                                   with_tf_idf_preprocessing_compute_iwcs,
                                                   with_word2vec_preprocessing_compute_iwcs)
    
    if (model_code == IWCS_BIDAF_OR_BERT_WITH_REVIEWS_CODE_NUMBER):
        source_name = "reviews_with_bidaf"
        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        compute_all_iwcs_and_save_result_in_a_file(reviews_df,
                                                   source_name,
                                                   file_path,
                                                   with_tf_idf_preprocessing_compute_iwcs,
                                                   with_word2vec_preprocessing_compute_iwcs)

if (len(sys.argv) == 2):
    model_code_chosen = int(sys.argv[1])
else:
    print_all_available_models()
    model_code_chosen = int(input("which model you want to chosen: "))
_init_model(model_code_chosen)
_compute_files_needed_for_model(model_code_chosen)