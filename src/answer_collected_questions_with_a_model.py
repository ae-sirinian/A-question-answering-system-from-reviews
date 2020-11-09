#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:57:20 2020

@author: aram
"""

from answering_and_saving_to_collected_questions import compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid
from answering_and_saving_to_collected_questions import compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs
from answering_and_saving_to_collected_questions import compute_all_answers_from_yelp_and_write_to_file_with_tf_idf
from read_and_write_functions import read_dataframe_from_csv_file
from read_and_write_functions import load_yelp_data
from word_vectors_creating import create_word_vectors_with_word2vec
from question_loading import load_questions_from_yelp
from allennlp.predictors.predictor import Predictor
import pandas as pd
import config
import sys
#from transformers import BertForQuestionAnswering
#from transformers import BertTokenizer

tips_df = None
reviews_df = None
model_code_chosen = None

WV_CENTROID_WITH_TIPS_CODE_NUMBER = 1
WV_CENTROID_WITH_REVIEWS_CODE_NUMBER = 2
WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER = 3
WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER = 4
WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 5
WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 6
#WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER = 7
#WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER = 8
#WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 9
#WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 10
IWCS_WITH_TIPS_CODE_NUMBER = 11
IWCS_WITH_REVIEWS_CODE_NUMBER = 12
IWCS_BIDAF_WITH_TIPS_CODE_NUMBER = 13
IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER = 14
IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 15
IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 16
#IWCS_BERT_WITH_TIPS_CODE_NUMBER = 17
#IWCS_BERT_WITH_REVIEWS_CODE_NUMBER = 18
#IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 19
#IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 20
TF_IDF_WITH_TIPS_CODE_NUMBER = 21
TF_IDF_WITH_REVIEWS_CODE_NUMBER = 22
TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER = 23
TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER = 24
#TF_IDF_BERT_WITH_TIPS_CODE_NUMBER = 25
#TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER = 26


def print_all_available_models():
    print("All available models are:")
    print(str(WV_CENTROID_WITH_TIPS_CODE_NUMBER) + ") \t" + "WCS+Without_BiDAF with Yelp's Tips")
    print(str(WV_CENTROID_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WCS+Without_BiDAF with Yelp's Reviews")
    print(str(WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "WCS+With_BiDAF with Yelp's Tips")
    print(str(WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WCS+With_BiDAF with Yelp's Reviews")
    print(str(WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "WCS+With_BiDAF with Yelp's Tips without rearranging after BiDAF")
    print(str(WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "WCS+With_BiDAF with Yelp's Reviews without rearranging after BiDAF")
    #print(str(WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "WV_Centroid+BERT with Tips")
    #print(str(WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "WV_Centroid+BERT with Reviews")
    #print(str(WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "WV_Centroid+BERT with Tips without rearranging after BiDAF")
    #print(str(WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "WV_Centroid+BERT with Reviews without rearranging after BiDAF")
    
    print(str(IWCS_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Tips")
    print(str(IWCS_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Reviews")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Tips")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Reviews")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "IWCS+With_BiDAF with Yelp's Tips without rearranging after BiDAF")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "IWCS+With_BiDAF with Yelp's Reviews without rearranging after BiDAF")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+BERT with Tips")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+BERT with Reviews")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "IWCS+BERT with Tips without rearranging after BiDAF")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "IWCS+BERT with Reviews without rearranging after BiDAF")
    
    print(str(TF_IDF_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Tips")
    print(str(TF_IDF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Reviews")
    print(str(TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Tips")
    print(str(TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Reviews")
    #print(str(TF_IDF_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+BERT with Tips")
    #print(str(TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+BERT with Reviews")


def _init_model(model_code): 
    global tips_df
    global reviews_df
    
    tips_df = pd.DataFrame()
    reviews_df = pd.DataFrame()
    config.wv = create_word_vectors_with_word2vec()
    config.index2word_set = set(config.wv.index2word)
    config.predictor = Predictor.from_path(config.BIDAF_MODEL_FILE_PATH)
    load_questions_from_yelp()
    file_path = None
    
    
    if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER):
        file_path = config.TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH
        tips_df = load_yelp_data(file_path)
    
    if (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH
        reviews_df = load_yelp_data(file_path)
    
    if (model_code == TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = config.TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        tips_df = load_yelp_data(file_path)
    
    """
    if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = config.TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        tips_df = load_yelp_data(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """
    
    if (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        reviews_df = load_yelp_data(file_path)
    
    """
    if (model_code == TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        reviews_df = load_yelp_data(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """
    
    if (model_code == IWCS_WITH_TIPS_CODE_NUMBER):
        file_path = config.FILE_PATH_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_IWCS
        tips_df = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER) or (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        tips_df = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER) or (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        tips_df = read_dataframe_from_csv_file(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """
    
    if (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.FILE_PATH_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_IWCS
        reviews_df = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER) or (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        reviews_df = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER) or (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_IWCS
        reviews_df = read_dataframe_from_csv_file(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """
    
    if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER):
        file_path = config.FILE_PATH_FOR_TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_CENTROIDS_WITH_PREPROCESSING
        tips_df = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER) or (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING
        tips_df = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER) or (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = config.FILE_PATH_FOR_TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING
        tips_df = read_dataframe_from_csv_file(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """
    
    if (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.FILE_PATH_FOR_REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_WITH_CENTROIDS_WITH_PREPROCESSING
        reviews_df = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER) or (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING
        reviews_df = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER) or (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = config.FILE_PATH_FOR_REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_WITH_CENTROIDS_WITH_PREPROCESSING
        reviews_df = read_dataframe_from_csv_file(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    """

def _compute_all_answers_from_yelp_and_write_to_file(model_code):
    bidaf_flag = False
    with_tf_idf_preprocessing_compute_iwcs = True
    with_word2vec_preprocessing_compute_iwcs = True
        
    if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER) or (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = False
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
        
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER) or (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = True
        bert_flag = False
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER) or (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = True
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    """
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) or (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        bidaf_flag = True
        bert_flag = False
        with_rearranging_after_bidaf = False
        with_rearranging_after_bert = False
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) or (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        bidaf_flag = False
        bert_flag = True
        with_rearranging_after_bidaf = False
        with_rearranging_after_bert = False
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_centroid(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    """
    
    if (model_code == IWCS_WITH_TIPS_CODE_NUMBER) or (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = False
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(
                tips_df, reviews_df, with_tf_idf_preprocessing_compute_iwcs,
                with_word2vec_preprocessing_compute_iwcs,
                bidaf_flag, bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
        
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER) or (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = True
        bert_flag = False
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(
                tips_df, reviews_df, with_tf_idf_preprocessing_compute_iwcs,
                with_word2vec_preprocessing_compute_iwcs,
                bidaf_flag, bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    
    """
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER) or (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = True
        with_rearranging_after_bidaf = True
        with_rearranging_after_bert = True
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(
                tips_df, reviews_df, with_tf_idf_preprocessing_compute_iwcs,
                with_word2vec_preprocessing_compute_iwcs,
                bidaf_flag, bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    """
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) or (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        bidaf_flag = True
        bert_flag = False
        with_rearranging_after_bidaf = False
        with_rearranging_after_bert = False
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(
                tips_df, reviews_df, with_tf_idf_preprocessing_compute_iwcs,
                with_word2vec_preprocessing_compute_iwcs,
                bidaf_flag, bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    
    """
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) or (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        bidaf_flag = False
        bert_flag = True
        with_rearranging_after_bidaf = False
        with_rearranging_after_bert = False
        compute_all_answers_from_yelp_and_write_to_file_with_word2vec_iwcs(
                tips_df, reviews_df, with_tf_idf_preprocessing_compute_iwcs,
                with_word2vec_preprocessing_compute_iwcs,
                bidaf_flag, bert_flag,
                rearranging_after_bidaf = with_rearranging_after_bidaf,
                rearranging_after_bert = with_rearranging_after_bert)
    """
    
    if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER) or (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = False
        compute_all_answers_from_yelp_and_write_to_file_with_tf_idf(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag)
        
    if (model_code == TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER) or (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = True
        bert_flag = False
        compute_all_answers_from_yelp_and_write_to_file_with_tf_idf(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag)
    
    """
    if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER) or (model_code == TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER):
        bidaf_flag = False
        bert_flag = True
        compute_all_answers_from_yelp_and_write_to_file_with_tf_idf(
                tips_df, reviews_df,
                with_bidaf = bidaf_flag, with_bert = bert_flag)
    """


if (len(sys.argv) == 2):
    model_code_chosen = int(sys.argv[1])
else:
    print_all_available_models()
    model_code_chosen = int(input("which model you want to chosen: "))
_init_model(model_code_chosen)
_compute_all_answers_from_yelp_and_write_to_file(model_code_chosen)