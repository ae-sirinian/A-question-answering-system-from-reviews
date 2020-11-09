#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:57:20 2020

@author: aram
"""

from query_answering import _answer_query_with_centroids
from query_answering import _answer_query_with_iwcs
from tf_idf import matching_score
from read_and_write_functions import read_dataframe_from_csv_file
from read_and_write_functions import load_yelp_data
from word_vectors_creating import create_word_vectors_with_word2vec
from question_loading import load_questions_from_yelp
from allennlp.predictors.predictor import Predictor
import pandas as pd
import config
#import sys
#from transformers import BertForQuestionAnswering
#from transformers import BertTokenizer

tips_df = None
reviews_df = None
model_code_chosen = None
with_tf_idf_preprocessing_compute_iwcs = True
with_word2vec_preprocessing_compute_iwcs = True

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

IWCS_WITH_TIPS_CODE_NUMBER = 7
IWCS_WITH_REVIEWS_CODE_NUMBER = 8
IWCS_BIDAF_WITH_TIPS_CODE_NUMBER = 9
IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER = 10
IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 11
IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF = 12
#IWCS_BERT_WITH_TIPS_CODE_NUMBER = 17
#IWCS_BERT_WITH_REVIEWS_CODE_NUMBER = 18
#IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 19
#IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT = 20

TF_IDF_WITH_TIPS_CODE_NUMBER = 13
TF_IDF_WITH_REVIEWS_CODE_NUMBER = 14
TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER = 15
TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER = 16
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
    #print(str(WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "WV_Centroid+BERT with Tips without rearranging after BERT")
    #print(str(WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "WV_Centroid+BERT with Reviews without rearranging after BERT")
    
    
    print(str(IWCS_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Tips")
    print(str(IWCS_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+Without_BiDAF with Yelp's Reviews")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Tips")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+With_BiDAF with Yelp's Reviews")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "IWCS+With_BiDAF with Yelp's Tips without rearranging after BiDAF")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "IWCS+With_BiDAF with Yelp's Reviews without rearranging after BiDAF")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "IWCS+BERT with Tips")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "IWCS+BERT with Reviews")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "IWCS+BERT with Tips without rearranging after BERT")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "IWCS+BERT with Reviews without rearranging after BERT")
    
    
    print(str(TF_IDF_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Tips")
    print(str(TF_IDF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+Without_BiDAF with Yelp's Reviews")
    print(str(TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Tips")
    print(str(TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+With_BiDAF with Yelp's Reviews")
    #print(str(TF_IDF_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "TF-IDF+BERT with Tips")
    #print(str(TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "TF-IDF+BERT with Reviews")

def _ask_user_a_yes_or_no_question(question):
    while True:
        answer = (input(question + "[y/n]: ")).lower()
        
        if (answer == "y" or answer == "yes" or answer == "yeah"):
            return True
        elif (answer == "n" or answer == "no" or answer == "nah"):
            return False
        else:
            continue

def _init_model(model_code): 
    global tips_df
    global reviews_df
    
    tips_df = pd.DataFrame()
    reviews_df = pd.DataFrame()
    config.wv = create_word_vectors_with_word2vec()
    config.index2word_set = set(config.wv.index2word)
    config.predictor = Predictor.from_path(config.BIDAF_MODEL_FILE_PATH)#TODO is it needed here?
    load_questions_from_yelp()#TODO is it needed?
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
    
    if (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = config.REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        reviews_df = load_yelp_data(file_path)
    
    """
    if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = config.TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH
        tips_df = load_yelp_data(file_path)
        config.bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        config.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
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

def _print_answers(answer):
    answers_text = answer["answer"].tolist()
    answers_similarity = answer["similarity"].tolist()
    for i in range(len(answers_text)):
        print(str(i) + ") " + answers_text[i], "\t :", answers_similarity[i])

def _answer_user_questions(model_code):
    global tips_df
    global reviews_df
    global with_tf_idf_preprocessing_compute_iwcs
    global with_word2vec_preprocessing_compute_iwcs
    
    while _ask_user_a_yes_or_no_question("Do you want to ask a question?"):
        query = (input("Query: "))
        business = (input("Business: "))
        num_of_answers = int(input("Max number of best answers: "))
        
        if (model_code == IWCS_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = True,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = True)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = True,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag,
                                              num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = True)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = True,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = True)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = True,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = True)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = False
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            _print_answers(answers)
        
        if (model_code == TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER):
            source_name = "tips_with_bidaf"
            source_df = tips_df
            bidaf_flag = True
            bert_flag = False
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            _print_answers(answers)
        
        """
        if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER):
            source_name = "tips_with_bert"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = True
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            _print_answers(answers)
        """
        
        if (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = False
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            _print_answers(answers)
        
        if (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews_with_bidaf"
            source_df = reviews_df
            bidaf_flag = True
            bert_flag = False
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            _print_answers(answers)
        
        """
        if (model_code == TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER):
            source_name = "reviews_with_bert"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = True
            
            answers = matching_score(num_of_answers, query, business,
                                                 source_df, source_name,
                                                 with_bidaf = bidaf_flag,
                                                 with_bert = bert_flag)
            
            if answers.size == 0:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_centroids(source_df, query,
                                                            business, bidaf_flag,
                                                            bert_flag,
                                                            num_of_answers,
                                                            rearranging_after_bidaf = False,
                                                            rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag, num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
            source_name = "tips"
            source_df = tips_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag, num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """
        
        if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = True
            bert_flag = False
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag, num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        
        """
        if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
            source_name = "reviews"
            source_df = reviews_df
            bidaf_flag = False
            bert_flag = True
            
            answers = _answer_query_with_iwcs(source_df, query, business,
                                              source_name, bidaf_flag,
                                              bert_flag, num_of_answers,
                                              with_tf_idf_preprocessing = with_tf_idf_preprocessing_compute_iwcs,
                                              with_word2vec_preprocessing = with_word2vec_preprocessing_compute_iwcs,
                                              rearranging_after_bidaf = False,
                                              rearranging_after_bert = False)
            
            if answers is None:
                print("Your query's business is not supported or doesn't exist in Yelp's database!!!")
                print("Please try again with another business name.")
                continue
            
            answers = answers[answers['similarity'].notna()]
            
            _print_answers(answers)
        """


print_all_available_models()
model_code_chosen = input("which model you want to chosen: ")
for code_chosen in model_code_chosen.split(", "):
    model_code_chosen = int(code_chosen)
    _init_model(model_code_chosen)
    _answer_user_questions(model_code_chosen)
