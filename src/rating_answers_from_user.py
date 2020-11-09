from math import isnan
import pandas as pd
import sys
import config


list_of_question_for_user_to_answer = []

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

def read_dataframe_from_csv_file(file_path):
    try:    
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("File " + file_path + " cannot be found.")
        sys.exit()

temp_dictionary_of_QBA_and_rating = {}
choosen_dataframe = pd.DataFrame()

ANSWERS_SCRAPPED_FROM_YELP_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/answers_only_from_first_question_pages.csv"

ANSWERS_FROM_REVIEWS_WITH_IWCS_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_iwcs.csv"
ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_iwcs_and_bidaf.csv"
ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_iwcs_and_bidaf_without_rearranging_after_bidaf.csv"
"""
ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_iwcs_and_bert.csv"
ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_iwcs_and_bert_without_rearranging_after_bert.csv"
"""

ANSWERS_FROM_REVIEWS_WITH_TF_IDF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_tf_idf.csv"
ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_tf_idf_and_bidaf.csv"
"""
ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_tf_idf_and_bert.csv"
"""

ANSWERS_FROM_REVIEWS_WITH_WV_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_wv.csv"
ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_wv_and_bidaf.csv"
ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_wv_and_bidaf_without_rearranging_after_bidaf.csv"
"""
ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_wv_and_bert.csv"
ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_reviews_with_wv_and_bert_without_rearranging_after_bert.csv"
"""

ANSWERS_FROM_TIPS_WITH_IWCS_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_iwcs.csv"
ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_iwcs_and_bidaf.csv"
ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_iwcs_and_bidaf_without_rearranging_after_bidaf.csv"
"""
ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_iwcs_and_bert.csv"
ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_iwcs_and_bert_without_rearranging_after_bert.csv"
"""

ANSWERS_FROM_TIPS_WITH_TF_IDF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_tf_idf.csv"
ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_tf_idf_and_bidaf.csv"
"""
ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_tf_idf_and_bert.csv"
"""

ANSWERS_FROM_TIPS_WITH_WV_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_wv.csv"
ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_wv_and_bidaf.csv"
ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_wv_and_bidaf_without_rearranging_after_bidaf.csv"
"""
ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_wv_and_bert.csv"
ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH = "../data/User_rating_of_answers/Answers_from_tips_with_wv_and_bert_without_rearranging_after_bert.csv"
"""


Answers_scrapped_from_yelp = read_dataframe_from_csv_file(ANSWERS_SCRAPPED_FROM_YELP_WITH_WVC_FILE_PATH)
#Answers_scrapped_from_yelp = Answers_scrapped_from_yelp.rename(columns = {"answer": "answer", 
#                                  "business_name": "business", 
#                                  "helpful": "similarity", "question": "question"})
#clist = ['question', 'business', 'answer', 'similarity']
#Answers_scrapped_from_yelp = Answers_scrapped_from_yelp[clist]
#Answers_scrapped_from_yelp = preprocess_for_answers_scrapped_from_yelp(Answers_scrapped_from_yelp)

Answers_from_reviews_with_iwcs = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_IWCS_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_iwcs_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_iwcs_and_bidaf_without_rearranging_after_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_reviews_with_iwcs_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_iwcs_and_bert_without_rearranging_after_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH)
"""

Answers_from_reviews_with_tf_idf = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_TF_IDF_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_tf_idf_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_reviews_with_tf_idf_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH)
"""

Answers_from_reviews_with_wv = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_WV_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_wv_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_wv_and_bidaf_without_rearranging_after_bidaf = read_dataframe_from_csv_file(
     ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_reviews_with_wv_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH)
Answers_from_reviews_with_wv_and_bert_without_rearranging_after_bert = read_dataframe_from_csv_file(
     ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH)
"""

Answers_from_tips_with_iwcs = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_IWCS_WITH_WVC_FILE_PATH)
Answers_from_tips_with_iwcs_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH)
Answers_from_tips_with_iwcs_and_bidaf_without_rearranging_after_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_tips_with_iwcs_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH)
Answers_from_tips_with_iwcs_and_bert_without_rearranging_after_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH)
"""

Answers_from_tips_with_tf_idf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_TF_IDF_WITH_WVC_FILE_PATH)
Answers_from_tips_with_tf_idf_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_tips_with_tf_idf_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH)
"""

Answers_from_tips_with_wv = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_WV_WITH_WVC_FILE_PATH)
Answers_from_tips_with_wv_and_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH)
Answers_from_tips_with_wv_and_bidaf_without_rearranging_after_bidaf = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH)
"""
Answers_from_tips_with_wv_and_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH)
Answers_from_tips_with_wv_and_bert_without_rearranging_after_bert = read_dataframe_from_csv_file(
    ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH)
"""


def print_all_available_files_to_rate():
    print("All available files to rate are:")
    print(str(WV_CENTROID_WITH_TIPS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(WV_CENTROID_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "B??????????")
    print(str(WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "B??????????")
    print(str(WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "B??????????")
    print(str(WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "B??????????")
    #print(str(WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "Answers from WV_Centroid+BERT with Tips")
    #print(str(WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "Answers from WV_Centroid+BERT with Reviews")
    #print(str(WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "Answers from WV_Centroid+BERT with Tips without rearranging after BiDAF")
    #print(str(WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "Answers from WV_Centroid+BERT with Reviews without rearranging after BiDAF")
    
    print(str(IWCS_WITH_TIPS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(IWCS_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "B??????????")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "B??????????")
    print(str(IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "B??????????")
    print(str(IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF) + ") \t" + "B??????????")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "Answers from IWCS+BERT with Tips")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "Answers from IWCS+BERT with Reviews")
    #print(str(IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "Answers from IWCS+BERT with Tips without rearranging after BiDAF")
    #print(str(IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT) + ") \t" + "Answers from IWCS+BERT with Reviews without rearranging after BiDAF")
    
    print(str(TF_IDF_WITH_TIPS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(TF_IDF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "??????????")
    print(str(TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER) + ") \t" + "B??????????")
    print(str(TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "B??????????")
    #print(str(TF_IDF_BERT_WITH_TIPS_CODE_NUMBER) + ") \t" + "Answers from TF-IDF+BERT with Tips")
    #print(str(TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER) + ") \t" + "Answers from TF-IDF+BERT with Reviews")

def _init_model(model_code):
    global choosen_dataframe
    
    if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """
    
    if (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """
    
    if (model_code == IWCS_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """
    
    if (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """
    
    if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """
    
    if (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER):preprocess_for_answers_scrapped_from_yelp
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    
    if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
        choosen_dataframe = read_dataframe_from_csv_file(file_path)
    """

def _clear_file_contents(file_path):
    f = open(file_path, "w+")
    f.close()

def write_dataframe_to_csv_file(file_path, dataframe):
    _clear_file_contents(file_path)
    
    with open(file_path, 'a') as f:
        dataframe.to_csv(f)

def preprocess_for_answers_scrapped_from_yelp(dataframe):
    temp_list_of_question = []
    temp_list_of_businesses = []
    temp_list_of_answer = []
    temp_list_of_similarity = []
    
    for index, row in dataframe.iterrows():
        temp_list_of_question.append(row['question'])
        
        temp_business_name = row['business']
        if " - Temporarily Closed" in temp_business_name:
            temp_business_name = temp_business_name.replace(" - Temporarily Closed", "")
        temp_list_of_businesses.append(temp_business_name)
        
        temp_list_of_answer.append(row['answer'])
        temp_list_of_similarity.append(row['similarity'])
    
    raw_data = {'question':temp_list_of_question,
                'business':temp_list_of_businesses,
                'answer':temp_list_of_answer,
                'similarity':temp_list_of_similarity
               }
    
    dataframe = pd.DataFrame(raw_data, columns = ['question', 'business', 'answer', 'similarity'])
    
    return dataframe

def _ask_user_a_yes_or_no_question(question):
    while True:
        answer = (input(question + "[y/n]: ")).lower()
        
        if (answer == "y" or answer == "yes" or answer == "yeah"):
            return True
        elif (answer == "n" or answer == "no" or answer == "nah"):
            return False
        else:
            continue
        
def init_temp_dictionary_of_QBA_and_rating():
    global temp_dictionary_of_QBA_and_rating
    collection_of_all_answers = []
    """
    collection_of_all_answers.append(Answers_scrapped_from_yelp)
    """
    
    collection_of_all_answers.append(Answers_from_reviews_with_iwcs)
    collection_of_all_answers.append(Answers_from_reviews_with_iwcs_and_bidaf)
    collection_of_all_answers.append(Answers_from_reviews_with_iwcs_and_bidaf_without_rearranging_after_bidaf)
    """
    collection_of_all_answers.append(Answers_from_reviews_with_iwcs_and_bert)
    collection_of_all_answers.append(Answers_from_reviews_with_iwcs_and_bert_without_rearranging_after_bert)
    """
    
    collection_of_all_answers.append(Answers_from_reviews_with_tf_idf)
    collection_of_all_answers.append(Answers_from_reviews_with_tf_idf_and_bidaf)
    """
    collection_of_all_answers.append(Answers_from_reviews_with_tf_idf_and_bert)
    """
    
    collection_of_all_answers.append(Answers_from_reviews_with_wv)
    collection_of_all_answers.append(Answers_from_reviews_with_wv_and_bidaf)
    collection_of_all_answers.append(Answers_from_reviews_with_wv_and_bidaf_without_rearranging_after_bidaf)
    """
    collection_of_all_answers.append(Answers_from_reviews_with_wv_and_bert)
    collection_of_all_answers.append(Answers_from_reviews_with_wv_and_bert_without_rearranging_after_bert)
    """
    
    collection_of_all_answers.append(Answers_from_tips_with_iwcs)
    collection_of_all_answers.append(Answers_from_tips_with_iwcs_and_bidaf)
    collection_of_all_answers.append(Answers_from_tips_with_iwcs_and_bidaf_without_rearranging_after_bidaf)
    """
    collection_of_all_answers.append(Answers_from_tips_with_iwcs_and_bert)
    collection_of_all_answers.append(Answers_from_tips_with_iwcs_and_bert_without_rearranging_after_bert)
    """
    
    collection_of_all_answers.append(Answers_from_tips_with_tf_idf)
    collection_of_all_answers.append(Answers_from_tips_with_tf_idf_and_bidaf)
    """
    collection_of_all_answers.append(Answers_from_tips_with_tf_idf_and_bert)
    """
    
    collection_of_all_answers.append(Answers_from_tips_with_wv)
    collection_of_all_answers.append(Answers_from_tips_with_wv_and_bidaf)
    collection_of_all_answers.append(Answers_from_tips_with_wv_and_bidaf_without_rearranging_after_bidaf)
    """
    collection_of_all_answers.append(Answers_from_tips_with_wv_and_bert)
    collection_of_all_answers.append(Answers_from_tips_with_wv_and_bert_without_rearranging_after_bert)
    """
    
    
    for dataframe in collection_of_all_answers:
        for index, row in dataframe.iterrows():
            new_key = (row['question'], row['business'], row['answer'])

            if new_key in temp_dictionary_of_QBA_and_rating:
                continue

            if 'human_rating' in dataframe.columns:
                if not (row['human_rating'] == None or isnan(row['human_rating'])):
                    temp_dictionary_of_QBA_and_rating[new_key] = row['human_rating']
                    continue

def fill_already_existing_ratings(target_dataframe):
    global temp_dictionary_of_QBA_and_rating
    temp_list_of_position = []
    temp_list_of_question = []
    temp_list_of_business = []
    temp_list_of_answer = []
    temp_list_of_similarity = []
    temp_list_of_execution_time_in_sec = []
    temp_list_of_human_rating = []
    
    for index, row in target_dataframe.iterrows():
        temp_list_of_position.append(row['Unnamed: 0'])
        temp_list_of_question.append(row['question'])
        temp_list_of_business.append(row['business'])
        temp_list_of_answer.append(row['answer'])
        temp_list_of_similarity.append(row['similarity'])
        temp_list_of_execution_time_in_sec.append(row['execution_time_in_sec'])
        new_key = (row['question'], row['business'], row['answer'])
        
        if new_key in temp_dictionary_of_QBA_and_rating:
            temp_list_of_human_rating.append(temp_dictionary_of_QBA_and_rating[new_key])
        else:
            temp_list_of_human_rating.append(None)
    
    raw_data = {'Unnamed: 0':temp_list_of_position,
                'question':temp_list_of_question,
                'business':temp_list_of_business,
                'answer':temp_list_of_answer,
                'similarity':temp_list_of_similarity,
                'execution_time_in_sec':temp_list_of_execution_time_in_sec,
                'human_rating':temp_list_of_human_rating
               }
    
    dataframe = pd.DataFrame(
        raw_data,
        columns = ['Unnamed: 0', 'question', 'business', 'answer', 'similarity', 'execution_time_in_sec', 'human_rating'])
    
    return dataframe

def add_ratings_to_answers(dataframe):
    global list_of_question_for_user_to_answer
    
    temp_list_of_position = []
    temp_list_of_question = []
    temp_list_of_business = []
    temp_list_of_answer = []
    temp_list_of_similarity = []
    temp_list_of_execution_time_in_sec = []
    temp_list_of_human_rating = []
    continue_rating_flag = True
    
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    
    for index, row in dataframe.iterrows():
        temp_list_of_position.append(row['Unnamed: 0'])
        temp_list_of_question.append(row['question'])
        temp_list_of_business.append(row['business'])
        temp_list_of_answer.append(row['answer'])
        temp_list_of_similarity.append(row['similarity'])
        temp_list_of_execution_time_in_sec.append(row['execution_time_in_sec'])
        new_key = (row['question'], row['business'], row['answer'])
        
        if 'human_rating' in dataframe.columns:
            if not (row['human_rating'] == None or isnan(row['human_rating'])):
                temp_list_of_human_rating.append(row['human_rating'])
                temp_dictionary_of_QBA_and_rating[new_key] = row['human_rating']
                continue
        
        if continue_rating_flag and ([row['question'], row['business']] in list_of_question_for_user_to_answer):
            while True:
                print("===========================================================")
                if _ask_user_a_yes_or_no_question("Do you want to continue rating?"):
                    print("question:")
                    print(row['question'])
                    print("")
                    print("answer:")
                    print(row['answer'])
                    try:
                        rating = int(input("Insert a rating in [1-5]: "))
                        if rating < 1 or rating > 5:
                            print("Error: Enter an integer from 1 to 5 please!")
                            continue
                        temp_list_of_human_rating.append(rating)
                        temp_dictionary_of_QBA_and_rating[new_key] = rating
                        break
                    except:
                        print("Error: Enter an integer from 1 to 5 please!")
                        continue
                else:
                    continue_rating_flag = False
                    temp_list_of_human_rating.append(None)
                    break
        else:
            temp_list_of_human_rating.append(None)
    
    raw_data = {'Unnamed: 0':temp_list_of_position,
                'question':temp_list_of_question,
                'business':temp_list_of_business,
                'answer':temp_list_of_answer,
                'similarity':temp_list_of_similarity,
                'execution_time_in_sec':temp_list_of_execution_time_in_sec,
                'human_rating':temp_list_of_human_rating
               }
    
    
    dataframe = pd.DataFrame(
        raw_data,
        columns = ['Unnamed: 0', 'question', 'business', 'answer', 'similarity', 'execution_time_in_sec', 'human_rating'])
    
    dataframe = dataframe.sort_values(by=['Unnamed: 0']).reset_index(drop=True)
    dataframe = dataframe.drop(["Unnamed: 0"], axis=1)
    return dataframe

def print_number_of_rated_answers(dataframe):
    global list_of_question_for_user_to_answer
    
    number_of_answers = 0
    number_of_rated_answers = 0
    
    for index, row in dataframe.iterrows():
        if [row['question'], row['business']] in list_of_question_for_user_to_answer:
            number_of_answers += 1
        
        if 'human_rating' in dataframe.columns:
            if not (row['human_rating'] == None or isnan(row['human_rating'])):
                number_of_rated_answers += 1
    
    percentage = number_of_rated_answers*100/number_of_answers
    print("Number of rated QBA: ", number_of_rated_answers, ", ", "Total number of QBA: ", number_of_answers)
    print("Completed: ", percentage, "%")

def save_choosen_dataframe(model_code):
    file_path = None
    
    if (model_code == TF_IDF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_WITH_WVC_FILE_PATH
    
    if (model_code == TF_IDF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_WITH_WVC_FILE_PATH
    
    if (model_code == TF_IDF_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == TF_IDF_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH
    """
    
    if (model_code == TF_IDF_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == TF_IDF_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_TF_IDF_AND_BERT_WITH_WVC_FILE_PATH
    """
    
    if (model_code == IWCS_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_TIPS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
    """
    
    if (model_code == IWCS_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITH_WVC_FILE_PATH
    
    if (model_code == IWCS_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_REVIEWS_WITH_IWCS_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
    """
    
    if (model_code == WV_CENTROID_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BIDAF_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BERT_WITH_TIPS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_TIPS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
    """
    
    if (model_code == WV_CENTROID_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BIDAF_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BIDAF):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BIDAF_WITHOUT_REARRANGING_AFTER_BIDAF_WITH_WVC_FILE_PATH
    
    """
    if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITH_WVC_FILE_PATH
    
    if (model_code == WV_CENTROID_BERT_WITH_REVIEWS_CODE_NUMBER_WITHOUT_REARRANGING_AFTER_BERT):
        file_path = ANSWERS_FROM_REVIEWS_WITH_WV_AND_BERT_WITHOUT_REARRANGING_AFTER_BERT_WITH_WVC_FILE_PATH
    """
    
    write_dataframe_to_csv_file(file_path, choosen_dataframe)

def add_best_questions_from_answers(dataframe_with_answers):
    global list_of_question_for_user_to_answer
    
    dataframe_with_answers_grouped = dataframe_with_answers.groupby(['question','business'])
    dataframe_with_answers_and_mean_of_similarities = dataframe_with_answers_grouped['similarity'].mean().to_frame()
    dataframe_with_answers_and_mean_of_similarities = dataframe_with_answers_and_mean_of_similarities.sort_values(by=['similarity'], ascending=False)
    
    i = 0
    for index, row in dataframe_with_answers_and_mean_of_similarities.iterrows():
        query = row.name[0]
        business = row.name[1]
        if (i >= config.NUMBER_OF_QUESTIONS_FOR_USER_TO_RATE/config.NUMBER_OF_MODELS_FOR_USER_TO_RATE):
            break

        if ([query, business] not in list_of_question_for_user_to_answer): 
            list_of_question_for_user_to_answer.append([query, business])
            i += 1

def compute_list_of_question_for_user_to_answer():
    if (config.NUMBER_OF_QUESTIONS_FOR_USER_TO_RATE % config.NUMBER_OF_MODELS_FOR_USER_TO_RATE != 0) or (config.NUMBER_OF_QUESTIONS_FOR_USER_TO_RATE < config.NUMBER_OF_MODELS_FOR_USER_TO_RATE):
        sys.exit()
    
    
    add_best_questions_from_answers(Answers_from_reviews_with_iwcs)
    add_best_questions_from_answers(Answers_from_reviews_with_iwcs_and_bidaf)
    add_best_questions_from_answers(Answers_from_reviews_with_iwcs_and_bidaf_without_rearranging_after_bidaf)

    add_best_questions_from_answers(Answers_from_reviews_with_tf_idf)
    add_best_questions_from_answers(Answers_from_reviews_with_tf_idf_and_bidaf)

    add_best_questions_from_answers(Answers_from_reviews_with_wv)
    add_best_questions_from_answers(Answers_from_reviews_with_wv_and_bidaf)
    add_best_questions_from_answers(Answers_from_reviews_with_wv_and_bidaf_without_rearranging_after_bidaf)

    add_best_questions_from_answers(Answers_from_tips_with_iwcs)
    add_best_questions_from_answers(Answers_from_tips_with_iwcs_and_bidaf)
    add_best_questions_from_answers(Answers_from_tips_with_iwcs_and_bidaf_without_rearranging_after_bidaf)

    add_best_questions_from_answers(Answers_from_tips_with_tf_idf)
    add_best_questions_from_answers(Answers_from_tips_with_tf_idf_and_bidaf)

    add_best_questions_from_answers(Answers_from_tips_with_wv)
    add_best_questions_from_answers(Answers_from_tips_with_wv_and_bidaf)
    add_best_questions_from_answers(Answers_from_tips_with_wv_and_bidaf_without_rearranging_after_bidaf)








if (len(sys.argv) == 2):
    model_code_chosen = int(sys.argv[1])
else:
    print_all_available_files_to_rate()
    model_code_chosen = int(input("which answers you want to rate: "))

init_temp_dictionary_of_QBA_and_rating()
_init_model(model_code_chosen)
choosen_dataframe = fill_already_existing_ratings(choosen_dataframe)
compute_list_of_question_for_user_to_answer()
print_number_of_rated_answers(choosen_dataframe)
choosen_dataframe = add_ratings_to_answers(choosen_dataframe)
print_number_of_rated_answers(choosen_dataframe)
save_choosen_dataframe(model_code_chosen)