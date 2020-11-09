#Author: Sirinian Aram Emmanouil

from textblob import TextBlob
from yelp_data_loading import load_yelp_data
import pandas as pd
import config
from multiprocessing import Pool
import numpy as np
from progress_bar import printProgressBar


NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA = 4
BOTH_WAYS = None
MEAN = None
STANDARD_DEVIATION = None

def _split_yelps_data_into_sentences(dataframe):
    global NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA
    
    temp_list_of_name = []
    temp_list_of_text = []
    number_of_rows = len(dataframe)
    i = 0
    
    printProgressBar(i, number_of_rows, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for index, row in dataframe.iterrows():
        passage = row['text']
        zen = TextBlob(passage)

        for sentence in zen.sentences:
            temp_list_of_name.append(row['name'])
            temp_list_of_text.append(str(sentence))
        
        i += 1
        printProgressBar(i, number_of_rows, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)

    raw_data = {'name':temp_list_of_name,
                'text':temp_list_of_text
               }

    dataframe = pd.DataFrame(
        raw_data,
        columns = ['name', 'text'])
    
    return dataframe

def _add_number_of_words_in_dataframe(dataframe):
    global NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA
    
    temp_list_of_name = []
    temp_list_of_text = []
    temp_list_of_number_of_words = []
    number_of_rows = len(dataframe)
    i = 0
    
    printProgressBar(i, number_of_rows, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for index, row in dataframe.iterrows():
        zen = TextBlob(row['text'])
        number_of_words = len(zen.words)
        
        temp_list_of_name.append(row['name'])
        temp_list_of_text.append(row['text'])
        temp_list_of_number_of_words.append(number_of_words)
        
        i += 1
        printProgressBar(i, number_of_rows, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)
    
    raw_data = {'name':temp_list_of_name,
                'text':temp_list_of_text,
                'number_of_words':temp_list_of_number_of_words
               }

    dataframe = pd.DataFrame(
        raw_data,
        columns = ['name', 'text', 'number_of_words'])
    return dataframe

def _remove_rows_from_dataframe_based_on_their_text_length(dataframe):
    global BOTH_WAYS
    global MEAN
    global STANDARD_DEVIATION
    global NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA
    
    both_ways = BOTH_WAYS
    mean = MEAN
    standard_deviation = STANDARD_DEVIATION
    
    temp_list_of_name = []
    temp_list_of_text = []
    temp_list_of_number_of_words = []
    min_length = mean - standard_deviation
    max_length = mean + standard_deviation
    #print("min_length : ", min_length)
    #print("max_length : ", max_length)
    number_of_rows = len(dataframe)
    i = 0
    
    printProgressBar(i, number_of_rows, prefix = 'Progress:',
                     suffix = 'Complete', autosize = True)
    for index, row in dataframe.iterrows():
        
        if both_ways:
            if row['number_of_words'] >= min_length and row['number_of_words'] <= max_length:
                temp_list_of_name.append(row['name'])
                temp_list_of_text.append(row['text'])
                temp_list_of_number_of_words.append(row['number_of_words'])
        else:
            if row['number_of_words'] >= min_length:
                temp_list_of_name.append(row['name'])
                temp_list_of_text.append(row['text'])
                temp_list_of_number_of_words.append(row['number_of_words'])
        i += 1
        printProgressBar(i, number_of_rows, prefix = 'Progress:',
                         suffix = 'Complete', autosize = True)
    
    raw_data = {'name':temp_list_of_name,
                'text':temp_list_of_text,
                'number_of_words':temp_list_of_number_of_words
               }

    dataframe = pd.DataFrame(
        raw_data,
        columns = ['name', 'text', 'number_of_words'])
    
    return dataframe

def _parallelize_dataframe(df, func, n_cores = 2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

def preprocess_yelp_data():
    global BOTH_WAYS
    global MEAN
    global STANDARD_DEVIATION
    global NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA
    
    tips = load_yelp_data(load_reviews = False, load_tips = True)
    reviews = load_yelp_data(load_reviews = True, load_tips = False)
    
    print("Creating new dataframe of Tips with passages split into sentences...")
    tips_split_into_sentences = _parallelize_dataframe(
            tips,
            _split_yelps_data_into_sentences,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Creating new dataframe of Tips with passages split into sentences done successfully!")
    print("Creating new dataframe of Reviews with passages split into sentences...")
    reviews_split_into_sentences = _parallelize_dataframe(
            reviews,
            _split_yelps_data_into_sentences,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Creating new dataframe of Reviews with passages split into sentences done successfully!")
    
    print("Adding number of words in Tips...")
    tips = _parallelize_dataframe(
            tips,
            _add_number_of_words_in_dataframe,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Adding number of words in Tips done successfully!")
    print("")
    print("Length of Tips: ", len(tips))
    print("Mean of Tips: ", round(tips["number_of_words"].mean()))
    print("Std of Tips: ", round(tips["number_of_words"].std()))
    print("Min of Tips: ", round(tips["number_of_words"].min()))
    print("Max of Tips: ", round(tips["number_of_words"].max()))
    print("")
    
    print("Adding number of words in Reviews...")
    reviews = _parallelize_dataframe(
            reviews,
            _add_number_of_words_in_dataframe,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Adding number of words in Reviews done successfully!")
    print("")
    print("Length of Reviews: ", len(reviews))
    print("Mean of Reviews: ", round(reviews["number_of_words"].mean()))
    print("Std of Reviews: ", round(reviews["number_of_words"].std()))
    print("Min of Reviews: ", round(reviews["number_of_words"].min()))
    print("Max of Reviews: ", round(reviews["number_of_words"].max()))
    print("")
    
    
    print("Adding number of words in Tips with passages split into sentences...")
    tips_split_into_sentences = _parallelize_dataframe(
            tips_split_into_sentences,
            _add_number_of_words_in_dataframe,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Adding number of words in Tips with passages split into sentences done successfully!")
    print("")
    print("Length of Tips split into sentences: ", len(tips_split_into_sentences))
    print("Mean of Tips split into sentences: ", round(tips_split_into_sentences["number_of_words"].mean()))
    print("Std of Tips split into sentences: ", round(tips_split_into_sentences["number_of_words"].std()))
    print("Min of Tips split into sentences: ", round(tips_split_into_sentences["number_of_words"].min()))
    print("Max of Tips split into sentences: ", round(tips_split_into_sentences["number_of_words"].max()))
    print("")
    
    
    print("Adding number of words in Reviews with passages split into sentences...")
    reviews_split_into_sentences = _parallelize_dataframe(
            reviews_split_into_sentences,
            _add_number_of_words_in_dataframe,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Adding number of words in Reviews with passages split into sentences done successfully!")
    print("")
    print("Length of Reviews split into sentences: ", len(reviews_split_into_sentences))
    print("Mean of Reviews split into sentences: ", round(reviews_split_into_sentences["number_of_words"].mean()))
    print("Std of Reviews split into sentences: ", round(reviews_split_into_sentences["number_of_words"].std()))
    print("Min of Reviews split into sentences: ", round(reviews_split_into_sentences["number_of_words"].min()))
    print("Max of Reviews split into sentences: ", round(reviews_split_into_sentences["number_of_words"].max()))
    print("")
    
    BOTH_WAYS = True
    #MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_IN_TIPS_SPLIT_INTO_SENTENCES
    #STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_IN_TIPS_SPLIT_INTO_SENTENCES
    MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS
    STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS
    print("Setting MEAN to : ", MEAN)
    print("Setting STANDARD_DEVIATION to : ", STANDARD_DEVIATION)
    print("Removing too short or too long sentences from Tips with passages split into sentences...")
    tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long = _parallelize_dataframe(
            tips_split_into_sentences,
            _remove_rows_from_dataframe_based_on_their_text_length,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Removing too short or too long sentences from Tips with passages split into sentences done successfully!")
    print("")
    print("Length of Tips split into sentences with rows removed if text length too short or too long: ", len(tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long))
    print("Mean of Tips split into sentences with rows removed if text length too short or too long: ", round(tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].mean()))
    print("Std of Tips split into sentences with rows removed if text length too short or too long: ", round(tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].std()))
    print("Min of Tips split into sentences with rows removed if text length too short or too long: ", round(tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].min()))
    print("Max of Tips split into sentences with rows removed if text length too short or too long: ", round(tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].max()))
    print("")
    tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long.drop(columns=["number_of_words"])
    
    
    BOTH_WAYS = True
    #MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_IN_REVIEWS_SPLIT_INTO_SENTENCES
    #STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_IN_REVIEWS_SPLIT_INTO_SENTENCES
    MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS
    STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS
    print("Setting MEAN to : ", MEAN)
    print("Setting STANDARD_DEVIATION to : ", STANDARD_DEVIATION)
    print("Removing too short or too long sentences from Reviews with passages split into sentences...")
    reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long = _parallelize_dataframe(
            reviews_split_into_sentences,
            _remove_rows_from_dataframe_based_on_their_text_length,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Removing too short or too long sentences from Reviews with passages split into sentences done successfully!")
    print("")
    print("Length of Reviews split into sentences with rows removed if text length too short or too long: ", len(reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long))
    print("Mean of Reviews split into sentences with rows removed if text length too short or too long: ", round(reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].mean()))
    print("Std of Reviews split into sentences with rows removed if text length too short or too long: ", round(reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].std()))
    print("Min of Reviews split into sentences with rows removed if text length too short or too long: ", round(reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].min()))
    print("Max of Reviews split into sentences with rows removed if text length too short or too long: ", round(reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long["number_of_words"].max()))
    print("")
    reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long.drop(columns=["number_of_words"])
    
    
    BOTH_WAYS = False
    #MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_IN_TIPS
    #STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_IN_TIPS
    MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS_SPLIT_INTO_SENTENCES
    STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS_SPLIT_INTO_SENTENCES
    print("Setting MEAN to : ", MEAN)
    print("Setting STANDARD_DEVIATION to : ", STANDARD_DEVIATION)
    print("Removing too short sentences from Tips...")
    tips_with_rows_removed_if_text_length_too_short = _parallelize_dataframe(
            tips,
            _remove_rows_from_dataframe_based_on_their_text_length,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Removing too short sentences from Tips done successfully!")
    print("")
    print("Length of Tips with rows removed if text length too short: ", len(tips_with_rows_removed_if_text_length_too_short))
    print("Mean of Tips with rows removed if text length too short: ", round(tips_with_rows_removed_if_text_length_too_short["number_of_words"].mean()))
    print("Std of Tips with rows removed if text length too short: ", round(tips_with_rows_removed_if_text_length_too_short["number_of_words"].std()))
    print("Min of Tips with rows removed if text length too short: ", round(tips_with_rows_removed_if_text_length_too_short["number_of_words"].min()))
    print("Max of Tips with rows removed if text length too short: ", round(tips_with_rows_removed_if_text_length_too_short["number_of_words"].max()))
    print("")
    tips_with_rows_removed_if_text_length_too_short.drop(columns=["number_of_words"])
    
    
    BOTH_WAYS = False
    #MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_IN_REVIEWS
    #STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_IN_REVIEWS
    MEAN = config.YELP_MEAN_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS_SPLIT_INTO_SENTENCES
    STANDARD_DEVIATION = config.YELP_STANDARD_DEVIATION_NUMBER_OF_WORDS_FOR_TIPS_AND_REVIEWS_SPLIT_INTO_SENTENCES
    print("Setting MEAN to : ", MEAN)
    print("Setting STANDARD_DEVIATION to : ", STANDARD_DEVIATION)
    print("Removing too short sentences from Reviews...")
    reviews_with_rows_removed_if_text_length_too_short = _parallelize_dataframe(
            reviews,
            _remove_rows_from_dataframe_based_on_their_text_length,
            n_cores = NUMBER_OF_CORES_FOR_PREPROCESSING_YELP_DATA)
    print("Removing too short sentences from Reviews done successfully!")
    print("")
    print("Length of Reviews with rows removed if text length too short: ", len(reviews_with_rows_removed_if_text_length_too_short))
    print("Mean of Reviews with rows removed if text length too short: ", round(reviews_with_rows_removed_if_text_length_too_short["number_of_words"].mean()))
    print("Std of Reviews with rows removed if text length too short: ", round(reviews_with_rows_removed_if_text_length_too_short["number_of_words"].std()))
    print("Min of Reviews with rows removed if text length too short: ", round(reviews_with_rows_removed_if_text_length_too_short["number_of_words"].min()))
    print("Max of Reviews with rows removed if text length too short: ", round(reviews_with_rows_removed_if_text_length_too_short["number_of_words"].max()))
    print("")
    reviews_with_rows_removed_if_text_length_too_short.drop(columns=["number_of_words"])
    
    
    print("Saving Tips split into sentences with rows removed if text length too short or too long...")
    tips_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long.to_json(
            config.TIPS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH,
            orient='records', lines=True)
    print("Saving Tips split into sentences with rows removed if text length too short or too long done sucessfully!")
    print("Saving Reviews split into sentences with rows removed if text length too short or too long...")
    reviews_split_into_sentences_with_rows_removed_if_text_length_too_short_or_too_long.to_json(
            config.REVIEWS_SPLIT_INTO_SENTENCES_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_OR_TOO_LONG_FILE_PATH,
            orient='records', lines=True)
    print("Saving Reviews split into sentences with rows removed if text length too short or too long done successfully!")
    print("Saving Tips with rows removed if text length too short...")
    tips_with_rows_removed_if_text_length_too_short.to_json(
            config.TIPS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH,
            orient='records', lines=True)
    print("Saving Tips with rows removed if text length too short done successfully!")
    print("Saving Reviews with rows removed if text length too short...")
    reviews_with_rows_removed_if_text_length_too_short.to_json(
            config.REVIEWS_WITH_ROWS_REMOVED_IF_TEXT_LENGTH_TOO_SHORT_FILE_PATH,
            orient='records', lines=True)
    print("Saving Reviews with rows removed if text length too short done successfully!")

preprocess_yelp_data()