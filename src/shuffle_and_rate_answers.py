#Author: Sirinian Aram Emmanouil

from read_and_write_functions import read_dataframe_from_csv_file
from read_and_write_functions import write_dataframe_to_csv_file
from math import isnan
import pandas as pd
import config

def _ask_user_a_yes_or_no_question(question):
    while True:
        answer = (input(question + "[y/n]: ")).lower()
        
        if (answer == "y" or answer == "yes" or answer == "yeah"):
            return True
        elif (answer == "n" or answer == "no" or answer == "nah"):
            return False
        else:
            continue

def _add_ratings_to_answers(dataframe):
    temp_list_of_position = []
    temp_list_of_question = []
    temp_list_of_business = []
    temp_list_of_answer = []
    temp_list_of_similarity = []
    temp_list_of_execution_time_in_sec = []
    temp_list_of_human_rating = []
    continue_rating_flag = True
    
    for index, row in dataframe.iterrows():
        temp_list_of_position.append(row['Unnamed: 0'])
        temp_list_of_question.append(row['question'])
        temp_list_of_business.append(row['business'])
        temp_list_of_answer.append(row['answer'])
        temp_list_of_similarity.append(row['similarity'])
        temp_list_of_execution_time_in_sec.append(row['execution_time_in_sec'])

        if 'human_rating' in dataframe.columns:
            if not (row['human_rating'] == None or isnan(row['human_rating'])):
                temp_list_of_human_rating.append(row['human_rating'])
                continue
        
        if continue_rating_flag:
            while True:
                if _ask_user_a_yes_or_no_question("Do you want to continue rating?"):
                    try:
                        rating = int(input("Insert a rating in [1-5]: "))
                        if rating < 1 or rating > 5:
                            print("Error: Enter an integer from 1 to 5 please!")
                            continue
                        temp_list_of_human_rating.append(rating)
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
    
    return dataframe

def shuffle_and_rate_answers():
    #TODO Add ifs for choosing the filepath of dataframe to load/rate
    file_path = "./asdasd.csv"
    
    
    dataframe = read_dataframe_from_csv_file(file_path)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe = _add_ratings_to_answers(dataframe)
    dataframe = dataframe.sort_values(by=['Unnamed: 0']).reset_index(drop=True)
    print("Number of answers rated : ", len(dataframe["human_rating"].dropna().tolist()))
    dataframe = dataframe.drop(['Unnamed: 0'], axis=1)
    write_dataframe_to_csv_file(file_path, dataframe)

shuffle_and_rate_answers()