PREPROCESS_YELP_DATA_CODE_NUMBER = 1
COMPUTE_MODEL_DATA_FILES_CODE_NUMBER = 2
PASS_QUERIES_TO_A_MODEL_CODE_NUMBER = 3
COMPUTE_ANSWERS_TO_COLLECTED_QUERIES_WITH_A_MODEL = 4
RATE_COMPUTED_ANSWERS = 5

def print_all_available_actions():
    print("All available options:")
    print(str(PREPROCESS_YELP_DATA_CODE_NUMBER) + ") \t" + "Preprocess Yelp data")
    print(str(COMPUTE_MODEL_DATA_FILES_CODE_NUMBER) + ") \t" + "Compute model data files")
    print(str(PASS_QUERIES_TO_A_MODEL_CODE_NUMBER) + ") \t" + "Pass queries to a model")
    print(str(COMPUTE_ANSWERS_TO_COLLECTED_QUERIES_WITH_A_MODEL) + ") \t" + "Compute answers to collected queries")
    print(str(RATE_COMPUTED_ANSWERS) + ") \t" + "Rate computed answers")

while True:
    print_all_available_actions()
    model_code_chosen = int(input("Which script do you want to run? [1-5]: "))
    
    if (model_code_chosen == PREPROCESS_YELP_DATA_CODE_NUMBER):
        import preprocessing_yelp_data
        #preprocessing_yelp_data.py
        break
    elif (model_code_chosen == COMPUTE_MODEL_DATA_FILES_CODE_NUMBER):
        import compute_model_files
        #compute_model_files.py
        break
    elif(model_code_chosen == PASS_QUERIES_TO_A_MODEL_CODE_NUMBER):
        import answer_user_queries
        #answer_user_queries.py
        break
    elif(model_code_chosen == COMPUTE_ANSWERS_TO_COLLECTED_QUERIES_WITH_A_MODEL):
        import answer_collected_questions_with_a_model
        #answer_collected_questions_with_a_model.py
        break
    elif(model_code_chosen == RATE_COMPUTED_ANSWERS):
        import rating_answers_from_user
        #rating_answers_from_user.py
        break
    else:
        print("Error: Your imput is invalid! Please try again")
