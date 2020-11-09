#author: Sirinian Aram Emmanouil
import json

BUSINESS_SOURCE_FILE_PATH = "/home/jana/Documents/yelp-dataset/yelp_academic_dataset_business.json"

def compute_business_id_name_dictionary():
    business_id_name_dict = dict()
    
    with open(BUSINESS_SOURCE_FILE_PATH, 'r', encoding = 'latin-1') as input_json_file:
        for line in input_json_file:
            data = json.loads(line)        
            business_id = data['business_id']
            business_name = data['name']
            tmp_dict = {business_name:business_id}
            business_id_name_dict.update(tmp_dict)
    
    return business_id_name_dict

def _ask_user_a_yes_or_no_question(question):
    while True:
        answer = (input(question + "[y/n]: ")).lower()
        
        if (answer == "y" or answer == "yes" or answer == "yeah"):
            return True
        elif (answer == "n" or answer == "no" or answer == "nah"):
            return False
        else:
            continue


business_id_name_dict = compute_business_id_name_dictionary()
while _ask_user_a_yes_or_no_question("Do you want to search for a business"):
	business_name = input("Enter the business name: ")
	if business_name in business_id_name_dict:
		print(business_id_name_dict[business_name])
		print("")
