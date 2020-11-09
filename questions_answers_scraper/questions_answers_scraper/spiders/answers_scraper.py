import time
import sys
import random
import scrapy
import json
import os
#import re
import pandas as pd
#from scrapy.utils.response import open_in_browser
from ..items import QuestionsAnswersScraperItem

MIN_DELAY_SECONDS_FOR_SCRAPPING_REQUEST = 20
MAX_DELAY_SECONDS_FOR_SCRAPPING_REQUEST = 30
BUSINESS_SOURCE_FILE_PATH = "/home/jana/Documents/yelp-dataset/yelp_academic_dataset_business.json"
ANSWER_LINKS_ONLY_FROM_FIRST_QUESTION_PAGES = False
ANSWER_LINKS_FILE_PATH = "/home/jana/Documents/question_answering_system/answer_links_scraper/answer_links_all_pages.csv"
ANSWER_LINKS_ONLY_FROM_FIRST_QUESTION_PAGES_FILE_PATH = "/home/jana/Documents/question_answering_system/answer_links_scraper/answer_links_only_first_pages.csv"
ALREADY_SCRAPPED_QUESTION_ANSWER_LINKS_FILE_PATH = "/home/jana/Documents/question_answering_system/questions_answers_scraper/already_scrapped_question_answer_links.json"
#already_scrapped_question_links = []
#######################
#business_id_and_name_dict = dict()
#already_scrapped_urls = []
#scrapped_urls = []
#scrapped_urls_without_already_scrapped_urls = []
#######################

def _read_urls_from_json(file_path):
    list_to_read = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as filehandle:
            list_to_read = json.load(filehandle)
    return list_to_read

def _write_urls_to_json(urls, file_path):
    with open(file_path, 'w') as filehandle:
        json.dump(urls, filehandle)

def _read_scrapped_urls(file_path):
    scrapped_urls = []
    if os.path.exists(file_path):
        scrapped_urls = list(pd.read_csv(file_path)['link'])
        print("scrapped_urls before: ", len(scrapped_urls))
        scrapped_urls = set(scrapped_urls)
        print("scrapped_urls after: ", len(scrapped_urls))
        scrapped_urls = list(scrapped_urls)
    else:
        print("")
        print("Urls from which scrapping starts not found")
        print("Make sure you have run the other scrapper first")
        print("")
    return scrapped_urls

def _compute_urls_for_scrapping(scrapped_urls, already_scrapped_urls):
    return [url for url in scrapped_urls if url not in already_scrapped_urls]

def _compute_business_id_and_name_dictionary(file_path):
    business_id_and_name_dict = dict()
    
    with open(file_path, 'r', encoding = 'latin-1') as input_json_file:
        for line in input_json_file:
            data = json.loads(line)        
            business_id = data['business_id']
            business_name = data['name']
            tmp_dict = {business_id:business_name}
            business_id_and_name_dict.update(tmp_dict)
    return business_id_and_name_dict

class AnswersSpider(scrapy.Spider):
    name = "answer"
    already_scrapped_urls = _read_urls_from_json(ALREADY_SCRAPPED_QUESTION_ANSWER_LINKS_FILE_PATH)
    print("")
    print("")
    print(len(already_scrapped_urls), " already_scrapped_urls :")
    #for i in range(len(already_scrapped_urls)):
    #    print(already_scrapped_urls[i])
    print("")
    print("")
    business_id_and_name_dict = _compute_business_id_and_name_dictionary(BUSINESS_SOURCE_FILE_PATH)
    #print("")
    #print("")
    #print(len(business_id_and_name_dict), " business_id_and_name_dict :")
    #print("")
    #print("")
    
    
    if ANSWER_LINKS_ONLY_FROM_FIRST_QUESTION_PAGES:
        scrapped_urls = _read_scrapped_urls(
            ANSWER_LINKS_ONLY_FROM_FIRST_QUESTION_PAGES_FILE_PATH) 
    else:
        scrapped_urls = _read_scrapped_urls(ANSWER_LINKS_FILE_PATH)
    
    scrapped_urls_without_already_scrapped_urls = _compute_urls_for_scrapping(scrapped_urls, already_scrapped_urls)
    print("")
    print("")
    print(len(scrapped_urls_without_already_scrapped_urls),
          " scrapped_urls_without_already_scrapped_urls :")
    for x in scrapped_urls_without_already_scrapped_urls:
        print(x)
    print("")
    print("")
    start_urls = scrapped_urls_without_already_scrapped_urls
    #print("")
    #print("")
    #print(len(start_urls), " start_urls :")
    #for x in start_urls:
    #    print(x)
    #print("")
    #print("")
    
    
    def parse(self, response):
        #if response.url not in already_scrapped_urls:
        #global business_id_and_name_dict
        #global already_scrapped_urls
        #global scrapped_urls
        #global scrapped_urls_without_already_scrapped_urls
        
        time.sleep(random.randint(MIN_DELAY_SECONDS_FOR_SCRAPPING_REQUEST,
                                  MAX_DELAY_SECONDS_FOR_SCRAPPING_REQUEST))
        ###########################################
        print("")
        print("")
        print(len(AnswersSpider.already_scrapped_urls), " already_scrapped_urls(2) :")
        #for i in range(len(already_scrapped_urls)):
        #    print(already_scrapped_urls[i])
        print("")
        print("")
        
        ############################################
        business_id = response.url.split("/")[-1]            
        if business_id in AnswersSpider.business_id_and_name_dict:
            print("")
            print("")
            print("Scrapping business with id: ", AnswersSpider.business_id_and_name_dict[business_id])
            print("")
            print("")
        
        items = QuestionsAnswersScraperItem()
        
        business_name_list = response.css(".breadcrumbs li:nth-child(1) a").css("::text").extract()
        question_list = response.css(".alternate").css("::text").extract()
        answers_list = response.css(".answers-list .u-break-word").css("::text").extract()
        helpful_list = response.css(".js-count").css("::text").extract()
        
        if ((len(business_name_list) != 0) and (len(question_list) != 0) and
            (len(answers_list) != 0) and (len(helpful_list) != 0)):
            
            temp_helpful_list = []
            for helpful_str in helpful_list:
                helpful = 0
                try:
                    helpful = int(helpful_str)
                except:
                    pass
                temp_helpful_list.append(helpful)
            
            helpful_list = temp_helpful_list
                        
            for i in range(len(answers_list)):
                items["business_name"] = business_name_list[0]
                items["question"] = question_list[0]
                items["answer"] = answers_list[i]
                items["helpful"] = helpful_list[i]
                yield items
            AnswersSpider.already_scrapped_urls.append(response.url)
            _write_urls_to_json(AnswersSpider.already_scrapped_urls, ALREADY_SCRAPPED_QUESTION_ANSWER_LINKS_FILE_PATH)