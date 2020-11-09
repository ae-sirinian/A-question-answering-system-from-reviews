import time
import sys
import random
import scrapy
import json
import os
import pandas as pd
#from scrapy.utils.response import open_in_browser
from ..items import AnswerLinksScraperItem

SCRAPE_ONLY_THE_FIRST_PAGES = False
MIN_DELAY_SECONDS_FOR_SCRAPPING_REQUEST = 20
MAX_DELAY_SECONDS_FOR_SCRAPPING_REQUEST = 30
REVIEWS_FROM_YELP_FILE_PATH = '/home/jana/Documents/question_answering_system/data/Free_data_from_Yelp/reviews.json'
TIPS_FROM_YELP_FILE_PATH = '/home/jana/Documents/question_answering_system/data/Free_data_from_Yelp/tips.json'
BUSINESS_SOURCE_FILE_PATH = "/home/jana/Documents/yelp-dataset/yelp_academic_dataset_business.json"
TARGET_ROOT_URL = "https://www.yelp.com/questions/"
ALREADY_SCRAPPED_BUSINESSES_FILE_PATH = "/home/jana/Documents/question_answering_system/answer_links_scraper/already_scrapped_businesses.json"
business_id_and_name_dict = dict()
business_name_and_id_dict = dict()
already_scrapped_businesses = []
links_buffer = []
scrapped_businesses_buffer = []
businesses_with_reviews_and_tips = []

def _read_list_from_json(file_path):
    list_to_read = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as filehandle:
            list_to_read = json.load(filehandle)
    return list_to_read

def _write_list_to_json(list_to_write, file_path):
    with open(file_path, 'w') as filehandle:
        json.dump(list_to_write, filehandle)

def _compute_business_name_and_id_dictionary(file_path):
    business_name_and_id_dict = dict()
    
    with open(file_path, 'r', encoding = 'latin-1') as input_json_file:
        for line in input_json_file:
            data = json.loads(line)        
            business_id = data['business_id']
            business_name = data['name']
            tmp_dict = {business_name:business_id}
            business_name_and_id_dict.update(tmp_dict)
    
    return business_name_and_id_dict

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

def _compute_tips_reviews_df():
    tips_df = pd.read_json(TIPS_FROM_YELP_FILE_PATH, lines=True)
    reviews_df = pd.read_json(REVIEWS_FROM_YELP_FILE_PATH, lines=True)
    tips_and_reviews_df = pd.concat([reviews_df, tips_df])
    tips_and_reviews_df = tips_and_reviews_df.drop(['pos_sent', 'neg_sent', 'neu_sent'], axis=1)
    
    return tips_and_reviews_df

def _compute_businesses_with_reviews_and_tips(business_name_and_id_dict):
    businesses_with_reviews_and_tips = []
    
    tips_and_reviews_df = _compute_tips_reviews_df()
    tips_and_reviews_df_grouped_by_name = tips_and_reviews_df.groupby("name")
    
    temp_name_list = []
    temp_count_list = []
    for name, group in tips_and_reviews_df_grouped_by_name:
        count = list(tips_and_reviews_df_grouped_by_name.get_group(name).count())[0]
        temp_count_list.append(count)
        temp_name_list.append(name)
    raw_data = {
        'name':temp_name_list,
        'count':temp_count_list
    }
    
    businesses_with_there_count_df = pd.DataFrame(
        raw_data, columns = ['name', 'count'])
    businesses_with_there_count_df = businesses_with_there_count_df.sort_values(
        by=['count'], ascending=False)
    businesses = list(businesses_with_there_count_df['name'])
    businesses_with_reviews_and_tips = [[business, business_name_and_id_dict[business]] 
                              for business in businesses 
                              if business in business_name_and_id_dict]
    
    return businesses_with_reviews_and_tips

def _compute_businesses_with_reviews_and_tips_without_already_scrapped_businesses(
        businesses_with_reviews_and_tips, already_scrapped_businesses):
    businesses_with_reviews_and_tips_without_already_scrapped_businesses = []
    
    businesses_with_reviews_and_tips_without_already_scrapped_businesses = [
        (TARGET_ROOT_URL + business_name_and_id[1]) 
        for business_name_and_id in businesses_with_reviews_and_tips 
        if business_name_and_id[1] not in already_scrapped_businesses]
    
    return businesses_with_reviews_and_tips_without_already_scrapped_businesses

class AnswerLinksSpider(scrapy.Spider):
    name = "answer_links"
    main_url = "https://www.yelp.com"
    #handle_httpstatus_list = [404, 503]
    handle_httpstatus_list = [404]
    handle_httpstatus_all = True
    
    business_id_and_name_dict = _compute_business_id_and_name_dictionary(
        BUSINESS_SOURCE_FILE_PATH)
    business_name_and_id_dict = _compute_business_name_and_id_dictionary(
        BUSINESS_SOURCE_FILE_PATH)
    already_scrapped_businesses = _read_list_from_json(
        ALREADY_SCRAPPED_BUSINESSES_FILE_PATH)
    businesses_with_reviews_and_tips = _compute_businesses_with_reviews_and_tips(
        business_name_and_id_dict)
    businesses_with_reviews_and_tips_without_already_scrapped_businesses = _compute_businesses_with_reviews_and_tips_without_already_scrapped_businesses(
        businesses_with_reviews_and_tips, already_scrapped_businesses)
    
    print("")
    print("")
    print(len(already_scrapped_businesses), " already_scrapped_businesses :")
    for already_scrapped_businesse in already_scrapped_businesses:
        if already_scrapped_businesse in business_id_and_name_dict:
            print(business_id_and_name_dict[already_scrapped_businesse], ": is already scrapped")
    print("")
    print("")
    
    print("")
    print("")
    print(len(businesses_with_reviews_and_tips),
          " businesses_with_reviews_and_tips")
    print(len(
        businesses_with_reviews_and_tips_without_already_scrapped_businesses), 
        " businesses_with_reviews_and_tips_without_already_scrapped_businesses :")
    for x in businesses_with_reviews_and_tips_without_already_scrapped_businesses:
        print(x)
    print("")
    print("")
    #sys.exit()
    start_urls = businesses_with_reviews_and_tips_without_already_scrapped_businesses
    #start_urls = ["https://www.yelp.com/questions/yQab5dxZzgBLTEHCw9V7_w",
    #              "https://www.yelp.com/questions/4JNXUYY8wbaaDmk3BPzlWw"]
    #start_urls = ["https://www.yelp.com/questions/yQab5dxZzgBLTEHCw9V7_w"]
    
    def parse(self, response):
        global links_buffer
        global scrapped_businesses_buffer
            
        business_id = response.url.split("/")[-1]
        if business_id not in already_scrapped_businesses:
            if business_id in business_id_and_name_dict:
                print("")
                print("")
                print("Scrapping business with id: ", business_id_and_name_dict[business_id])
                print("")
                print("")
            error_404 = response.css(".page-status").css("::text").extract()
            #error_503 = response.css("h2").css("::text").extract()
            if len(error_404) == 0:
                            
                time.sleep(random.randint(MIN_DELAY_SECONDS_FOR_SCRAPPING_REQUEST,
                                          MAX_DELAY_SECONDS_FOR_SCRAPPING_REQUEST))
                #open_in_browser(response)
                items = AnswerLinksScraperItem()
                
                temp_links = response.css(".question-details").xpath("@href").extract()
                
                links_buffer.append([(AnswerLinksSpider.main_url + temp_link) for temp_link in temp_links])
                scrapped_businesses_buffer.append(business_id)
                
                if SCRAPE_ONLY_THE_FIRST_PAGES:
                    for links in links_buffer:
                        for link in links:
                            items["link"] = link
                            yield items
                    already_scrapped_businesses.append(scrapped_businesses_buffer[0])
                    _write_list_to_json(already_scrapped_businesses, ALREADY_SCRAPPED_BUSINESSES_FILE_PATH)     
                    links_buffer = []
                    scrapped_businesses_buffer = []
                else:
                    next_page = response.css(".next").xpath("@href").extract()
                    
                    if len(next_page) != 0:
                        next_page_link = AnswerLinksSpider.main_url + next_page[0]
                        yield response.follow(next_page_link, callback=self.parse)
                    else:
                        for links in links_buffer:
                            for link in links:
                                items["link"] = link
                                yield items
                        already_scrapped_businesses.append(scrapped_businesses_buffer[0])
                        _write_list_to_json(already_scrapped_businesses, ALREADY_SCRAPPED_BUSINESSES_FILE_PATH)
                        links_buffer = []
                        scrapped_businesses_buffer = []
            else:
                print("")
                print("There is a problem occurring with business with id: ",
                      business_id)
                print("Scrapper will ignore this one as if it doesn't exist.")
                print("")
                
                already_scrapped_businesses.append(business_id)
                _write_list_to_json(already_scrapped_businesses,
                                    ALREADY_SCRAPPED_BUSINESSES_FILE_PATH)