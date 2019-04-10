import pandas as pd
import numpy as np 

import spacy 
import en_core_web_sm 

from nltk.parse import CoreNLPParser
import datetime

path_to_data = './data/'
cities = pd.read_csv(path_to_data + 'us_cities_states_counties.csv')  
cities['City'] = cities['City'].apply(lambda x: str(x))
ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner') 
parser = CoreNLPParser(url='http://localhost:9000')
def formatted_entities(classified_paragraphs_list):
    entities = []

    for classified_paragraph in classified_paragraphs_list:
        for entry in classified_paragraph:
            entry_value = entry[0]
            entry_type = entry[1]
            
            if entry_type == 'LOCATION': 
                entities.append(entry_value) 
    return entities 


currentDT = datetime.datetime.now()
print (str(currentDT))

count = 0
passed = 0
for i, city in enumerate(cities['City'].unique()):
    try:         
        city_ = parser.tokenize(city)     
        classified_paragraphs_list = ner_tagger.tag_sents([city_]) 
        formatted_result = formatted_entities(classified_paragraphs_list)  
        if len(formatted_result)>0:
            count+=1
    except Exception as e:  
        passed +=1
        print(i, city, 'error:', e)
        pass
    if i% 100 == 0: print (i, count, passed, city, city_, 'result:', ' '.join(formatted_result)) 
print(f'Stanford knows {count} out of {cities.City.unique().shape[0]}')
print('couldnt process:', passed)

currentDT = datetime.datetime.now()
print (str(currentDT))
