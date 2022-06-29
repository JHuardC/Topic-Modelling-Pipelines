# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:48:15 2021

@author: Student User
"""

import logging
import pathlib as plib
import pandas as pd
from functools import partial
from webscraping_functions import get_text_json

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO
)

agent = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582'}

### construct keys for url extraction
base_name = 'data__attributes__'
keys = ['action', 
        'background', 
        'additional_details', 
        'created_at', 
        'moderation_threshold_reached_at', 
        'opened_at', 
        'updated_at', 
        'topics', 
        'state']

json_keys = [f'{base_name}{el}' for el in keys]
output_keys = dict(zip(json_keys, keys))

get_text_json_partial = partial(
        get_text_json, 
        agent = agent, 
        structure = json_keys, 
        out_keys = output_keys
)

########### Load URLs into Pandas from csv ###########
directory = plib.Path(__file__).parent.parent.parent
urls = pd.read_csv(directory.joinpath('data', 'petition_meta.csv'),usecols=['URL'])

get_petition_samples = """
########### Load URLs into Pandas from csv ###########
url_sample = urls.sample(frac = 0.1, random_state = 42)

########### Apply get_text_json to url_sample ###########

petitions = url_sample['URL'].apply(get_text_json_partial)
petitions = pd.DataFrame(petitions.to_list())
petitions.to_csv(directory.joinpath('data', 'petitions_sample_text.csv'),index = False)
"""

get_all_petitions = """"""
########### Apply get_text_json to urls ###########

# writing to csv
import csv

with open(
        directory.joinpath('data', 'all_petitions_data.csv'), 
        'w'
) as f:
        # Using DictWriter to pass records to csv files individually
        d_writer = csv.DictWriter(f, keys)
        for row in urls.itertuples(index = True, name = 'js'):
                logging.info(
                        f'Record {row.Index}, retrieving petition data for {row.URL}.'
                )
                pet_dict = get_text_json_partial(row.URL)
                d_writer.writerow(pet_dict)

logging.info('Complete.')