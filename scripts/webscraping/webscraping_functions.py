# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021

@author: Joe L/Joe HC/Steven H

All the webscraping functions are stored here. Their application is not yet certain in the project when we switch
to email data.

"""

from typing import Any, Optional
import urllib.request as r
from bs4 import BeautifulSoup as Soup
import json
import pandas as pd


def __get_parse_website_soup(url: str, agent = dict[str, str], timeout: Optional[int] = None, parse: str = "html.parser") -> Soup:

    """Makes a call to a specified url and parses the webpage using beautiful soup"""

    req = r.Request(url = url, headers = agent)
    with r.urlopen(req, timeout = timeout) as response:
        page = response.read()

    page = Soup(page, parse)

    return page


def __get_parse_website_json(url: str, agent = dict[str, str], timeout: Optional[int] = None
                            ) -> dict[str, Any]:

    """
    Assumes webpage has an associated json.
    Makes a call to a specified url, appending .json extension to the address.
    Parses the response to a python dictionary.
    """

    req = r.Request(url = f'{url}.json', headers = agent)
    with r.urlopen(req, timeout = timeout) as response:
        page = response.read()

    page = json.loads(page)

    return page


########### collect relevant text from url(s) ###########

def find_text(input_text, input_structure):
    output = dict()

    for search in input_structure:

        if len(search) == 3:

            output[search[0]] = input_text.find_all(search[1])[search[2]].text

        else:

            output[search[0]] = input_text.find_all(search[1], class_=search[2])[search[3]].text

    return output


def get_text(urls, agent, structure, df_url_col=None):
    if type(structure) is not list:
        raise TypeError('Structure data type not accepted')
    if not all([type(el) is list for el in structure]):
        raise TypeError('Structure data type not accepted')

    output = []

    if type(urls) is pd.core.frame.DataFrame:

        if urls.shape[1] == 1:
            df_url_col = urls.columns[0]
        elif df_url_col is None:
            raise TypeError('No column name provided.')
        elif type(df_url_col) is not str or type(df_url_col) is not int:
            raise TypeError('Invalid df column reference')
        elif type(df_url_col) is int:
            df_url_col = urls.columns[df_url_col]

        for i, row in urls.iterrows():
            page = __get_parse_website_soup(row[df_url_col], agent)

            page_text = find_text(page, structure)

            output.append(page_text)

    elif any([type(urls) is el for el in [list, tuple, pd.core.series.Series]]):

        for addr in urls:
            page = __get_parse_website_soup(addr, agent)

            page_text = find_text(page, structure)

            output.append(page_text)

    elif type(urls) is str:

        page = __get_parse_website_soup(addr, agent)

        page_text = find_text(page, structure)

        output.append(page_text)

    else:

        raise TypeError('url datatype not recognised')

    return output


def __extract_json(json: dict[str, Any], structure: list[str], out_keys: Optional[dict[str, str]] = None) -> dict[str, str]:

    """
    Extracts data from a dictionary (json).
    Can handle nested dictionaries using __ between key names.
    Returns a flattened dictionary of selected items, whose key names can be modified by out_keys.
    """

    if out_keys is None:
        output_keys = dict()
    else:
        output_keys = out_keys

    output = dict()

    for key in structure:
        data = json.copy()
        for sub_key in key.split('__'):
            data = data.get(sub_key)
        output.update({output_keys.get(key, key): data})

    return output


def get_text_json(url: str, agent: dict[str, str], structure: list[str], 
                    timeout: Optional[int] = None, out_keys: Optional[dict[str, str]] = None) -> dict[str, str]:

    """
    Assumes webpage has an associated json.
    Makes a call to a specified url, appending .json extension to the address.
    
    Extracts data from a dictionary (json).
    Can handle nested dictionaries using __ between key names.
    Returns a flattened dictionary of selected items, whose key names can be modified by out_keys.
    """

    dict_page = __get_parse_website_json(url = url, agent = agent, timeout = timeout)

    return __extract_json(dict_page, structure, out_keys)