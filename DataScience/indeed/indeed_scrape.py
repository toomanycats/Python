######################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
from indeed import IndeedClient
import pandas as pd

pub_id = "2818745629107228"

client = IndeedClient(pub_id)
params = {'q': "data science",
        'l': location,
        'format': 'json',
        'userip': '1.2.3.4',
        'useragent' :'Mozilla/%2F4.0%28Firefox%29',
        'v':'2',
        'start':'0',
        'limit':'25',
        'st':'employer',
        'fromage':age
        }


def get_urls():
    df = pd.read_csv('/home/daniel/git/Python2.7/DataScience/indeed/us_postal_codes.csv', dtype=str)

    locations = df['Postal Code'].dropna()
    loc_samp = locations.sample(100)
