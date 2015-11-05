######################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
from indeed import IndeedClient
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

pub_id = "2818745629107228"

client = IndeedClient(pub_id)
params = {'q': "data science",
          'l': '',
          'format': 'json',
          'userip': '1.2.3.4',
          'useragent' :'Mozilla/%2F4.0%28Firefox%29',
          'v':'2',
          'start':'0',
          'limit':'25',
          'st':'employer'
         }


def get_urls():
    df = pd.read_csv('/home/daniel/git/Python2.7/DataScience/indeed/us_postal_codes.csv', dtype=str)

    locations = df['Postal Code'].dropna()
    loc_samp = locations.sample(100)

    urls = []
    for loc in loc_samp:
        params['loc'] = loc
        data = client.search(**params)
        urls.append(data['results']['url'])

    return urls

def open_url(url):
    try:
        response = urllib2.urlopen(url)
        content = response.read()
        return content

    except urlib2.HTTPError:
        return

    finally:
        response.close()

def parse_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    summary = soup.find('span', {'summary'})

    return summary.get_text()

def main():
    df = pd.DataFrame()
    df['url'] = get_urls()
    df['summary'] = df['url'].apply(parse_content)

    matrix, features = vectorizer(df['summary'])

def vectorizer(corpus):
    vectorizer = TfidfVectorizer(max_features=30,
                                lowercase=True,
                                max_df=0.7,
                                min_df=0.1,
                                use_idf=True,
                                stop_words='english',
                                norm='l2',
                                ngram_range=(1, 1),
                                analyzer='word',
                                decode_error='ignore',
                                strip_accents='unicode'
                                )

    matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()

    return matrix, features
