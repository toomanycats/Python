######################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
#TODO: add logger
#TODO: use MRJob mapper for parsing

import ConfigParser
import json
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from nltk import stem
from nltk import tokenize

toker = tokenize.word_tokenize
stemmer = stem.SnowballStemmer('english')

class Indeed(object):
    def __init__(self):
        self.config_path = "/home/daniel/git/Python2.7/DataScience/indeed/tokens.cfg"
        self.load_config()
        self.api = 'http://api.indeed.com/ads/apisearch?publisher=%(pub_id)s&chnl=%(channel_name)s&l=%(loc)s&q=%%22data%%20science%%22&start=0&fromage=30&limit=25&st=employer&format=json&co=us&fromage=360&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'

    def load_config(self):
        config_parser = ConfigParser.RawConfigParser()
        config_parser.read(self.config_path)
        self.pub_id = config_parser.get("id", "pub_id")
        self.channel_name = config_parser.get("channel", "channel_name")

    def get_urls(self):
        df= pd.read_csv('/home/daniel/git/Python2.7/DataScience/indeed/us_postal_codes.csv', dtype=str)
        df.dropna(inplace=True)

        locations = df['Postal Code'].dropna()
        loc_samp = locations.sample(100)

        urls = []
        for loc in loc_samp:
            api = self.api %{'pub_id':self.pub_id,
                             'loc':loc,
                             'channel_name':self.channel_name
                             }

            try:
                response = urllib2.urlopen(api)
                data = json.loads(response.read())
                response.close()

                try:
                    urls.extend([item['url'] for item in data['results']])

                except Exception, err:
                    print err
                    continue

            except urllib2.HTTPError, err:
                print err
                continue

            except Exception, err:
                print err
                continue

        return urls

    def get_content(self, url):
        if url is None:
            return None

        try:
            response = urllib2.urlopen(url)
            content = response.read()
            response.close()

            return content

        except urllib2.HTTPError, err:
            print err
            return None

        except Exception, err:
            print err
            return None

    def len_tester(self, word_list):
        new_list = []
        for word in word_list:
            if len(word) < 3:
                continue
            else:
                new_list.append(word)

        return new_list

    def tokenizer(self, string):
        if string is None:
            return None

        words = toker(string)
        words = self.len_tester(words)
        words = map(stemmer.stem, words)

        return " ".join(words)

    def parse_content(self, url):
        content = self.get_content(url)

        if content is None:
            return None

        content = content.decode("ascii", "ignore")
        soup = BeautifulSoup(content, 'html.parser')

        try:
            summary = soup.find('span', {'summary'})

        except AttributeError:
            summary = soup.find_all('span')
            if summary is None:
                return None

        bullets = summary.find_all("li")
        if bullets is not None:
            skills = bullets
        else:
            skills = summary

        output = [item.get_text() for item in skills]

        if len(output) > 0:
            return " ".join(output)
        else:
            return None

    def main(self):
        df = pd.DataFrame()
        df['url'] = self.get_urls()
        df['summary'] = df['url'].apply(lambda x:self.parse_content(x))
        df['summary_toke'] = df['summary'].apply(lambda x: self.tokenizer(x))

        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True, how='any', axis=0)

        matrix, features = self.vectorizer(df['summary_toke'])
        fea = pd.DataFrame(features)
        fea.to_csv("/home/daniel/git/Python2.7/DataScience/indeed/features.txt", index=False)

        df['assignments'] = self.cluster(matrix)
        df.to_csv('/home/daniel/git/Python2.7/DataScience/indeed/data_frame.csv', index=False)

    def vectorizer(self, corpus, max_features=100, max_df=1.0, min_df=0.2, n_min=2):
        vectorizer = TfidfVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    lowercase=True,
                                    use_idf=False, # consider using CountVectorizer
                                    stop_words='english',
                                    norm='l2',
                                    ngram_range=(n_min, 3),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode'
                                    )

        matrix = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()

        return matrix, features

    def cluster(self, matrix):

        #cos_dist = cosine_distances(matrix)
        euc_dist = euclidean_distances(matrix)

        km = KMeans(n_clusters=3,
                    #init='random',
                    init='k-means++',
                    n_init=100,
                    max_iter=1000,
                    tol=1e-7,
                    precompute_distances=False,
                    verbose=0,
                    random_state=1,
                    copy_x=True,
                    n_jobs=1
                    )

        assignments = km.fit_predict(euc_dist)

        return assignments

if __name__ == "__main__":
    indeed = Indeed()
    indeed.main()
