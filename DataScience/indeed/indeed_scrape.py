######################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
import json
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from sklearn.cluster import KMeans
from sklearn import metrics
import ipdb

class Indeed(object):
    def __init__(self):
#http://api.indeed.com/ads/apisearch?publisher=2818745629107228&q=%22data%20science%22&start=0&limit=4000&st=employer&format=json&co=us&fromage=360&userip=1.2.3.4&useragent=Mozilla/%2F4.0%28Firefox%29&v=2
        self.pub_id = "2818745629107228"
        self.api = r'http://api.indeed.com/ads/apisearch?publisher=%(pub_id)s&l=%(loc)s&q=%%22data%%20science%%22&start=0&limit=25&st=employer&format=json&co=us&fromage=360&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'


    def get_urls(self):
        df= pd.read_csv('/home/daniel/git/Python2.7/DataScience/indeed/us_postal_codes.csv', dtype=str)
        df.dropna(inplace=True)

        locations = df['Postal Code'].dropna()
        loc_samp = locations.sample(100)

        urls = []
        for loc in loc_samp:
            api = self.api %{'pub_id':self.pub_id,
                            'loc':loc
                            }

            try:
                response = urllib2.urlopen(api)
                data = json.loads(response.read())
                response.close()

                try:
                    url = [item['url'] for item in data['results']]

                except Exception, err:
                    print err
                    continue

            except urllib2.HTTPError, err:
                print err
                continue


            urls.append(url)

        return urls

    def open_url(self, url):
        try:
            response = urllib2.urlopen(url)
            content = response.read()
            return content

        except urlib2.HTTPError:
            return

        finally:
            response.close()

    def parse_content(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        summary = soup.find('span', {'summary'})

        return summary.get_text()

    def main(self):
        ipdb.set_trace()
        df = pd.DataFrame()
        df['url'] = self.get_urls()
        df['summary'] = df['url'].apply(self.parse_content)
        df = df.duplicated(keep='last')

        matrix, features = self.vectorizer(df['summary'])
        print features

        df['assignments'] = self.cluster(matrix)
        grp = df.groupby('assignments')
        print grp

    def vectorizer(self, corpus):
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

    def cluster(self, matrix):

        cos_dist = cosine_distances(matrix)
        euc_dist = euclidean_distances(matrix)

        km = KMeans(k=3,
                    init='random',
                    #init='k-means++',
                    n_init=100,
                    max_iter=1000,
                    tol=1e-7,
                    precompute_distances=False,
                    verbose=0,
                    random_state=1,
                    copy_x=True,
                    n_jobs=4
                    )

        assignments = km.fit_predict(cos_dist)

        return assignments

if __name__ == "__main__":
    indeed = Indeed()
    indeed.main()
