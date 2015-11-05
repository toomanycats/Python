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
from nltk import stem
from nltk import tokenize

toker = tokenize.word_tokenize
stemmer = stem.SnowballStemmer('english')

class Indeed(object):
    def __init__(self):
        self.pub_id = "2818745629107228"
        self.api = r'http://api.indeed.com/ads/apisearch?publisher=%(pub_id)s&l=%(loc)s&q=%%22data%%20science%%22&start=0&fromage=30&limit=25&st=employer&format=json&co=us&fromage=360&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'

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
                    urls.extend([item['url'] for item in data['results']])

                except Exception, err:
                    print err
                    continue

            except urllib2.HTTPError, err:
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

        except urlib2.HTTPError:
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
        words = toker(string)
        words = self.len_tester(words)
        words = map(stemmer.stem, words)

        return " ".join(words)

    def parse_content(self, url):
        content = self.get_content(url)

        if len(content) > 0:
            content = content.decode("ascii", "ignore")
            soup = BeautifulSoup(content, 'html.parser')
            summary = soup.find('span', {'summary'})

            return summary.get_text()

        else:
            return None

    def main(self):
        df = pd.DataFrame()
        df['url'] = self.get_urls()
        df['summary'] = df['url'].apply(lambda x:self.parse_content(x))
        df['summary_toke'] = df['summary'].apply(lambda x: self.tokenizer(x))
        df.drop_duplicates(inplace=True)

        matrix, features = self.vectorizer(df['summary_toke'])
        print features

        df['assignments'] = self.cluster(matrix)
        df.to_csv('/home/daniel/git/Python2.7/DataScience/indeed/data_frame.csv', index=False)

        grp = df.groupby('assignments')
        print grp.describe()

    def vectorizer(self, corpus, max_features=100, max_df=0.65, min_df=0.2):
        vectorizer = TfidfVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    to_lower=True,
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

        assignments = km.fit_predict(cos_dist)

        return assignments

if __name__ == "__main__":
    indeed = Indeed()
    indeed.main()
