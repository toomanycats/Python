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
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
from nltk import stem
from nltk import tokenize
import re

toker = tokenize.word_tokenize
stemmer = stem.SnowballStemmer('english')


class Indeed(object):
    def __init__(self):
        self.add_loc = None
        self.stop_words = None
        self.num_samp = 300
        self.zip_code_file ='/home/daniel/git/Python2.7/DataScience/indeed/us_postal_codes.csv'
        self.df = pd.DataFrame()
        self.config_path = "/home/daniel/git/Python2.7/DataScience/indeed/tokens.cfg"
        self.query = None
        self.locations = None

    def add_stop_words(self):
        if self.stop_words is not None:
            words = self.stop_words.split(" ")
            self.stop_words = ENGLISH_STOP_WORDS.union(words)

    def build_api_string(self):
        if self.query is None:
            print "query cannot be empty"
            raise ValueError

        # beware of escaped %
        query = self.format_query(self.query)
        prefix = 'http://api.indeed.com/ads/apisearch?'
        pub = 'publisher=%(pub_id)s'
        chan = '&chnl=%(channel_name)s'
        loc = '&l=%(loc)s'
        query = '&q=%s' % query
        start = '&start=0'
        frm = '&fromage=30'
        limit = '&limit=25'
        site = '&st=employer'
        format = '&format=json'
        sort = '&sort=0'
        country = '&co=us'
        suffix = '&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'

        self.api = prefix + pub + chan + loc + query + start + frm + limit + \
                   site + format + country + sort + suffix

    def format_query(self, query):
         return "%%20".join(query.split(" "))

    def load_config(self):
        '''loads a config file that contains tokens'''
        config_parser = ConfigParser.RawConfigParser()
        config_parser.read(self.config_path)
        self.pub_id = config_parser.get("id", "pub_id")
        self.channel_name = config_parser.get("channel", "channel_name")

    def load_zipcodes(self):
        '''loads the zip code file and returnes a list of zip codes'''
        self.df_zip = pd.read_csv(self.zip_code_file, dtype=str)
        self.df_zip.dropna(inplace=True, how='all')
        locations = self.df_zip['Postal Code'].dropna(how='any')

        return locations

    def get_city_url_content_stem(self):
        ind = 0
        for zipcode in self.locations:
            url_city = self.get_url(zipcode)
            if url_city is not None:
                for item in url_city:
                    self.df.loc[ind, 'zipcode'] = str(zipcode)
                    self.df.loc[ind, 'url'] = item[0]
                    self.df.loc[ind, 'city'] = item[1]
                    content = self.parse_content(item[0])
                    self.df.loc[ind, 'summary'] = content
                    self.df.loc[ind, 'summary_stem'] = self.stemmer_(content)
                    ind += 1

    def get_urls(self):
        urls = []
        for item in self.locations:
            url = self.get_url(item)
            if url is not None:
                urls.extend(url)

        return urls

    def get_url(self, location):
        '''method good for use with MapReduce'''

        api = self.api %{'pub_id':self.pub_id,
                         'loc':location,
                         'channel_name':self.channel_name
                        }

        try:
            response = urllib2.urlopen(api)
            data = json.loads(response.read())
            response.close()

            urls = []
            urls.extend([ (item['url'], item['city']) for item in data['results']])

        except urllib2.HTTPError, err:
            print err
            return None

        except Exception, err:
            print err
            return None

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

    def stemmer_(self, string):

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
        soup = BeautifulSoup(content, 'lxml')

        try:
            summary = soup.find('span', {'summary'})

        except AttributeError:
            summary = soup.find_all('span')

        if summary is None:
            return None

        bullets = summary.find_all("li")

        if bullets is not None and len(bullets) > 2:
            skills = bullets
        else:
            skills = summary

        try:
            output = [item.get_text() for item in skills]

        except AttributeError:
            return None

        if len(output) > 0:
            return " ".join(output)
        else:
            return None

    def parse_zipcode_beg(self):
        '''locs are zipcode prefixes, like:902, provided as string'''
        pat = '^%s' %self.add_loc
        obj = re.compile(pat)

        self.df_zip['include'] = self.df_zip['Postal Code'].apply(lambda x: 1 if obj.match(x) else 0)
        zips = self.df_zip[self.df_zip['include']==1]['Postal Code']

        return zips.tolist()

    def handle_locations(self):
        '''main method for setting up the locations for the API call'''
        locations = self.load_zipcodes()
        locations = locations.sample(self.num_samp).tolist()
        if self.add_loc is not None:
            for loc in self.parse_zipcode_beg():
                locations.append(loc)

        locations = np.unique(locations)
        np.random.shuffle(locations)
        # if we stop the program and take the avaiable output we'd like it to
        # be radomly sampled

        return locations

    def save_data(self):

        self.df.drop_duplicates(subset=['url'], inplace=True)
        self.df.dropna(inplace=True, how='any', axis=0)

        self.df.to_csv('/home/daniel/git/Python2.7/DataScience/indeed/data_frame.csv',
                        index=False,
                        encoding='utf-8')

    def main(self):
        '''Run all the steps to collect data and produce a bi-gram
        bar plot of the keyword counts. Terminating the program execution
        with control C, will save whatever data was collected. To make this
        save-on-quit feature more usable, the locations are shuffled prior to
        getting the content.'''

        self.load_config()
        self.build_api_string()
        self.add_stop_words()
        # allow the user to specify locations
        if self.locations is None:
            self.locations = self.handle_locations()

        try:
            self.get_city_url_content_stem()

        except KeyboardInterrupt:
            print "Quiting job, saving data."
            self.save_data()
            raise

        #self.save_data()

    def vectorizer(self, corpus, max_features=200, max_df=0.8, min_df=0.1, n_min=2):
        vectorizer = CountVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    lowercase=True,
                                    stop_words=self.stop_words,
                                    ngram_range=(n_min, 3),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode'
                                    )

        matrix = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()

        return matrix, features

    def plot_sorted_subsection(self, fea, mat, start, end):
        '''Plot the next n highest counts in sorted order from n to m'''

        m = mat.toarray().sum(axis=0)

        ind_sort = np.argsort(m)
        m = m[ind_sort]

        f = np.array(fea)
        f = f[ind_sort]

        range_ = end - start
        if range_ < 0:
            raise ValueError

        x = np.arange(range_)

        plt.xticks(x, f, rotation=90, fontsize=14)
        plt.bar(x, m[start:end], align='center')

    def plot_features(self, features, matrix):
        x = np.arange(len(features))
        tot = np.array(matrix.sum(axis=0))
        tot = np.squeeze(tot)

        plt.bar(x, tot, align='center', alpha=0.5)
        plt.xticks(x, features, rotation='vertical', fontsize=15)
        plt.grid(True)
        plt.xlim(-1, len(x))

        plt.ylabel("Counts", fontsize=15)
        string = "Key Word Count: %i Features" %len(features)
        plt.title(string,  fontsize=15)
        plt.tight_layout()

        plt.show()

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

    def find_words_in_radius(self, series, keyword, radius):

        words_in_radius = []

        for string in series:
            test = string.decode('ascii', 'ignore')
            test = tokenize.word_tokenize(test)
            test = np.array(test)

            kw_ind = np.where(test == keyword)[0]
            if len(kw_ind) == 0: #empty
                continue

            src_range = self._find_words_in_radius(kw_ind, test, radius)

            temp = " ".join(test[src_range])
            words_in_radius.append(temp)

        return words_in_radius

    def _find_words_in_radius(self, kw_ind, test, radius):
            # can be more than one kw in a string
            lower = kw_ind - radius
            upper = kw_ind + radius

            src_range_tot = np.empty(0, dtype=np.int16)

            for i in range(lower.shape[0]):
                src_range = np.arange(lower[i], upper[i])

                # truncate search range to valid values
                # this operation also flattens the array
                src_range = src_range[(src_range >= 0) & (src_range < test.size)]
                src_range_tot = np.hstack((src_range_tot, src_range))

            return np.unique(src_range_tot)


if __name__ == "__main__":
    ind = Indeed()
    ind.main()
