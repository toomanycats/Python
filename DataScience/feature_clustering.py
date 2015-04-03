"""This program used TF-IDF vectorization to
cluster words used in a document. A gephi friendly
graphml file is output.
"""
import ipdb
from os import path
from optparse import OptionParser

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
import numpy as np
import re

from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import nltk
from nltk import tokenize
from nltk import stem

import networkx as nx
import MiscTools

misctools = MiscTools.MiscTools()
stemmer = stem.SnowballStemmer('english')
pat = "^[NVJR]" # defined here for speed
reg = re.compile(pat)

class Visualizations(object):

    def __init__(self, data):
        self.data = data

    def make_graph(self, words, dist):
        graph = nx.Graph()
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                w = dist[i,j]
                graph.add_edge(word1, word2, weight=float(w))

        return graph

    def prep_dim_reduction_for_gephi(self, pos):
        dim = pos.shape[0]
        dist = np.zeros((dim, dim), dtype=np.float32)
        for ind, row in enumerate(pos):
            diff = row - pos
            dist[ind, :] = np.sqrt(np.sum(diff**2, axis=1))

        return dist

    def get_dim_reduction(self, n_components=2):
        '''
        http://stackoverflow.com/questions/16990996/multidimensional-scaling-fitting-in-numpy-pandas-and-sklearn-valueerror
        '''
        similarities = cosine_distances(self.data)
        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=n_components,
                           max_iter=1000,
                           eps=1e-6,
                           random_state=seed,
                           dissimilarity="precomputed",
                           n_jobs=-1,
                           n_init=8
                           )

        pos = mds.fit(similarities).embedding_

        return pos

    def show_in_2D(self, features, kw=None):
        pos = self.get_dim_reduction(2)

        num_annotes = pos.shape[0]
        lower = -num_annotes / 2 - 20
        upper = -lower + 20
        pts = np.arange(lower, upper, 5, dtype=int)
        rpos = np.random.choice

        x = pos[:,0]
        y = pos[:,1]

        plt.figure()
        plt.scatter(x, y,
                    marker='o',
                    s=300,
                    alpha=0.65
                    )

        bbox_dict = dict(boxstyle='round, pad=0.3',
                         fc='yellow',
                         alpha=0.4
                         )

        for label, x, y in zip(features, pos[:, 0], pos[:, 1]):
            plt.annotate(label,
                         xy=(x,y),
                         xytext=( rpos(pts), rpos(pts) ),
                         #xytext=(-20, 20),
                         textcoords = 'offset points',
                         ha='right',
                         va='bottom',
                         bbox=bbox_dict,
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0')
                         )

        if kw is None:
            plt.title('Word Clustering with %i words.' %len(features))

        else:
            plt.title("word Clustering with Key Word:%s and %i words." %(kw, len(features)))

    def show_3D_in_2D(self, features, kw, radius, pos_filter):
        '''
        http://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot
        '''

        pos = self.get_dim_reduction(3)

        num_annotes = pos.shape[0]
        lower = -num_annotes / 2 - 20
        upper = -lower + 20
        pts = np.arange(lower, upper, 5, dtype=int)
        rpos = np.random.choice

        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        plt.figure()
        plt.scatter(x, y,
                    marker='o',
                    c=z,
                    s=300,
                    cmap = plt.get_cmap('Spectral'),
                    alpha=0.65
                    )

        bbox_dict = dict(boxstyle='round, pad=0.3',
                         fc='yellow',
                         alpha=0.4
                         )

        for label, x, y, z in zip(features, pos[:, 0], pos[:, 1], pos[:,2]):
            plt.annotate(label,
                         xy=(x,y),
                         #xytext=(-25, 20),
                         xytext=( rpos(pts), rpos(pts) ),
                         #xytext=(-20, 20),
                         textcoords = 'offset points',
                         ha='right',
                         va='bottom',
                         bbox=bbox_dict,
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0')
                         )

        plt.colorbar()
        if kw is None:
            plt.title('Word Clustering with %i words.' %len(features))

        else:
            string = "Word Clustering with Key Word:%(kw)s, radius:%(rad)i, POS Filter:%(pos)s  %(n_words)i words."
            string = string %{'kw':kw,
                              'rad':radius,
                              'pos':pos_filter,
                              'n_words':len(features)
                              }

            plt.title(string)


class DocumentClustering(object):
    def __init__(self):
        pass
        ### Documents ###
        ### k-means ###
#         matrix = self.feature_extraction(df[self.df_key], norm='l1', max_features=100)
#         matrix_dim = self.dim_reduction(matrix, n_components=30)
#         n = np.arange(0, matrix.shape[0])
#         assignments = self.kmean_cluster(matrix_dim, n, 10)
#         df['Doc_Labels'] = assignments
#
#         self.print_text_by_label(df, label=0)

    def kmean_cluster(self, data, k=12):
        km = KMeans(k)

        assignments = km.fit_predict(data)

        return  assignments

    def dbscan_plotting(self, matrix, min_samp):
        n_clusters = np.zeros(18, dtype=np.float32)
        noise = n_clusters.copy()
        eps_array = noise.copy()

        for  i, eps in zip(range(18), np.arange(0.05, 0.9, 0.05)):
            db_labels = self.dbscan_cluster(matrix, eps, min_samp)
            n_clusters[i] = max(db_labels)
            noise[i] = db_labels[db_labels == -1].size
            eps_array[i] = eps

        plt.figure()
        plt.plot(eps_array, np.log(n_clusters+1), 'go')
        plt.plot(eps_array, np.log(noise+1), 'rx')

    def dbscan_cluster(self, matrix, eps, min_samp):
        # min_samples being 2 essentially turns this into hierarchical single linkage.
        # similar to http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html
        # Want to use full DBSCAN api anyways for future testing when min_samples > 2 (clustering retweets for example).

        db = DBSCAN(eps=eps, min_samples=min_samp)
        db.fit(matrix.toarray())

        return db.labels_

    def dim_reduction(self, matrix, n_components):
            svd_ = TruncatedSVD(n_components)
            lsa = make_pipeline(svd_, Normalizer(copy=False))

            out = lsa.fit_transform(matrix)


            explained_variance = svd_.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}".format(int(explained_variance * 100)))

            return out

    def print_text_by_label(self, df, label):
        text =  df[df['Doc_Labels'] == label]['Contents']
        for line in text:
            print line + '\n********\n'


class WordClustering(object):

    def __init__(self, file_path, df_key, ngram_range, radius, key_word, stop_words, join_words, pos_filter):
        self.input_file = file_path
        self.df_key = df_key
        ngram_range_str_list = ngram_range.split(',')
        self.ngram_range = [ int(ngram_range_str_list[0]), int(ngram_range_str_list[1]) ]
        self.radius = radius
        self.key_word = key_word
        self.additional_stop_words = ['http', 'rt', 'amp', 'don'] + stop_words.split(',')
        self.set_join_words(join_words)
        self.pos_filter = pos_filter

    def set_join_words(self, join_words):
        default_words = ['on line']
        if join_words is not None:
            self.words_to_join = join_words.split(',') + default_words
        else:
            self.words_to_join = default_words

        self.words_to_join = map(lambda word: word.replace(' ', ''), self.words_to_join )

    def add_to_stop_words(self):
        self.stop_words = ENGLISH_STOP_WORDS.union(self.additional_stop_words)

    def feature_extraction(self, df_series, norm, max_features):
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          max_df=0.65,
                                          min_df=0.01,
                                          stop_words=self.stop_words,
                                          use_idf=True,
                                          norm=norm,
                                          ngram_range=(self.ngram_range[0], self.ngram_range[1]),
                                          analyzer='word',
                                          decode_error='ignore',
                                          strip_accents='unicode'

                                          )


        out = self.vectorizer.fit_transform(df_series)
        self.features = self.vectorizer.get_feature_names()

        return out

    def load_excel(self, File, sheetname):
        df = pd.read_excel(File, sheetname=sheetname, encoding='utf-8')

        return df

    def find_words_in_radius(self, series, keyword):
        if self.radius == 0:
            return series

        words_in_radius = []

        for string in series:
            test = string.encode('ascii', 'ignore')
            test = tokenize.word_tokenize(test)
            test = np.array(test)

            kw_ind = np.where(test == keyword)[0]
            if len(kw_ind) == 0: #empty
                continue

            #replace all fuzzy matched words with the original
            # keyword
            test[kw_ind] = self.key_word

            src_range = self._find_words_in_radius(kw_ind, test)

            temp = " ".join(test[src_range])
            #temp = temp.replace(keyword, '')
            words_in_radius.append(temp)

        return words_in_radius

    def _find_words_in_radius(self, kw_ind, test):
            # can be more than one kw in a string
            lower = kw_ind - self.radius
            upper = kw_ind + self.radius

            #src_range = np.zeros((lower.shape[0], len(test)), dtype=np.int16)
            # if there's only one kw match, reduce the dim
            #src_range = np.squeeze(src_range)

            src_range_tot = np.empty(0, dtype=np.int16)
            for i in range(lower.shape[0]):
                src_range = np.arange(lower[i], upper[i])

                # truncate search range to valid values
                # this operation also flattens the array
                src_range = src_range[(src_range >= 0) & (src_range < test.size)]
                src_range_tot = np.hstack((src_range_tot, src_range))

            return np.unique(src_range_tot)

    def ward(self, dist_cos, dist_eucl, features):
        # scipy's ward implementation
        plt.figure()
        plt.subplot(121)
        plt.title("Cosine")
        link = ward(dist_cos)
        dendrogram(link, orientation="right", labels=features)
        plt.tight_layout()

        plt.subplot(122)
        plt.title("Euclidean")
        link = ward(dist_eucl)
        dendrogram(link, orientation="right", labels=features)
        plt.tight_layout()

    def join_words(self, series):
        '''Some words should be joined in preprocessing, i.e., "whole foods",
        "salad bar". '''

        for ind, item in enumerate(series):
            temp = item
            for word in self.words_to_join:
                temp = re.sub(pattern=word, repl=word, string=temp, flags=re.I)

            series.loc[ind] = temp

        return series

    def stemmer(self, series):
        # run the stemmer on each doc in the series
        for ind, doc in enumerate(series):
            words = tokenize.word_tokenize(doc)
            stems = map(stemmer.stem, words)
            new_doc = " ".join(stems)
            series.iloc[ind] = new_doc

        return series

    def _get_pos(self, string):
        test = string.encode('ascii', 'ignore')
        test = tokenize.word_tokenize(test)
        pos = nltk.pos_tag(test)

        return pos

    def _filter_pos(self, pos):
        out = []
        for item in pos:
            if reg.match(item[1]):
                out.append(item[0])

        return out

    def reduce_parts_of_speech(self, series):
        for ind, string in enumerate(series):
            pos = self._get_pos(string)
            words = self._filter_pos(pos)

            series.loc[ind] = " ".join(words)

        return series

    def main(self):

        if  path.splitext(self.input_file)[1] == ".xlsx":
            df = self.load_excel(self.input_file, sheetname=0)

        elif  path.splitext(self.input_file)[1] == ".csv":
            df = misctools.load_csv(self.input_file, skip_rows=0)

        else:
            raise Exception("File type unknown or not accepted.Extension: %s" % path.splitext(self.input_file)[1])

        df.dropna(axis=0, how='any', subset=[self.df_key], inplace=True)
        df.drop_duplicates(subset=self.df_key, inplace=True)

        # deal with a date string getting thrown in and
        # causing a heterogenious data type, and  datetime objects
        series = df[self.df_key].astype(unicode)
        # deal with things like: "whole foods" and "Wal Mart
        # should be "wholefoods"
        series = self.join_words(series)
        # return all words in the corpus to their stem
        series = self.stemmer(series)
        # reduce pos to nouns, verbs and adjectives
        if self.pos_filter:
            series = self.reduce_parts_of_speech(series)

        self.add_to_stop_words()
        if self.radius > 0:
            words_in_rad = self.find_words_in_radius(series, keyword=self.key_word)
            matrix = self.feature_extraction(words_in_rad, norm='l2', max_features=50)
        else:
            matrix = self.feature_extraction(series, norm='l2', max_features=50)

        vis = Visualizations(matrix.T)
        #ipdb.set_trace()
        positions = vis.get_dim_reduction(3)
        distances = vis.prep_dim_reduction_for_gephi(positions)
        graph = vis.make_graph(self.features, distances)
        #graph.write_graphml('/Users/dcuneo/Desktop/feature_cluster.graphml')

        f = open('/Users/dcuneo/Desktop/word_cluster.graphml', 'w')
        nx.GraphMLWriter(graph).dump(f)
        f.close()

        #cos_dist = cosine_distances(matrix.T)
        #euc_dist = euclidean_distances(matrix.T)

        #self.ward(cos_dist, euc_dist, self.features)

        #plt.show()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file-path", dest="file_path", type="string",
                      help="path to csv or xlsx file")

    parser.add_option("-c", "--column-name", dest="df_key", default="Title", type="string",
                      help="""The name of the column which contains the conversation to be analyzed. Typically
                      its Title or title or titles""")

    parser.add_option("-n", "--ngram-range", dest="ngram_range", default="1,1", type="string",
                      help="""The number of words in the ngram to use when searching the corpus for
                      terms. It needs to be a range, i.e., "1,2" or "2,2", or "2,3". The default is "1,1". """)

    parser.add_option("-r", "--radius", dest="radius", default=0, type="int",
                      help="""you may optionally restrict your search in behind and in front of a word, "in a radius".
                      All the words in the 'radius' will be searched. Default is 0 which turns this feature off. """)

    parser.add_option("-w", "--key-word", dest="key_word", type="string", default='None',
                      help="""The key word is used with the radius feature. This is the word you will be centered around. Fuzzy match
                      option pertains to this word.""")

    parser.add_option("-s", "--stop-words", dest="stop_words", default="", type="string",
                      help="""These are the additional stop words you want to include. The default already includes 'http' and 'rt.'
                      These words are entered as like so: "cancer, drug, medicine".""")

    parser.add_option("-j", "--join-words", dest="join_words",
                      help="""A list of words that should be joined, that can contain a space, ie 'whole foods'. """)

    parser.add_option("-p", "--pos", dest="pos_filter", action="store_true", default=False,
                      help="""Filter the bag of words to only contain: Nouns, Adjectives, Verbs and Adverbs. Prog will run slow. """)

    (options, args) = parser.parse_args()

    term = WordClustering(options.file_path,
              options.df_key,
              options.ngram_range,
              options.radius,
              options.key_word,
              options.stop_words,
              options.join_words,
              options.pos_filter
              )

    if options.key_word != 'None' and options.radius == 0:
        raise Exception("You have set a keyword, but no search radius.")

    # file_path, df_key, ngram_range, radius, fuzzy_match, key_word, stop_words
    term.main()

