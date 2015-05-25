import re
import urllib2
import MiscTools
from bs4 import BeautifulSoup as bs4 
from scipy.signal import welch
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_clustering import DocumentClustering
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

misctools = MiscTools.MiscTools()
doc_clus = DocumentClustering()

input = r"C:\Users\dcuneo\TwitterTsExp\twitter_pull_karenw_dates_labels_title.csv"
outfile = r"C:\Users\dcuneo\TwitterTsExp\twitter_pull_karenw_dates_labels_full_text.csv"

class TwitterTimeSeries(object):
    def __init__(self, csv_input=None, csv_output=None):
        self.csv_input = csv_input
        self.csv_output = csv_output
        
    def _get_soup(self, url):
        response = urllib2.urlopen(url)
        html = response.read()
        response.close()
     
        soup = bs4(html)
    
        return soup
    
    def get_date(self, url):
    
        try:
            soup = self._get_soup(url)
          
            tweet_timestamp = soup.find_all('a', 'tweet-timestamp')
            
            date_string = tweet_timestamp[0].get('title')
            
            date_form = '%I:%M %p - %d %b %Y'
            date = pd.datetools.datetime.strptime(date_string, date_form)
            
            return date
        
        except:
            return pd.NaT 
    
    def get_title(self, url):
        soup = self._get_soup(url)
        
        title = tweet_text = soup.find_all('p', 'js-tweet-text')
        title = title[0].get_text('title').encode('ascii', 'ignore')
        
        return title
    
    def clean_title(self, title):
        # clean off the title@title<...>title
        #title#title<...>title
        
        pat = r'(?P<beg>.*)(title@title)(?P<word>\w+)(title)(?P<end>.*)'
        while re.search(pat, title):
                title = re.sub(pat, r'\g<beg>\g<word>\g<end>', title)
            
        pat = r'(?P<beg>.*)(title#title)(?P<word>\w+)(title)(?P<end>.*)'
        while re.search(pat, title):
                title = re.sub(pat, r'\g<beg>\g<word>\g<end>', title)
    
        pat = r'(?P<beg>.*)(#title)(?P<word>\w+)(title)(?P<end>.*)'
        while re.search(pat, title):
            title = re.sub(pat, r'\g<beg>\g<word>\g<end>', title)
        
        title = re.sub("\s(\w*http.*)", r'', title)
    
        return title
    
    def vectorizer(self, doc_series):
        stop_words = ENGLISH_STOP_WORDS.union(['karenmpd', 'RT','tinyurl','www','rt', 'twitter'])    
        vectorizer = TfidfVectorizer(max_features=200,
                                     max_df=0.50,
                                     min_df=0.01,
                                     use_idf=True,
                                     ngram_range=(1, 4),
                                     analyzer='word',
                                     decode_error='ignore',
                                     strip_accents='unicode',
                                     stop_words=stop_words           
                                     )
        
        out = vectorizer.fit_transform(doc_series)   
        features = vectorizer.get_feature_names()
    
        return out, features
    
    def make_df(self):
        tweet_links = self.df['tweetlink']
        
        dates = []
        titles = []
        
        # texts is already there...!!
        for link in tweet_links:
            dates.append( self.get_date(link) )
            t =  self.get_title(link) 
            titles.append( self.clean_title(t) )
            
        self.df['clean_title'] = pd.Series(titles)    
        self.df['dates'] = pd.Series(dates)    
        self.df.dropna(axis=0, how='any', subset=['text', 'tweetlink', 'dates', 'clean_title'], inplace=True)
    
    def print_labeled_titles(self, label):
        labs = self.df['doc_labels']
        mask = labs == label
        
        titles = self.df.clean_title[mask]
        for line in titles:
            print line + '\n\n'
          
    def get_label_ts(self, label):
        labs = self.df['doc_labels']
        
        mask = labs == label
        mask = mask.apply(lambda x: 1 if x else 0)
    
        return mask
    
    def fft(self, data, nperseg, nfft, plot=False, interval=5):
        freq, y = welch(x=data, 
                        fs=1.0, 
                        window='hanning', 
                        nperseg=nperseg,
                        noverlap=None, # default is 50%
                        nfft=nfft, 
                        detrend='linear',
                        return_onesided=True, 
                        scaling='density',
                        axis=-1
                        )
             
        if plot:
            self.plot_fft(freq, y, interval)
            
        return freq, y
    
    def _db(self, y):
        '''y sold be output of "density" --> V**2/Hz '''
        y = np.sqrt(y)
        y /= np.sum(y)
        
        db = np.log2(y)
        
        return db
    
    def plot_fft(self, f, y, interval):
        #minorLocator = MultipleLocator(interval/5)
        majorLocator  = MultipleLocator(interval)
       
        ax = plt.subplot()
        ax.xaxis.set_major_locator(majorLocator)
        #ax.xaxis.set_minor_locator(minorLocator)
        
        tck =  np.round( 1/f[0::interval], 1 )
        plt.xticks(f[0::interval], tck, rotation="vertical")
        
        #y = self._db(y)
        plt.plot(f, y)
        plt.show()    

    def down_sample(self, label_ts):    
        events_df = pd.DataFrame(data=label_ts)
        events_df.set_index(pd.DatetimeIndex(self.df.dates), inplace=True)
        
        events_df = events_df.resample(rule='D', how=np.sum)
        events_df.fillna(value=0, inplace=True)
      
        return events_df.squeeze()
    
    def clean_soup(self, soup):
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out
        
        # get text
        text = soup.get_text()
        
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        text = text.encode('ascii', 'ignore')
        return text
     
    def main(self, input, outfile):
        
        self.df = misctools.load_csv(input)
        self.df.drop_duplicates(subset=['text'], inplace=True)  
        self.df.dropna(axis=0, how='any', subset=['link'], inplace=True)   
        #file_ = open("C:\Users\dcuneo\TwitterTsExp\web_text.txt", "w")   

        text = []
        for ind, url in enumerate(self.df['link']): 
            try:
                soup = self._get_soup(url)
                #text = soup.p.get_text()
                text.append(self.clean_soup(soup))
                #file_.write(text)
                #file_.write('\n\n') 
                print ind
            except:
                print "url failed:%s" %url
        
        #file_.seek(0)
        matrix, features = self.vectorizer(text)
        np.save(r"C:\Users\dcuneo\TwitterTsExp\article_matrix.npy", matrix)
        #file_.close()
        
        matrix = doc_clus.dim_reduction(matrix, n_components=150)
    
        assignments = doc_clus.kmean_cluster(matrix, 5) 
        
        self.df['doc_labels'] = pd.Series(assignments)
        
        self.df = self.df.groupby(by=['doc_labels', 'dates']).max()
        
        self.df.to_csv(path_or_buf=outfile, encoding='utf-8')       
        
if __name__ == "__main__":
    twit = TwitterTimeSeries()
    twit.main(input, outfile)
        
        
        
    
    
    