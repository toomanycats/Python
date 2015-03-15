from functools import partial
import csv
from multiprocessing import Pool
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
import nltk
import re
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle

stemmer = nltk.stem.SnowballStemmer('english')
pat = "^(NN)" # defined here for speed
reg = re.compile(pat)
regular = 0.0 

hash_ob = re.compile('.*(#[A-Za-z]{2,}).*')

# file paths for testing
words_csv = "/Users/dcuneo/cus_matched_words.pkl"

def reduce_to_hash_tag(text):
    if hash_ob.search(text):
        groups = hash_ob.search(text).groups()
        red_text = []
        for tag in groups:
            red_text.append(tag)

        return " ".join(red_text)
    else:
        return " "

def remove_RT(text):
    pat = "^(RT @WholeFoods).*"
    if re.match(pat, text):# can use match
        return ''
    else:
        return text

def stemmer_(text):
    #pdb.set_trace()
    # run the stemmer on each doc in the brand
    if isinstance(text, unicode) or isinstance(text, str):
        text = nltk.tokenize.word_tokenize(text)
    stems = map(stemmer.stem, text)
    text = " ".join(stems)
        
    return text    

def _get_pos(string):
    test = string.encode('ascii', 'ignore')
    test = np.array(nltk.tokenize.word_tokenize(test))
    
    lens = np.array(map(len, test), dtype=np.int16)
    keep = np.where(lens > 3)[0]
    test = test[keep]

    pos = nltk.pos_tag(list(test))

    return pos

def _filter_pos(pos):
    out = []
    for item in pos:
        if reg.match(item[1]):
            out.append(item[0])
    
    # keep shape if no nouns are found
    if len(out) > 0:
        return out
    else:
        return " "

def reduce_parts_of_speech(text):
    text = remove_urls(text)

    pos = _get_pos(text)
    words = _filter_pos(pos)
    words = np.unique(words) 
    
    words = stemmer_(words)

    return words    

def test_for_tags(brand_tags, text):
    cus_tags = np.array(reduce_to_hash_tag(text))
    count = 0
    match =[]
    intersect_tags = np.intersect1d(brand_tags, cus_tags)
    
    for tag in intersect_tags:
        if tag in cus_tags:
            count += 1
            match.append(tag)
    
    return match, count

def test_for_words(args):
    words = args[0]
    tweet = args[1]
    count  = regular #avoid NaN's
    #pdb.set_trace() 

    match_words = []
    for word in words.split(' '):
        if word in tweet:
            count += 1
            match_words.append(word)

    return count, match_words

def add_lag(cov, lag):
    hits = cov.nonzero()[0]
    
    n = hits.size - 1
    for ind in hits:
        if ind + lag < n:
            cov.loc[ind:ind+lag] += 1
        else:
            cov.loc[ind:] += 1
        
    return cov     

def remove_urls(text):
    pat = '(http*).*'
    
    text = re.sub(pattern=pat, repl='', string=text) 

    return text

def unique_cus_words(text):
    text = text.encode('ascii', 'ignore')
    words = nltk.tokenize.word_tokenize(text)
    words = list(np.unique(words))

    text = " ".join(words)

    return text            

def save_date_vector(ind):
    dates = np.unique(ind)
    dates = [d[0:10] for d in dates]
    np.savetxt('/Users/dcuneo/date_vector.txt', dates, fmt="%s")
    print "saved date vector"

def plot_data(cus, brand, dates, words):
    ind = np.where(cus != 0)[0]
    
    plt.subplot(211)
    plt.title("Customer word adoption")
    plt.plot(brand)
    plt.xticks(ind, dates, rotation='vertical')

    plt.subplot(212)
    plt.plot(cus)
    plt.xticks(ind, words, rotation='vertical', fontsize=9)

    plt.tight_layout()

def hash_tag_analysis(brand, cus):
    ### hash analysis ###
    pool = Pool(processes=2)

    brand_tags = pool.map(reduce_to_hash_tag, brand)
    brand_tags = np.unique(brand_tags)
    print "brand hash tags done"
     
    # customer tags
    test = partial(test_for_tags, brand_tags) 
    out = pool.map(test, cus)
   
    tag_match = map(lambda out:out[0], out)
    tag_cnt = map(lambda out:out[1], out)

    plt.plot(tag_cnt)
    plt.show()

    return

def save_cus_words(file_, match_words):
    words = []
    for w in match_words:
        if len(w) > 0:
            temp = " ".join(np.unique(w))
            temp = str(temp)
            words.append(temp)
        else:
            words.append(" ")

    f = open(file_, "w")
    pickle.dump(words, f)
    f.close()

def cus_word_adoption(brand, cus):
    pool = Pool(processes=2)
    brand = pool.map(reduce_parts_of_speech, brand)
    print "Done with brand POS tagging." 

    # customer preprocess
    cus = pool.map(reduce_parts_of_speech, cus)
    print "Done with customer POS tagging." 
    
    args = zip(brand, cus)
    out  = pool.map(test_for_words, args) 
    print "Done with testing for words" 
    
    match_cnt = map(lambda out: out[0], out) 
    match_words = map(lambda out: out[1], out) 
    
    save_cus_words(words_csv, match_words)

    match_cnt = np.array(match_cnt, dtype=np.float32) 
    brand_cov = np.array(map(len, brand), dtype=np.float32)

    return match_cnt, brand_cov

def main():
    brand_ts = r"/Users/dcuneo/Downloads/brand.xlsx"
    brand = pd.read_excel(brand_ts) 
    brand.set_index(pd.DatetimeIndex(brand['Date(ET)']), inplace=True)

    cus_ts = r"/Users/dcuneo/Downloads/customer.xlsx"
    cus = pd.read_excel(cus_ts, skip_footer=0) 
    cus['title'] = cus['title'].astype(unicode)# avoids unexp date obj
    cus['title'] = pd.Series(map(remove_RT, cus['title']))
    cus.set_index(pd.DatetimeIndex(cus['dateet']), inplace=True)
   
    # Intersection of the time frames 
    ind = pd.DatetimeIndex.intersection(cus.index, brand.index)
    save_date_vector(ind)
    date_ind = pd.DataFrame(ind, columns=['dates'])
    
    brand = pd.merge(left=date_ind, right=brand, left_on=['dates'], right_on=['Date(ET)'])
    brand.set_index(pd.DatetimeIndex(brand['Date(ET)']), inplace=True)
    
    cus  = pd.merge(left=date_ind, right=cus, left_on=['dates'], right_on=['dateet'])
    cus.set_index(pd.DatetimeIndex(cus['dateet']),inplace=True)
    
    brand = brand["Title"]     
    brand = brand.groupby(brand.index).agg(sum)
    print "brand grouped"

    cus = cus['title'] 
    cus = cus.groupby(cus.index).agg(sum)
    print "cus grouped"
   
    cus_count, brand_cov = cus_word_adoption(brand, cus)
    np.save("/Users/dcuneo/brand_cov.npy", brand_cov)
    np.save("/Users/dcuneo/match_cnt.npy", cus_count)
    
    #plot_data(cus_count, brand_cov, date_ind, brand)

    # GLM
    model = sm.GLM(cus_count, brand_cov, family=sm.families.Gaussian())   
    res = model.fit()
    
    print(res.summary())
   
    cus_count[cus_count ==0] = 0.01
    corr = np.corrcoef(cus_count, brand_cov)
    print corr



if __name__ == "__main__":
    main()
