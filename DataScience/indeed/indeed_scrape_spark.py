
# coding: utf-8

# # Applying Data Science to DS Job Hunting

# ## Indeed API
# Indeed.com offers a publisher's API for adding links in a web page or app. I decided to use this API
# to gather a sample of job posting from which to scrape a list of skills.
# 
# The API will only return a maximum of 25 url's, so one needs a trick to get a significant amount of data. The trick I'm using now, is to query by zipcode. There are ~43K in the US so that's going to hopefully bring us some hits.
# For now, I'm using 500 randomly selected samples of the ~43K zipcodes, returning from 0 to 25 urls from each.
# 
# ## Parsing Out Skills
# To parse out what I think are the skills, I use BeautifulSoup to iterate over the sections locating the bulleted points:
# <span class='summary'>
# <li> SQL
# <li> Python
# <li> AWS
# </span>
# 
# Visual inspection indicates that most of the time, an employer will use a list to itemize the position skills.
# It would be cool to run a second supporting project that tries to verify this. How many job posting contain any itemized lists versus those that do not ?
#  
# ### Stop Words
# I wanted a way to add new stop words. The word "data" obviously shows up many times and is not helpful.
# 
# ## Begin Analysis
# ### Bar Plot 
# To count up the parsed skill tokens, I employ SciKit-Learn's CountVectorizer and produce a simple bar plot output.
# 
# ### Locations
# For this example, I'm using all the zipcodes that start with '9' and 100 randomly selected samples. 

# In[1]:

from pyspark import  SparkContext
sc = SparkContext( 'local', 'pyspark')


# In[83]:

import pandas as pd
import indeed_scrape
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (15, 8)

get_ipython().magic(u'matplotlib inline')
ind = indeed_scrape.Indeed()
ind.query = "data science"
ind.add_loc = '90' # will add regex-ed zip codes
ind.num_samp = 0
ind.stop_words = "director media relations"


# In[84]:

ind.add_stop_words()
ind.load_config()
ind.build_api_string()
locs = ind.handle_locations()


# In[ ]:

locsRDD = sc.parallelize(locs)
url_city = locsRDD.flatMap(lambda x: ind.get_url(x))
out = url_city.collect()


# In[ ]:

df = pd.DataFrame()
df['url'] = [item[0] for item in out]
df['city'] = [item[1] for item in out]


# In[ ]:

skills = sc.parallelize(df['url'])
s = skills.map(ind.parse_content)
df['skill'] = s.collect()


# In[ ]:

stem = sc.parallelize(df['skill'])
st = stem.map(ind.tokenizer)
df['skill_stem'] = st.collect()
df.to_csv("/home/daniel/git/Python2.7/DataScience/indeed/data_frame_gary.csv")


# ## Coming Back 

# In[ ]:

df = pd.read_csv("data_frame_gary.csv")


# Take a look at how many job postings were returned.

# In[ ]:

df = df.drop_duplicates().dropna()
df['url'].count()


# ### bi-grams

# In[ ]:

plt.figure(figsize=(15, 8))
matrix, features = ind.vectorizer(df['skill_stem'])
ind.plot_features(features, matrix)


# ## Monogram 
# Above a Bi-gram analysis was performed by default. Let's include single words in the n-gram range,
# (1,2), and using a corpus that has been stemmed with NLTK.

# In[ ]:

corpus_stem = df['skill_stem']
mat, fea = ind.vectorizer(corpus_stem, n_min=1)
plt.figure(figsize=(15,8))
ind.plot_features(fea, mat)


# ## Explore High Count Words 
# The word "experience" showed up with a high count. I want to know if there's more to that. Experience with a platform, technology, SQL or jusy previous analytic experiece. NLP is a deep rabbit hole, and I only peered a short ways down for this project. 
# 
# My word radius method gathers words to the left and right of a chosen keyword, and builds a corpus from within that radius. Then I apply the CountVectorizer again.
# 
# You'll notice that I need to write code to remove the keyword that was searched for, from the anlaysis.
# Next iteration...

# ### Experience

# In[ ]:

plt.figure(figsize=(10,5))
# adjust stop words
ind.stop_words = "experi"
ind.add_stop_words()

words_in_radius = ind.find_words_in_radius(corpus_stem, 'experi', 5)
mat, fea = ind.vectorizer(words_in_radius, max_features=30, n_min=1)
ind.plot_features(fea, mat)


# ### Skills

# In[ ]:

plt.figure(figsize=(10,5))
# adjust stop words
ind.stop_words = "skill"
ind.add_stop_words()

words_in_radius = ind.find_words_in_radius(corpus_stem, 'skill', 5)
mat, fea = ind.vectorizer(words_in_radius, max_features=30, n_min=1)
ind.plot_features(fea, mat)


# ## Job Postings Per City

# In[ ]:

grp = df.groupby('city')
grp['url'].count().sort_values()[-20:].plot('bar', alpha=0.5, figsize=(14,8), grid=True)

