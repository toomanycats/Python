
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import re


# In[3]:

df = pd.read_csv("train.csv")
df.head(n=3)


# In[4]:

format = '%Y-%m-%d  %I:%M:S'
df['DateTime'] = df['DateTime'].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))
df['Hour'] = df['DateTime'].apply(lambda x: x.hour)
df['Day_of_week'] = df['DateTime'].apply(lambda x: x.dayofweek)


# In[5]:

obj = re.compile("(?P<num>\d+)\s*(?P<word>\w+)")
def convert_age(string):
    match = obj.search(string)
    if match:
        return match.groups()
    else:
        None, None



# In[7]:

convert_dict = {None:0,
                'day':1,
                'days':1,
                'month':30,
                'months':30,
                'year':365,
                'years':365,
                'week':7,
                'weeks':7
               }


# In[8]:

for i in range(df.shape[0]):
    try:
        string = df.loc[i, 'AgeuponOutcome']
        num, word = convert_age(string)
        num_days = int(num) * convert_dict[word]
        df.loc[i, 'AgeuponOutcome_inDays'] = num_days
    except:
        df.loc[i, 'AgeuponOutcome_inDays'] = 0
        continue


# In[9]:

df.head(n=3)


# In[10]:

df['has_name_bool'] = df['Name'].apply(lambda x: 1 if x is not None else 0)


# In[11]:

color = df.groupby('Color').count()['AnimalID'].sort(inplace=False)


# In[12]:

color.sample(n=10).head()


# In[13]:

obj = re.compile('(\w+)\s*(\w+)*/*(\w+)*')
def break_up_colors(string):
    match = obj.search(string)
    if match:
        colors = match.groups()
        return colors
    else:
        return None


# In[14]:

colors = []
for c in color.index:
    colors +=break_up_colors(c)

colors = np.unique(colors)
print (colors)


# In[15]:

dff = pd.DataFrame(columns=colors, data=np.zeros((df.shape[0], colors.shape[0]), np.int8))


# In[16]:

for i in range(df.shape[0]):
    color_string = df.loc[i, 'Color']
    color_string_tuple = break_up_colors(color_string)
    dff.loc[i, color_string_tuple] = 1


# In[17]:

breed = df.groupby('Breed').count()['AnimalID']

breeds = []
for b in breed.index:
    breeds +=break_up_colors(b)

breeds = np.unique(breeds)
print (breeds)[0::4]


# In[18]:

df_breed = pd.DataFrame(columns=breeds, data=np.zeros((df.shape[0], breeds.shape[0]), np.int8))


# In[19]:

for i in range(df.shape[0]):
    breed_string = df.loc[i, 'Breed']
    breed_string_tuple = break_up_colors(breed_string)
    df_breed.loc[i, breed_string_tuple] = 1


# In[20]:

df_breed = pd.get_dummies(df_breed)
df_breed.shape


# In[21]:

df_final = pd.get_dummies(df[['OutcomeType', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome',
                              'Breed', 'Hour', 'Day_of_week']])
df_final = pd.concat([df_final, dff, df_breed, df['AgeuponOutcome_inDays']], axis=1)


# Name, OutcomeType, OutcomeSubtype, AnimalType, SexuponOutcom, Breed, (colors dummies), has_name_bool
# hour, Day_of_week, AgeuponOutcome_inDays, breeds

# In[22]:

df_final.head(n=1)


# In[23]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn import metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import cross_validation


# In[24]:

X = df_final[['OutcomeType_Adoption',
              'OutcomeType_Died',
              'OutcomeType_Euthanasia',
              'OutcomeType_Return_to_owner',
              'OutcomeType_Transfer']]

Y = df_final.drop(X.columns, axis=1)


# In[38]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, train_size=0.3, random_state=0)


# In[39]:

clf = RandomForestClassifier(random_state=0, n_jobs=1)
clf.fit(X_train, y_train)


# In[36]:

y_pred = clf.predict(X_test)
error = metrics.classification_report(y_test, y_pred)
print error


# In[ ]:

#cv = cross_validation.KFold(df_final.shape[0], 100)
#params = {'min_samples_split': np.arange(3,10),
#          'max_features': ['sqrt', 'log2', None]
#         }


# In[ ]:

#cross_validation.cross_val_score(clf, X, Y, cv=cv)
#gs = RandomizedSearchCV(clf, params, cv=cv, n_jobs=5)
#gs = GridSearchCV(clf, cv=cv, n_jobs=6)
#gs.fit(X, Y)
#best_parameters, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
#for param_name in sorted(params.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))

