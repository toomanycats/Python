import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pandas.tseries.tools import to_datetime
import datetime

from time import sleep

class MiscTools(object):
    
    def __init__(self):
        pass
       
    def load_csv(self, file_, encoding=None, skiprows=0):
        if encoding is None:
            encodings = ['utf-8', 'cp1252', 'ISO-8859-15', 'ascii']
        
        else:
            encodings = [encoding]
        
        
        for code in encodings:
            try:
                df = pd.read_csv(file_, 
                                 error_bad_lines=False, 
                                 warn_bad_lines=True,
                                 encoding=code,
                                 skiprows=skiprows
                                 )
        
                print "Success using: %s" %code    
                return df
            
            except UnicodeDecodeError, err:
                print err
                print "Trying another encoding."
        
    def choose_num_sing_values(self, matrix, delta_perc=0.96):
        '''Compute the SVD for the matrix and return the number
        of eigen vectors that changes less than given 'delta' '''
        
        S = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        
        sqrs = S**2
        diff_sqrs = np.hstack( (0, np.diff(sqrs, n=1)) )
        
        cum_sum_diffs = np.cumsum(diff_sqrs)
        
        cum_sum_diffs /= cum_sum_diffs.min()
        
        num = np.where(cum_sum_diffs > delta_perc)[0]
        
        plt.plot(num)
        
    def get_date_from_url(self, url):
        # http://mobilewallpapersfullhd.blogspot.com/2014/02/240x320-mobile-wallpapers-mobile_28.html
        # this will match the 24 as a day. but since I'm filing in missing days
        # with a random int, who cares.
        #pat = r'(20[0-1][0-5]([-_/]?)[0-9]{2}(?:\2[0-9]{2})?)'
        pat = r'(20[0-1][0-5]([-_/]?)[0-9]{2}(?P<day>(?:\2[0-9]{2}[-_/])?))'
        ob = re.compile(pat)
        
        if ob.search(url):
            grp = ob.search(url).group()
        
        else:
            return None
        
        print grp

        # deal with 201503 to_datetime won't handle that
        if len(grp) == 6 and re.match('[20[0-1][0-5][0-9]{2}', grp):
            grp = grp[0:4] + ' ' + grp[4:]
    
        grp = re.sub('_', '/', grp) # fail to match return orig string
        date = to_datetime(grp)
        
        # confirm that datetime was able to parse
        if isinstance(date, datetime.datetime):
            # check to see if there's a "day" group
            # datetime will use the last day
            print date
        
            print ob.search(url).groups()#('day')
            
            if  len(ob.search(url).group('day')) == 0:
                day = np.random.randint(0, 29) # kis for now    
                new_date = "%s/%s/%s" %(date.year, date.month, str(day))
                date = to_datetime(new_date)
                print date
                return date
        else:
            return None
    
    def flatten(self, x):
        ''' Flatten an arbitrary number of nested lists into 1D list
         Taken from: http://lemire.me/blog/archives/2006/05/10/flattening-lists-in-python/
         Fairly standard functional type recursion...
        '''
        
        if isinstance(x, list):
            return sum( map(self.flatten, x) )
        else:
            return x
#         if not isinstance(x, list):
#             return x
#         
#         elif len(x) is 0:
#             return []
#         
#         elif isinstance(x[0], list):
#             return self.flatten(x[0]) + self.flatten(x[1:])
#         
#         else:
#             return [x[0]] + self.flatten(x[1:])