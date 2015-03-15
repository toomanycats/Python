from optparse import OptionParser
import glob
import traceback
import csv
from os import path
import re
import numpy as np

def one_file(csv_file):
    
    #Response Label Regex: 3rd Column ( 0 index )
    acr_pat = '.*across.*'
    wit_pat = '.*within.*'
    
    acr_match = re.compile(acr_pat, flags=re.IGNORECASE | re.DOTALL)
    wit_match = re.compile(wit_pat, flags=re.IGNORECASE | re.DOTALL)
    
    wrong_pat = '.*wrong.*response.*'
    corr_pat = '.*correct.*response.*'
    
    wrong_match = re.compile(wrong_pat, flags=re.I | re.DOTALL)
    correct_match = re.compile(corr_pat, flags=re.I | re.DOTALL)
    
    # catch a failed experiment:
    fail_pattern = '.*AFTER PRETEST, PSYSCOPE UNEXPECTEDLY QUIT: USE PRETEST FROM HERE AND TEST FROM NEXT RUN.*'
    fail = re.compile(fail_pattern, flags=re.I | re.DOTALL)
    
    f = open(csv_file, 'r')
    reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    
    test_row = reader.next()
    
    f.seek(0)
    
    if test_row[-1] == 'NA':  # original
        category_ind = 3 
        response_ind = 4 
    
    elif re.match('.*tive.*|.*cohol.*|.*bev.*', test_row[3], flags=re.I | re.DOTALL):
        category_ind = 4 
        response_ind = 5   
    
    elif re.match('.*emo.*|.*alc.*|.*bev.*', test_row[3], flags=re.I | re.DOTALL):
        category_ind = 4
        response_ind = 5
        
    else:
        raise Exception, "The number of columns in the CSV file is wrong or changing. Or, there's a header line a the top."
    
    for row in reader:     
        if fail.match(' '.join(row)):
            raise Exception, """String was found in CSV file:
    AFTER PRETEST, PSYSCOPE UNEXPECTEDLY QUIT: USE PRETEST FROM HERE AND TEST FROM NEXT RUN"""        
    
        try:
        
            if  acr_match.match(row[category_ind]) and correct_match.match(row[category_ind]):
                search_match('across', row, response_ind)
                 
            elif wit_match.match(row[category_ind]) and correct_match.match(row[category_ind]):
                search_match('within', row, response_ind)    
        
        except IndexError:
            if len(row) == 0:
                pass
            else:
                raise Exception, "Row has issues: %s" %row     
            
    
    f.close()
        
    print "File: %s" %csv_file    
    print " *******************************************************"    
    print "Alcohol:  D measure: %f  mean within: %f  mean across: %f  " %(compute_stat(alc), alc.wit_mean(), alc.acr_mean())
    print "Beverage: D measure: %f  mean within: %f  mean across: %f  " %(compute_stat(bev), bev.wit_mean(), bev.acr_mean())
    print "Positive: D measure: %f  mean within: %f  mean across: %f  " %(compute_stat(pos), pos.wit_mean(), pos.acr_mean())
    print "Negative  D measure: %f  mean within: %f  mean across: %f  " %(compute_stat(neg), neg.wit_mean(), neg.acr_mean())
    print " *******************************************************"    

def compute_stat(class_ob):
    n1 = class_ob.wit_num()
    std1 = class_ob.wit_std()
    mu1 = class_ob.wit_mean()
    
    n2 = class_ob.acr_num()
    std2 = class_ob.acr_std()
    mu2 = class_ob.acr_mean()
    
    numerator = (mu2 - mu1) * (n1 + n2 - 1)
    
    denom = np.sqrt( ((n1-1) * std1**2 + (n2-1) * std2**2) + ( 0.25 * (n1 + n2) * (mu2 - mu1)**2) )
                     
    return numerator / denom                 
                      
def search_match(Type, row, response_ind):
    for pat  in stim_dict.iterkeys():
            if re.match(pat, row[2], flags=re.I | re.DOTALL):
                try:
                    stim_dict[pat].push_val(Type, np.int(row[response_ind]))
                except:
                    if np.int(row[response_ind + 1]):
                         stim_dict[pat].push_val('within', np.int(row[response_ind + 1]))
                return

class category(object):
    def __init__(self):
        self.acr_values = []
        self.wit_values = []

    def acr_mean(self):
        return np.mean(self.acr_values)
    
    def wit_mean(self):
        return np.mean(self.wit_values)
    
    def acr_num(self):
        return len(self.acr_values)
    
    def wit_num(self):
        return len(self.wit_values)
    
    def acr_std(self):
        return np.std(self.acr_values, ddof=1)
    
    def wit_std(self):
        return np.std(self.wit_values, ddof=1)
                      
    def push_val(self, Type, value):
        if Type == 'across':
            self.acr_values.append(value)
            
        elif Type == 'within':
            self.wit_values.append(value)                                
            
alc = category()
bev = category()
pos = category()
neg = category()

stim_dict = {'.*beer.*':alc,
             '.*booze.*': alc,
             '.*liquor.*':alc,
             '.*wine.*':alc,
             '.*cheerful.*':pos,
             '.*glad.*':pos,
             '.*happy.*':pos,
             '.*lucky.*':pos,
             '.*aggressive.*':neg,
             '.*furious.*':neg,
             '.*mad.*':neg,
             '.*angry.*':neg,
             '.*juice.*':bev,
             '.*lemonade.*':bev,
             '.*soda.*':bev,
             '.*water.*':bev
                 }

parser = OptionParser()
parser.add_option("-d", "--csv-dir", dest="csv_dir",
                  help="Path to the directory containing the csv files.")

(options, args) = parser.parse_args()
csv_dir = options.csv_dir 

pattern = path.join(csv_dir, '*.csv')
csv_files = glob.glob(pattern)

if csv_files < 1:
    raise Exception, "No CSV files where found."

failed_list = []
for f in csv_files:
    try:
        one_file(f)
    except:
        mess = tb = tb = traceback.format_exc()
        failed_list.append([f, mess])
 
if failed_list is not None:
    print "\n************FAILURE REPORT **********"
    for f in failed_list:
        print "Failed File: %s \n" %f[0] 
        print "Reason: %s" %f[1]
    
    
   

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    