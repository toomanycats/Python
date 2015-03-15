import csv
import MiscTools
from numpy import loadtxt
from os import path

root = '/fs/cl10/dpc/Data/Resting_Test2'
outfile = '/fs/u00/dpc/sql_table_info.txt'
file_obj = open(outfile, 'w')
misctools = MiscTools.MiscTools()

#cat ./NCANDA_S00721/group_covariate_NCANDA_S00721.txt
#{"h_alc": "Negative", "sub_id": "NCANDA_S00721", "gender": "Male", "age": 13.24, "site": "GE", "label": "B-00475-M-9"}


file_list = loadtxt('/fs/cl10/dpc/Data/Resting_Test2/cov_list_mod.txt', 'str')
cov_sample = misctools.load_dict_from_json( path.join(root, file_list[0]) )
csv_writer = csv.DictWriter(file_obj, cov_sample.keys(), delimiter=',')


def write_one(infile):
    cov = misctools.load_dict_from_json(infile)
    csv_writer.writerow(cov)

for infile in file_list:
    infile = path.normpath(path.join(root, infile))
    print infile
    write_one(infile)
    
file_obj.close()    
              
              


    
    
    
    
