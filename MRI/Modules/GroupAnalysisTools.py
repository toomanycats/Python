import PyXnatTools
import StaticPaths
import csv
import MiscTools
import PhysioCorrect
import PySQL
from os import path
from numpy import delete, where, empty, concatenate, loadtxt

pyx = PyXnatTools.PyXnatTools()
misctools = MiscTools.MiscTools()
pysql = PySQL.PySQL()

class LookupCovariates(object):
    def __init__(self, sub_id, time_pt='base'):
        self.site_scanner_dict = {'duke':'GE',
                                  'sri':'GE',
                                  'ucsd':'GE',
                                  'ohsu':'SEIMENS',
                                  'upmc':'SEIMENS'
                                   }
        
        self.static_p = StaticPaths.StaticPaths()
        self.static_p.set_ncanda_full_paths(sub_id, time_pt)
        
        self.sub_id = sub_id
        self.rs_path = self.static_p.rs_sri24_bold_stdy
        self.site, self.label = pyx.get_site_from_sub_ID(self.sub_id)
        
        self.clinical_file = self.static_p.clinical 
        self.demographics = self.static_p.demographics
         
        self.set_demo_info() 
         
    def get_site(self):
        return {'site':self.site}

    def get_family_history(self):
        clin_obj = open(self.clinical_file)
        reader = csv.reader(clin_obj, delimiter=',')
        first_line = reader.next()
        second_line = reader.next()
        clin_obj.close()
        
        FH_ind = first_line.index('fh_alc')
        FH = second_line[FH_ind] # family history of alcohol
        
        # do not need to set 0's for absence of a cov, b/c the array was initialized as zeros
        if FH.lower() == 'n' or FH.lower() == '': # family history alc use
            return {'fh_alc_pos':0, 
                    'fh_alc_neg':1
                    }
        
        elif FH.lower() == 'y' or FH.lower() == 'p': #positive ?
            return {'fh_alc_pos':1,
                    'fh_alc_neg':0
                    }  
        
        else:
            raise Exception, "FH code:%s not recognized. Sub Label:%s  Sub ID: %s" %(FH, self.label, self.sub_id)
                                              
    def get_scanner_cov(self):    
        scanner = self.site_scanner_dict[self.site]
     
        if scanner.lower() == 'ge':
            return {'scanner_GE':1,
                    'scanner_SIEMENS':0
                    }
            
        elif scanner.lower() == 'seimens':
            return {'scanner_GE':0,
                    'scanner_SIEMENS':1
                    }   
         
    def get_gender(self):    
        try:
            gender_ind = self.first_line_demo.index('sex')
        
        except ValueError:
            print "Gender not identified by 'sex' keyword in demo file: %s" %self.static_p.demographics
            
        gender = self.second_line_demo[gender_ind]
        
        if gender.lower() == 'f':
            return  {'Female':1,
                     'Male':0
                     }
    
        elif gender.lower() == 'm':
            return {'Female':0,
                    'Male':1
                    } 
    
        else:
            raise Exception, "Not f/F or m/M" %(self.label, self.sub_id)
    
    def set_demo_info(self):
        measures_obj = open(self.demographics, 'r')
        reader = csv.reader(measures_obj, delimiter=',')
        self.first_line_demo = reader.next()
        self.second_line_demo = reader.next()
        measures_obj.close()
        
    def get_age(self):    
        age_ind = self.first_line_demo.index('visit_age')
        age = float(self.second_line_demo[age_ind]) 
       
        return {'age':age}
     
    def get_label(self):
        return {'label':self.label}         

    def get_physio(self):
        physio_dir = self.static_p.physio_dir
        physio = PhysioCorrect.Main(physio_dir=physio_dir,
                                 num_vols_drop=5, # num drop not needed for one site with delay in recording time
                                 rs_path=self.rs_path, #used to get TR 
                                 output_dir='', # preprocess_only arg will se this to tempdir
                                 phy_denoised_out='', # can't be None b/c join
                                 preprocess_only=True # critical argument
                                 )
        try:
            physio.main()
            file_list_dict = physio.file_list_dict
     
            card = 1
            resp = 1
            
            if file_list_dict['card'] is None:
                card = 0
            
            if file_list_dict['resp'] is None:
                resp = 0
            
            # exceptions are raised when the physio data are missing vital
            # file commponents, e.g. header, footer, or other things I forgot about.
            #TODO: have stack trace returned and logged in the SQL db and log
        except:
            return {'card':'False', 'resp':'False'}

        return {'card':card, 'resp':resp}


class NcandaGroupCov(LookupCovariates):
    def __init__(self, sub_id, time_pt):
        self.sub_id = sub_id
        LookupCovariates.__init__(self, sub_id, time_pt)
        
    def get_cov(self):
        self.cov = self.get_family_history()
        self.cov.update(self.get_scanner_cov())
        self.cov.update(self.get_gender())
        self.cov.update(self.get_age())
#         self.cov.update(self.get_physio())
#         self.cov.update(self.get_label())        
#         self.cov = {'sub_id':self.sub_id}        
#         self.cov = self.get_site()        
        return self.cov
        
        
class WriteCovCsv(object):
    def __init__(self, output_path):
        self.output_path = output_path
        self.file_obj = open(self.output_path, 'w')
    
    def get_writer(self, cov_dict_keys):    
        self.csv_writer = csv.DictWriter(self.file_obj, cov_dict_keys, delimiter=',')
    
    def write_header(self):
        self.csv_writer.writeheader()
    
    def write_row_to_csv(self, cov_dict):
        self.csv_writer.writerow(cov_dict)
        
    def close_file_object(self):
        self.file_obj.close() 
            
        
class AllNcandaSubjects(WriteCovCsv):#TODO: make MySQLdb version using insert instead of making CSV file
    '''The list of subject ID's can be had from the mySQL data base or a text file list. Sometimes the DB is being 
    developed and is not a good option. A simple "ls /fs/ncanda-share/pipeline/cases | sort" is enough. There's a try catch
    in the code to avoid problems when data is missing. '''
    
    def __init__(self, output_path, sub_id_list=None, use_data_base=False):
        WriteCovCsv.__init__(self, output_path)
        if use_data_base:
            self.sub_id_list = pysql.get_complete_sub_list()
        
        elif sub_id_list is not None and not use_data_base: 
            self.sub_id_list = loadtxt(sub_id_list, 'str')    

        else:
            raise Exception, "Must use either a list of sub ids in a text file, or a data base."
#             static = StaticPaths.StaticPaths()
#             regex = '^NCANDA_S[0-9]{5}'
#             sub_id_paths = misctools.build_file_list(search_path=static.ncanda_root, 
#                                                          regex=regex, 
#                                                          string=False
#                                                          )
#         
#             self.sub_id_list = [p.split('/')[-1] for p in sub_id_paths]

        self.output_path = output_path
        self.find_remove_missing_sub()
        
        
    def get_dict_keys(self):
        cov = NcandaGroupCov(self.sub_id_list[0])
        cov_dict = cov.get_cov()
        
        return sorted(cov_dict.keys())
    
    def get_all_sub_cov(self):
        '''This is the main method.'''
        cov_dict_keys = self.get_dict_keys()
        writer = WriteCovCsv(self.output_path)
        writer.get_writer(cov_dict_keys)
        writer.write_header() 
        
        for i in range(self.sub_id_list.size):
         
            cov = NcandaGroupCov(self.sub_id_list[i])
            cov_dict = cov.get_cov()
            writer.write_row_to_csv(cov_dict)
            print i/float(self.sub_id_list.size) * 100
            
        writer.close_file_object()           
        
    def find_remove_missing_sub(self):
        missing = []
        static_p = StaticPaths.StaticPaths()
        
        for sub_id in self.sub_id_list:
            static_p.set_ncanda_full_paths(sub_id, time_point='base')#TODO: allow other time points        
                    
            if not path.exists(static_p.rs_sri24_bold_stdy):
                missing.append(sub_id)
        
        
        missing_ind = empty(0)
        
        for sub in missing:
            ind = where(self.sub_id_list == sub)[0]
            missing_ind = concatenate((missing_ind, ind), axis=0)
        
        self.sub_id_list = delete(self.sub_id_list, missing_ind)

            
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
        
        