'''
Created on Nov 8, 2013

@author: dpc
'''

from MiscTools import MiscTools
from ImageTools import ImageTools
import re
from os import path
from pyxnat import Interface
import logging
import StaticPaths

user_home = path.expanduser('~')

log = logging.getLogger(__name__)
imagetools = ImageTools()

class PyXnatTools(object):
    def __init__(self):
        self.get_config_path()
        self.get_interface()
        self.misctools = MiscTools()
        
    def get_config_path(self):
        self.conf = path.join( path.join( user_home, '.server_config/ncanda.cfg') )
    
    def get_interface(self):
        '''Create interface from stored configuration'''
        self.interface = Interface( config=self.conf )#take dict for keyword args
        #interface._memtimeout = 0
    
        return self.interface

    def get_subject_ids_like(self, like_string):
        columns = ['xnat:subjectData/SUBJECT_ID']
        criteria = [('xnat:subjectData/SUBJECT_LABEL','LIKE', like_string),'AND',
                    ('xnat:subjectData/SUBJECT_LABEL','NOT LIKE','%-00000-P-%'), 'AND',
                    ('xnat:subjectData/SUBJECT_LABEL', 'NOT LIKE', '%-99999-P-%')]
        subject_keys = self.interface.select('xnat:subjectData' , columns ).where(criteria).get('subject_id')
        
        return subject_keys
    
    def query_interface(self, table, cols, criteria, return_field):
        
        query = self.interface.select(table, cols).where(criteria).get(return_field)
        
        return query

    def join_sub_keys_with_mr_session_keys(self, sub_keys, select_columns):#, return_field):
        ''' "cols_to_match" are the cols of the second table you want to return. "sub_keys" 
        is the output of another query, likely the subject id's.
        
        Returns a list of dicts with keys: "site", "sess_id","sub_id " '''
    
        join_data = []
        
        for key in sub_keys:
            criteria = [('xnat:mrSessionData/SUBJECT_ID','LIKE', key)]
            join_data += self.interface.select( 'xnat:mrSessionData', select_columns ).where( criteria ).items()#get(return_field)
                        
        return join_data
  
    def get_exp_keys_for_sub_keys(self, sub_keys):
        columns = ['xnat:mrSessionData/SESSION_ID']
        exp_keys = self.join_sub_keys_with_mr_session_keys(sub_keys, columns)
        
        exp_keys = [i[0] for i in exp_keys]
        # exp_keys starts as a list of tuples, b/c xnat thinks there could
        # be more than one experiment for a given subject. But we only use xnat for ncanda
        return exp_keys 

    def get_subject_label_for_exp_key(self,exp_key):
        '''Given an experiment key, return a subject label. ''' 
        json_ob = self.interface.select('xnat:mrSessionData',['xnat:mrSessionData/SUBJECT_ID','xnat:mrSessionData/PROJECT']).where([('xnat:mrSessionData/SESSION_ID','=', exp_key)])
        #list of dicts
        sub_key = json_ob.data[0]['subject_id']
        project = json_ob.data[0]['project']
        
        proj = self.interface.select.project(project)
        sub_label = proj.subject(sub_key).attrs.get('xnat:subjectData/label')
    
        return sub_label  
    
    def get_resource_file_paths(self):
        '''Get a list of dicts of resource files. Keys are 
        "fname", "uri" and "size" .'''
        
        interface = self.get_interface()
        
        table = 'xnat:subjectData'
        cols = [table + '/SUBJECT_ID'] 
        criteria = [(table + '/SUBJECT_LABEL','NOT LIKE','%-%-P-%')]
        return_field = 'subject_id'
        
        sub_ids = self.query_interface(table, cols, criteria, return_field)
        
        columns_to_match_against = ['xnat:mrSessionData/SESSION_ID']
        matches = self.join_sub_keys_with_mr_session_keys(sub_ids, columns_to_match_against)
        
        experiment_keys = self.misctools.flatten_double_nested(matches)
        
        resource_files = []
        # get experiemtn keys from session keys
        for exp_key in experiment_keys:
            resource_keys = interface.select.experiment(exp_key).resources().get()#returns a list of resource keys
            
            if resource_keys != []:
                
                for key in resource_keys:
                    files = interface._get_json( '/data/experiments/%(exp_key)s/resources/%(resource_key)s/files?format=json' %{'exp_key':exp_key,'resource_key':key} ) 
                    resource_files += [ {'fname':file['Name'], 'uri':file['URI'], 'size':file['Size']} for file in files ]
    
        return resource_files        

    def get_phantom_subject_keys(self, interf=None):
        if interf is None:
            interf = self.interface
            
        columns = ['xnat:subjectData/SUBJECT_ID']
        criteria = [('xnat:subjectData/SUBJECT_LABEL','LIKE','%-99999-P-%')]
        phantom_keys = interf.select('xnat:subjectData' , columns ).where(criteria).get('subject_id')
        
        return phantom_keys

    def get_all_experiment_keys(self):
        '''Get a list of all experiment_id's aka keys. '''
        exp_ob = self.interface.select.experiments()
        exp_keys = exp_ob.get()   
    
        return exp_keys
    
    def get_nifti_path(self, experiment_key):
        '''Given a unique experiment_id, or key, return the path to the nifti files
        associated with that session. '''
        
        xml = self.interface.select.experiment(experiment_key).get()     
        pat = '.*<xnat:resource label="nifti".*URI="(?P<uri>/fs/.*/nifti)/nifti_catalog.xml".*'
        ob = re.compile(pat, re.DOTALL)
        mat = ob.match(xml)
        if mat:
            uri = mat.group('uri')
        else:
            raise Exception, "The key you entered does not point to an experiment which contains nifti files."
            
        dirs = uri.split('/')
        del(dirs[0])#first index is empty b/c of delimiter choice
        index = dirs.index('archive')
        shortened_path = dirs[index:]
        path_prefix = ['/fs','ncanda-xnat']
        
        full_path_parts = path_prefix + shortened_path
        
        nii_path = path.normpath( '/'.join(full_path_parts) )    
        
        return nii_path
    
    def get_data_from_server(self, exp_key, File, dl_path, Type='QA'):
        '''Given an experiment key, the file type e.g. 'phantom.xml', and type, 'QA'
         download the data. '''
        
        data_path = self.interface.select.experiment(exp_key).resource(Type).file(File).get_copy(dl_path)
        
        return data_path
    
    def get_xml_for_exp_id(self, expkey):
        xml = self.interface.select.experiment(expkey).get()
        
        return xml
    
    def get_custom_variable(self, expkey, field_name ):                                                                                                                                                          
        regex = '.*<xnat:field name="%s">(?P<field_value>.*?)</xnat:field>' % field_name                                                                                                                                       
        xml = self.get_xml_for_exp_id(expkey)
        match = re.match(regex, xml, flags=re.DOTALL)                                                                                                                                                                           
        
        if match:                                                                                                                                                                                                     
            value = re.sub( '.*<!--.*-->\s?', '', match.group('field_value'),                                                                                                                                                  
                            flags=re.DOTALL ) 
            
            if value is not ' ' and value is not None and value is not '':
                return value
            
            else:
                return None
                                                                                                                                                                                        
        else:
            return None
    
    def get_site_from_sub_ID(self, sub_id):
        '''Get the site for a single sub id. '''
        sub_keys = [sub_id]# takes a list
        exp_key = self.get_exp_keys_for_sub_keys(sub_keys)
        
        if exp_key is None:
            raise Exception, "Sub key returned no experiment key, i.e., exp_key list is None. sub_key:%s" %sub_keys
           
        sub_label = self.get_subject_label_for_exp_key(exp_key[0])
        
        first_letter = sub_label[0]
        
        site = {'B':'sri',
                'C':'duke',
                'A':'upmc',
                'D':'ohsu',
                'E':'ucsd'
                }
        
        return site[first_letter], sub_label    
        
    def get_spiral_uris(self, xnat_eid_list):

        for xnat_eid in xnat_eid_list:
            resource_dict_list = self.interface._get_json( '/data/experiments/%s/resources/?format=json' %xnat_eid )
            
            spiral_uri = ''
            for res in resource_dict_list:
                if res['label'] == 'spiral':
                    resource_id = res['xnat_abstractresource_id']
                    eid = res['cat_id']
                    obj = self.interface._get_json('/data/experiments/%s/resources/%s/files?format=json' %(eid, resource_id))    
                    file_path = obj[0]['Name']
                    
                    spiral_uri = "/".join([eid, resource_id, file_path])
                
                spiralrest_uri = ''  
                if res['label'] == 'spiralrest':
                    resource_id = res['xnat_abstractresource_id']
                    eid = res['cat_id']
                    obj = self.interface._get_json('/data/experiments/%s/resources/%s/files?format=json' %(eid, resource_id))    
                    file_path = obj[0]['Name']
                    
                    spiralrest_uri = "/".join([eid, resource_id, file_path])
                        
        return (spiral_uri, spiralrest_uri)
    
    def export_spiral_files(self, resource_location, to_directory, tmp_dir = "/tmp"):
        # resource location contains results dict with path building elements
        # NCANDA_E01696/27630/spiral.tar.gz
        resource_location = resource_location.split('/')
        xnat_eid = resource_location[0]
        resource_id = resource_location[1]
        resource_file_path = resource_location[2]
        
        tar_tmp = path.join( tmp_dir, 'tarfile_dir' )
        extracted_tmp = path.join( tmp_dir, 'extracted_dir' )
        
        experiment = self.interface.select.experiment( xnat_eid )
        
        tmp_file_path = experiment.resource( resource_id ).file( resource_file_path ).get_copy( tar_tmp )
    
        self.misctools.untar_to_dir( tmp_file_path, extracted_tmp )
        spiral_E_file = self.misctools.glob_for_files(extracted_tmp, pattern='E*.7', num_limit=1) 
        
        imagetools.make_nifti_from_spiral(spiral_E_file, to_directory) 
    
    def get_sub_id_from_exp_id(self, expkey):
        xml = self.get_xml_for_exp_id(expkey)    
    
        # <xnat:subject_ID>NCANDA_S00311</xnat:subject_ID>

        pat = "<xnat:subject_ID>(NCANDA_S[0-9]{5})</xnat:subject_ID>" 
        re_obj = re.compile(pat)
    
        match = re_obj.search(xml)
        
        return match.group(1)
    
    def check_baseline_existance(self, expkeys):
        phan_mat_ob = re.compile('[A-Z]-[9]{5}-P-.*|[A-Z]-[0]{5}-P-.*')
        out = []
        for key in expkeys:

            sub_id = self.get_sub_id_from_exp_id(key)
            
                        # exclude phantoms
            sub_label = self.get_subject_label_for_exp_key(key)
            if phan_mat_ob.search(sub_label):
                continue
                          
            if sub_id is None:
                out.append('%s:None' %key)
                                   
            static_paths = StaticPaths.StaticPaths()  
            static_paths.set_ncanda_full_paths(sub_id, time_point='base')
            baseline_path = static_paths.baseline
            
            if path.exists(baseline_path):
                out.append("%s:%s:True" %(sub_id, key))
            
            else:
                out.append("%s:%s:False" %(sub_id, key))
              
        return out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    