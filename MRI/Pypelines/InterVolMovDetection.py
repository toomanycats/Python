'''
Created on Sep 13, 2013

@author: dpc
'''
from os import path
from glob import glob 

import numpy as np
import matplotlib.pyplot as plt

import Tools


imagetools = Tools.ImageTools()
pyx = Tools.PyXnatTools()
sig = Tools.SignalProcess()

class InterVolMovDet(object):
    def __init__(self, path_to_single_sub=None):
        self.path_to_single_sub = path_to_single_sub

    def get_vol_list(self, p=None):
        '''Use glob module to grab all image.nii.gz files for a 4D 
        data set. '''
        if p is None:
            p = self.path_to_single_sub
        
        P = path.join(p, '*.nii.gz')
        vols = glob(P)
       
        return vols
    
    def compute_one_subject(self,list_of_vols):
        '''Returns a numpy array of xcorr arrays, one for each vol. '''
        # initialize the numpy array
        vol = list_of_vols[0]
        del(list_of_vols[0])
        slice_diff_vec = self.compute_one_vol(vol)
        
        for vol in list_of_vols:
            slice_diff = self.compute_one_vol(vol)
            #concatentate
            slice_diff_vec = np.dstack((slice_diff_vec, slice_diff))
    
        return slice_diff_vec     
    
    def compute_one_vol(self, infile):
        '''Returns a numpy array of xcorr for a single vol.'''
    
        image = imagetools.load_image(infile)
        data = image.get_data()
        
        # xcorr between slices on one vol
        slice_diff = np.zeros(data.shape)
        #for i in np.arange(data.shape[2]-1):
        slice1 = np.squeeze(data[:,:,17])
        slice2 = np.squeeze(data[:,:,18])
        slice_diff = slice2 - slice1
        
        return slice_diff
    
    def make_nii_paths(self, keys):
        '''using the experiment keys, find the path to the NiFTI files. '''
        
        nii_paths = []
        for key in keys:
            try:
                p = pyx.get_nifti_path(key)
                nii_paths.append(p)
            except Exception:
                pass
            
        return  nii_paths
    
    def filter_nii_paths_for_rs_data(self, nii_paths):
        '''Weed out paths that do not contain a directory named:
        "9_ncanda-rsfmri-v1". That is, paths that do not contain reseting
        state nifti files. Doesn't appear to work flawlessly...  '''
        
        rs_full_paths = []
        for p in nii_paths:
            rs_path = path.join(p, '9_ncanda-rsfmri-v1')
            if path.exists(rs_path):
                rs_full_paths.append(rs_path)
                
        return rs_full_paths        
           
    def save_data(self, out_file_path, Data):
        '''Saves data as a list of dicts, into numpy archive format.'''    
        
        np.save(out_file_path, Data)   
                           
    def compute_data_for_all_nii_files_in_DB_and_save(self, out_file_path):
        '''Looks up ALL experiments and runs the motion detection routine on them.
        and saves the data structure. '''
        keys = pyx.get_all_experiment_keys()
        
        nii_paths = self.make_nii_paths(keys)
        
        rs_full_paths = self.filter_nii_paths_for_rs_data(nii_paths)
    
        # list of dicts, one for each subject    
        Data = []
        for p in rs_full_paths:
            data = {}
            list_of_vols = self.get_vol_list(p)
            
            xcorr_vec_list = self.compute_one_subject(list_of_vols)
            
            data['path'] = p
            data['ts_std'] = np.std(xcorr_vec_list, axis=1)#compute along cols
            data['xcorr_vols'] = xcorr_vec_list
            
            Data.append(data)
        
        self.save_data(out_file_path, Data)

    def look_for_outliers_in_slice(self, slice_diff_vec):
        '''Find the indices of potential bad volumes. This index should
        then be used with the path data to build the full path to the offending
        image. '''
         
        d = np.zeros(slice_diff_vec.shape[2])
        for i in xrange(0, slice_diff_vec.shape[2]):
            d[i] =  np.sum( (slice_diff_vec[:,:,i])**2 )  
               
        threshold = d.mean() + d.std()
        bad_slice_ind = np.where((d > d.mean() + d.std() )| (d < d.mean() - d.std()) )[0]
        
        return bad_slice_ind, d

    def match_bad_ind_to_path(self, bad_ind, sub_data):
        '''Given a single bad vol index, return a full path to that vol.'''
        sub_path = sub_data['path']
        string = 'image%04d.nii.gz' %bad_ind
        full_path = path.join(sub_path, string)

        return full_path
















