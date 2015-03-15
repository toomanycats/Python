'''
Created on Oct 10, 2013

@author: dpc
'''

from os import path
import numpy as np
import ImageTools
import SignalProcessTools
import StaticPaths
import logging
import json
import MiscTools
import nibabel.eulerangles as nie
from scipy.linalg import qr as qr_factor
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
import nipy.algorithms.statistics.empirical_pvalue as empirical_pvalue
from nipy.modalities.fmri.glm import GeneralLinearModel as glm
from scipy import ndimage, stats
from pynfft import NFFT
from nilearn.decomposition.canica import CanICA
from operator import itemgetter
from itertools import groupby

log_method = MiscTools.log_method
log = logging.getLogger('module_logger')
invert = np.linalg.inv
imagetools = ImageTools.ImageTools()
sigtools = SignalProcessTools.SignalProcessTools()
misctools = MiscTools.MiscTools()
save_new_image = ImageTools.ImageTools().save_new_image   

# used just for module methods
static_paths = StaticPaths.StaticPaths()

class BandwidthTR(object):
    def __init__(self, Type, rs_file):
        '''Object containing import bandwidth and TR information. By supplying arg rs_file
        you can ste the TR using a 4D resting state data file.'''
        
        self.TR = imagetools.get_pixdims(rs_file)[4]
        
        if Type == 'human':
            self.f_lp = 0.009
            self.f_ub = 0.09
            
        elif Type == 'rat':
            self.f_lp = 0.002
            self.f_ub = 0.01
         
        else:
            raise Exception," Type not understood: %s.  Choices are 'rat' and 'human'. " %Type   


class DetrendMovementNoise(object):

    def __init__(self, root_dir, rs_4D_path, confound_cleaned_output, mask, sub_id, time_point='base', 
                 Type='human', wm_rs=None, aux_tissue=None, csf_rs=None, move_par_type=None):
        
        self.root_dir = root_dir
        self.wm_rs = wm_rs
        self.aux_tissue = aux_tissue
        self.csf_rs = csf_rs
        
        self.mask_path = mask
        self.rs_path = rs_4D_path
        self.Type = Type
        self.clean_rs_out_path = confound_cleaned_output
         
        # additional options
        self.global_mean = True
        self.use_abs_motion_con = True
        self.move_par_type = move_par_type
        self.aux_conf = False 
        
        self.der_all_confounds = True
        self.der_tissue_confounds_only = False
            
        self.motion_f2f_parameters_file = path.join(self.root_dir, 'motion_f2f_params.txt')
        
        self.static_paths = StaticPaths.StaticPaths()
        self.static_paths.set_ncanda_full_paths(sub_id=sub_id, time_point=time_point)
        
        self.num_vols_ref = 260
        # starts with 274 or 275, 5 dummies at acq. I rewrote Torsten's
        # script to apply all 275(4) volumes then drop 15 later
        
        # aux confound options
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        self.z0 = None
        self.z1 = None

    def get_bandwidth_and_TR(self):    
        bw_tr = BandwidthTR(self.Type, self.rs_path)
               
        self.f_lp = bw_tr.f_lp
        self.f_up = bw_tr.f_ub
        self.TR = bw_tr.TR
        self.sample_rate = np.round(np.float(1 / self.TR), decimals=4)
    
    def load_data_from_images(self):
        self.mask = imagetools.load_image(self.mask_path)
        self.mask_data = self.mask.get_data()
        self.mask_data[np.isnan(self.mask_data)] = 0
        
        self.rs_obj = imagetools.load_image(self.rs_path)
        self.rs_data =  np.array(self.rs_obj.get_data(), dtype=np.float32)
        self.rs_data[np.isnan(self.rs_data)] = 0
        
        self.data_dim = self.rs_data.shape
        
    def main(self):
        self.get_bandwidth_and_TR()
        # sets the self.mats_path variable
        if self.move_par_type and self.move_par_type == 'mcflirt':
            # new method, is to re-compute the motion confound using first vol as ref
            # I don't trust the mcflirt params for motion confounds.
            self.mats_path = path.join(self.root_dir, 'motion_params.mat')
                  
        elif self.move_par_type and self.move_par_type == 'cmtk':
            self.mats_path = misctools.glob_for_files(self.root_dir, '*.f2f', num_limit=1)
        
        else:
            raise Exception, "Motion parameter file not specified correctly."
        
        # sets self.rs_data and mask data, and clears nan's
        self.load_data_from_images()
        
        # get confounds
        log.debug("getting confounds")
        confounds = self.get_all_confounds()
        
        # regress out the confounds
        log.debug("Removing confounds")
        data, ind = self.detrend_data(confounds)
 
        log.debug("Saving cleaned rs data back to disc.")
        self.save_clean_image(data, ind)  
  
    def get_comp_of_confound(self, roi, num_comp):
        _,_,V = np.linalg.svd(roi)
        
        con = V[:, 0:num_comp]
        
        return np.mean(con, axis=1)
        
    def get_tissue_confound(self, tissue_mask):
        tiss_mask_obj = imagetools.load_image(tissue_mask)
        tiss_mask_data = tiss_mask_obj.get_data()
        
        # deal with the occasional x,y,z,1 shaped data
        tiss_mask_data = np.squeeze(tiss_mask_data)
        
        tiss_data = np.zeros((self.rs_data.shape[3], 1))
        for i in range(self.rs_data.shape[3]):
            tiss_data[i] = np.mean(self.rs_data[:,:,:,i] * tiss_mask_data)
        
        #np.save(path.join(self.root_dir, 'wm_mean_ts.npy'), wm_data)
        return tiss_data
            
    def get_global_mean_confound(self):
        mean = np.zeros((self.rs_data.shape[3], 1))
        for i in range(mean.shape[0]):
            mean[i] = (self.mask_data * self.rs_data[:,:,:,i]).mean()
        
        return mean
        
    def get_aux_confound(self):
        x0 = self.x0
        x1 = self.x1
        y0 = self.y0
        y1 = self.y1
        z0 = self.z0
        z1 = self.z1
        
        roi = self.rs_data[x0:x1,y0:y1,z0:z1,:]
        
        if roi.ndim != 4:
            raise Exception, " Aux confound file containing coord does not have the proper ammount. Dim of roi was less than 3."
        
        roi = np.squeeze(roi)
        
        if roi.ndim == 4:
            confound = roi.mean(0).mean(0).mean(0)
        elif roi.ndim == 3:
            confound = roi.mean(0).mean(0)
        elif roi.ndim == 2:
            confound = roi.mean(0)  
        elif roi.ndim == 1:
            confound = roi
        
        confound = np.reshape(confound, (confound.shape[0], 1))
        
        return confound
    
    @log_method
    def compute_derivative_of_confounds(self, confounds, order, win):   
        der = np.zeros_like(confounds, dtype=np.float32)
        
        for i in range(confounds.shape[1]): 
            der[:,i] = sigtools.savitzky_golay(confounds[:,i], window_size=win, order=order, deriv=1, rate=1)
    
        return der
    
    def get_motion_f2f_params_mcflirt(self, mats_path):
        mats = self.load_mat_files(mats_path)
        deltaMat = self.compute_temp_der(mats)
        deltaTrans = self.get_delta_trans(mats)
        thx, thy, thz = self.get_rots(deltaMat)
        # rows=samples and col=features
        confounds = deltaTrans
        confounds = np.hstack((confounds, thx))
        confounds = np.hstack((confounds, thy))
        confounds = np.hstack((confounds, thz))  
        
        return confounds
    
    def get_aux_parc_tissue_conf(self):
        label_data = imagetools.load_image(self.aux_tissue).get_data()
        labels = np.unique(label_data)[1:] # toss 0
        
        tiss_comp = np.zeros((self.data_dim[3], labels.size), np.float32)
        
        for i, lab in enumerate(labels):
            ind = label_data == lab
            roi = self.rs_data[ind,:]
#            delta_sigma_ev = sigtools.examine_eigen_values_per_label(roi, lab)
            tiss_comp[:, i] = roi.mean(axis=0)
            
#            try:
#                 if lab == 16:
#                     thres = 0.70
#                 thres = 0.25    
#                 num_comp = np.where(delta_sigma_ev < thres)[0][1]
#             except IndexError:
#                 tiss_comp[:, i] = roi.mean(axis=0) 
#                 log.info('Label:%s  mean taken' %lab)
#                 continue
#            
#            log.info('label:%s :num comp:%i' %(lab, num_comp))
#            tiss_comp[:, i] = self.get_comp_of_confound(roi, num_comp)
            
        return tiss_comp    
    
    def get_all_confounds(self):
        confounds = [] # dummy to allow concat without motion params
        
        if  self.move_par_type == 'mcflirt':
            # use the name space var b/c we could also ask for other motion      
            # parameters, such as abs motion from 1 st frame
            confounds = self.get_motion_f2f_params_mcflirt(self.mats_path)
            np.savetxt(self.motion_f2f_parameters_file, confounds)
        
        if self.move_par_type == 'cmtk':
            confounds = self.get_cmtk_move_par()   
                 
        if self.global_mean:       
            global_mean = self.get_global_mean_confound()
            confounds = np.hstack((confounds, global_mean))
        
        if self.aux_conf:
            aux = self.get_aux_confound()
            confounds = np.hstack((confounds, aux))
            
        if self.wm_rs is not None:
            wm_confound = self.get_tissue_confound(self.wm_rs)
            confounds = np.hstack((confounds, wm_confound))
        
        if self.aux_tissue is not None:
            aux_confound = self.get_aux_parc_tissue_conf()
            confounds = np.hstack((confounds, aux_confound))
        
        if self.csf_rs is not None:
            csf_confound = self.get_tissue_confound(self.csf_rs)
            confounds = np.hstack((confounds, csf_confound))
        
        if self.der_tissue_confounds_only:
            tiss_confounds = np.hstack((wm_confound, aux_confound))
            der = self.compute_derivative_of_confounds(tiss_confounds, order=2, win=5)
            confounds = np.hstack((confounds, der))
        
        if self.use_abs_motion_con:
            abs_motion_con = self.get_abs_motion_params(self.mats_path)
            confounds = np.hstack((confounds, abs_motion_con))
        
        if self.der_all_confounds:
            der = self.compute_derivative_of_confounds(confounds, order=1, win=3)
            confounds = np.hstack((confounds, der))     
        
        np.savetxt(path.join(self.root_dir, 'all_confounds.txt'), confounds)
        
        return confounds
    
    def save_clean_image(self, data, ind):
        clean_4D = np.zeros_like(self.rs_data, dtype=data.dtype)
        clean_4D[ind] = data                 
        
        # save the final image output    
        save_new_image(clean_4D, self.clean_rs_out_path, self.rs_obj.coordmap)
    
    def detrend_data(self, confounds):
        
        # band pass filter the confounds per Hallquist
        ## Transpose the confounds n/c filter uses row major order   
        confounds = sigtools.butter_bandpass(self.f_lp, self.f_up, self.sample_rate, confounds.T, order=5)
        
        # back to col major order for clean method coming up
        confounds = confounds.T
                
        roi, ind, _, _ = imagetools.get_roi(self.rs_path, self.mask_path)
        
        roi = sigtools.butter_bandpass(self.f_lp, self.f_up, self.sample_rate, roi, order=5)
        
        # conditioning for qr decomp 
        # col maj, detrend and norm along rows, axis=0
        sig = np.std(confounds, ddof=1, keepdims=True, axis=0)
        sig[sig == 0] = 1
        confounds /= sig
        confounds -= np.mean(confounds, keepdims=True, axis=0)
        # row maj, detrend and norm along col, axis=1
        roi -= roi.mean(axis=1, keepdims=True)
        sig = np.std(roi, axis=1, ddof=1, keepdims=True)
        sig[sig==0] = 1
        roi /= sig

        Q = qr_factor(confounds, mode='economic')[0]
        roi = roi.T
        roi -= np.dot(Q, np.dot(Q.T, roi))
        
        roi = roi.astype(np.float32, copy=False)
        
        log.debug("Saving orthog confounds to text file.")
        np.savetxt(path.join(self.root_dir, 'confounds_orthog.txt'), Q)
        
        roi[np.isnan(roi)] = 0
        
        return roi.T, ind                                 

    def load_mat_files(self, base_path):
                
        mats = np.zeros((self.data_dim[3],4,4))
        # only load in the mats we need. dropping 1 to have 268 even num, means toss MAT_0000
        for i in range(0, self.data_dim[3]): 
            index = i 
            mat_path = "MAT_%04d" %index
            fp = path.join(base_path, mat_path)
            mats[i,:,:] = np.loadtxt(fp)
    
        log.info('mat files loaded.')
        log.debug('Number of mat files loaded: %i' %mats.shape[0])
        
        return mats  
    
    def get_delta_trans(self, mats):
        deltaTrans = np.zeros((self.data_dim[3], 3))  
        for i in range(1,mats.shape[0]):
            deltaTrans[i,:] = mats[i,:3,3] - mats[i-1,:3,3] 
        
        log.info('translations computed')
        log.debug('Number of translation points: %i:' %deltaTrans.shape[0])
        
        return deltaTrans
    
    def get_trans(self, mats):
        trans = np.zeros((self.data_dim[3], 3))  
        for i in range(1,mats.shape[0]):
            trans[i,:] = mats[i,:3,3] 
        
        return trans
    
    
    def compute_temp_der(self, mats):
        deltaMat = np.zeros((self.data_dim[3],4,4))
        for i in range(1,len(mats)-1):
            deltaMat[i,:,:] =  np.dot( mats[i+1,:,:], invert(mats[i,:,:]) )
      
        log.info('Derivatives of affines computed.')
        log.debug('Number of affines derivatives: %i' %deltaMat.shape[0])
        
        log.debug("temp der computed")
        
        return deltaMat
    
    def get_rots(self, deltaRotmat):
        theta_x = np.zeros((self.data_dim[3],1)) 
        theta_y = theta_x.copy()
        theta_z = theta_x.copy()
        
        for i, mat in enumerate(deltaRotmat):
            rots = nie.mat2euler(mat[:3,:3])
            theta_z[i] = rots[0]
            theta_y[i] = rots[1]
            theta_x[i] = rots[2]
    
        log.debug('Rots computed.')
        
        return theta_x, theta_y, theta_z

    def get_cmtk_move_par(self):
        par = np.loadtxt(self.mats_path, dtype=np.float32)
        
        log.debug("cmtk p2p file loaded.")
        return par
    
    def get_abs_motion_params(self, mats_path):
        mats = self.load_mat_files(mats_path)
        trans = self.get_trans(mats)
        thx, thy, thz = self.get_rots(mats)
        # rows=samples and col=features
        confounds = trans
        confounds = np.hstack((confounds, thx))
        confounds = np.hstack((confounds, thy))
        confounds = np.hstack((confounds, thz))  
        
        rms = self.get_abs_motion_rms(confounds)
        
        return rms[:, np.newaxis]

    def get_abs_motion_rms(self, confounds):   
        motion = MotionAnalysis(self.root_dir, '', self.mask_path, '',  confounds)
        rms = motion.calculate_trajectory_dpc()
        
        return rms
        

class MakeMeanTimeSeries(object):

    def __init__(self, root_dir, rs_data_path, parc_path=None, Type='human'):
        self.root_dir = root_dir
        self.parc_path = parc_path    
        self.rs_path = rs_data_path
        
        self.averages_path = path.join(self.root_dir,'roi_averages.npy')    
        self.roi_dict_path = static_paths.roi_dict
    
        bw_tr = BandwidthTR(Type, self.rs_path)
        self.TR = bw_tr.TR
       
        self.erode_bool = False # incase you need to first erode the parc image
        self.parc_ero = path.join(self.root_dir, 'parc_ero.nii.gz')

    def load_data(self):
        if self.parc_path is not None:
            
            self.parc_obj = imagetools.load_image(self.parc_path)
            self.parc_data = self.parc_obj.get_data()
            self.parc_data = np.squeeze(self.parc_data)
            self.parc_erode = self.parc_path.replace('.nii.gz', '_eroded.nii.gz')
    
        self.rs_obj = imagetools.load_image(self.rs_path)
        self.rs_data = self.rs_obj.get_data()
        self.rs_dim = self.rs_data.shape 
    
        self.roi_dict = load_roi_dict(self.roi_dict_path)
    
    def main(self):
        self.load_data()
        
        if self.erode_bool:
            self.erode_roi()
        
        self.average_roi_tc()
            
    def average_roi_tc(self):
        roi_names = self.roi_dict.keys()
        roi_avg_data = np.empty( (len(roi_names), self.rs_data.shape[3]), dtype=np.float32 ) 
        roi_avg_data[:] = np.nan
        
        for name in roi_names:
            index  = self.roi_dict[name]['index']
            label_match = self.parc_data == self.roi_dict[name]['label']  
            
            if not np.all(np.bitwise_not(label_match)):# no match found, will return all False's
                temp = self.rs_data[label_match,:]
                                
                temp = temp[np.all(temp, axis=1)]# avoid a vector of all zeros, at a given voxel
                roi_avg_data[index,:] = np.mean(temp, axis=0)
        
        roi_avg_data -= roi_avg_data.mean(axis=1, keepdims=True) # center
        norm = roi_avg_data.std(axis=1, keepdims=True) # deal with this sep to keep Warnings from being raised
        norm[ norm == 0 ] = 1
        roi_avg_data /= norm
        
        np.save(self.averages_path, roi_avg_data)     

    def erode_roi(self):
        struct = generate_binary_structure(rank=3, connectivity=3)
        temp = np.zeros_like(self.parc_data, dtype=np.uint8)
        ero = np.zeros_like(self.parc_data, dtype=np.uint8)
        roi_names = self.roi_dict.keys() 
        
        for name in roi_names:
            #index  = self.roi_dict[name]['index']
            mask = self.parc_data == self.roi_dict[name]['label']  
            temp[mask] = 1
            out = binary_erosion(temp, structure=struct, mask=mask)
            ero[out] = self.roi_dict[name]['label'] 
            temp = np.zeros_like(self.parc_data, dtype=np.uint8)
            
        imagetools.save_new_image(ero, self.parc_ero, self.parc_obj.coordmap)                   
        self.parc_data = ero   
                          
    def get_average_from_rsREL(self):
        rsREL = static_paths.rsREL    
        roi_data = imagetools.load_image(rsREL).get_data()
        rois = np.unique(roi_data)[1:] # toss the zero
        
        roi_avg_data = np.zeros( (len(rois), self.rs_data.shape[3]), dtype=np.float32 ) 

        for i, roi in enumerate(rois):
            # make mask 
            label_match = roi_data == roi  
            temp = self.rs_data[label_match,:]
            temp = temp[np.all(temp, axis=1)]# avoid a vector of all zeros, at a given voxel
            roi_avg_data[i,:] = np.mean(temp, axis=0)
        
        roi_avg_data -= roi_avg_data.mean(axis=1, keepdims=True) # center
        norm = roi_avg_data.std(axis=1, keepdims=True) # deal with this sep to keep Warnings from being raised
        norm[ norm == 0 ] = 1
        roi_avg_data /= norm
        
        return roi_avg_data
    

class ProcessTimeSeriesRoi(object):
    def __init__(self, root_dir, roi_averages_path, Type, rs_image_path, 
                 parc_path, rs_mask_path):
        
        self.root_dir = root_dir
        self.parc_path = parc_path
        self.rs_mask_path = rs_mask_path
        self.rs_image_path = rs_image_path 
        self.roi_averages_path = roi_averages_path
        self.roi_dict_path = static_paths.roi_dict
        self.corr_path = path.join(self.root_dir, 'corr.npy') 
        
        self.Type = Type
    
    def load_data(self):    
        self.roi_averages = np.load(self.roi_averages_path)
        
        if self.Type == 'human':
            self.roi_dict = load_roi_dict(self.roi_dict_path)
            roi_names = self.roi_dict.keys() 
          
        elif self.Type == 'rat':
            roi_names = [str(num) for num in range(0, self.roi_averages.shape[1])]
            roi_names = np.array(roi_names)

        else:
            raise Exception, "Type not understood. Choices are 'human' or 'rat', you entered:%s" %self.Type
        
    def coherence(self):
        pass

    def correlation(self, p_thres=0.05):
        self.load_data()
        
        roi = self.roi_averages

        corr = np.corrcoef(roi)
                
#         enn = empirical_pvalue.NormalEmpiricalNull(corr.ravel())
#         corr_value_thres = enn.uncorrected_threshold(p_thres) 
#         corr[corr < corr_value_thres] = 0 
        
        np.save(self.corr_path, corr)
        
        return corr 

    def seed_analysis(self, seed, targets, method='corr'):
        pass
     
    @classmethod        
    def create_vol_for_vis(self, roi_dict, parc_path, corr):
        parc_obj =  imagetools.load_image(parc_path)
        parc_data = parc_obj.get_data()
        
        output = np.zeros(parc_data.shape)
        
        for name in roi_dict.keys():
            ind = parc_data == roi_dict[name]['label']
            i = roi_dict[name]['index']
            output[ind] = corr[i,0]
            
        return output          
    

class VoxelWiseAnalysis(object):
    def __init__(self, root_dir, rs_path, rs_mask_path, Type='human'):
        
        self.global_corr_thres = 0.80
        #self.Threshold_Corr_values_Bool = False # not gonna use this probably but here some good code here I think
        self.root_dir = root_dir
        self.rs_path = rs_path
        self.rs_mask_path = rs_mask_path
        
        bw_obj = BandwidthTR(Type, rs_path)
        self.TR = bw_obj.TR
         
        self.corr_binary = None
        self.corr = None
        self.corr_path_binary = path.join(self.root_dir, 'corr_binary.npy')
        self.corr_path = path.join(self.root_dir, 'corr.npy')
        self.ind = None
        self.roi_shape = None
        self.rs_coordmap = None
 
        self.deg_map_path = path.join(self.root_dir, 'deg_map.nii.gz')
        self.local_clus_coef_path = path.join(self.root_dir, 'local_clustering_coefs.nii.gz')
        self.tot_conn_path =  path.join(self.root_dir, 'total_connectivity.nii.gz')
        
        self.state = self.make_rand_gen_state_obj()
        
    def make_rand_gen_state_obj(self):
        part1 = 'MT19937'
        part2 = np.array([1182714280, 1154484342, 2894605489, 2592055667, 2601979739,
       1382299672,  563442768,  928512024, 3937575551, 3475438645,
       3817282362, 2354294932, 1151478423, 3072296329, 4216855466,
       3717602631, 2510943908, 3990243076, 1251349971, 2656581336,
        111895012, 3968560765, 1860573550, 2584797309, 4161220661,
       3062992462, 2652247617, 3778673430, 1613379143, 2940913901,
       1424416248, 3620417765, 3443591665,  983908586, 2801693793,
       1464159934, 2895688681, 2826333938, 1366598615, 1788599490,
         89403214,  985138631,  785799752,  989236887,  829042603,
       4239635445, 1329671194, 1264616579, 3546566488, 3626899858,
       1585505319,  630931866, 3183080558, 2575604150, 2259661910,
       2487356094, 2128343589, 1708054433, 1344331963, 3130599328,
        628805348, 4283754272, 2222402726, 1071587731,  163478789,
       1529010700, 1726152244, 3730399955, 2069722123, 1999103875,
       4240970593,  139556252, 2585078257, 1440756860, 3940624917,
       2386311378, 2301569054, 3090695190, 3900281933,  347700411,
       1989299056, 3813504153, 3694283701, 2136732343, 3207762111,
       3571342637, 3168022184, 1477493222, 2906550355, 2416222675,
       3413338363,  481569546, 3554283766, 3752038842,  383608904,
       2892929149, 3675379846, 3092704085, 2765464669, 2515385998,
       2580452856, 4073239655, 1672008072, 4229326597, 3794255789,
       2084504482, 1491877385,  928947085, 2929252050,  315010387,
        897597611,  851013784, 1439157770, 2985338953,  459395022,
        676460704, 2153202071, 1919090736,  253318147, 3958652646,
       1908215306,  374634240, 2778382314, 2201836981, 2472846347,
       2237987585, 2026015150, 2911282560,  723688953, 1751554584,
       2353926674, 4224374597,  538842412, 3730597704, 4014996081,
       2146785297, 1537121885, 2134267039, 1490202337, 2710577256,
       1974423587, 1104548025,  423468379, 3783033594, 2012129875,
       3769933805, 4088266439, 1547449555, 3593260496, 2012113290,
       3583785237, 3540386963,  980540578, 4022473116,  542663002,
       3124432577,  647345004, 2397498471,  842874357, 2321456408,
       2144023525, 4184704483, 2708349273,  695391106, 3927563620,
       2045467753,  260478605, 3116962068, 3099910952, 2133573524,
       2555458447, 3996787233, 1467168357, 2148259759, 1791564699,
       2595680918, 3928634795, 2320609040,  455551825, 3866074069,
       2824600016, 3650870404, 3705192023, 1911798906, 2745094018,
       3988445969, 1348665343, 1256481530, 4002710636, 3021425450,
       1777175522, 1295475500, 2609397659, 1902960044, 2208011762,
       1943745087, 2635455184, 3035635346,  175289012, 2072427282,
        103586674,  144301109, 1702214577, 2066151504, 2173705196,
       1636976230, 2197924803, 4274343053, 4238402685, 1469547662,
       4291724691, 3999219328, 1192030825, 2929069147, 3377140724,
        540938403,  371940811, 1876171536, 3896288484, 4014586550,
       2043733082, 1056323182, 3957221696, 2036664053, 1546410684,
       1856770233,  408047266, 3665592455, 3290729405, 1634163783,
       2687492235, 2597161687, 3785362257, 3860110046, 1205510541,
       4086028249, 3379018414, 4123164786, 4213490899, 2110939703,
       3322109679, 3810487031, 3741561766, 1221648232, 3090724887,
       1788175087, 3736071354, 3585021307,  911529034,  958454186,
       1294794644, 3434239563, 2714102795,  188431233, 1494118175,
        218094142,  478863827, 3785254623,  588337181, 1617705236,
       2410651322, 2693136085, 2346779319, 1837822102,   23332956,
       2642261696, 1702119333, 3361775425, 1045298862, 3004048956,
       1249958963, 1474577843, 2534415831,  647785688, 4276172856,
        614413926,  327148393, 1495156390, 2324257529, 1378385280,
       1757955650,  245213614, 1266814065, 2941677349, 2823401033,
        856129537, 1777058573, 2989533277, 2641902069, 1188436063,
       1201660200, 3435529820, 2651513698, 1874151674, 3215561810,
       1688937326,  857637079, 1791537712, 3108875114, 2665111432,
       3828876161, 3489958086, 1392880226, 2709584278, 3607287114,
       4144777296, 3537489360,  971197029, 2525429660, 2071310203,
       2186919798, 1162079712, 3057291408,  147578478, 2633342143,
       2861901277, 2141204177, 2339223071,  657949072, 2458817397,
       2738787194,  240462203, 3004498794, 1021210380,  725880253,
        471926068, 1816079504, 1325958091,  913585024,  994086316,
       2588331202, 1307739904,  907816040, 2149807195, 1632577801,
       3323011514, 1461399226, 2904878213, 4284751368, 1270979616,
       3279272699,  388579887, 2420283339, 3858361219, 2911037880,
        459198946, 3620880342, 1311868795,  349218715, 2976886635,
       1748969970, 3623592431, 1378630860, 2798910053, 3825370915,
        519989869, 4192096223, 3357319857, 1381845564, 1754058839,
        897836002, 1597890398, 1799287625, 3373515480, 1729735701,
       1574165703, 2473153110,  253902870,  529798678, 4214997936,
        575430939, 3230684945, 3052962979, 1292109498, 2863376414,
       1402760525, 1157902985, 3592490210, 1529784935, 2360398415,
       1890701464, 2247614588, 2279841552, 1222464523, 2698223234,
       3230013366, 1389191191, 4094817741, 2013689150, 1286496670,
       1718755653, 4225194029,  582879204, 3014832773,  871335165,
       3248530035,  767451870,   20627327, 2963601209, 1679104870,
       3286613480, 4219522169,  206286170, 1797914428, 1840996027,
        805712901, 3799037905, 2305225128, 1159499263,  521367284,
       3168834513, 2992713937, 3373275001, 1945537549, 2636064957,
       2081109680, 1534868428, 1152121329, 1394914759, 3432209022,
       1359602780, 2570978252, 1251001808,  670723389, 1409141088,
       3597081058, 3579943444,  515532964, 1354553423, 1778026732,
       2361985239, 1780797068,  715013031, 2171236250, 3012657693,
       4289646271, 2291692832, 2929846694,   23659542,  783132023,
       3836369833, 2060583002, 1836130143, 1253137784,  843043987,
       1591793339, 4197224467, 2843665503, 2621318873, 3438630149,
       1445878572, 1794213436, 1572909406,  426835182,  299122019,
       2950062516, 1831796247, 2084110323,  161122705,  346806045,
       2907876783, 2608383819, 1087350114,  327270677, 2150431220,
       1469960910, 2526072363, 2141264328, 1903543382, 2306106640,
       2844150808, 3446542431, 1492620839,   73661523, 3249651389,
       3885831048, 1792786324, 3500524732, 3670494711, 2564535075,
        524107180, 1902106451,  835733183, 1004517298, 3507632282,
       2679954397, 2540578494, 1606369001,  152227983, 1937208892,
       1445898703,  222400966, 1054247590, 2143885542, 2930315034,
        539600905,  864036034, 4162765419,  622635892,  923382298,
        602869787, 3705136783, 3063436101, 3397971819, 2489667735,
       2840553484, 3678166221, 1637264174, 3963590766, 1344194319,
       1533595420, 4005719461,  555262503,  184210495, 4094383456,
       2697821014, 1790631386, 2048670065, 3066753750, 1804859845,
       1723546915, 3627813259, 2395950407, 3628608368,  260844475,
       2998526247, 2950592355, 2457460370, 1089449122, 3662153130,
       1573787428,  354136421, 3624551704, 1586711440,  913261533,
       2422189039, 2933636884, 1491544419,  361497893, 1941442035,
       1951120789, 4064358918, 3571711103,  489084338, 1462298315,
       3761220014, 3510439607, 3499301661,  563618040, 3224919853,
        415672215, 3373075169,  748587823, 2442712650, 4288060881,
       1665420583, 1097409284, 1267429684, 2246466218,  231923858,
        104217425, 3801263085, 2723124289, 1824309142, 3710476639,
       2666539577,  194874852, 1530542433, 2990856022,  188065701,
       3308789885, 3360993613, 1724347465, 2403700090, 3718297905,
        805632654, 4151957024,  139143078, 2426465538, 3640224198,
       3080180936, 2536800996, 2708599696, 3580062370, 2394725947,
        530556019,  955584375, 4008624806, 2185128133,  278990029,
       4013157202, 1261011887,  427158192, 2935528730, 1246904555,
       1741939921, 2848561671, 1367892364,  163332197, 4238194041,
       2064463374, 1660009048,   98456248, 3053405985, 4064936311,
       4273801410, 2775877688, 3961350906,  990095392, 3462465798,
       3096993108, 2711044637,  699377893,  539574239, 1950052368,
       1667713671,  162768162, 4135339878, 4239493774, 3891933754,
       2505979596,  470325593,  859051992,  243336714], dtype=np.uint32)
        part3 = 332
        
        return (part1, part2, part3) 
 
    def _check_roi_mask_parameters(self):
        try:
            assert self.roi_shape is not None
            assert self.ind is not None
            assert self.rs_coordmap is not None
            assert self.i_ind is not None
        
        except:
            _, self.ind, self.roi_shape, self.rs_coordmap = imagetools.get_roi(self.rs_path, self.rs_mask_path)
                
            self.i_ind = np.nonzero(self.ind)[0]
            self.j_ind = np.nonzero(self.ind)[1]
            self.k_ind = np.nonzero(self.ind)[2]
    
    def _check_for_binary_corr(self):
        if self.corr_binary is None:
            if path.exists(self.corr_path_binary):
                self.corr_binary = np.load(self.corr_path_binary)
                self._check_roi_mask_parameters()
            
            else:
                self._check_for_corr()
                self.make_binary_corr(self.global_corr_thres)

    def _check_for_corr(self):
        if self.corr is None:
            if path.exists(self.corr_path):
                self.load_corr_matrix()    
                self._check_roi_mask_parameters()
                
            else:
                self.compute_corr()    
                  
    def compute_corr(self):               
        '''scale types are: position or size'''
        log.debug("Loading ROI.")
        # using class var for the roi which could be large
        # forcing it to be called by reference when sent to various methods.
        self.roi, self.ind, self.roi_shape, self.rs_coordmap = imagetools.get_roi(self.rs_path, self.rs_mask_path)
        
#         if not np.all(self.roi):
#             num_of_zeros = self.roi[self.roi == 0].size
#             self.roi[self.roi == 0] = np.random.randn(num_of_zeros)
#             log.info("ROI getting zeros replaced with random numbers.")
        
        log.debug("Computing corr.")
        self._corr() # sets self.corr
        
        self.corr -= self.corr.mean()
        self.corr /= self.corr.max(keepdims=True)
        np.fill_diagonal(self.corr, 0)
        
        np.save(self.corr_path, self.corr)
        
    def make_binary_corr(self, thres):    
        corr = self.corr.copy()
        
        corr[np.isnan(corr)] = 0
        corr[corr >= thres] = 1
        corr[corr < 1 ] = 0
    
        self.corr_binary = corr
                         
        np.save(self.corr_path_binary, self.corr_binary)

    def compute_degree(self):
        self._check_for_binary_corr()
        
        deg_map = np.zeros(self.roi_shape[0:3], dtype=np.float32)
        deg_map[self.ind] = self.corr_binary.sum(axis=0)        
        
        imagetools.save_new_image(deg_map, self.deg_map_path, self.rs_coordmap, log_this=True)

    def _corr(self):
        if not self.roi.flags.f_contiguous:
            self.roi = np.asfortranarray(self.roi)
        
        if self.roi.dtype != np.float32:
            self.roi = self.roi.astype(np.float32, copy=False)
                    
        n =  float(self.roi.shape[1] - 1)
        
        log.debug("Computing Dot product")
        cov = np.dot(self.roi, self.roi.T)
        cov /= n
        self.roi = []
        #cov = blas.sgemm(alpha=1.0, beta=n, a=self.roi, n=self.roi, trans_b=True)
        #cov /= n
        
        sigma = np.diag(cov)
        sigma = sigma[:, np.newaxis]
        
        outer = np.sqrt(np.dot(sigma, sigma.T))
        #outer = blas.sgemm(alpha=1.0, a=sigma, n=sigma, trans_b=True)
        del(sigma)
        
        #outer = np.sqrt(outer)
      
        cov /= outer
        
        self.corr = cov

    def compute_total_conn(self):
        self._check_for_binary_corr()
    
        np.fill_diagonal(self.corr_binary, 0)# zero the diagonal
        coefs = np.zeros(self.corr_binary.shape[0], np.float32)
        
        for i in range(self.corr_binary.shape[0]):
            row = self.corr_binary[i,:]
            sub_conn_ind = np.nonzero(row)[0]
            
            #build array or cols ( same as rows )
            cols = self.corr_binary[:, sub_conn_ind]
            
            #find connections bw components with dot prod
            coefs[i] = row.dot(cols).sum()   
        
        new = np.zeros(self.roi_shape[0:3], dtype=np.float32)
        new[self.ind] = coefs
        imagetools.save_new_image(new, self.tot_conn_path, self.rs_coordmap, log_this=True)
        
    def compute_local_cluster_coef(self, thres=2):
        '''http://en.wikipedia.org/wiki/Clustering_coefficient'''
        if not path.exists(self.tot_conn_path):
            self.compute_total_conn()
            
        self._check_for_binary_corr()
        
        np.fill_diagonal(self.corr_binary, 0)# zero the diagonal
        deg = self.corr_binary.sum(axis=1)
        deg[deg < thres] = 1e4
        denom = deg * (deg -1)
        
        tot_conn_roi,_,_,_ = imagetools.get_roi(self.tot_conn_path, self.rs_mask_path)       
        tot_conn_roi /= denom
        
        new = np.zeros(self.roi_shape[0:3], dtype=np.float32)
        new[self.ind] = tot_conn_roi
        imagetools.save_new_image(new, self.local_clus_coef_path, self.rs_coordmap, log_this=True)
            
        return new

    def load_corr_matrix(self):
        self.corr = np.load(self.corr_path)

    def pvalue_ideas(self):    
        tri = np.tril(self.corr)
        #del(corr)
        half = tri[tri != 0]
        
        if half.size > 2e6:
            samp_size = 2e5
            
            ran_obj = np.random
            ran_obj.set_state(self.state)
            
            samp = np.zeros((5, samp_size), dtype=np.float32)
            
            #sample with replacement
            for i in range(5):
                ind = ran_obj.random_integers(0, half.size, samp_size)
                samp[i] = half[ind]
            
            del(half)
         
            enn = empirical_pvalue.NormalEmpiricalNull(samp.ravel())
            del(samp)
                
        else:
            enn = empirical_pvalue.NormalEmpiricalNull(half)
            
        enn.learn()
        #mu = enn.mu
        sigma = enn.sigma
        
        self.corr[self.corr < sigma] = 0       
            
    def compute_deg_with_dist(self, corr_thres, dist_thres):
        self._check_for_corr()
        self._check_roi_mask_parameters()
        
        world_dim = imagetools.get_pixdims(self.rs_path)[1:4]
        world_dim[2] = np.round(world_dim[2], decimals=1) # dim3 sometimes 4.99999
        
        corr = self.corr.copy()
        del(self.corr) # for memory
        
        corr[corr > corr_thres ] = 1 #TODO: p value sig testing ?
        corr[corr < 1 ] = 0
        
        out = np.zeros(corr.shape[0], dtype=np.float64)
        
        for i in range(corr.shape[0]):
            row = corr[i,:]
            pts = np.nonzero(row)[0] #ind of  non zero connections
            co_2 = np.squeeze(self._get_ijk(i))#starting location i,j,k
            points = self._get_ijk(pts)#make tuples of i,j,k coordinates target
            dists = self._get_dist(points, co_2, world_dim)
        
            dists[dists < dist_thres] = 0
            dists[dists > 0] = 1
            out[i] = dists.sum()
        
        new = np.zeros(self.roi_shape[0:3])
        new[self.ind] = out
        
        basename = 'deg_dist_thres_%.2fth_%2.1fmm.nii.gz' %(corr_thres, dist_thres)
        outfile = path.join(self.root_dir, basename)
        imagetools.save_new_image(new, outfile, self.rs_coordmap, log_this=True)

    def compute_deg_cont_thres(self, corr_thres_min=0.20, corr_thres_deg=0.7):
        self._check_for_corr()
        self._check_roi_mask_parameters()
        
        world_dim = imagetools.get_pixdims(self.rs_path)[1:4]
        world_dim[2] = np.round(world_dim[2], decimals=1) # dim3 sometimes 4.99999
        
        corr = self.corr.copy()
        corr[corr < corr_thres_min] = 0 # exp shows that below .30 is questionable for across brain
        del(self.corr) #for safety, since self.corr is used else where 
        
        outfile = path.join(self.root_dir, 'deg_cont_thres_mm.nii.gz')    
        
        out = np.zeros(corr.shape[0], dtype=np.float64)   
        
        
        for i in range(corr.shape[0]):
            co_2 = np.squeeze(self._get_ijk(i))#starting location i,j,k
            
            row = corr[i, :]
            ind = np.arange(corr.shape[0])
            points = self._get_ijk(ind)#make tuples of i,j,k coordinates target
            
            dists = self._get_dist(points, co_2, world_dim)
            weight = (corr_thres_deg - corr_thres_min) / dists.max()
            
            adj_corr = row * (dists * weight * row + 1)
            
            adj_corr[adj_corr >= corr_thres_deg] = 1
            adj_corr[adj_corr < 1] = 0
            
            out[i] = adj_corr.sum()

        new = np.zeros(self.roi_shape[0:3])
        new[self.ind] = out
        
        imagetools.save_new_image(new, outfile, self.rs_coordmap, log_this=True)

    def compute_deg_with_multi_dist(self, A=200, corr_thres=0.30):
        self._check_for_corr()
        self._check_roi_mask_parameters()
        
        world_dim = imagetools.get_pixdims(self.rs_path)[1:4]
        world_dim[2] = np.round(world_dim[2], decimals=1) # dim3 sometimes 4.99999
                
        corr = self.corr.copy()
        
        corr[corr >= 0.70 ] = 1
        corr[corr < 1] = 0 
        out = corr.sum(0)
        
        corr = self.corr.copy()
        del(self.corr) # for memory     
           
        corr[corr >= 0.7] = 0
        corr[corr < corr_thres] = 0
        
        m = A / 0.4
        b = A - m * corr_thres       
        
        for i in range(corr.shape[0]):
            co_2 = np.squeeze(self._get_ijk(i))#starting location i,j,k
            
            row = corr[i, :]
            ind = np.nonzero(corr[i,:])[0]
            points = self._get_ijk(ind)#make tuples of i,j,k coordinates target
            
            dists = self._get_dist(points, co_2, world_dim)

            dist_thres =  -m * row[ind] + b# dist thres computed for each corr value
            dists[dists < dist_thres] = 0
            out[i] += dists.sum()
        
        new = np.zeros(self.roi_shape[0:3])
        new[self.ind] = out
        name =  'deg_0.70_cont_%fth_%fmm.nii.gz' %(corr_thres, A)
        out = path.join(self.root_dir, name)
        imagetools.save_new_image(new, out, self.rs_coordmap, log_this=True)

    def _get_dist(self, points, co_2, dim):
        dist = np.zeros(points.shape[0], np.float32)
       
        for i in range(points.shape[0]):
            co_1 = points[i,:]
            a = dim[0]
            b = dim[1]
            c = dim[2]
            
            dist[i] = np.sqrt( a**2*(co_2[0]-co_1[0])**2 + b**2*(co_2[1]-co_1[1])**2 + c**2*(co_2[2]-co_1[2])**2 )
          
        return dist
    
    def _get_ijk(self, pts):
        if isinstance(pts, int):
            return (self.i_ind[pts], self.j_ind[pts], self.k_ind[pts])
        else:
            n = len(pts)    
        
        points = np.zeros((n,3))
        
        for i in range(n):
            points[i,:] = (self.i_ind[pts[i]], self.j_ind[pts[i]], self.k_ind[pts[i]])
         
        return points
        
    def seed_analysis_from_ijk(self, i, j, k, *args):
        self._check_for_corr()
        self._check_roi_mask_parameters()
        
        # get pixdim for i, j ,k
        row_num = np.where((self.i_ind==i) & (self.j_ind==j) & (self.k_ind==k))[0]
        
        if row_num.shape[0] != 1:
            raise Exception, "The i,j,k co ordinates do not lie within the brain."
        
        new = np.zeros(self.roi_shape[0:3], dtype=np.float32)
        new[self.ind] =  np.squeeze(self.corr[row_num,:])
        name = 'seed_corr_%i_%i_%i.nii.gz' %(i,j,k)
        imagetools.save_new_image(new, path.join(self.root_dir, name), self.rs_coordmap, log_this=True)
        
        return row_num        


class MotionAnalysis(object):
    
    def __init__(self, root_dir, bold, brain_mask, gm,  f2f_motion_file):
        self.root_dir = root_dir
        self.bold = bold
        self.brain_mask = brain_mask
        self.gm = gm
        self.motion_file = f2f_motion_file
        self.motion_thres = 0.35
            
    def calculate_trajectory_dpc(self):
        ''' motion file is from PyConn.Detrend. it is the frame to frame
        motion and is 6 cols with a row for each time point. Or, it is a 
        numpy array time x voxels''' 
        if isinstance(self.motion_file, str):
            motion = np.loadtxt(self.motion_file)
        elif isinstance(self.motion_file, np.ndarray):
            motion = self.motion_file
        else:
            raise TypeError, "motion arg should be a txt file or a np array"
              
        # vox dim in mm
        pixdims = imagetools.get_pixdims(self.brain_mask)[1:4]
        
        mask_data = imagetools.load_image(self.brain_mask).get_data()
        
        # distance map to closet vox outside mask
        distances = ndimage.distance_transform_edt(mask_data)
        
        dist = np.where(distances == distances.max())
        
        # max dist is worst case
        distx = dist[0].max()
        disty = dist[1].max()
        distz = dist[2].max()
        
        #largest diameter in each axis
        dia = 2 * np.array((distx, disty, distz)) * pixdims
        dia = np.tile(dia, (motion.shape[0],1))
        
        trans = motion[:,0:3]
        rots = dia * motion[:,3:] 
                 
        trans_rms = np.sqrt(np.mean(trans**2, axis=1))
        rots_rms = np.sqrt(np.mean(rots**2, axis=1))
             
        trajectory = trans_rms + rots_rms
     
        return trajectory
    
    def convert_spm_motion_to_vol2vol_file(self, rp_file):
        
        orig = np.loadtxt(rp_file)
        
        trans = orig[:, 0:3]
        rots = orig[:, 3:]
        
        num_vols = orig.shape[0]
        
        mats = np.zeros((num_vols, 4, 4))    
        
        # put params back into Matrices
        for i in range(num_vols):
            mats[i, 3, :] = np.array((0 ,0, 0, 1))
            mats[i, 0:3, 3] = trans[i,:]
            # euler2mat( z,y,x)
            mats[i, 0:3, 0:3] = nie.euler2mat(rots[i, 2], rots[i, 1], rots[i, 0])  
        
        detrend = DetrendMovementNoise(root_dir=None, rs_4D_path=None, confound_cleaned_output=None, mask=None)
        detrend.data_dim = np.array((0, 0, 0, num_vols))

        delta_mat = detrend.compute_temp_der(mats)
  
        rotx, roty, rotz  = detrend.get_rots(delta_mat)
 
        trans = detrend.get_trans(delta_mat)

        return np.concatenate((trans, rotx, roty, rotz), axis=1)

    def get_rms_std_prime_per_vol(self):
        roi, _, _,_ = imagetools.get_roi(self.bold, self.brain_mask)
        
        sig_prime = np.hstack(((0), np.diff(roi.std(0), axis=1)))
        
        rms_sig_prime = np.sqrt(np.mean(sig_prime**2))
        
        return rms_sig_prime

    def caluculate_trajectory_power(self):
        motion = np.loadtxt(self.motion_file, np.float16)
        
        mot_dr = np.hstack(((0), np.diff(motion, axis=0)))
                           
        traj = np.sqrt(np.mean((mot_dr[:,0:3]*0.5)**2, axis=1))
                           
        traj += np.sqrt(np.mean(mot_dr[:,3:]**2, axis=1))
        
        return traj
    
    def main_dpc(self):
        
        traj = self.calculate_trajectory_dpc()
        # prune traj b/c of possible vol drop 
        num_vols = imagetools.get_dim(self.bold)[4]
        
        drop = traj.size - num_vols
        if drop < 0:
            raise Exception, "Something weird happened adjusting drop points for traj."
        
        traj = traj[drop:]
        
        traj_copy = traj.copy()
        
        _, sigma = sigtools.get_mean_signal_with_mask(self.bold, self.gm, True)
        
        sigma -= sigma.mean()
        sigma /= sigma.std()
        
        traj -= traj.mean()
        traj /= traj.std()
        
        test = sigma * traj
        
        ind = np.where((test < 0) & (traj_copy > self.motion_thres))[0]
        
        ind_chunks = self._get_contiguous(ind)
        ind_chunks = self.join_chunks(ind_chunks)
        
        bad_ind = []
        for tup in ind_chunks:
            if len(tup) > 2:
                #tup = [tup[0-1] + 1] + [tup[0] - 1] + tup
                bad_ind.append(tup)
            
        bad_ind = np.array(imagetools.flatten(bad_ind))
        bad_ind = np.unique(bad_ind)
        
        if not len: # empty return False
            return None, traj_copy, test            
 
        else:
            return bad_ind, traj_copy, test

    def main_power_censor(self):
        #dvars = self.get_rms_std_prime_per_vol()[0]
        
        traj = self.calculate_trajectory_power()
        # prune traj b/c of possible vol drop 
        num_vols = imagetools.get_dim(self.bold)[4]
        
        drop = traj.size - num_vols
        if drop < 0:
            raise Exception, "Something weird happened adjusting drop points for traj."
        
        traj = traj[drop:]
        
        traj_copy = traj.copy()
    
        keep_ind = np.where(traj < 0.3)[0]
        toss_ind = np.where(traj > 0.3)[0]
        
        
        cont_ind = self._get_contiguous(keep_ind)    
        
        for chunk in cont_ind:
            if len(chunk) < 5:
                toss_ind = np.hstack((toss_ind, chunk))     
                
        keep_ind = np.arange(traj.size)
        keep_ind = np.delete(keep_ind, toss_ind)
        
        cont_ind = self._get_contiguous(keep_ind)    
        
        total_len = 0
        for  chunk in cont_ind:
            if len(chunk) >= 50:
                total_len += len(chunk)
        
        if len(toss_ind) > 0: 
            return toss_ind, traj_copy, None # this None is for consist with other main
        
        else:
            return None, traj_copy, None            

                          
    def join_chunks(self, chunks):
        new_chunk = []
        for i in range(len(chunks)-1):
            c1 = chunks[i][-1]
            c2 = chunks[i+1][0]
            if c2 - c1 > 1 and c2 - c1 < 4:
                new_chunk.append(chunks[i] + chunks[i+1])
                
        return new_chunk + chunks        
                
                            

    def _get_der_trig(self, roi, Stat, extrema_type):
        
        if Stat == 'std':
            signal = np.std(roi, axis=0) 
        
        elif Stat == 'mean':
            signal = np.mean(roi, axis=0) 
        
        signal -= signal.mean()# this is just for humans
        
        #signal = sigtools.low_pass_filter(signal, 0.1, 1/2.2, 2)
         
        #signal_der1 = np.hstack( (0 , np.diff(signal)) )
        signal_der1 = sigtools.savitzky_golay(y=signal, window_size=3, order=1, deriv=1)
        signal_der2 = np.hstack( (0 , np.diff(signal_der1)) )
        #signal_der2 = sigtools.savitzky_golay(y=signal, window_size=5, order=2, deriv=2)
        
        test = signal_der1[0:-1] * signal_der1[1:]
        test = np.insert(test, [0], 0)
        signal_trig = np.zeros_like(signal)
        
        signal_der2 = sigtools.low_pass_filter(signal_der2, 0.1, 1/2.2, 2)
        
        if extrema_type == 'max': 
            maxima = np.where((test < 0) & (signal_der2 < 0))[0]
            signal_trig[maxima] = 1
            

        elif extrema_type == 'min': 
            minima = np.where((test < 0) & (signal_der2 > 0))[0]
            signal_trig[minima] = 1

        return signal_trig, signal_der2
        
    def correlate_traj_mean(self, traj, motion_div):
        '''For line fitting: use traj_bins as "x", and scipy.polyfit(traj_mu, sigma_mu)
        for the line. You can use "w=errors" for a weighted fit. '''
        roi, _,_,_ = imagetools.get_roi(self.bold, self.brain_mask)
        mu = roi.mean(0)
        
        range_ = traj.max() - traj.min()
        num_bins = np.int16(np.round(range_ / motion_div, decimals=0)) #mm
        traj_bins = np.linspace(traj.min(), traj.max(), np.int8(num_bins))
        
        new_ind = np.digitize(traj, traj_bins)
        
        traj_mu = np.zeros(num_bins, np.float32)
        mean_mu = np.zeros(num_bins, np.float32)
        errors = np.zeros(num_bins, np.float32)
        
        for i in range(num_bins):
            get_vals = lambda vec: vec[new_ind == i]
            
            traj_mu[i] = np.mean(get_vals(traj))
            
            mean_mu[i] = np.mean(get_vals(mu))
            
            vals = get_vals(mu)
            errors[i] = np.std( vals[vals != 0], ddof=0 )
        
        traj_mu[np.isnan(traj_mu)] = 0 # if no new_ind matched, a nan occurs, same as 0 for this
        mean_mu[np.isnan(mean_mu)] = 0
        errors[np.isnan(errors)] = 0
        
        return traj_bins, traj_mu, mean_mu, errors

    def _get_contiguous(self, chunk):
        out = []
        for _, g in groupby(enumerate(chunk), lambda (i,x):i-x):
            out.append( map(itemgetter(1), g) )
           
        return out    


class NFFT_Interp(object):
    def __init__(self, mask, fourD_file, output, bad_ind):
        self.mask = mask
        self.fourD_file = fourD_file
        self.output = output
        self.bad_ind = bad_ind.astype(np.int16)# cheap insurance
      
        bw_tr = BandwidthTR('human', self.fourD_file)  
        
        self.f_low = bw_tr.f_lp
        self.f_upper = bw_tr.f_ub
        self.TR = bw_tr.TR
    
    def main(self):
        
        roi, ind, shape, coord = imagetools.get_roi(self.fourD_file, self.mask)   
        
        x = np.linspace(-0.5, 0.5, roi.shape[1])   # 'continuous' time/spatial domain; -0.5<x<+0.5
        nodes = np.delete(x, self.bad_ind)
        
        M = nodes.size                   # number of nodes
        N =  M                    # number of Fourier coefficients
        
        plan = NFFT(N, M)
        plan.x = nodes
        plan.precompute()
        
        values = np.delete(roi, self.bad_ind, axis=1)
        
        output = np.zeros_like(roi)
        
        ### start loop over all time points here ###
        ### crude test ###
        
        for i in range(roi.shape[0]):
        
            plan.f = values[i,:]
        
            f_hat = plan.adjoint()
            
            half = f_hat.shape[0] / 2
            
            if np.mod(f_hat.shape[0], 2.0) != 0:
                pad = shape[3] - half - 1
                
            else:
                pad =  shape[3] - half    
            
            f_hat = 2 * f_hat[half:]
            
            # interpolation requires extra zeros
            # in this case they are places AFTER the data, not in
            # the middle
            f_hat = np.concatenate( (f_hat, np.zeros(pad)), axis=0)
            
            F = np.fft.ifft(f_hat)
            
            output[i,:] = F.real
    
        ### output ###
        sig = np.std(output, keepdims=True, axis=1)
        sig[sig == 0] = 1
        output /= sig
        
        # for varying reasons, the ts come out backwards and 
        # shifted wrt to the np fft covention.
        # this fixes it so no one hates on me        
        output = np.fliplr(output)
        output = np.fft.fftshift(output, axes=1)
        
        new = np.zeros(shape)
        new[ind] = output
        imagetools.save_new_image(new, self.output, coord)
             

class GLM(object):
    def __init__(self, root_dir, infile, mask, design):
        self.root_dir = root_dir
        self.roi.T, self.ind, shape, self.coord = imagetools.get_roi(infile, mask)
        self.glm = glm    
        self.design = design
    
        self.new = np.zeros(shape[0:3])
    
        self.residual_image = path.join(self.root_dir, 'glm_residual.nii.gz' )
     
    def glm(self):
        self.model = self.glm(self.design)
        self.model.fit(self.roi)
          
    def z_stat(self, con):    
        z = self.model.contrast(con).z_score()
        
        return z
     
    def make_residual_image(self):
        mse = self.model.contrast(np.eye(self.design.shape[1])) 
    
        self.new[self.ind] = np.sqrt(mse)
        imagetools.save_new_image(self.new, self.residual_image, self.coord)


class ICA(object):
    def __init__(self, root_dir, sub_path_list, brain_mask, output):
        '''sub_file_list is the output from PySQL get sub by gender.'''
        self.root_dir = root_dir
        self.file_list = sub_path_list
        self.output = output
        self.brain_mask = brain_mask
        
    def _append_root_dir(self):
        inputs = []
        
        for row in self.file_list:
            joined_path = path.join(self.root_dir, row[0], row[1])
            normed_path = path.normpath(joined_path)
            inputs = np.hstack((inputs, normed_path))    
            
        return inputs    
        
    def main(self, n_comp=20, fwhm=0., threshold=None):
        inputs = self._append_root_dir()
        
        canica = CanICA(n_components=n_comp, 
                        smoothing_fwhm=fwhm,
                        memory="nilearn_cache", 
                        memory_level=5,
                        threshold=threshold, 
                        verbose=0, 
                        random_state=0,
                        mask=self.brain_mask
                        )
         
        canica.fit(inputs)
 
        components_img = canica.masker_.inverse_transform(canica.components_)
    
        components_img.to_filename(self.output)    


class ReplaceWithNoise(object):
    def __init__(self, mask, fourD_file, output, bad_ind):
        self.mask = mask
        self.fourD_file = fourD_file
        self.output = output
        self.bad_ind = bad_ind
    
    def _load_data(self):
        self.roi, self.ind, self.shape, self.coord = imagetools.get_roi(self.fourD_file, self.mask) 

    def process_roi(self):
        
        beg_contig, end_contig = get_contiguous_regions(self.bad_ind) # class method
                    
        seg_starts = self.bad_ind[beg_contig]
        seg_ends = self.bad_ind[end_contig]

        n = len(seg_starts)
        
        for i in range(n):
            num = seg_ends[i] - seg_starts[i] + 1
            noise =  np.random.randn(self.roi.shape[0], num) 
            noise /= noise.std(keepdims=True, axis=1)
            self.roi[:,seg_starts[i]:seg_ends[i]+1] = noise

    def main(self):
        self._load_data()
        self.process_roi()
            
        new = np.zeros(self.shape)
        new[self.ind] = self.roi
        imagetools.save_new_image(new, self.output, self.coord)    
            

def save_roi_dict(root_dir, roi_dict):
        f = open(path.join(root_dir, 'roi_dict.json'), 'w')
        json.dump(roi_dict, f)  
        f.close()

def generate_roi_name_to_index_map(root_dir, parc_path):
        '''The dict returned here, maps the roi ascii name, with
        the label in the parc file, and the array index, of the
        roi_names array. The roi averages are computed in a loop
        based on the roi_names array order. '''
        
        # bug in np.loadtxt won't allow unpack to diff dtypes; converters doesn't work either
        labels, names = np.loadtxt(static_paths.roi_names, dtype=str, unpack=True) 
        labels = np.array(labels, dtype=int)
        
        # delete any roi names, labels that we don't want to include
        ind = np.where( labels == 0)[0]         
        names = np.delete(names, ind)   
             
        # what we really need for ease of programming, is a map of the roi string name, 
        # to the index that stores the numeric label. 
        
        # make roi string name to index map ( from roi_labels )
        roi_dict = {}
        for index, name in enumerate(names):
            roi_dict[name] = {'index':index, 'label':labels[index]}
        
        save_roi_dict(root_dir, roi_dict)

def load_roi_dict(roi_dict_path):
    f = open(roi_dict_path)
    roi_dict = json.load(f)
    f.close()
    
    return roi_dict
        
def roi_dict_index_lookup(roi_dict, index):

    for k,v  in roi_dict.iteritems():
        if v['index'] == index:
            return k
    
    return None    
        
def get_contiguous_regions(orig_bad_ind):
    '''Only works if the indicies have at least two monotonically incr values. '''
    bad_ind = np.hstack((orig_bad_ind, np.int16(1e3)))
    beg_contig = []   
    end_contig = []
    
    j = 0
    i = 0
    while j < bad_ind.size - 2: #2 b/c we added a dummy
        
        if bad_ind[i+1] == bad_ind[i] + 1:
            beg_contig.append(i)
        
            for j in range(i+1, bad_ind.size - 1):
                if bad_ind[j+1] != bad_ind[j] + 1:
                    end_contig.append(j) # plus 1 is b/c  non-inclusive
                    i = j + 1
                    break
        else:
            i += 1

    return beg_contig, end_contig
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    