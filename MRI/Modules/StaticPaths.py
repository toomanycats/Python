import ConfigParser
import logging
from os import path

log = logging.getLogger(__name__)

class StaticPaths(object):
    
    def __init__(self, Type='production'):
        if Type == 'production':
            # sets the path prefix for the location of the atlas files
            self.atlas_type = 'production_atlas'
            self.config_path = '/fs/cl10/dpc/CopyOfRepoForCluster/python/Configs/Config.cfg'
        
        elif Type == 'corpus6':
            self.atlas_type = 'corpu6_atlas'
            self.config_path = '/fs/corpus6/dpc/workspace/python/Configs/Config.cfg'
        
        elif Type == 'laptop':
            self.atlas_type = 'laptop_atlas'
            self.config_path = '/home/daniel/workspace/python/Configs/Config.cfg'
        
        else:
            raise Exception, "Config Type is not understood."        
    
        config_parser = ConfigParser.RawConfigParser()
        config_parser.read(self.config_path)
        
        # python path text file to source when working with cluster 
        self.environment_variables_for_pypeline = config_parser.get('environment_scripts', 'environment_variables_for_pypeline')
        
        # path prefix to atlas files
        self.atlas_path = config_parser.get(self.atlas_type, 'atlas_path')
        
        #####################
        #### ATLAS PATHS ####
        #####################
        
        self.roi_names        = path.join(self.atlas_path, 'atlas/roi_names.txt')
        self.parc_116_native  = path.join(self.atlas_path, 'atlas/parc116plus.nii.gz')
        self.SRI24            = path.join(self.atlas_path, 'atlas/spgr.nii.gz')
        self.rat              = path.join(self.atlas_path, 'atlas/rat_16roi.nii.gz')
        self.roi_dict         = path.join(self.atlas_path, 'atlas/roi_dict.json')
        self.sri24_4mm_brain_mask       = path.join(self.atlas_path, 'atlas/sri24_4mm_brain_mask.nii.gz')
        self.sri24_4mm_tissues       = path.join(self.atlas_path, 'atlas/sri24_4mm_tissues.nii.gz')
        self.sri24_1mm_tissues = path.join(self.atlas_path, 'atlas/sri24_1mm_tissues.nii.gz')
        self.sri24_2mm_wm_ero = path.join(self.atlas_path, 'atlas/sri24_2mm_wm_ero.nii.gz')
        self.avg152T1 = path.join(self.atlas_path, 'atlas/avg152T1.nii')
        self.avg152T1_brain = path.join(self.atlas_path, 'atlas/avg152T1_brain.nii')
        self.avg152T2 = path.join(self.atlas_path, 'atlas/avg152T2.nii')
        self.avg152T2_brain = path.join(self.atlas_path, 'atlas/avg152T2_brain.nii')
        self.rsREL = path.join(self.atlas_path, 'atlas/rsREL.nii.gz')
        
        ###########################
        #### NCANDA FILE PATHS ####
        ###########################
        
        # project root
        self.ncanda_root = '/fs/ncanda-share/pipeline/cases'
        
        # baseline paths
        self.baseline = "standard/baseline"
        
        #one year follow up
        self.one_year = "standard/followup_1y"
        
        # structural
        self.t1_native = "structural/native/t1.nii.gz"
        self.t1_native_brain = "structural/stripped/t1_brain.nii.gz"
        self.t1_native_seg = "structural/segment/t1_seg.nii.gz"
        
        self.t1_sri24 = 'structural/reslice/sri24/affine/t1.nii.gz'
        self.t1_sri24_brain = 'structural/reslice/sri24/affine/t1_brain.nii.gz'
        self.t1_sri24_seg = 'structural/reslice/sri24/affine/t1_seg.nii.gz'
        
        self.wm_core_sri24 = 'restingstate/reslice/sri24_4mm/wm_core_mask.nii.gz'
        
        # resting state scans:  bold_4d.nii.gz  bold_mean.nii.gz  bold.nii.gz  wm_core_mask.nii.gz
        self.rs_sri24_bold_full = 'restingstate/reslice/sri24_4mm/bold.nii.gz'
        self.rs_sri24_bold_stdy = 'restingstate/reslice/sri24_4mm/bold_4d.nii.gz'
        self.rs_sri24_bold_mean = 'restingstate/reslice/sri24_4mm/bold_mean.nii.gz'
        
        self.rs_native_vols = 'restingstate/native/rs-fMRI'
        
        # resting state motion parameters from mcflirt
        ### this it he "par" file. The 6 parameters straight from affine
        #### matrices, NOT the frame 2 frame computed version
        self.rs_motion_par = 'restingstate/mcflirt/bold_unwarped_mcf.par'
        
        # measures and info for group covariates
        self.clinical = 'measures/clinical.csv'
        self.demographics = 'measures/demographics.csv'
      
        # physio
        self.physio_glob = "restingstate/native/physio/*data*"
        self.physio_dir = "restingstate/native/physio"
        # mcflirt mats from coreg of 4d fmri
        self.mcflirt_mats = "restingstate/mcflirt/bold_unwarped_mcf.mat"
      
        # build a path like this
        # rs_bold = os.path.join(self.ncanda_root, self.baseline %<sub_id>, self.rs_sri24_bold_stdy)
        
        ### NCANDA Free Surfer Outputs
        self.aparc_aseg = 'freesurfer/%(sub_id)s_standard_baseline/mri/aparc+aseg.mgz'
        self.rawavg = 'freesurfer/%(sub_id)s_standard_baseline/mri/rawavg.mgz'
        
        
    def set_ncanda_full_paths(self, sub_id, time_point):
            
        time_pt = {'base':self.baseline,
                   'one_year':self.one_year
                  }
        # directories
        self.base_dir = path.join(self.ncanda_root, sub_id, time_pt[time_point])
        
        # files and images  
        self.rs_native_vols = path.join(self.base_dir, self.rs_native_vols)     
        self.rs_sri24_bold_stdy = path.join(self.base_dir, self.rs_sri24_bold_stdy)
        self.rs_sri24_bold_mean = path.join(self.base_dir, self.rs_sri24_bold_mean)
        
        self.t1_native = path.join(self.base_dir, self.t1_native)
        self.t1_native_brain = path.join(self.base_dir, self.t1_native_brain)
        self.t1_native_seg = path.join(self.base_dir, self.t1_native_seg)
        
        self.t1_sri24 = path.join(self.base_dir, self.t1_sri24)
        self.t1_sri24_brain = path.join(self.base_dir, self.t1_sri24_brain)        
        self.t1_sri24_seg = path.join(self.base_dir, self.t1_sri24_seg)
        
        self.mcflirt_mats = path.join(self.base_dir, self.mcflirt_mats)
        self.rs_motion_par = path.join(self.base_dir, self.rs_motion_par)
        self.wm_core_sri24 = path.join(self.base_dir, self.wm_core_sri24)
        
        self.clinical = path.join(self.base_dir, self.clinical)
        self.demographics = path.join(self.base_dir, self.demographics)
        self.physio_dir = path.join(self.base_dir, self.physio_dir )    

        self.aparc_aseg = path.join(self.base_dir, self.aparc_aseg %{'sub_id':sub_id})
        self.rawavg = path.join(self.base_dir, self.rawavg %{'sub_id':sub_id})
       
        self.physio_glob = path.join(self.base_dir, self.physio_glob %{'sub_id':sub_id})
        self.physio_dir = path.join(self.base_dir, self.physio_dir %{'sub_id':sub_id})












