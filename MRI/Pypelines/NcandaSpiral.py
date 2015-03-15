from os import path, rename, makedirs
from shutil import copy
from MiscTools import MiscTools
from ImageTools import ImageTools
from SignalProcessTools import SignalProcessTools
import PhysioCorrect
import PyXnatTools
import PyConn
import logging

misctools = MiscTools()
imagetools = ImageTools()
sigtools = SignalProcessTools()
pyx = PyXnatTools.PyXnatTools()

log = logging.getLogger(__name__)


# sub_label = 'B-00100-F-3-20140410'
# sub_id = pyx.get_subject_ids_like(sub_label[0:11])

sandbox_path = '/fs/ncanda-share/spiral'

#########################
### Structural  Paths ###
#########################
struct_base = '/fs/ncanda-share/pipeline/cases' 
structural = 'standard/baseline/structural/reslice/subject_t1/t2.nii.gz'

##########################
### XNAT paths to data ###
##########################
archive_base = '/fs/ncanda-xnat/archive/sri_incoming/arc001' 
log.debug("Base path:%s" %archive_base)

resources = 'RESOURCES/spiral/Spiral.tar.gz'
log.debug("Resources Path:%s" %resources)

####################
### AUX varibles ###
####################
num_vol_drop = 0

class SpiralTask(object):
    def __init__(self, sub_label):
        self.sub_label = sub_label
        self.sub_id = pyx.get_subject_ids_like(sub_label[0:11])
        
        self.root_dir = path.join(sandbox_path, sub_label)
        self.log_path = path.join(self.root_dir, 'logfile.txt') 
        
        self.inputs_dir = path.join(self.root_dir, 'Inputs')
        self.inter_dir = path.join(self.root_dir, 'Intermediates')
        self.final_dir = path.join(self.root_dir, 'Final')
        self.struct_dir = path.join(self.root_dir, 'Structural')

        ######################################
        ### Image file paths for all steps ###
        ######################################
        self.input_physio_dir = self.inputs_dir
        
        ## intermediate images ##
        self.nifti_file = path.join(self.inter_dir, 'time_series.nii.gz')
        self.phy_denoised_out = path.join(self.inter_dir, self.nifti_file.replace('.nii.gz','_phy.nii.gz'))  
        self.slice_time_out = self.phy_denoised_out.replace('.nii.gz','_slctm.nii.gz')
        self.nifti_file_coreg = self.slice_time_out.replace('.nii.gz', '_coreg.nii.gz')
        self.nifti_file_drift_corr = self.nifti_file_coreg.replace('.nii.gz','_drift.nii.gz')
        self.mean = path.join(self.inter_dir, 'ts_mean.nii.gz')
        self.nifti_file_drift_corr = self.nifti_file_coreg.replace('.nii.gz','_drift.nii.gz')
        self.confound_cleaned_output = path.join(self.final_dir, 'motion_cleaned.nii.gz')

        ## structual images ##
        self.mask = path.join(self.struct_dir, 'mask.nii.gz')
        
        self.wm = path.join(self.struct_dir, 'wm_t1.nii.gz')
        self.wm_erode = self.wm.replace('.nii.gz', '_eroded.nii.gz')
        self.wm_erode_mean = path.join(self.struct_dir, 'wm_eroded_mean.nii.gz')
        
        self.gm = path.join(self.struct_dir, 'gm_t1.nii.gz')
        self.gm_mean = self.gm.replace('.nii.gz', '_mean.nii.gz')
        
        self.t2_t1 = path.join(struct_base, self.sub_id, structural)
        log.info("T2 in T1 space Path:%s" %self.t2_t1)
        
        self.tissue_map_t1 = path.join(struct_base, self.sub_id, 'standard/baseline/structural/segment/t1_seg.nii.gz')
        log.info("Tissue map path:%s" %self.tissue_map_t1)

        ## coreg affine matrix output dirs ##
        self.mat_file = self.nifti_file_coreg.replace('.nii.gz', '.mat')
        self.t2_t1_to_mean_xform = path.join(self.struct_dir, 't2_t1_to_mean.xform')

        log.info("Sub label:%s" %sub_label)
        
        log.info("Sub ID:%s" %self.sub_id)

    def eva_truncated_version(self):
        self.make_dirs() 
        self.untar_and_copy_over()
        self.set_E_file()
        self.change_name_of_physio_file() 
        self.make_nifti()
        self.physio_correct()
        self.convert_to_analyse()
        # copy the structural images over
        base = path.join(struct_base, self.sub_id, 'standard/baseline/structural/native')
        copy(path.join(base, 't1.nii.gz'), self.struct_dir)
        copy(path.join(base, 't2.nii.gz'), self.struct_dir)

        base = path.join(struct_base, self.sub_id, 'standard/baseline/structural/segment')
        copy(path.join(base, 't1_seg.nii.gz'), self.struct_dir)

    def convert_to_analyse(self):
        num_vols = imagetools.get_number_of_vols(self.phy_denoised_out)
        split_dir = path.join(self.inter_dir, 'NiftiSplit')
        
        if not path.exists(split_dir):
            makedirs(split_dir)
             
        split_output_base = path.join(split_dir ,'nifti_split_')
        imagetools.split_4D_to_vols(self.phy_denoised_out, split_output_base, log_this=True)
        
        files = misctools.glob_for_files(split_dir, "*.nii.gz", num_vols, log_this=True)
        if files is None:
            log.error("No split nifti files were found.")
        
        n = 0
        for f in files:
            outfile = path.join(self.final_dir, 'physio_corrected_%03i.hdr' %n)
            imagetools.convert_image_format(f, outfile)
            n += 1
    
    def make_dirs(self):
        for directory in self.inputs_dir, self.inter_dir, self.struct_dir, self.final_dir:
            if not path.exists(directory):
                makedirs(directory)

    def motion_detrend(self):
    
        ## detrend outputs ##
        detrend = PyConn.DetrendMovementNoise(self.inter_dir, self.nifti_file_drift_corr, self.confound_cleaned_output, 
                                              self.mask, Type='human', wm_rs=self.wm_erode_mean, move_par_type='mcflirt')
        cleaned = detrend.clean_rs_out_path
        self.cleaned_sm = cleaned.replace('.nii.gz', '_sm.nii.gz')

        log.info("Remove the motion and other confounds. Output:%s" %self.confound_cleaned_output)
        detrend.main()  

    def untar_and_copy_over(self):
        tarfile = path.join(archive_base, self.sub_label, resources)
        misctools.untar_to_dir(tarfile, self.inputs_dir)

    def change_name_of_physio_file(self):
        physio = imagetools.glob_for_files(self.inputs_dir, pattern='P*.physio', num_limit=1)
        if physio is None:
            msg = "Physio files not found."
            log.exception(msg)
            raise Exception, msg
        
        new_physio_name = path.join(self.inputs_dir, 'Spiral.physio')
        rename(physio, new_physio_name)
        log.info("Change name of physio file from P*.physio to :%s" %new_physio_name)

    def set_E_file(self):
        self.spiral_file = misctools.glob_for_files(self.inputs_dir, pattern='E*.7', num_limit=1, log_this=True)

    def make_nifti(self):
        imagetools.make_nifti_from_spiral(self.spiral_file, self.nifti_file)
        imagetools.fix_slice_dur(self.nifti_file)

    def physio_correct(self):  
        log.info("Physio Correct:%s" %self.phy_denoised_out)      
        physio_corr = PhysioCorrect.Main(self.input_physio_dir, num_vol_drop, self.nifti_file, 
                                         self.inter_dir, self.phy_denoised_out)
        physio_corr.main()

    def slice_timer(self):
        TR = imagetools.get_pixdims(self.nifti_file)[4]
        imagetools.slice_timing_correct(self.phy_denoised_out, self.slice_time_out, TR=TR, order='seq')

    def mcflirt(self):
        imagetools.mcflirt(self.slice_time_out, self.nifti_file_coreg, options='mean', mats=True)

    def main(self):
        self.make_dirs()
        self.untar_and_copy_over()
        self.set_E_file()
        self.change_name_of_physio_file() 
        self.make_nifti()
        self.physio_correct()

        #############################
        ### make mean time series ###
        ############################# 
        imagetools.make_mean_image_from_4D(self.nifti_file_coreg, self.mean)
         
        ####################
        ### BET the mean ###
        ####################
        imagetools.bet(self.mean, self.mean, options='-R')
         
        ####################
        ### erode the wm ###
        ####################
        imagetools.get_single_tissue_from_seg(self.tissue_map_t1, 'wm', self.wm)
        imagetools.erode_image_box(self.wm, self.wm_erode, num_vox=5, dim=3)
         
        ###############################
        ### coreg the T2_t1 to mean ###
        ###############################
        log.info("Coreg the T2 in T1 space to the time series mean. Xform path:%s" %self.t2_t1_to_mean_xform)
        imagetools.coregister_t2_to_fmri_mean(static=self.mean, moving=self.t2_t1, xform=self.t2_t1_to_mean_xform)
         
        ##############################################
        ### down sample wm erode  to ts mean space ###
        ##############################################
        imagetools.reformatx(self.wm_erode_mean, moving=self.wm_erode, reference=self.mean, xform=self.t2_t1_to_mean_xform, interp='pv')
         
        ### ####################################
        ### get gm mask for testing purposes ###
        ########################################
        imagetools.get_single_tissue_from_seg(self.tissue_map_t1, 'gm', self.gm)
        imagetools.reformatx(self.gm_mean, moving=self.gm, reference=self.mean, xform=self.t2_t1_to_mean_xform, interp='pv')
        imagetools.binarize(self.gm_mean, self.gm_mean)
         
        #############################
        ### make time series mask ###
        #############################
        imagetools.make_mask_from_non_zero(self.mean, self.mask)
         
        ########################
        ### drift correction ###
        ########################
        sigtools.remove_global_drift(self.nifti_file_coreg, self.mask, self.nifti_file_drift_corr, seg_len=51, deg=8)
         
        ############################
        ### mask the time series ###
        ############################
        imagetools.apply_mask(self.nifti_file_drift_corr, self.mask, self.nifti_file_drift_corr)
         
        ######################
        ### Detrend Python ###
        ######################
        self.motion_detrend()
        
        ######################
        ### spatial smooth ###
        ######################
        imagetools.spatial_smooth_mask(self.cleaned, self.cleaned_sm, self.mask, fwhm=4.0)
         
        ########################
        ### voxwise analysis ###
        ########################
        vox = PyConn.VoxelWiseAnalysis(self.root_dir, self.cleaned_sm, self.mask)
        vox.compute_degree() 