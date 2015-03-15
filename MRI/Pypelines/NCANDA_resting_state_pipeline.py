#!/usr/bin/python

from optparse import OptionParser
from os import path, makedirs, rename, remove
import numpy as np
import shutil 
import PyConn
import ImageTools
import SignalProcessTools
import MiscTools
import StaticPaths
import PhysioCorrect
import LoggingTools
from nitime.utils import percent_change
import traceback
from scipy.signal import detrend, welch
import GroupAnalysisTools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import GroupAnalysisTools
import PyXnatTools

static_paths = StaticPaths.StaticPaths()
imagetools = ImageTools.ImageTools()
sigtools = SignalProcessTools.SignalProcessTools()
misctools = MiscTools.MiscTools()
pyx = PyXnatTools.PyXnatTools()

    
class ProcessNCANDAResting(object):
    def __init__(self, sub_id, root_dir, physio_correct, time_point):
        self.sub_id = sub_id
        self.root_dir =  path.join(root_dir, sub_id)
        self.time_point = time_point
        self.physio_correct = physio_correct

        self.static_paths = static_paths
        self.static_paths.set_ncanda_full_paths(self.sub_id, self.time_point)      
 
        self.group_cov_path_txt = path.join(self.root_dir, 'group_covariate_' + self.sub_id + '.txt') 
        
        group_cov = GroupAnalysisTools.LookupCovariates(self.sub_id)
        self.scanner = group_cov.get_scanner_cov()

        #inputs
        self.bold = path.join(self.root_dir, 'bold_mcf_unw.nii.gz')
        self.bold_mean = path.join(self.root_dir, 'bold_mean.nii.gz')
        self.brain_mask = path.join(self.root_dir, 'brain_mask.nii.gz')
        self.brain_mask_pre_sm = self.brain_mask.replace('.nii.gz', '_pre_sm.nii.gz')
        #output
        
        # intermediates
        self.t1_seg = path.join(self.root_dir, 't1_seg_bold_1mm.nii.gz')# registered to bold mean in 1mm iso
        self.wm = path.join(self.root_dir, 'wm.nii.gz')
        self.gm = path.join(self.root_dir, 'gm.nii.gz')
        self.csf = path.join(self.root_dir, 'csf.nii.gz')
        self.T1_2_bold_xform = path.join(self.root_dir, 'T1_2_bold_mean.affine')
        
        self.parc_tissues = path.join(self.root_dir, 'parc_tissues.nii.gz')
        
    def set_up_log(self):

        log_path = path.join(self.root_dir, 'log.txt')
        self.log = LoggingTools.SetupLogger(log_path, 'info').get_module_logger()
        
        if pyx: 
            try:
                site, label = pyx.get_site_from_sub_ID(self.sub_id)
                self.log.info('Subject ID: %s' %self.sub_id)
                self.log.info('Subject Label: %s' %label)
                self.log.info('Site:%s' %site)    
           
            except:
                self.log.info("XNAT query bombed for some reason. Didn't get site decoded and subject label.")
        
        self.log.info('Subject ID:%s' %self.sub_id)
            
    def setup_dirs(self):
        if not path.exists(self.root_dir):
            makedirs(self.root_dir)      
                    
    def call_shell_script_to_make_subspace_unwarped_aligned(self):
        cmd = "resting_state_unwarp_mcflirt.sh %(root_dir)s %(sub_id)s"
        cmd = cmd %{'root_dir':self.root_dir,
                    'sub_id':self.sub_id
                    }
        
        output = imagetools.call_shell_program(cmd)
        self.log.info(output)

    def fix_bold_header_for_physio_corr(self, infile):    
        ### ORDER MATTERS ###    
        
        #fix intent code
        image = imagetools.load_image(infile)
        hdr = image.header.copy()
        hdr['intent_code'] = 0
        imagetools.save_new_image_clone(image.get_data(), hdr, self.bold, coordmap=image.coordmap)
        
        # fix pixdim4 = TR
        imagetools.fix_pixdim4(infile, TR=2.2)
        
        #fix sform/qform code
        self.fix_form_code(infile)
        
        #fix slice duration
        imagetools.fix_slice_dur(infile)
        
    def fix_form_code(self, infile):
        q_code, s_code = imagetools.get_qs_form_code(infile)
        if q_code == 1 and s_code == 2: # no need to fix
            return
        
        cmd = "fslorient -setsformcode 2 %s && fslorient -setqformcode 1 %s" 
        cmd = cmd %(infile, infile)
        imagetools.call_shell_program(cmd)

    def _drop_frames(self, infile):
        outfile = infile.replace('.nii.gz', '_drp.nii.gz')                  
        
        num_vols = imagetools.get_dim(self.bold)[4]
        num_drop = num_vols - self.num_vols_ref
        
        if num_drop < 0:
            raise ValueError, "num drop < 0."
        
        imagetools.fsl_roi(infile, outfile, front_drop=num_drop, end_drop=0)
    
        return outfile
    
    def _slice_timing(self, infile):
        outfile = infile.replace('.nii.gz', '_tmg.nii.gz')
        
        # all EPI in this pipeline are interleaved.
        # GE is 'even': 2,4,6,8,...1,3,5,7,...
        # Siemens is 'odd' when num slices in Z is odd: 1,3,5,7...2,4,6,8...
        # Siemens is 'even' when num slice in Z is even
          
        if self.scanner['scanner_GE'] == 1:
            order = 'even'
        
        elif self.scanner['scanner_SIEMENS'] ==1 :
            num_z = imagetools.get_dim(infile)[2]
            
            if num_z % 2 == 0: # even num Z vox
                order = 'even'
            
            else:    
                order = 'odd'
        
        else:
            raise Exception, "Scanner name not recognized:%s" %self.scanner        
            
        imagetools.slice_timing_correct(infile, outfile, order)
        
        return outfile
             
    def _polyfit(self, infile):
        outfile = infile.replace('.nii.gz','_hi.nii.gz')
        
        self.fix_form_code(infile)# get roi will not mask correctly if s form code != 2
        
        # lowest freq in band of interest: 0.009
        # T = 111 s, which is 50.5 frames
        sigtools.remove_global_drift(infile, self.brain_mask, outfile, seg_len = 51) 
        self.fix_bold_header_for_physio_corr(outfile)
        
        return outfile
    
    def _physio_correct(self, infile):
        physio_files = imagetools.find_files(self.static_paths.physio_dir, 
                                             '*data*', 
                                             Type='f', 
                                             maxdepth=1
                                             )
        
        for f in physio_files:
            dst = path.join(self.root_dir, path.basename(f))
            shutil.copyfile(f, dst)
        
        num_vols_drop = imagetools.get_dim(infile)[4] - self.num_vols_ref
        denoised_out = infile.replace('.nii.gz', '_phy.nii.gz')
        # physio_dir, num_vols_drop, rs_path, output_dir, phy_denoised_out
        pc = PhysioCorrect.Main(self.root_dir, num_vols_drop, infile, self.root_dir, denoised_out)
        pc.main()
        outfile = pc.denoised_out
        
        return outfile
    
    def _despike(self, infile):
        outfile = infile.replace('.nii.gz','_dspk.nii.gz')   
                
        imagetools.afni_despike(infile, outfile, new=True)   
        self.fix_form_code(outfile)
        
        return outfile
    
    def _detrend(self, infile):
        outfile = infile.replace('.nii.gz', '_reg.nii.gz')
        detrend_mv = PyConn.DetrendMovementNoise(root_dir=self.root_dir, 
                                                 rs_4D_path=infile, 
                                                 confound_cleaned_output=outfile, 
                                                 mask=self.brain_mask, 
                                                 sub_id=self.sub_id,
                                                 time_point='base', 
                                                 Type='human', 
                                                 wm_rs=self.wm, 
                                                 aux_tissue=self.parc_tissues,
                                                 csf_rs=None,
                                                 move_par_type='mcflirt'
                                                 )
        
        detrend_mv.global_mean = False
        detrend_mv.use_abs_motion_con = True 
        detrend_mv.der_tissue_confounds_only = True
        
        detrend_mv.num_vols_ref = self.num_vols_ref # cheap insurance
        detrend_mv.main()
        outfile = detrend_mv.clean_rs_out_path
        
        self.motion_params_file = detrend_mv.motion_f2f_parameters_file
        
        return outfile
    
    def _motion_analysis(self, infile):
        motion = PyConn.MotionAnalysis(self.root_dir, infile, self.brain_mask, self.gm, self.motion_params_file)
        motion.motion_thres = self.motion_thres
        self.rem_ind, self.traj, self.motion_test = motion.main_power_censor()
        
        self.rem_file = path.join(self.root_dir, 'removed_ind.txt')
        self.motion_traj_file = path.join(self.root_dir, 'motion_traj.txt')
        
        np.savetxt(self.motion_traj_file, self.traj, fmt='%f')
        
        # if no frames are bad, rem_file is None
        if self.rem_ind is None:
            f = open(self.rem_file, 'w')
            f.write('None')
            f.close()
            
        else:
            np.savetxt(self.rem_file, self.rem_ind, fmt='%i')

        # spoof an outfile   
        return infile
       
    def _NFFT(self, infile):
        if self.rem_ind is None:
            self.log.info("There were no bad motion frames for this subject, NFFT skipped.")
            return infile # spoof an outfile for the next process
                     
        outfile = infile.replace('.nii.gz', '_NFFT.nii.gz')
        
        nfft = PyConn.NFFT_Interp(self.brain_mask, infile, outfile, self.rem_ind)
        nfft.main()
        
        return outfile
    
    def _spatial_smooth(self, infile):
        outfile = infile.replace('.nii.gz','_sm.nii.gz')
     
        imagetools.spatial_smooth(infile, outfile, fwhm=6)
        ### Smooth the mask ! ###
        # copy the original
        shutil.copyfile(self.brain_mask, self.brain_mask_pre_sm)
        imagetools.make_mask_for_smoothed_time_series(outfile, self.brain_mask)
        
        self.fix_form_code(outfile)    
    
        return outfile
    
    def _percent_signal(self, infile):
        outfile = infile.replace('.nii.gz', '_psc.nii.gz')
        # the signal has been centered and z scored
        # of the detrending process
        
        #first un-center the signal
        roi, ind, shape, coord = imagetools.get_roi(infile, self.brain_mask)
        roi -= roi.min(axis=1, keepdims=True)
        roi = percent_change(roi, 1)
        new = np.zeros(shape, dtype=np.float32)
        new[ind] = roi
        imagetools.save_new_image(new, outfile, coord, log_this=True)
        
        return outfile
    
    def pre_process(self, infile):  
        
        try:               
            for step in self.processing_steps:
                infile = self.steps_dict[step](infile)
        
            return infile
          
        except Exception:
            msg = traceback.print_exc()
            self.log.error(msg)
            raise Exception
            
    def get_design_matrix_rsREL(self, infile):
        mean_conn = PyConn.MakeMeanTimeSeries(self.root_dir, infile)
        mean_conn.load_data()
        out = mean_conn.get_average_from_rsREL()
        
        return out.T # design matrices are col vectors

    def _convolution_filter_lowpass(self, infile):
        window=10
        
        outfile = infile.replace('.nii.gz', '_lopss.nii.gz')
        
        roi, ind, shape, coord = imagetools.get_roi(infile, self.brain_mask)
        
        num = roi.shape[1] + window - 1
        out = np.zeros((roi.shape[0], num), dtype=np.float32)
 
        for i in range(roi.shape[0]):
            out[i,:] = sigtools.smooth(roi[i,:], window)

        new = np.zeros((shape[0], shape[1], shape[2], out.shape[1]),dtype=np.float32)
        new[ind] = out
        
        imagetools.save_new_image(new, outfile, coord)
        
        return outfile

    def _bandpass(self, infile):
        outfile = infile.replace('.nii.gz', '_bp.nii.gz')
        
        roi, ind, shape, coord = imagetools.get_roi(infile, self.brain_mask)
        
        out = sigtools.butter_bandpass(0.009, 0.09, 1/2.2, roi)
        
        new = np.zeros(shape)
        new[ind] = out
        
        imagetools.save_new_image(new, outfile, coord)
        
        return outfile

    def _replace_noise(self, infile):
        if self.rem_ind is None:
            self.log.info("There were no bad motion frames for this subject, NFFT skipped.")
            return infile # spoof an outfile for the next process
        
        outfile = infile.replace('.nii.gz', '_noise.nii.gz')
        noise = PyConn.ReplaceWithNoise( self.brain_mask, infile, outfile, self.rem_ind)
        noise.main()
        
        return outfile

    def _reduce_sub_root_dir_to_proj_root(self, root_dir):
        root_dir = self.root_dir.split('/')[1:6]
        root_dir = path.join('/', *root_dir)
     
        return root_dir

    def prep_tissue_mask(self, infile, reference, tissue_type, erode=0):
        temp = imagetools.mk_temp_file_name(suffix='.nii.gz')
        
        try:
            imagetools.get_single_tissue_from_seg(self.t1_seg, tissue_type,  temp)
            
            if erode > 0:
                imagetools.erode_image_box(temp, temp, num_vox=erode, dim=3)
                infile = infile.replace('.nii.gz', '_ero.nii.gz')
            
            imagetools.resample_tissue_mask_to_ref(infile=temp, 
                                                   ref=reference, 
                                                   outfile=infile
                                                   ) 
                        
        except Exception:
            msg = traceback.print_exc()
            raise Exception, msg
            
        finally:
            if path.exists(temp):
                remove(temp)            

    def prep_tissue_masks(self):
        temp = imagetools.mk_temp_file_name(suffix='.nii.gz')
        
        self.prep_tissue_mask(self.gm, self.bold_mean, tissue_type='gm', erode=1)
        self.prep_tissue_mask(self.wm, self.bold_mean, tissue_type='wm', erode=5)
        self.prep_tissue_mask(self.csf, self.bold_mean, tissue_type='csf', erode=2)
        
        try:
            
            # root_dir, sub_id, time_point, t1_seg_bold_1mm, T1_2_bold_1mm_affine
            fs = FreeSurferROIs(self.root_dir, self.sub_id, self.time_point, self.t1_seg, self.T1_2_bold_xform)
            fs.make_parc_rois(self.bold_mean, self.parc_tissues) 
            
            
        except Exception:
            msg = traceback.print_exc()
            raise Exception, msg
            
        finally:
            if path.exists(temp):
                remove(temp)
       
    def coregister_seg_to_bold_1mm(self):
        
        # make 1mm bold mean ref
        # reg stripped T1 to bold mean native
        # reformat using 1mm bold ref
        
        bold_mean_1mm = imagetools.mk_temp_file_name(suffix='.nii.gz')
    
        try:
            cmd = 'convertx --resample 1 %s %s' %(self.bold_mean, bold_mean_1mm)
            imagetools.call_shell_program(cmd)
            
            # t1_native_brain is stripped by voting
            imagetools.coregister_struct_to_bold_mean(self.bold_mean, self.static_paths.t1_native_brain, self.T1_2_bold_xform, Type='T1')
            
            cmd = 'reformatx --pv --pad-out 0 --ushort -o %(outfile)s --floating %(floating)s %(ref)s %(xform)s '
            cmd = cmd %{'outfile':self.t1_seg,
                        'floating':self.static_paths.t1_native_seg,
                        'ref':bold_mean_1mm,
                        'xform':self.T1_2_bold_xform
                        }
            
            imagetools.call_shell_program(cmd)
        
        except Exception, err:
            raise Exception, err
        
        finally:    
            remove(bold_mean_1mm)
            
    def turn_seg_into_mask(self):
        cmd = "reformatx -o %(outfile)s --floating %(float)s %(target)s"
        cmd = cmd %{'outfile':self.brain_mask,
                    'float':self.t1_seg,
                    'target':self.brain_mask
                    }
        
        imagetools.call_shell_program(cmd)
        
        imagetools.binarize(self.brain_mask, self.brain_mask)


class ProcessNCANDARestingSub(ProcessNCANDAResting):
    def __init__(self, sub_id, root_dir, physio, time_point):
        ProcessNCANDAResting.__init__(self, sub_id, root_dir, physio, time_point)
        self.time_point = time_point
             
    def single_sub_setup(self):
             
        self.setup_dirs()
        self.set_up_log()
        
#         self.call_shell_script_to_make_subspace_unwarped_aligned()
        
        imagetools.make_mean_image_from_4D(self.bold, self.bold_mean)
        
        imagetools.bet(self.bold_mean, self.bold_mean, self.brain_mask, options='-f 0.3 -R -m')
        
        self.fix_form_code(self.brain_mask)
        
        self.coregister_seg_to_bold_1mm()
        
        self.turn_seg_into_mask()
        
        imagetools.apply_mask(self.bold, self.brain_mask, self.bold)
        self.fix_bold_header_for_physio_corr(self.bold)    
        
        imagetools.apply_mask(self.bold_mean, self.brain_mask, self.bold_mean)
            

class ProcessNCANDARestingDpc(ProcessNCANDARestingSub):
    def __init__(self, sub_id, root_dir, physio, time_point):
        ProcessNCANDARestingSub.__init__(self, sub_id, root_dir, physio)
        self.time_point = time_point
        self.motion_thres = 0.35
        self.num_vols_ref = 260    
        
        self.all_switch = False        
#         if aux_infile is not None:
#             self.aux_infile = path.join(self.root_dir, aux_infile)
#         
        # processing flow control
        self.steps_dict = {'drop':self._drop_frames,
                           'polyfit':self._polyfit,
                           'physio':self._physio_correct,
                           'despike':self._despike,
                           'detrend':self._detrend,
                           'motion_analysis':self._motion_analysis,
                           'NFFT':self._NFFT,
                           'smooth':self._spatial_smooth,
                           'psc':self._percent_signal,
                           'bandpass':self._bandpass,
                           'noise':self._replace_noise,
                           'slice_timing':self._slice_timing
                           }   
    
    def main(self):
        self.set_up_log()

        # call the base class main
        self.single_sub_setup()
        
        self.prep_tissue_masks()
    
        self.motion_params_file = imagetools.glob_for_files(self.root_dir, 'motion_f2f_params.txt', num_limit=1)
        # where the self.steps are run
        self.outfile = self.pre_process(self.bold)
        
        root_dir = self._reduce_sub_root_dir_to_proj_root(self.root_dir)
            
        ### QC metrics
        bold_bf = path.join(self.root_dir, 'bold_mcf_unw_dspk.nii.gz') # needs 15 frames dropped
        bold_af = path.join(self.root_dir, 'bold_mcf_unw_dspk_reg_drp.nii.gz')
        
        qc = QcMetrics(self.root_dir, 
                       self.sub_id, 
                       bold_bf=bold_bf, 
                       bold_af=bold_af, 
                       mask=self.gm, 
                       traj_file=self.motion_traj_file, 
                       rem_ind_file=self.rem_file,
                       motion_thres=self.motion_thres,
                       motion_test=self.motion_test
                       )
        qc.main()
          
        ### deg map post procssing for sanity checks ###
        vox = ProcessVox(self.sub_id, root_dir, self.outfile, self.brain_mask, global_corr=0.75)
        vox.main()


class ConnSingleSubject(ProcessNCANDARestingSub):
    def __init__(self, sub_id, root_dir, physio, time_point):
        ProcessNCANDARestingSub.__init__(self, sub_id, root_dir, physio, time_point)
        self.time_point = time_point
        
        self.motion_file = path.join(self.root_dir, 'motion_params.par')
        self.program_name = path.join(self.root_dir, 'single_sub_conn.m')
        self.cov_csv_path = path.join(self.root_dir, 'cov_file.csv')
       
    def generate_calling_script(self):
        # %% is the escape sequence for using a literal '%'
        script = '''addpath('/fs/p00/spm8');
%%addpath('/fs/p00/conn13o');
%% we use the modifed conn TB b/c the function, conn_setup_wizard for v 13.o
%% has a bug, and it's corrected in my modified copy in my repo.
addpath('/fs/cl10/dpc/CopyOfRepoForCluster/matlab/MATLAB/conn');

%% This global var is for removing sub that failed the preprocessing
%% by setting it to 1 ( True ) , we say telling conn to not remove any
%% failures. It's not really applicable here so much as it was for the total
%% processing script for the NCANDA 500, but we must set it up to use the modified
%% conn TB version

global remove_failures_done;
remove_failures_done = 1;

T1 = '%(T1_seg)s';
bold = '%(bold)s';
GM = '%(GM)s';
WM = '%(WM)s';
CSF = '%(CSF)s';
motion_file = '%(motion_file)s';

single_sub_connectivity_only(bold, T1, GM, WM, CSF, motion_file)
exit
'''
        script = script %{'T1_seg':self.t1_seg,
                          'bold':self.bold,
                          'GM':self.gm,
                          'WM':self.wm,
                          'CSF':self.csf,
                          'motion_file':self.motion_file,
                          'cov_file':self.cov_csv_path
                          }
        
        return script
        
    def main(self):
        self.setup_dirs()
        self.set_up_log()
        
        self.single_sub_setup()
                
        imagetools.make_mean_image_from_4D(self.bold, self.bold_mean)
        
        imagetools.bet(self.bold_mean, self.bold_mean, self.brain_mask, options='-f 0.3 -R -m')
        
        self.fix_form_code(self.brain_mask)
        
        self.coregister_seg_to_bold_1mm()
        
        self.turn_seg_into_mask()
        
        imagetools.apply_mask(self.bold, self.brain_mask, self.bold)
        #self.fix_bold_header_for_physio_corr(self.bold)    
        
        imagetools.apply_mask(self.bold_mean, self.brain_mask, self.bold_mean)
        
        # do not erode, Connt TB will erode
        imagetools.get_single_tissue_from_seg(self.t1_seg, 'gm',  self.gm)
        imagetools.get_single_tissue_from_seg(self.t1_seg, 'wm',  self.wm)
        imagetools.get_single_tissue_from_seg(self.t1_seg, 'csf',  self.csf)
        
        #self.remove_unused_files()
        
        # Conn batch mode can't seem to handle the smoothing
        imagetools.spatial_smooth(self.bold, self.bold, fwhm=6)
        
        self.make_cov_csv_file()

        # spm doesn't open gzip
        self.ungzip_all_inputs()
        
        # conn can't parse the ext, .par
        rename(self.motion_file, self.motion_file.replace('.par','.txt'))
        self. motion_file = self.motion_file.replace('.par','.txt')

        self.run_matlab_calling_script()
     
    def run_matlab_calling_script(self):
         
        calling_script = self.generate_calling_script()
        imagetools.write_shell_script(self.program_name, calling_script)
        
        output = imagetools.call_matlab_program(self.program_name)        
        self.log.info(output)

    def remove_unused_files(self):
        try:
            shutil.rmtree(path.join(self.root_dir, 'motion_params.mat'))
            remove(path.join(self.root_dir, 'T1_2_bold_mean.affine'))
            remove(path.join(self.root_dir, 't1_seg_bold_1mm.nii.gz'))
        except:
            pass    
        
    def make_cov_csv_file(self):
        cov = GroupAnalysisTools.NcandaGroupCov(self.sub_id, time_pt='base')
        covariates = cov.get_cov()# returns a dict
        
        writer = GroupAnalysisTools.WriteCovCsv(self.cov_csv_path)
        header_keys = sorted(covariates.keys())
        writer.get_writer(header_keys)
        writer.write_header()
        writer.write_row_to_csv(covariates)
        writer.close_file_object()

    def ungzip_all_inputs(self):
        _, self.bold = imagetools.check_if_gzipped(self.bold, unzip_if_true=True)
        _, self.t1_seg = imagetools.check_if_gzipped(self.t1_seg, unzip_if_true=True)
        _, self.gm = imagetools.check_if_gzipped(self.gm, unzip_if_true=True)
        _, self.wm = imagetools.check_if_gzipped(self.wm, unzip_if_true=True)
        _, self.csf = imagetools.check_if_gzipped(self.csf, unzip_if_true=True)


class ProcessVox(ProcessNCANDAResting):
    def __init__(self, sub_id, root_dir, bold, mask, global_corr):
        ProcessNCANDAResting.__init__(self, sub_id, root_dir)
        self.bold = bold
        self.mask = mask
        self.global_corr = global_corr
        
    def main(self):
        self.set_up_log()        
        self.vox_analysis()
        
    def vox_analysis(self):   
        self.log.info('Making degree map.')
        self.vox_methods = PyConn.VoxelWiseAnalysis(self.root_dir, self.bold, self.mask) 
        self.vox_methods.global_corr_thres = self.global_corr
        
        self.vox_methods.compute_degree()

    def make_seed_img_frm_highest_deg_map_vox_ijk(self):
        deg_data = imagetools.load_image(self.deg_map_smoothed).get_data()
        
        max_inds = np.nonzero(deg_data == deg_data.max())
        
        self.vox_methods._check_roi_mask_parameters() # sets up the self.roi, self.ind_i, etc
        self.vox_methods.compute_deg_cont_thres()
        self.vox_methods.seed_analysis_from_ijk(*max_inds)
        
             
class ProcessRoi(object):
    def __init__(self):
        pass
        

class FreeSurferROIs(object):
    
    def __init__(self, root_dir, sub_id, time_point, t1_seg_bold_1mm, T1_2_bold_1mm_affine):
        self.root_dir = root_dir
        self.sub_id = sub_id
        self.time_point = time_point

        self.static_paths = static_paths
        self.static_paths.set_ncanda_full_paths(self.sub_id, self.time_point)        
        
        self.t1_seg_bold_1mm = t1_seg_bold_1mm
        self.T1_2_bold_1mm_affine = T1_2_bold_1mm_affine
        self.aparc_aseg_bold_1mm = path.join(self.root_dir, 'aparc+aseg_bold_1mm.nii.gz')# nice to keep for other rois'
        
        self.list_of_parc_labels = [4, 10, 11, 12, 49, 50, 51, 17, 24, 53]
        
    def make_aparc_aseg_bold_1mm(self, aparc_aseg_bold_1mm):
        # the rawavg in fs dir is in reg with native t1_brain 
        aparc_aseg_native_t1_brain = imagetools.mk_temp_file_name(suffix='.nii.gz')
        
        try:
            cmd = "mri_label2vol --seg %(aparc_aseg)s --temp %(template)s --o %(outfile)s --regheader %(aparc_aseg)s"
    
            cmd = cmd %{'aparc_aseg':self.static_paths.aparc_aseg,
                        'outfile':aparc_aseg_native_t1_brain, # covnersion to .nii.gz is done here
                        'template':self.static_paths.rawavg
                        }
            
            imagetools.call_shell_program(cmd)
         
            # resample the parc to the T1 segmentation that is reg to bold mean, in 1mm iso
            cmd = "reformatx --pv --pad-out 0 -o %(outfile)s --floating %(aparc_aseg_native_t1_brain)s %(t1_seg_bold_1mm)s  %(T1_2_bold_1mm_affine)s"
            cmd = cmd %{'outfile':aparc_aseg_bold_1mm,
                        'aparc_aseg_native_t1_brain':aparc_aseg_native_t1_brain,
                        't1_seg_bold_1mm':self.t1_seg_bold_1mm,
                        'T1_2_bold_1mm_affine':self.T1_2_bold_1mm_affine
                        }
            
            imagetools.call_shell_program(cmd)
            
        finally:
            if path.exists(aparc_aseg_native_t1_brain):
                remove(aparc_aseg_native_t1_brain)     
        
    def make_parc_rois(self, bold_mean, outfile):
        
        if not path.exists(self.aparc_aseg_bold_1mm):
            self.make_aparc_aseg_bold_1mm(self.aparc_aseg_bold_1mm)

        temp_file = imagetools.mk_temp_file_name(suffix='.nii.gz')

        try:
            parc_hdr = imagetools.load_image(self.aparc_aseg_bold_1mm)
            parc_data = parc_hdr.get_data()
            
            new = np.zeros_like(parc_data)
            
            for label in self.list_of_parc_labels:
                ind = parc_data == label
                new[ind] += label
            
            imagetools.save_new_image(new, temp_file, parc_hdr.coordmap )
                
            imagetools.resample_tissue_mask_to_ref(temp_file, bold_mean, outfile)    
        
        finally:
            remove(temp_file)


class QcMetrics(object):
   
    def __init__(self, root_dir, sub_id, bold_bf, bold_af, mask, traj_file, rem_ind_file, motion_thres, motion_test=None):
        self.bold_bf = bold_bf
        self.bold_af = bold_af
        self.mask = mask
        self.traj = np.loadtxt(traj_file, float)
        self.figure_name = path.join(root_dir, 'QC_plots_%s.png' %sub_id) 
        try:
            self.rem_ind = np.loadtxt(rem_ind_file, int)
        except ValueError:
            self.rem_ind = []
        
        self.sub_id = sub_id 
        self.motion_thres = motion_thres
        self.motion_test = motion_test

    def main(self):
        roi_bf, roi_af, gm_mu, gm_std, frq, mag_traj, mag_mu = self.make_metrics()
        self.subplots(roi_bf, roi_af, gm_mu, gm_std, frq, mag_traj, mag_mu)
        
    def make_metrics(self):
    
        roi_bf,_,_,_ = imagetools.get_roi(self.bold_bf, self.mask)        
        roi_bf = np.float32(roi_bf)
        
        roi_bf -= roi_bf.mean(axis=1, keepdims=True)
        sig = roi_bf.std(axis=1, ddof=10, keepdims=True)
        sig[sig == 0] = 1
        roi_bf /= sig
        
        roi_af,_,_,_ = imagetools.get_roi(self.bold_af, self.mask)
        roi_af = np.float32(roi_af)
        sig = roi_bf.std(axis=1, ddof=10, keepdims=True)
        sig[sig == 0] = 1
        roi_af /= sig
    
        # bf data doesn't have the 15 frames dropped yet.
        # that happens after reg b/c of filter steady state response time
        num_dropped = roi_bf.shape[1] - roi_af.shape[1]
        if num_dropped < 0:
            raise Exception, "Something is wrong. The number of time points dropped was too many." 
    
        roi_bf = roi_bf[:, num_dropped:]  
    
        gm_mu = roi_af.mean(axis=0)
        gm_mu -= gm_mu.mean(axis=0)
        
        gm_std = roi_af.std(axis=0)
        gm_std -= gm_std.mean()
    
      
        traj = self.traj.copy()
        traj[traj > self.motion_thres] = 0
        frq, mag = welch(traj, fs=1/2.2, window='hanning', nperseg=100, noverlap=50, nfft=512, detrend='linear', 
                                      return_onesided=True, scaling='density', axis=-1)

        _, mag_mu = welch(gm_mu, fs=1/2.2, window='hanning', nperseg=100, noverlap=50, nfft=512, detrend='linear', 
                         return_onesided=True, scaling='density', axis=-1)

        #frq, mag = sigtools.periodogram_1D(detrend(self.traj, type='constant'), 1/2.2)
    
        return roi_bf, roi_af, gm_mu, gm_std, frq, mag, mag_mu
    
    def subplots(self, roi_bf, roi_af, gm_mu, gm_std, frq, mag_traj, mag_mu): 
        fig = plt.figure()
        x_len = self.traj.size
        
        ax1 = fig.add_subplot(5,1,1)
        ax1.plot(self.traj,'k')
        ax1.hlines(self.motion_thres, 0, 260,'y')
        ax1.set_title('F2F Trajectory: %s, Removed Ind:%i, Mot Thres:%f' %(self.sub_id, len(self.rem_ind), self.motion_thres))# uselen b/c None type returns 0
        ax1.set_ylabel('mm', fontsize='xx-small' )
        ax1.set_xlim(0, x_len)
        ax1.tick_params(labelsize=5)
        ax1.tick_params(axis='x', which='both', bottom='off', labelbottom='off')   
                            
        ax2 = fig.add_subplot(5,1,2)
        ax2.imshow(roi_bf, origin='lower', interpolation='nearest', aspect='auto')
        ax2.set_title('Time Courses Before Regressions', fontsize='xx-small')
        ax2.tick_params(axis='both', which='both', bottom='off', labelleft='off', labelbottom='off')
        
        ax3 = fig.add_subplot(5,1,3)
        ax3.imshow(roi_af, origin='lower', interpolation='nearest', aspect='auto')
        ax3.set_title('Time Courses After Regressions', fontsize='xx-small')
        ax3.tick_params(axis='both', which='both', bottom='off', labelleft='off', labelbottom='off')
        
        ax4 = fig.add_subplot(5,1,4)
        ax4.plot(frq, mag_mu)
        ax4.vlines(0.009, mag_mu.min(), mag_mu.max(),'r')
        ax4.vlines(0.09, mag_mu.min(), mag_mu.max(),'r')
        #ax4.plot(self.motion_test)   
#        ax4.hlines(0, 0, 260)
#         if len(self.rem_ind) != 0:
#             ax4.vlines(self.rem_ind, self.motion_test.min(), self.motion_test.max(), 'r')
        ax4.set_title('FFT of Mean Global Signal ( GM mask).', fontsize='xx-small')
#        ax4.set_xlim(0, x_len)
        ax4.tick_params(axis='both', which='both', bottom='off', labelleft='off', labelbottom='off')
       
        minorLocator = MultipleLocator(10)
        ax5 = fig.add_subplot(5,1,5)
        ax5.plot(frq, mag_traj)   
        ax5.set_title('FFT of Trajectory', fontsize='xx-small')
        ax5.set_xlabel('Frequency', fontsize='xx-small')
        ax5.set_ylabel('Power Density')
        ax5.tick_params(labelsize=5)
        ax5.xaxis.set_minor_locator(minorLocator)
        ax5.vlines(0.009, mag_traj.min(), mag_traj.max(),'r')
        ax5.vlines(0.09, mag_traj.min(), mag_traj.max(),'r')
        plt.savefig(self.figure_name)   

    def fslview_overlay(self):
        # make 1mm deg map
        cmd = 'reformatx -o %(outfile)s --floating %(deg_map)s %(aparc_1mm_bold)s'
        
        cmd = "fslview deg_map_1mm.nii.gz -t 0.5 -l 'render3' aparc+aseg_bold_1mm.nii.gz  -t 0.5 -l 'Random-Rainbow'"
                
if __name__ ==  "__main__":
    def raise_implementation_error(*args):
        raise NotImplementedError     
        
    parser = OptionParser()
    
    
    parser.add_option("-r", "--root-dir", dest="root_dir",
                       help="Path where the experiment will be put.")
    
    parser.add_option("-t", "--type", dest="Type", default='vox',
                       help="Types are 'roi-dpc', 'vox-dpc', 'conn_single' ")
    
    parser.add_option("-c", "--sub_id", dest="sub_id",
                       help="NCANDA sub_id ID, i.e. NCANDA_S00033")
    
    parser.add_option("-p", "--physio-off", dest="physio_correct", action='store_false',
                       help="Turn physio correction step ON/OFF. Default is off. No arg, just presence of the flag")
    
    parser.add_option("-P", "--physio-on", dest="physio_correct", action='store_true',
                       help="Turn physio correction step ON/OFF. Default is off. No arg, just presence of the flag")
    
    parser.add_option("--step-A", dest="step_A_bool", action='store_true', 
                      help="If the option is present, the step A will be run and used as input.")
    
    parser.add_option("--time-point", dest="time_point", default='base',
                       help="Default is 'base'. Choices are 'base' and 'one_year' ")
    
    (options, args) = parser.parse_args()
    
    if options.step_A_bool:
        step_A = ProcessNCANDAResting(options.sub_id, options.root_dir, options.physio_correct, options.time_point)
        step_A.set_up_log()
        step_A.call_shell_script_to_make_subspace_unwarped_aligned()
    
    else:
        pass
    
    proc_dict = {'conn_single':ConnSingleSubject,
                 'vox-dpc':ProcessNCANDARestingDpc,
                 'roi-dpc':raise_implementation_error

                 }
    
    process_class = proc_dict[options.Type](options.sub_id, options.root_dir, options.physio_correct, 
                                            options.time_point
                                           )
    
    process_class.main()
    

    
    
   
 














