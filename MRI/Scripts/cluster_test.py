# import PhysioCorrect
import os
import ImageTools
import SignalProcessTools
# import GroupAnalysisTools
# import ComputeTools
import PyConn
import numpy as np
import SetupCtbProcessing
import NCANDA_resting_state_pipeline
import matplotlib.pyplot as plt

imagetools = ImageTools.ImageTools()
sigtools = SignalProcessTools.SignalProcessTools() 
 
 
root_dir = '/fs/cl10/dpc/Data/Test/NCANDA_S00073'
sub_id = 'NCANDA_S00073'
infile = os.path.join(root_dir, 'bold_mcf_unw_dspk_reg_drp.nii.gz')
outfile = os.path.join(root_dir, 'test.nii.gz')
mask =  os.path.join(root_dir, 'brain_mask.nii.gz')
motion = os.path.join(root_dir, 'motion_f2f_params.txt')
motion_traj_file = os.path.join(root_dir, 'motion_traj.txt')
rem_file = os.path.join(root_dir, 'removed_ind.txt')
gm = os.path.join(root_dir, 'gm_ero.nii.gz')
wm =  os.path.join(root_dir, 'wm_ero.nii.gz')
aux_tissue = os.path.join(root_dir, 'parc_tissues.nii.gz')
rem_list = os.path.join(root_dir, 'removed_ind.txt')
t1_seg_bold_1mm = os.path.join(root_dir, 't1_seg_bold_1mm.nii.gz')
T1_2_bold_1mm_affine = os.path.join(root_dir, 'T1_2_bold_mean.affine')
bold_mean = os.path.join(root_dir, 'bold_mean.nii.gz')

bold_bf = os.path.join(root_dir, 'bold_mcf_unw_dspk.nii.gz') # needs 15 frames dropped
bold_af = os.path.join(root_dir, 'bold_mcf_unw_dspk_reg_drp.nii.gz')
        
qc =  NCANDA_resting_state_pipeline.QcMetrics(root_dir, 
                                               sub_id, 
                                               bold_bf=bold_bf, 
                                               bold_af=bold_af, 
                                               mask=gm, 
                                               traj_file=motion_traj_file, 
                                               rem_ind_file=rem_file,
                                               motion_thres=0.35
                                               )
qc.main()

#  root_dir, rs_4D_path, confound_cleaned_output, mask, sub_id, time_point='base', Type='human', wm_rs=None, aux_tissue=None, csf_rs=None, move_par_type=None):
# det = PyConn.DetrendMovementNoise(root_dir, infile, outfile, mask, sub_id, wm_rs=wm, aux_tissue=aux_tissue, move_par_type='mcflirt')
# det.main()

# fs = NCANDA_resting_state_pipeline.FreeSurferROIs(root_dir, sub_id, 'base', t1_seg_bold_1mm, T1_2_bold_1mm_affine)
# fs.make_ventricle_rois(bold_mean)


# motion = PyConn.MotionAnalysis(root_dir, infile, mask, gm, motion)
# ind, traj, test = motion.main()
# mu, sig = sigtools.get_mean_signal_with_mask(infile, mask, True)
# mu -= mu.mean()
# sig -= sig.mean()
# sig /= sig.std()
#plt.plot(mu*7,'-o')
# plt.plot(sig,'-*')
# plt.vlines(ind, sig.min(), sig.max(),'y')
# plt.plot(sig*traj)
# plt.show()


# ctb = SetupCtbProcessing.StepA(root_dir, 'NCANDA_S00033')
# ctb.main()

# motion = PyConn.MotionAnalysis(root_dir, infile, mask, gm, motion)
# bad_ind = motion.main_volt()
# print bad_ind



# bad_ind = np.array(([0,1,2,3,4,11,12,13,14,25,26,27,50,51,52,67,68]),dtype=np.int16)

# noise = PyConn.ReplaceWithNoise(mask, infile, outfile, bad_ind)
# noise.main()


# imagetools.make_mask_for_smoothed_time_series(infile, outfile)

# nfft = PyConn.NFFT_Interp( mask, infile, outfile, rem_list)
# nfft.main()

# import NCANDA_resting_state_pipeline
# 
# test = NCANDA_resting_state_pipeline.ProcessNCANDARestingAtlas('NCANDA_S00119', root_dir, physio_correct=False)
# test.get_group_cov_per_sub()

# cov = GroupAnalysisTools.AllNcandaSubjects('/fs/u00/dpc/all_cov.csv', '/fs/cl10/dpc/Data/CTB_Test/sub_id_list.txt')
# cov.get_all_sub_cov()

#     
# motion_ = PyConn.MotionAnalysis(root_dir, infile, mask, gm, motion)
# ind, gm = motion_.main()
# # 
# import matplotlib.pyplot as plt
# x = np.arange(gm.size)
# plt.plot(x, gm)
# plt.vlines(ind, -0.1, 0.1)
# plt.show()

# rem_ind = np.loadtxt(rem_list, np.int16)
# nfft = PyConn.NFFT_Interp(mask, infile, outfile, rem_ind)
# nfft.main()



# parc = os.path.join(root_dir, 'parc116.nii.gz')
# aves = os.path.join(root_dir, 'roi_averages.npy')

# mot = PyConn.MotionAnalysis()
# 
# test = mot.convert_spm_motion_to_vol2vol_file(infile)
# np.savetxt('new_convrt_test.txt', test)

# log_file = os.path.join(root_dir, "log_test.txt")
# outfile = os.path.join(root_dir, 'log_test_image.nii.gz')
# import LoggingTools
# log = LoggingTools.SetupLogger(log_file).get_module_logger()
# 
# import PyXnatTools
# pyx = PyXnatTools.PyXnatTools()
# 
# resources = 'NCANDA_E01696/27630/spiral.tar.gz'
# 
# import export_mr_sessions_pipeline
# to_dir = os.path.join(root_dir, 'stroop')
# export_mr_sessions_pipeline.export_spiral_files(pyx.interface, resources, to_dir)

#sigtools.smooth(np.ones(100), window_len=5)

#bw = PyConn.BandwidthTR('human', infile)

# vox = PyConn.VoxelWiseAnalysis(root_dir, infile , mask)
# vox.seed_analysis_from_ijk(42, 27, 16)
# vox.compute_degree()
# vox.compute_deg_with_multi_dist(A=100, corr_thres=0.30)
# vox.compute_deg_cont_thres(A=100, corr_thres=0.30)
# vox.compute_deg_with_multi_dist(12)



# row_num = vox.seed_analysis_from_ijk(30, 16, 23)
# print row_num
# vox.compute_deg_cont_thres(A=60)
# vox.compute_deg_with_multi_dist(55)

# roi = PyConn.MakeMeanTimeSeries(root_dir, infile, parc, 'human')
# roi.main()
# 
# roi = PyConn.ProcessTimeSeriesRoi(root_dir, aves ,'human', infile, parc, mask)
# corr = roi.correlation()
#  
# plt.imshow(corr, interpolation='nearest', origin='lower')
# plt.show()

# vox = PyConn.VoxelWiseAnalysis(root_dir, infile, mask, corr_thres=0.70 )
# vox.seed_analysis_from_ijk(28, 51, 19)
# vox.seed_analysis_from_ijk(29, 52, 18)
# vox.compute_deg_with_dist(corr_thres=0.50, dist_thres=55)
# vox.compute_local_cluster_coef(thres=15)
























                        




  




















                                         
