from os import path, remove, rename, makedirs
from shutil import copy
import PyConn
import logging
from ImageTools import ImageTools
from SignalProcessTools import SignalProcessTools
from ImageTools import np

source_rs_path = "/fs/cl02/torsten/rfMRI_rats/reformat/G2_13_warp/warp/rsfMRI_Rat_%02d/rsfMRI_Rat_%02d_session_%02d_gre_crop/%03d" #%(rat_num, rat_num, ses_num, sub session)
f2f_path = "/fs/cl02/torsten/rfMRI_rats/motion/session_%02d/rsfMRI_Rat_%02d/rat_axial_rsfmri_600x/%03d/rigid_motion.f2f" #%(session01, rat num, sub session num)

### methods ###
def smooth_avg_deg_maps(file_path_list, outfile):   
    outfiles = [] 
    for i, infile in enumerate(file_path_list):
        outfiles.append('/tmp/deg_map_sm_%i.nii.gz' %i)
        imagetools.spatial_smooth(infile, outfiles[i], fwhm=0.35)
    
    imagetools.average_4D_nipy(list_of_files=outfiles, outfile=outfile) 
    for f in outfiles:
        remove(f)
    
def copy_files_one_rat():
    infile = source_rs_path %(ses_num, rat_num)                                           
    imagetools.merge_vols_to_4D(infile, 'image.*\.nii\.gz', rs_4D_path, TR=1.65 )
                          
    # cp the tiss files
    copy(path.join(template_dir, 'rat_csf.nii.gz'), csf)
    copy(path.join(template_dir, 'rat_wm.nii.gz'), wm)
    copy(path.join(template_dir, 'rat_gm.nii.gz'), gm)

    # copy the struct files         
    copy(path.join(template_dir, 'late_brain.nii.gz'), late)
    copy(path.join(template_dir, 'brainmask.nii.gz'), mask)
#    copy(path.join(template_dir, 'seed_mask.nii.gz'), seed_mask)
       
    # copy the f2f file
    f2f = f2f_path %(ses_num, rat_num, sub_session) 
    
    copy(f2f, f2f_file)

def preprocess_one_rat():
    
    # downsample 
    imagetools.down_sample_iso(rs_4D_path, rs_4D_down, dim=0.5)
    
    # make the mean image
    imagetools.make_mean_image_from_4D(rs_4D_down, rs_mean)
    
    # coreg the Late brain image to the rs mean
    imagetools.coregister_images(rs_mean, late, late, bold_root_dir, dof=9, 
                                 cost="nmi", init='com')
    
    # apply the same transform to the brain mask 
    apply_xform(mask)  

    # erode, xform, threshold wm and csf
    imagetools.erode_image_box(infile=wm, outfile=wm, num_vox=3, dim=3)
    imagetools.erode_image_box(infile=csf, outfile=wm, num_vox=3, dim=3)
    
    apply_xform(wm)
    apply_xform(csf)
    
    imagetools.threshold_mask(mask=wm, outfile=wm, amt=0.50)
    imagetools.threshold_mask(mask=csf, outfile=csf, amt=0.50)

def process_one_rat():
    
    # remove time frames
    infile = rs_4D_down
    outfile = infile.replace('.nii.gz','_steady.nii.gz')
    num_drop = 5
    
    imagetools.fsl_roi(infile, outfile, num_drop, 0)
    
    # truncate the par file to match the rs data
    truncate_f2f_file(num_drop)
    
    infile = outfile
    outfile = infile.replace('.nii.gz','_drift.nii.gz')

    sigtools.remove_global_drift(infile, mask, outfile, seg_len=225, deg=2)

    
    # afni despike
    infile = outfile
    outfile = infile.replace('.nii.gz','_dspk.nii.gz')
     
    imagetools.afni_despike(infile, outfile)
    
    # spatial smooth
    infile = outfile
    outfile = infile.replace('.nii.gz', '_smooth.nii.gz')
     
    imagetools.spatial_smooth_mask(infile, outfile, mask, 0.45)
     
    # detrend noise
    infile = outfile
    
    detrend_mv = PyConn.DetrendMovementNoise( bold_root_dir, infile, mask, 'rat', wm_rs=wm, csf_rs=None)
    detrend_mv.move_par_type = 'cmtk'
    detrend_mv.aux_conf = False
    detrend_mv.global_mean = False
    detrend_mv.main()
    
    # cleaned output
    clean = detrend_mv.clean_rs_out_path
    
    # make new mask from one vol of the cleaned output. The brainmask.nii.gz doesn't fit well enough
    # and the zero time vectors will cause errors in the correlation computation
    
    new_mask = path.join(bold_root_dir, 'cleaned_image_mask.nii.gz')
    imagetools.make_mean_image_from_4D(infile=clean, outfile=new_mask)
    imagetools.make_mask_from_non_zero(new_mask, new_mask)
    
    # make a map of degree
    infile = outfile
    deg = PyConn.VoxelWiseAnalysis(bold_root_dir, infile, new_mask)#use cleaned > 0 as mask 
    deg.compute_degree(corr_thres=0.75)
    
    # seed analysis
    #apply_xform(seed_mask) # use this is the mask was created with another late image not reg 
     
    # outfile = path.join(bold_root_dir, 'seed_analysis.nii.gz')
    # vox  = PyConn.VoxelWiseAnalysis(bold_root_dir, clean, new_mask)
    # seed = vox.prepare_seed_from_mask(seed_mask)
    # # seed = vox.prepare_seed_from_coord(31, 36, 21, 24, 26, 31)
    # corr_map, coord = vox.voxel_wise_seed(seed)
    # imagetools.save_new_image(corr_map, outfile, coord)

def truncate_f2f_file(num_drop):
    f = open(f2f_file)
    lines = f.readlines()
    f.close()
    
    lines = lines[num_drop:]
    f = open(f2f_file.replace('.f2f','_trun.f2f'),'w')
    
    for line in lines:
        f.write(line)
    
    f.close()
    rename(f2f_file, f2f_file.replace('.f2f','.orig'))

def apply_xform(infile):
     
    cmd = "cd %s && reformatx -o %s --floating %s --cubic %s affine.xform"
    cmd = cmd %(bold_root_dir,
                infile, 
                infile, 
                rs_mean
                )
    
    imagetools.call_shell_program(cmd) 
    log.debug('apply xform cmd: %s' %cmd)



#### MAIN Script Loops ####
imagetools = ImageTools()
sigtools = SignalProcessTools()

root_dir = "/fs/corpus6/dpc/python_conn/Rat/"
template_dir = "/fs/corpus6/dpc/python_conn/Rat/Rat_ReProcess/Structural"
rat_num = 1
ses_dir = 'Sessions_02%i' %1
# /fs/cl02/torsten/GR_A_rats/motion/session_02/GR_A_Rat_01/rat_axial_greEPI_400x
source_rs_path = '/fs/cl02/torsten/GR_A_rats/motion/session_02%i/GR_A_Rat_02%i/rat_axial_greEPI_400x'

for ses_num in range(1,4):
                
    rat_dir = path.join(root_dir, 'GR_A_Rat_%02i' %rat_num)
    sub_sess_num = "%03i" %1 
    bold_root_dir = path.join(root_dir, ses_dir, rat_dir, sub_sess_num)   
       
    if not path.exists(bold_root_dir):
        makedirs(bold_root_dir)
       
    log_filename = path.join(bold_root_dir, 'logfile.log')
    log_level_dict = {'debug':10, 'info':20, 'error':40}
    level = 'debug'
    logging.basicConfig(filename=log_filename, level=log_level_dict[level])
    log = logging
       
    rs_4D_path = path.join(bold_root_dir, 'rs_bold.nii.gz')
    rs_4D_down = rs_4D_path.replace('.nii.gz','_0.5mm.nii.gz')
    rs_mean = rs_4D_down.replace('.nii.gz','_mean.nii.gz')
    
    late = path.join(bold_root_dir, 'late_brain.nii.gz')
    csf = path.join(bold_root_dir, 'rat_csf.nii.gz')
    wm = path.join(bold_root_dir, 'rat_wm.nii.gz')
    gm = path.join(bold_root_dir, 'rat_gm.nii.gz')
    mask = path.join(bold_root_dir, 'brainmask.nii.gz')
    f2f_file = path.join(bold_root_dir, 'rigid_motion.f2f')
    seed_mask = path.join(bold_root_dir, 'seed_mask.nii.gz')
    
         
    copy_files_one_rat()
    preprocess_one_rat()
    process_one_rat()

    # session analysis
    ## CONTROLS  
#     outfile = path.join(root_dir, ses_dir, 'degree_map_ctrl_mean_%i.nii.gz' %ses_num)
#     deg_maps = []
#     for rat in 2,4,5,6,9,12,13,14,20,21:
#         deg_maps.append(path.join(root_dir, ses_dir, 'Rat_%02d' %rat, 'degree_map.nii.gz'))
#     
#     smooth_avg_deg_maps(deg_maps, outfile)                            
#     
#     ## TREATED
#     outfile = path.join(root_dir, ses_dir, 'degree_map_treated_mean_%i.nii.gz' %ses_num)
#     deg_maps = []
#     for rat in 1,3,7,10,15,16,22,23:
#         deg_maps.append(path.join(root_dir, ses_dir, 'Rat_%02d' %rat, 'degree_map.nii.gz'))
#     
#     smooth_avg_deg_maps(deg_maps, outfile)     












