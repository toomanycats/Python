#!/usr/bin/python

from optparse import OptionParser
from nipype.algorithms import rapidart
import scipy
import numpy as np
import ImageTools
from os import path

imagetools = ImageTools.ImageTools()

def make_conn_tb_compatible_cov_file(num_vols, outlier_file, output_path):

    outliers =  np.loadtxt(outlier_file, int)
    num_outliers = outliers.shape[0]
    
    covariate = np.zeros((num_vols, num_outliers), np.int16)

    for i in range(num_outliers):
        covariate[outliers[i], i] = 1
    
    np.savetxt(output_path, covariate, fmt='%f') 
    #scipy.io.savemat(output_path, {'R':covariate}, oned_as='column')

def main(bold_4d, mask, motion_file, z_threshold, norm_threshold):

    art = rapidart.ArtifactDetect()
        
    art.inputs.realigned_files = bold_4d
    art.inputs.realignment_parameters = motion_file
    art.inputs.parameter_source = 'FSL'
    art.inputs.norm_threshold = norm_threshold
    art.inputs.use_differences = [True, False]
    art.inputs.zintensity_threshold = z_threshold
    art.inputs.mask_type = 'file'
    art.inputs.mask_file = mask
    art.inputs.bound_by_brainmask=True
    
    art.run()
     
    # get the rapid_art output in new format for CTB:
    dirname = path.dirname(bold_4d)
    name =  path.basename(bold_4d).replace('.nii','')
    # 'art.wr_bold_outliers.txt', output from rapid art, that we need 
    # to convert to the MATLAB Art style covariate.
    new_cov_fname = "art.%s_outliers.txt" %name
    
    art_spoof_path = path.join(dirname, 'art_regression_outliers_r_bold.txt')
    
    num_vols = imagetools.get_number_of_vols(bold_4d)
    make_conn_tb_compatible_cov_file(num_vols, new_cov_fname, art_spoof_path)
    
if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("--bold", dest="bold_4d",
                      help="Full path to the bold images in 4D container.")
    
    parser.add_option("--mask", dest="mask",
                      help="Brain mask for the bold.")

    parser.add_option("--motion_file", dest="motion_file",
                      help="FSL mcflirt .par motion estimate produced from '-plots' arg.")
    
    parser.add_option("--z_thres", dest="z_threshold", type='float',
                      help="Z normed threshold for the brain mean per volume.")
    
    parser.add_option("--norm_thres", dest="norm_threshold", type='float',
                      help="Threshold in mm for motion.")

    (options, args) = parser.parse_args()
 
    main(options.bold_4d, options.mask, options.motion_file, options.z_threshold, options.norm_threshold)    


