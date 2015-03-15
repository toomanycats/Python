#!/usr/bin/python

from optparse import OptionParser
from os import getcwd, path

import PhysioCorrect
from Tools import ImageTools

imagetools = ImageTools()

def main(out_dir, num_vols_drop, fourD_target_file, denoised_fname,
         card_file, resp_file, TR):
    '''This program is used to preprocess physio physio and then run it through the
    afni plugin called 3dretroicor.  The cardiac signal will be transformed into a trigger 
    the repiratory into a smoothed and centered wave form. These new outputs are saved as a 
    truncated_card.txt file. This program could be modified for use with different scanners, but
    as it is now, only supports the SRI scanner that Eva uses.'''
    
    # wrap in try catch, b/c if there's a failure, the input file
    # will be unzipped and then you'll have a *.nii that is confusing for
    # non programmer to trouble shoot

    # first, fix the header of the input file, pixdim4 and slice duration
    imagetools.fix_pixdim4(fourD_target_file, TR)
    imagetools.fix_slice_dur(fourD_target_file)# uses the new fixed TR

    
    # setup full path for output file
    denoised_path = path.join(out_dir, denoised_fname)
    
    # setup the dict ob that is used in the PhysioCorrect class
    # to hold boolean values ( None = False ) and paths ( True )
    dict_of_files = {'card':card_file, 
                     'card_trig':None,
                     'resp':resp_file,
                     }
    # generate a trigger for the cardiac file and filter the resp                  
    preproc = PhysioCorrect.PreprocessSriUcsd(out_dir, num_vols_drop, fourD_target_file)
    preproc.dict_of_files = dict_of_files # in the pypeline the dict of files is populated in a Main() class
    preproc.main()
    
    # if validation of the card or resp data fails,
    # then the dict entry for that data is set to None.
    # Otherwise, the dict_of_files gets updated with the paths to the
    # truncated/processed physio files for use in 3dretroicor
    
    # apply 3dretroicor
    denoise = PhysioCorrect.Denoise(out_dir, preproc.dict_of_files, fourD_target_file, denoised_path)        
    denoise.main() 
    
if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("-n", "--num-drop", dest="num_vols_drop", type=int, default=0,
                      help="The number of fMRI vols to drop to reach steady state.")
    
    parser.add_option("-o", "--out-file", dest="denoised_out_path", type=str,
                      help="The path of the output file.") 
    
    parser.add_option("-i", "--in-file", dest="fourD_target_file", type=str,
                      help="The path of the input file, which needs to be a 4D nifti.")  
    
    parser.add_option("-c", "--card", dest="card_file",type=str, default=None,
                         help="The path to the cardiac data files in txt format.")  
       
    parser.add_option("-r", "--resp", dest="resp_file",type=str, default=None,
                        help="The path to the respiratory data files in txt format.")     
    
    parser.add_option("-O", "--out-dir", dest="out_dir",type=str, default=getcwd(),
                        help="The root directory where all outputs will go. The default is the dir the script is called in.")
    
    parser.add_option("-T", "--TR", dest="TR",type=float, default=2.2,
                        help="The repetition time of the fMRI sequence. MUST BE IN SECONDS, i.e. TR=2.2, not 2200")          

    (options, args) = parser.parse_args()

    main( options.out_dir, options.num_vols_drop, options.fourD_target_file, options.denoised_out_path,
         options.card_file, options.resp_file, options.TR)





















