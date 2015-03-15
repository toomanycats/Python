#!/bin/python

from optparse import OptionParser
from os import getcwd, path, makedirs, rename
from shutil import rmtree, copyfile
import glob
import Tools
import PhysioCorrect
import traceback

imagetools = Tools.ImageTools()

def convert2analyze(infile, outfile):
    cmd = "convertx %s %s" %(infile, outfile)
    imagetools.call_shell_program(cmd)

def main(output_root, dicom_input, num_vols_drop, card_file, resp_file, TR):

    nii_dir = path.join(output_root, 'Nifti')
    if not path.exists(nii_dir):
        makedirs(nii_dir)
    
    analyze_dir = path.join(output_root, 'Corrected')
    if not path.exists(analyze_dir):
        makedirs(analyze_dir)
    
    # make nii's from dcm's
    output_pattern = path.join(nii_dir, 'image_%n.nii')
    imagetools.convert_dcm_to_nii(dicom_input, output_pattern, tol=1e-3)
    
    #copy the physio files into the root dir
    card_local = path.join(output_root, 'card_orig.txt')
    resp_local = path.join(output_root, 'resp_orig.txt')
    
    copyfile(card_file, card_local)
    copyfile(resp_file, resp_local)
    
    # 4D file for 3dretroicor
    fourD_file = path.join(nii_dir, 'image_four_dim.nii.gz')
    imagetools.merge_vols_to_4D(nii_dir, regex='image_[0-9]+\.nii.*', outfile=fourD_file, TR=TR)
    
    # fix the header of the input file, pixdim4 and slice duration
    imagetools.fix_slice_dur(fourD_file)# uses the new fixed TR
    
    # setup the dict ob that is used in the PhysioCorrect class
    # to hold boolean values ( None = False ) and paths ( True )
    dict_of_files = {'card':card_local, 
                     'card_trig':None,
                     'resp':resp_local
                     }
    
    # generate a trigger for the cardiac file and filter the resp                  
    preproc = PhysioCorrect.PreprocessSriUcsd(nii_dir, num_vols_drop, fourD_file)
    preproc.dict_of_files = dict_of_files # in the pypeline the dict of files is populated in a Main() class
    preproc.main()
    
    # apply 3dretroicor
    denoised_fname = path.join(nii_dir, 'image_4D_phy_corr.nii')
    denoise = PhysioCorrect.Denoise(nii_dir, preproc.dict_of_files, fourD_file, denoised_fname)        
    denoise.main() 
    
    #move preprocessed physio files from NiFTI dir to root dir
    rename(path.join(nii_dir,'truncated_card_data_trig.txt'), path.join(output_root,'truncated_card_data_trig.txt'))
    rename(path.join(nii_dir,'truncated_card_data.txt'), path.join(output_root,'truncated_card_data.txt'))
    rename(path.join(nii_dir,'truncated_resp_data.txt'), path.join(output_root,'truncated_resp.txt'))
    
    # split files for spm
    tmp_out = '/tmp/Physio'
    makedirs(tmp_out)
    output_base = path.join(tmp_out, 'physio_corrected_')
    
    imagetools.split_4D_to_vols(denoised_fname, output_base)
    
    # convert to analyze
    file_list = glob.glob(output_base + '*')
    for file in file_list:
        out = path.join(analyze_dir, path.basename(file).replace('.nii.gz','.hdr'))
        convert2analyze(file, out)
        #imagetools.ungzip(out.replace('.hdr','.img.gz'))

    # free memory and the path name for another run
    rmtree(tmp_out) 
        
        
if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("-o", "--output-dir", dest="output_dir",default=getcwd(),
                      help="Default is the current directory you are calling from.")
    
    parser.add_option("-i", "--dicom-input-dir", dest="dicom_dir",
                      help="The full path the directory containing the dicoms.")
    
    parser.add_option("-n", "--num-drop", dest="num_vols_drop", type=int, default=0,
                      help="Default=0. The number of fMRI vols to drop to reach steady state.")
    
    parser.add_option("-c", "--card", dest="card_file",type=str, default=None,
                         help="The name of the cardiac data files in txt format.")  
       
    parser.add_option("-r", "--resp", dest="resp_file",type=str, default=None,
                        help="The name of the respiratory data files in txt format.")     
    
    parser.add_option("-T", "--TR", dest="TR",type=float, default=2.2,
                        help="Default=2.2 s.The repetition time of the fMRI sequence. MUST BE IN SECONDS, i.e. TR=2.2, not 2200")      

    (options, args) = parser.parse_args()

    main(options.output_dir, options.dicom_dir, options.num_vols_drop, options.card_file, options.resp_file, options.TR)
