#!/usr/bin/python

from os import path
import ImageTools
import numpy as np
from optparse import OptionParser
import os
import LoggingTools
import tempfile
from shutil import rmtree

imagetools = ImageTools.ImageTools()

### GLOBAL VARIABLES ###
image_type_dict_choices = ['fieldmap', '4d', 'vols', 'structural', 'other']

def get_list_of_series_desc_for_uniq_dicom_dir(dicom_dirs, dicom_identifier):
    
    series_descriptions = []
       
    for dcm_dir in dicom_dirs:
        dicom_example = imagetools.find_files(dcm_dir, dicom_identifier, maxdepth=1)[0]
        series_descriptions.append(imagetools.get_series_description(dicom_example))   
    
    series_descriptions = list(set(series_descriptions))
    series_descriptions.sort()
    return series_descriptions  
        
def get_input():
    msg = ','.join(image_type_dict_choices)
    key = raw_input("What type of scan? %s:" %msg)
    if key not in image_type_dict_choices:
        print "You typed a value that is not one of the four, try again."
        key = get_input()
    
    return key

def display_series_desc_and_store_answers_in_dict(series_descriptions):  

    image_type_dict = {}
    
    for desc in series_descriptions:
        print desc    
        val = get_input()
        image_type_dict[desc] = val
    
    review_image_types(image_type_dict, series_descriptions)
         
    return image_type_dict                

def review_image_types(image_type_dict, series_descriptions):
    print "***REVIEW image type  for each series description***\n"
    
    for k,v in image_type_dict.iteritems():
        print "%s : %s" %(k,v)
        
    Bool = raw_input('\n Are these correct?(y/n):')
    
    if Bool == 'y':
        return
    
    elif Bool == 'n':
        display_series_desc_and_store_answers_in_dict(series_descriptions)   
        
    else:
        print "answer not understood, select 'y' or 'n'. "
        review_image_types(image_type_dict, series_descriptions)
                
def convert_dicom_2_nii(dicom_dir, output):
        
    cmd = "dcm2image --tolerance 1e-3  -rv -O %(output)s %(dicom_dir)s"
    
    cmd = cmd %{'output':output,
                           'dicom_dir':dicom_dir
                          }
    
    # dcm2image, will create a full path if it doesn't exist
    stdout = imagetools.call_shell_program(cmd)
    print stdout
    
    return output

def search_for_all_dicoms_return_uniq_dir(root_dir, dicom_identifier):
    cmd = "find %(root_dir)s -type f -iname '%(dicom_id)s' | xargs -n 1 dirname | uniq"
    cmd = cmd %{'root_dir':root_dir,
                'dicom_id':dicom_identifier
                }
 
    output = imagetools.call_shell_program(cmd)
    
    output = np.array(output.split(), str)
    
    return output

def get_nii_output_path(dicom_path, root_dir, study, dicom_identifier, series_description, dicom_example, image_type):
    delimiter_index = dicom_path.split('/').index(study)
    # skip the study name (+1), and the old series number (-1)
    remaining = dicom_path.split('/')[delimiter_index + 1 : -1]
    
    raw_index = remaining.index('raw')
    # keep the same path structure just add the 'nifti' dir at the same level as 'raw'
    remaining[raw_index] = 'nifti'
    
    if image_type == 'fieldmap' or image_type == 'vols':
        output = path.join(root_dir, '/'.join(remaining), series_description, series_description + '_%n.nii' ) 
    
    elif image_type == '4d':
        output = path.join(root_dir, '/'.join(remaining), series_description + '_4d.nii' ) 
    
    elif image_type == 'structural':
        output = path.join(root_dir, '/'.join(remaining), series_description + '.nii') 
    
    # image_type 'other' shouldn't make it this far
    else:
        raise Exception, "Image type not understood:%s" %image_type

    return output
                       
def set_environment_var_for_no_gzip():
    os.environ['CMTK_WRITE_UNCOMPRESSED'] = '1'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI'

def check_for_existing_image_type_dict_json(image_type_dict_json_file):
    
    if path.exists(image_type_dict_json_file):
        return True
    else:
        return False

def manage_image_type_dict(dicom_dirs, dicom_identifier, image_type_dict_json_file):
    Bool = check_for_existing_image_type_dict_json(image_type_dict_json_file)
    
    if Bool:
        image_type_dict = imagetools.load_dict_from_json(image_type_dict_json_file)
        
        # check to see if any new dicoms with new series descriptions
        # were found, that are not in the existing dict
        series_descriptions = get_list_of_series_desc_for_uniq_dicom_dir(dicom_dirs, dicom_identifier)
        image_type_dict = check_for_new_image_types(image_type_dict, series_descriptions)
        
        imagetools.save_dict_as_json(image_type_dict, image_type_dict_json_file)
        
        return image_type_dict
        
    else:
        series_descriptions = get_list_of_series_desc_for_uniq_dicom_dir(dicom_dirs, dicom_identifier)
        image_type_dict = display_series_desc_and_store_answers_in_dict(series_descriptions)
    
        imagetools.save_dict_as_json(image_type_dict, image_type_dict_json_file)
    
        return image_type_dict

def check_for_nii_existance(nii_path):
    # EPI_Metamemory1_%n.nii , Axial_fast_SPGR.nii
    name = path.basename(nii_path)
    name = name.replace('.nii','').replace('_%n','')
    pattern = "%s*.nii" %name
    
    niis = imagetools.glob_for_files(root_dir=path.dirname(nii_path), 
                              pattern=pattern, 
                              num_limit=999, # overkill but whatever
                              log_this=False
                              )
    
    if niis is not None:
        return True
    else:
        return False
    
def check_for_new_image_types(image_type_dict, series_descriptions):
    
    for desc in series_descriptions:
        if not image_type_dict.has_key(desc):
            print desc    
            val = get_input()
            image_type_dict[desc] = val
            
    return image_type_dict        

def make_4d_file(dcm_dir, nii_path, TR, log):
    try:
        if not path.exists(path.dirname(nii_path)):
            os.makedirs(path.dirname(nii_path))
        
        temp_dir = tempfile.mkdtemp(dir='/tmp', prefix='dicom')
        temp_path = path.join(temp_dir, 'fmri.nii')
        convert_dicom_2_nii(dcm_dir, temp_path)  
        
        imagetools.merge_vols_to_4D(path_to_vols=temp_dir, 
                            regex= '.*\.nii',
                            outfile=nii_path, 
                            TR=TR, 
                            drop=0, 
                            log_this=True
                            )

        rmtree(temp_dir)      

    except Exception, msg:
        log.error(msg)
        print msg    
        rmtree(temp_dir)

def main(root_dir, dicom_identifier, gzip_bool, log_level):
    log_path = path.join(root_dir, 'dicom_conversion_log.txt')
    log = LoggingTools.SetupLogger(log_path, log_level).get_basic_logger()
      
    if gzip_bool:
        set_environment_var_for_no_gzip()
    
    dicom_dirs = search_for_all_dicoms_return_uniq_dir(root_dir, dicom_identifier)
    dicom_dirs.sort()
    
    image_type_dict_json_file = path.join(root_dir, 'image_type_dictionary.json')
    image_type_dict = manage_image_type_dict(dicom_dirs, dicom_identifier, image_type_dict_json_file)

    study = root_dir.split('/')[-1]
    log.info('Study: %s' %study)
    
    for dcm_dir in dicom_dirs:
        
        dicom_example = imagetools.find_files(dcm_dir, dicom_identifier, maxdepth=1)[0]
        dicom_header = imagetools.get_dicom_header_object(dicom_example)# object with __str__ 
        series_description = dicom_header.SeriesDescription.replace(' ','_')
        image_type = image_type = image_type_dict[series_description]
        
        # continue early, to avoid checking for existance
        if image_type == 'other':
            continue
        
        nii_output = get_nii_output_path(dcm_dir, root_dir, study, dicom_identifier, series_description, dicom_example, image_type)
        # don't overwrite existing Nifti's from eariler run of this program
        Bool = check_for_nii_existance(nii_output)
        
        if Bool:
            continue
        
        try:
            log.info("%s\n%s\n\n" %(nii_output, series_description))
            
            if image_type == '4d':
                make_4d_file(dcm_dir, nii_output, dicom_header.RepetitionTime, log)
            else:    
                convert_dicom_2_nii(dcm_dir, nii_output)      
    
        except Exception, msg:
            log.error(msg)
            print msg    
        
if __name__ == "__main__":
    
    general_help = """This program expects the file heirarchy to have some basic structure. 
/path_to_here/<study>/<time_point>/raw/<sub_id>/<dicom_series_dir> 
     
Example:    
/fs/cl07/fmri/METACOL/METACOL_3Told_3monthFU/raw/1891_08092013/007
 
The NiFTI's will created in a new directory:
/fs/cl07/fmri/METACOL/METACOL_3Told_3monthFU/nifti/1891_08092013/007
 
'METACOL' is the study.
 
'METACOL_3Told_3monthFU' and 'METACOL_750_0baseline' are the time points.
 
'raw' is where the dicoms are.
 
'1891_08092013'  is a sub_id directory
 
'007' is a dicom directory and scan sequence number.
 
Inside the numbered dicom directories ( i.e. 010, or 007 ), are dicoms but we don't know for certain what their
series decription is.
 
This program, finds ALL dicoms and takes a single dicom as an example, then looks up the 'SeriesDescription'
from the dicom header. 
 
On first use, the program will ask you to mark which series are 'fieldmap', 'fmri', 'structural' or 'other'.
Series marked 'other' are not converted into NiFTI files.
 
The answers to those questions above, will be saved in a file at the 'study' level and reused later, when the
program is run again.
 
The program will NOT overwrite existing NiFTI files. It checks first, for an existing file(s) and skips 
converting dicom directores for which NiFTI files already exist.
 
Typical GE scanner usage:
dicom_2_nii_batch.py --input-root /fs/cl07/fmri/METACOL
 
Example when the dicoms do not have the .dcm file extension. In this case the keyword is MRDC.
dicom_2_nii_batch.py --input-root /fs/cl07/fmri/METACOL --dicom-identifier '*MRDC*'
      
"""

    parser = OptionParser(epilog=general_help)
    
    parser.add_option("--input-root", dest="root_dir", help="path to project root, e.g. /fs/cl07/fmri/METACOL") 
    
    parser.add_option("--dicom-identifier", dest="dicom_identifier", default='*.dcm',
                      help="The dicom file extension *.dcm ( default ). Siemens uses 'MRDC' in the file name.") 
    
    parser.add_option("--allow-gzip", dest="gzip_bool", action='store_true', default='store_false',
                       help="Allow outputs to gzipped, *.nii.gz")
    
    parser.add_option("--log-level", dest="log_level", default='info',
                       help="""This program can log many outputs helpful for debugging and getting
                       dicom header info. The default is set to 'info'. Use this option and set to
                       'debug' for more information.""")
    
    (options, args) = parser.parse_args()
    
    try:
        main(options.root_dir, options.dicom_identifier, options.gzip_bool, options.log_level)          
    
    except KeyboardInterrupt:
        cmd = 'rm -rf /tmp/dicom*'
        try:
            imagetools.call_shell_program(cmd)  
        except:
            pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
