from os import path
import os
import re
import ImageTools

imagetools = ImageTools.ImageTools()    
       
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'    
    
root_dir = '/fs/fmri/alcg_2014'

dir_list = os.listdir(root_dir)

dir_pattern = '[0-9]{4}_[0-9]{8}'

re_obj = re.compile(dir_pattern)

sub_list = []

# 0288_10122011/0288_10122011_fmri/swufmri003.hdr

for date in dir_list:
    if re_obj.search(date):
        fmri_path = path.join(root_dir, date, date + '_fmri')
        sub_list.append(fmri_path)
        
for sub in sub_list:
    outfile = path.join(root_dir, sub, 'swu_fmri_4d.nii')
    
    if path.exists(outfile):
        print "skipping:%s" %sub
        continue
    
    files = imagetools.glob_for_files(sub, 
                              pattern='swufmri*.hdr', 
                              num_limit=135
                              )
    if files is not None:
    
        imagetools.merge_vols_to_4D(path_to_vols=sub, 
                                    regex='swufmri[0-9]{3}.hdr', 
                                    outfile=outfile,
                                    TR=2.2
                                    )
                
        
        