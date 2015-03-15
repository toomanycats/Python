#!/usr/bin/python

import os
import numpy as np
import re
import nibabel
from optparse import OptionParser
import ImageTools

imagetools = ImageTools.ImageTools()

def flirt(infile, ref, output_mat):
    
    cmd = "flirt -in %(infile)s -ref %(ref)s -omat %(output_mat)s -dof 6 -cost normmi"
    cmd = cmd %{'infile':infile,
                'ref':ref,
                'output_mat':output_mat
                }
    
    imagetools.call_shell_program(cmd)

def make_initial_alignment(nii_file, ref_file, output_xform):
    '''[options] ReferenceImage FloatingImage OutputXform'''
    cmd = "make_initial_affine  --native-space --centers-of-mass %(ref_file)s  %(nii_file)s %(output_xform)s"
    cmd = cmd %{'ref_file':ref_file,
                'nii_file':nii_file,
                'output_xform':output_xform
                }
    
    imagetools.call_shell_program(cmd, catch_errors=True, catch_warnings=False)

def read_cmtk_xform_file(xform_file):
#     $ cat initi_align.xform 
#     ! TYPEDSTREAM 2.4
#     
#     affine_xform {
#             xlate 1.240130376 17.33253053 28.51091201 
#             rotate -0 -0 -0 
#             scale 1 1 1 
#             shear 0 0 0 
#             center 0 0 0 
#     }

    f = open(xform_file, 'r')
    lines = f.readlines()
    f.close()
    
    trans = lines[3]
    
    # (?P<name>...)
    pattern = '\s+xlate\s(?P<xtrans>-*[0-9]+(.[0-9])*)\s+(?P<ytrans>-*[0-9]+(.[0-9])*)\s+(?P<ztrans>-*[0-9]+(.[0-9])*)\s+'

    match = re.search(pattern,  trans)
    
    xtrans = np.float16(match.group('xtrans'))
    ytrans = np.float16(match.group('ytrans'))
    ztrans = np.float16(match.group('ztrans'))

    return np.array((xtrans, ytrans, ztrans))

def put_trans_into_matrix(trans):
    
    matrix = np.zeros((4,4), np.float16)
    
    np.fill_diagonal(matrix, 1)
    
    matrix[0:3, 3] = -trans # negate, b/c we want to correct by that much
    
    return matrix

def get_s_form(nii_file):
    header = imagetools.load_image(nii_file)
    
    sform = header.header.get_sform()
    
    return sform

def main(nii_file, ref):
    
    root_dir = os.path.dirname(nii_file)
    output_xform = os.path.join(root_dir, 'initial_alignment.xform')
    
    make_initial_alignment(nii_file, ref, output_xform)
    xform = read_cmtk_xform_file(output_xform)
    T = put_trans_into_matrix(xform)
    
    M = get_s_form(nii_file)
    
    #flirt(nii_file, mni_reference, output_xform)
    #T = np.loadtxt(output_xform, np.float32)
    
    # T \dot M = M'
    M_prime = T.dot(M)
    
    image = nibabel.load(nii_file)
    
    image.set_sform(M_prime, 2)
    
    nibabel.save(image, nii_file)
    
    imagetools.set_sform_code(nii_file, 2)

if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("--nii", dest="nii_file", help="path to the nifti image") 
    
    parser.add_option("--ref", dest="ref", help="path to the reference image") 
    
    (options, args) = parser.parse_args()
    
    main(options.nii_file, options.ref)   
    
    
    
    
    


