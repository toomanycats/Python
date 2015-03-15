'''
Created on Oct 14, 2013

@author: dpc
'''


from optparse import OptionParser
import logging
from os import path
from shutil import copyfile
import ImageTools
import MiscTools
import LoggingTools

log = logging.getLogger(__name__)
imagetools = ImageTools.ImageTools()
misctools = MiscTools.MiscTools()


def down_sample(target, reference, outfile):
        if outfile[-7:] == '.nii.gz':
            outfile = outfile[:-7]
            
        log.info('Running down_sample_mask.sh')
        cmd = 'flirt -in %(target)s -ref %(reference)s -out %(outfile)s -interp trilinear'
        cmd = cmd %{'target':target,
                    'reference':reference,
                    'outfile':outfile
                    }
        
        misctools.call_shell_program(cmd)

def make_mean_image(image1, image2, outfile):
    cmd = 'imagemath --in %(image1)s %(image2)s --average --out %(outfile)s'        
    cmd = cmd %{ 'image1':image1,
                 'image2':image2,
                 'outfile':outfile
                 }    
    
    misctools.call_shell_program(cmd) 
        
class SiemensUnWarp(object):
    
    def __init__(self, root_dir, rs_4D_data, mag1_image, mag2_image, phase_image, dwell_time):
        self.root_dir = root_dir
        
        self.deltaTE = 2.46 #s
        self.dwell =  dwell_time #275e-6  #s Oregon, for Pittsburg is U/K 
        
        #inputs
        self.mag1_image = mag1_image
        self.mag2_image = mag2_image
        self.phase_image = phase_image
        self.rs_4D_data = rs_4D_data    
        
        # outputs
        self.rs_4D_masked = self.rs_4D_data.replace('.nii.gz','_masked.nii.gz')
        self.intensity_corr_vol = path.join(self.root_dir, 'intensity_correction_vol.nii.gz')
        self.shiftmap = path.join(self.root_dir, 'shiftmap.nii.gz')
        
        #intermediate files
        self.mean_mag =  path.join(self.root_dir, 'mean_mag.nii.gz')# gets overwritten when mask is applied
        self.mask_rs = path.join(self.root_dir, 'mask_rs.nii.gz')
        
        self.unwrapped_phase_map = path.join(self.root_dir, 'unwrapped_phase_map.nii.gz')
        self.unwrapped_phase_map_rs = path.join(self.root_dir, 'unwrapped_phase_map_rs.nii.gz')
        self.ones_vol = path.join(self.root_dir, 'ones.nii.gz')
           
    def main(self):
        make_mean_image(self.mag1_image, self.mag2_image, self.mean_mag)
        
        hi_res_mask = path.join(self.root_dir, 'hi_res_mask.nii.gz')
        log.info("Running BET on the mean mag, to make the mask_rs.")
        imagetools.make_bet_mask(self.root_dir, self.mean_mag, hi_res_mask)
        
        #make rs mask
        down_sample(hi_res_mask, self.rs_4D_data, self.mask_rs)   
        imagetools.threshold_mask( self.mask_rs, self.mask_rs)
        imagetools.binarize( self.mask_rs, self.mask_rs)
        imagetools.apply_mask( self.rs_4D_data, self.mask_rs, self.rs_4D_masked)
       
        log.info('Unwrapping the phase map.')  
        imagetools.apply_mask( self.mean_mag, hi_res_mask, self.mean_mag )      
        imagetools.run_fsl_prepare_field_map(self.root_dir, self.phase_image, self.mean_mag, self.unwrapped_phase_map, self.deltaTE )
        imagetools.apply_mask( self.unwrapped_phase_map, hi_res_mask, self.unwrapped_phase_map)    
        
        #make the ones vol for inten corr map
        imagetools.binarize(self.mask_rs, self.ones_vol)
        
        #down sample the unwrapped phase map
        down_sample(self.unwrapped_phase_map, self.rs_4D_data ,self.unwrapped_phase_map_rs)
        
        log.info("Calling Fugue to make shiftmap and inten corr map.")
        log.debug("dwell: %f" %self.dwell)
        imagetools.get_shiftmap_inten_corr_fugue(self.root_dir, self.unwrapped_phase_map_rs, self.dwell, self.shiftmap, self.ones_vol, self.intensity_corr_vol)         
                           

class GeUnWarp(object):
    ''' Class for getting the shiftmap and intensity correction map to
    fix epi images from GE scanners. '''
    def __init__(self, root_dir, rs_4D_data, real_image1, imag_image1, real_image2, imag_image2, mag_image1, mag_image2, dwell_time):
        self.dwell = dwell_time #388e-6  #Eva auditory  #198e-6  NCANDA resting state  #s
        #self.deltaTE = 2 #ms 
        
        self.root_dir = root_dir
        self.rs_4D_data = rs_4D_data
        
        #intermediate files
        self.real_image1 = path.join(self.root_dir, 'real_image1.nii.gz')
        self.imag_image1 = path.join(self.root_dir, 'imag_image1.nii.gz')
        
        self.real_image2 = path.join(self.root_dir, 'real_image2.nii.gz')
        self.imag_image2 = path.join(self.root_dir, 'imag_image2.nii.gz')
        
        self.mag_image1 = path.join(self.root_dir, 'mag_image1.nii.gz')
        self.mag_image2 = path.join(self.root_dir, 'mag_image2.nii.gz')
        self.mean_mag = path.join(self.root_dir, 'mean_mag.nii.gz')
        
        #copy files
        copyfile(real_image1, self.real_image1)
        copyfile(real_image2, self.real_image2)
        copyfile(imag_image1, self.imag_image1)
        copyfile(imag_image2, self.imag_image2)
        copyfile(mag_image1, self.mag_image1)
        copyfile(mag_image2, self.mag_image2)
        
        # reorient images into LAS
#         imagetools.reorient_image(real_image1, self.real_image1, orientation='LAS')   
#         imagetools.reorient_image(real_image2, self.real_image2, orientation='LAS')   
#         imagetools.reorient_image(imag_image1, self.imag_image1, orientation='LAS')   
#         imagetools.reorient_image(imag_image2, self.imag_image2, orientation='LAS')   
#         imagetools.reorient_image(mag_image1,  self.mag_image1,  orientation='LAS')   
#         imagetools.reorient_image(mag_image2,  self.mag_image2,  orientation='LAS')   
    
        # original orientation reference
        self.orientation_ref = mag_image1
    
        #processing intermediates created in the class
        self.mask = path.join(self.root_dir, 'mag_mask.nii.gz')
        self.mask_rs = path.join(self.root_dir, 'mag_mask_rs.nii.gz')
        
        self.ratio_imag = path.join(self.root_dir, 'ratio_imag.nii.gz')
        self.ratio_real = path.join(self.root_dir, 'ratio_real.nii.gz')
        self.ratio_complex = path.join(self.root_dir, 'ratio_complex.nii.gz')
    
        self.unwrapped_phase_map = path.join(self.root_dir, 'unwrapped_phase_map.nii.gz')
        self.unwrapped_phase_map_rs = path.join(self.root_dir, 'unwrapped_phase_map_rs.nii.gz')
        self.ones_vol = path.join(self.root_dir, 'ones.nii.gz')
        
        #outputs
        self.shiftmap = path.join(self.root_dir, 'shiftmap.nii.gz') 
        self.intensity_corr_vol = path.join(self.root_dir, 'intensity_correction_vol.nii.gz')
        
    def main(self):
        
        # mean the mag image
        make_mean_image(self.mag_image1, self.mag_image2, self.mean_mag)
        # make a mask of the mean mag
        imagetools.make_bet_mask(self.root_dir, self.mean_mag, self.mask)
        # mask the mean_mag
        imagetools.apply_mask( self.mean_mag, self.mask, self.mean_mag )      
        
        # take the ratio of the real and imag images 1 and 2
        self.complex_ratio()
        #combine  the ratio files into one complex image vol
        self.make_comp_valued_image()
        
        # prelude
        imagetools.call_prelude(self.root_dir, self.ratio_complex, self.mask, self.unwrapped_phase_map)
        
        # convert Hz to rad
        self.convert_Hz_to_rad()
        
        #make the ones vol for inten corr map
        down_sample(self.mask, self.rs_4D_data, self.mask_rs)
        imagetools.add_scalar(self.mask_rs, self.ones_vol, amt=100)
        imagetools.binarize( self.ones_vol, self.ones_vol)
    
        #down sample the unwrapped phase map
        down_sample(self.unwrapped_phase_map, self.rs_4D_data, self.unwrapped_phase_map_rs)
        
        # fugue
        imagetools.get_shiftmap_inten_corr_fugue(self.root_dir, self.unwrapped_phase_map_rs, self.dwell, self.shiftmap, self.ones_vol, self.intensity_corr_vol, 'y' )         
    
        #copy geom from mag1 to fieldmap
        rs = ImageTools.nipy.load_image(self.rs_4D_data)
        shift = ImageTools.nipy.load_image(self.shiftmap)
        imagetools.save_new_image(shift.get_data(), self.shiftmap, rs.coordmap)
        
        inten = ImageTools.nipy.load_image(self.intensity_corr_vol)
        imagetools.save_new_image(inten.get_data(), self.intensity_corr_vol, rs.coordmap)
        
        #imagetools.copy_geometry(self.rs_4D_data, self.shiftmap)   
        #imagetools.copy_geometry(self.rs_4D_data, self.intensity_corr_vol)   
    
    def complex_ratio(self):
        #cmtk imagemath --in ${tmpdir}/r1.nii ${tmpdir}/i1.nii ${tmpdir}/r2.nii ${tmpdir}/i2.nii --complex-div --out ${tmpdir}/fmap_i.nii --pop --out ${tmpdir}/fmap_r.nii
        cmd = 'imagemath --in %(real1)s %(imag1)s %(real2)s %(imag2)s --complex-div --out %(outfile_i)s --pop --out %(outfile_r)s'
        cmd = cmd %{'real1':self.real_image1,
                    'imag1':self.imag_image1,
                    'real2':self.real_image2,
                    'imag2':self.imag_image2,
                    'outfile_i':self.ratio_imag,
                    'outfile_r':self.ratio_real
                    }
        
        misctools.call_shell_program(cmd)
        
    def make_comp_valued_image(self):
        #fsl fslcomplex -complex ${tmpdir}/fmap_r.nii ${tmpdir}/fmap_i.nii ${tmpdir}/fmap_c.nii
         
        cmd = 'fslcomplex -complex %(field_map_real)s %(field_map_imag)s %(outfile)s'
        cmd = cmd %{ 'field_map_real':self.ratio_real,
                     'field_map_imag':self.ratio_imag,
                     'outfile':self.ratio_complex
                     }    
         
        misctools.call_shell_program(cmd)
    
    def convert_Hz_to_rad(self):  
        ''' convert unwrapped phase from radians to Hz '''
        #cmtk imagemath --in ${tmpdir}/unwrap.nii --scalar-mul 500 --scalar-mul -0.159154943 --out ${tmpdir}/fieldmap.nii
        cmd = '''#!/usr/bin/bash
imagemath --in %(unwrapped_phase_map)s --scalar-mul 500 --scalar-mul -0.159154943  --out %(outfile)s 
'''    
        cmd = cmd %{'unwrapped_phase_map':self.unwrapped_phase_map,
                    'outfile':self.unwrapped_phase_map
                    }
        
        misctools.call_shell_program(cmd)

if __name__ == "__main__":
    parser = OptionParser()
    #choose scanner for logic
    parser.add_option('-s', '--scanner', dest='scanner', help='The scanner choice will setup the proper command line args. GE or SIEMENS')
    parser.add_option("-d", "--root-dir", dest="root_dir", help="Path where this step is performed.")
    parser.add_option("-t", "--time-series", dest="rs_4D_path", help="Path to aligned rs data in 4D format.")
    parser.add_option("-r", "--real1", dest="real1", default=None, help="The first real valued vol containing field information.") 
    parser.add_option("-R", "--real2", dest="real2", default=None, help="The second real valued vol containing field information.") 
    parser.add_option("-i", "--imag1", dest="imag1", default=None, help="The first imaginary valued vol containing field information.") 
    parser.add_option("-I", "--imag2", dest="imag2", default=None, help="The second imaginary valued vol containing field information.") 
    parser.add_option("-m", "--mag1", dest="mag1", help="The first magnitude vol created from field vols..") 
    parser.add_option("-M", "--mag2", dest="mag2", help="The second magnitude vol created from field vols..") 
    parser.add_option("-p", "--phase", dest="phase", default=None, help="Path to phase image.")
    parser.add_option("-w", "--dwell_time", dest="dwell", default=None, help="Dwell time from dicom:dcmdump I0001.dcm | grep -i effective")
     
    (options, args) = parser.parse_args()
    log = LoggingTools.SetupLogger(options.root_dir).get_logger()   
    
    if options.scanner == 'SIEMENS':    
        unwarp = SiemensUnWarp(options.root_dir, options.rs_4D_path, options.mag1, options.mag2, options.phase, options.dwell)
        unwarp.main()        
        
    elif options.scanner == 'GE':     
        unwarp = GeUnWarp(options.root_dir, options.rs_4D_path, options.real1, options.real1,
                           options.imag1, options.imag2, options.mag1, options.mag2, options.dwell)
        unwarp.main() 
         
    else:
        raise Exception, "The options are only SIEMENS or GE, you entered: %s" %options.scanner 

    # apply the unwarping to the 4D file
    time_series_file = unwarp.rs_4D_data
    shiftmap = unwarp.shiftmap
    inten_corr_vol = unwarp.intensity_corr_vol
    outfile = path.join(unwarp.root_dir, 'unwarped_4D_rs.nii.gz')
    root_dir = unwarp.root_dir
    
    imagetools.unwarp_epi(time_series_file, shiftmap, inten_corr_vol, outfile, root_dir)    
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        