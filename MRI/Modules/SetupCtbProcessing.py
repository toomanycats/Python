import ImageTools
import StaticPaths
from os import path, makedirs, remove
from optparse import OptionParser
from shutil import copyfile
from PyConn import DetrendMovementNoise
from numpy import savetxt

imagetools = ImageTools.ImageTools()


# First scenario
# Take outputs from Step A ( aka Torstens pipeline) and 
# prepare them for entry into CTB.


class StepA(object):
    '''This class, is for taking the outputs from the Step A part of the resting state, pipeline
    , AKA "Torsten Pipeline", and gets them ready for analysis in the Conn Toolbox. '''
    
    def __init__(self, root_dir, sub_id, recompute_motion=True, time_point='base'):
        self.root_dir = root_dir
        self.sub_id = sub_id
        self.sub_dir = path.join(self.root_dir, self.sub_id)
        
        self.static_paths = StaticPaths.StaticPaths()
        self.static_paths.set_ncanda_full_paths(sub_id, time_point) 
    
        self.bold = path.join(self.sub_dir, 'bold_4d.nii.gz')
        self.bold_mean = path.join(self.sub_dir, 'bold_mean.nii.gz')

        self.motion_par = path.join(self.sub_dir, 'mcflirt_motion_par.txt')
       
        self.t1 = path.join(self.sub_dir, 't1_brain.nii.gz')
        self.wm = path.join(self.sub_dir, 'wm_ero_4mm.nii.gz')
        self.gm = path.join(self.sub_dir, 'gm_4mm.nii.gz')
        self.csf = path.join(self.sub_dir,'csf_4mm.nii.gz')
    
        self.inputs_exist = False
        
        self.archive_list = [self.bold, self.gm, self.wm, self.t1]    
    
    def test_existance_of_inputs(self):
        if path.exists(self.static_paths.rs_sri24_bold_stdy) and \
           path.exists(self.static_paths.t1_sri24_brain)     and \
           path.exists(self.static_paths.wm_core_sri24):
            
            self.inputs_exist = True
    
        else:
            raise Exception, "Missing inputs." 
    
    def main(self):
        
        self.test_existance_of_inputs()
        
        if self.inputs_exist:
            
            self.make_directories()
            
            self.copy_files()
            
            self.get_motion_params_wrt_first()
            
            self.check_and_truncate_bold_vol_num()
            
            self.make_tissue_maps()
            
            imagetools.spatial_smooth(infile=self.bold, 
                                      outfile=self.bold, 
                                      fwhm=4.0
                                      )
            
            self.archive_image_orientations()
            
            for image in self.archive_list:
                self.remove_orientation(image)
            
            gzip_list = imagetools.build_file_list(search_path=self.sub_dir, 
                                                   regex='.*\.nii\.gz', 
                                                   string=False # return list not cs string
                                                   )
            
            for File in gzip_list:
                imagetools.check_if_gzipped(File, unzip_if_true=True)
      
    def make_directories(self):
        
        if not path.exists(self.sub_dir):
            makedirs(self.sub_dir)
    
    def truncate_motion_par_file(self, num_drop):
        f = open(self.motion_par,'r')
        lines = f.readlines()
        f.close()
        
        lines = lines[num_drop:]
        
        temp_file = self.motion_par.replace('.txt','_trun.txt')
        new_f = open(temp_file,'w')
        
        for line in lines:
            new_f.write(line)
        
        new_f.close()
        
        copyfile(temp_file, self.motion_par)
        remove(temp_file)
    
    def check_and_truncate_bold_vol_num(self):
        num_vol = imagetools.get_number_of_vols(self.bold)
        
        if num_vol > 269:
            diff = num_vol - 269
            imagetools.fsl_roi(infile=self.bold, 
                               outfile=self.bold, 
                               front_drop=diff, 
                               end_drop=0
                               )
            self.truncate_motion_par_file(diff)
    
        elif num_vol == 269:
            return
        
        else:
            raise Exception, "Subject has number of volumes less than 269 which is weird."
    
    def copy_files(self):
        copyfile(self.static_paths.t1_sri24_brain, self.t1)
        
        copyfile(self.static_paths.rs_sri24_bold_mean, self.bold_mean)
        
        copyfile(self.static_paths.rs_sri24_bold_stdy, self.bold)
        
    def make_tissue_maps(self):
        try:
            temp = imagetools.mk_temp_file_name(suffix='.nii.gz')
                
            imagetools.get_single_tissue_from_seg(self.static_paths.t1_sri24_seg, 'gm',  temp)
            imagetools.erode_image_box(temp, temp, num_vox=2, dim=3)
      
            res = imagetools.get_pixdims(self.bold_mean)[1]
            self.down_sample_iso(temp, temp, res)
            
            imagetools.binarize(temp, self.gm)
           
            imagetools.get_single_tissue_from_seg(self.static_paths.t1_sri24_seg, 'wm',  temp)
            # inplace gm1mm
            imagetools.erode_image_box(temp, temp, num_vox=2, dim=3)
      
            self.down_sample_iso(temp, temp, res)
            
            imagetools.binarize(temp, self.wm)        
            
        except Exception, err:
            raise Exception, err
            
        finally:
            remove(temp)    
        
    def down_sample_iso(self, infile, outfile, res):
        cmd = "convertx --resample-exact %(res)s %(infile)s %(outfile)s"
        cmd = cmd %{'infile':infile,
                    'outfile':outfile,
                    'res':res
                    }
    
#         cmd = "flirt -in %(infile) -ref %(ref)s -o %(outfile) --applyisoxfm %(res)s"
#         cmd = cmd %{'infile':infile,
#                     'ref':infile,
#                     'outfile':outfile,
#                     'res':res
#                     }        
            
        imagetools.call_shell_program(cmd)        

    def remove_orientation(self, infile):
        cmd = "fslorient -deleteorient %s" %infile
        
        imagetools.call_shell_program(cmd) 
        
    def archive_image_orientations(self):
        output = path.join(self.sub_dir, 'orientation_archive.zip')
        
        zip_list = " ".join(self.archive_list)
        
        # -j junk paths, that is, don't create full paths in the archive
        cmd = "zip  -j %s  %s" %(output, zip_list)
        
        imagetools.call_shell_program(cmd)

    def get_motion_params_wrt_first(self):
        detrend = DetrendMovementNoise(root_dir=self.sub_dir,
                                       rs_4D_path='', 
                                       confound_cleaned_output='', 
                                       mask='',
                                       sub_id=self.sub_id, 
                                       time_point='base',
                                       Type='human', 
                                       wm_rs=None, 
                                       gm_rs=None, 
                                       csf_rs=None, 
                                       move_par_type=None
                                       )
        
        mats_path = detrend.get_native_motion_mats_first_vol()
        
        confounds = detrend.get_motion_f2f_params_mcflirt(mats_path)        
        
        savetxt(self.motion_par, confounds)
  
        
                
                
if __name__ ==  "__main__":
    parser = OptionParser()
    # root_dir, sub_id, time_point='base'
    parser.add_option("-s", "--sub-id", dest="sub_id",
                       help="Subject ID, like NCANDA_S00033")
    
    parser.add_option("-r", "--root-dir", dest="root_dir",
                       help="Root path where the experiment will be put.")
    
    parser.add_option("-t", "--time-point", dest="time_point", default='base',
                       help="Time point, such as 'base' or '1year' .")
    
    (options, args) = parser.parse_args()
    
    run = StepA(options.root_dir, options.sub_id, options.time_point)
    run.main()
       
 
    
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        