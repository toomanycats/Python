#!/usr/bin/python

from optparse import OptionParser
import PhysioCorrect
import LoggingTools
import sys
import traceback

def denoise():
    physio.denoise()     
    
def preprocess():
    physio.preprocess()    

if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("-p", "--physio-dir", dest="physio_dir", type=str, default=None,
                      help="Dir where the physio files are read from.Not required with 'denoise' stage.")
    
    parser.add_option("-n", "--num-drop", dest="num_vols_drop", type=int, default=0,
                      help="The number of fMRI vols that were dropped before this tool was run. Default is zero.")
    
    parser.add_option("-i", "--input", dest="fourD_file", type=str,
                      help="The full path of the input file, which needs to be a 4D nifti. Required for both stage.")    
            
    parser.add_option("-o", "--output", dest="out_path",
                      help="The dir of preprocessed physio data files and where the 3dretroicor script will be places. Used in both stages..") 
    
    parser.add_option("-d", "--denoised", dest="denoised_output", type=str,default=None,
                      help="Full path of the physio corrected 4D file.") 
    
    parser.add_option("-s", "--stage", dest="stage", type=str,
                  help="Stage is either 'pre' for preprocess, or 'denoise' to apply 3dretroicor.") 
    
    parser.add_option("-l", "--log_path", dest="log_path", type=str,
                  help="Full path to where you want the log file to be written.") 
    
    parser.add_option("-L", "--log_level", dest="log_level", type=str, default='info',
                  help="Logging level. Default is 'info'. Set to debug for trouble shooting.") 
    
    (options, args) = parser.parse_args()

    # physio_dir, num_vols_drop, rs_path, output_dir, denoised_out
    physio = PhysioCorrect.Main(options.physio_dir, options.num_vols_drop, options.fourD_file, 
                                options.out_path, options.denoised_output) 
    
    log = LoggingTools.SetupLogger(options.log_path, options.log_level).get_module_logger()
    log.info("Stage:%s" %options.stage)
    log.info("Number drop:%i" %options.num_vols_drop)
    
    try:
        if options.stage == 'pre':
            preprocess()
    
        elif options.stage == 'denoise':
            denoise()
        
        else:
            msg =  "Stage option not a valid choice:%  Choose either 'pre' or 'denoise'" %options.stage
            log.exception(msg)
            raise Exception, msg
    except:
        log.error("Physio Correct ended early with failure. Track back below:")
        msg = traceback.format_exc()
        log.error(msg)
        sys.exit(1)    
    
    
    
    
    
    
    
    
    
    
    
    
    
