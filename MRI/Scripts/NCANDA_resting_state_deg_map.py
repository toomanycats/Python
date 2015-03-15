import NCANDA_resting_state_pipeline
from optparse import OptionParser

def main(root_dir, infile, sub_id):
        
    vox = NCANDA_resting_state_pipeline.ProcessVox(sub_id, root_dir, infile)
    vox.main()
            
        # mask is found by program
if __name__ ==  "__main__":
    parser = OptionParser()
    
    parser.add_option("-r", "--root-dir", dest="root_dir", 
                       help="Path where the experiment will be put.")
    
    parser.add_option("-s", "--subject_path", dest="infile")
                       
    parser.add_option("-i", "--sub_id", dest="sub_id")                   
    
    (options, args) = parser.parse_args()
    
    main(options.root_dir, options.infile, options.sub_id)



     
      
        