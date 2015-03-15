from MiscTools import MiscTools
import os
import re
import numpy as np
from time import sleep
import StaticPaths
try:
    import PySQL
    pysql = PySQL.PySQL()
except:
    pysql = ''
        
import traceback

class ClusterTools(MiscTools):
    def __init__(self):
        MiscTools.__init__(self)
        self.static_paths = StaticPaths.StaticPaths()        
             
    def _call_w_qsub(self, program, options):
        cmd = 'echo "hostname;source %(cluster_paths)s;%(program)s" | qsub  %(options)s'
        
        cmd = cmd %{'program':program,
                    'options':options,
                    'cluster_paths':self.static_paths.environment_variables_for_pypeline
                    }           
        
        # helpful when running in ipython, which is what I normally do
        print cmd
        
        self.call_shell_program(cmd, catch_warnings=False)
                  
    def qsub_10gb_mem(self, program, name=None, aux_option=''):

        if name is None:
            options = ' %(aux_option)s'
        else:
            options = " -N %(name)s %(aux_option)s " %{'name':name,
                                                       'aux_option':aux_option
                                                      }
    
        options += " -l vmem=10gb,nodes=1:ppn=2 "
    
        self._call_w_qsub(program, options)

    def qsub_generic(self, program, name=None, aux_option=''):
    
        if name is None:
            options = ' %(aux_option)s'
        else:
            options = " -N %(name)s %(aux_option)s " %{'name':name,
                                                       'aux_option':aux_option
                                                      }
            options += " -l vmem=4gb,nodes=1:ppn=1 "
            
        self._call_w_qsub(program, options)

    def run_FS_auto_recon_mask(self, root_dir, sub_list):
        program = "auto_recon_wrapper_one_sub.sh %(path)s %(sub)s"
    
        for i in range(sub_list.size):   
            program_out = program %{'path':root_dir, 
                                'sub':sub_list[i]
                                }
            
            print sub_list[i]
            sleep(5)
            
            self.qsub_8gb_mem(program_out)

        # path="$1/${sub_id}_head"
        # sub_id=$2
        # 
        # cd $path
        # check_last_op
        # 
        # echo "HOST:"
        # hostname
        # 
        # echo "SUB:"
        # echo $sub_id
        # 
        # echo -e "\nfirst recon call\n"
        # recon-all -sd $path -autorecon1 -noskullstrip -s $sub_id
        # check_last_op
        # 
        # echo -e "link 1\n"
        # cp ${sub_id}/mri/T1.mgz ${sub_id}/mri/brainmask.auto.mgz
        # check_last_op
        # 
        # echo -e "link 2\n"
        # cp  ${sub_id}/mri/brainmask.auto.mgz ${sub_id}/mri/brainmask.mgz
        # check_last_op
        # 
        # echo -e "calling autorecon2\n"
        # recon-all -sd $path -autorecon2 -autorecon3 -s $sub_id
        # check_last_op

 
    def setup_FS_auto_recon_mask(self, root_dir, t1_source_path, sub_list=None, start=None, end=None):
        '''if you want the t1 source path for ncanda, use: StaticPaths.StaticPaths.ncanda_base_line_t1'''
                    
        if sub_list is None:
            sub_list = self.generate_sub_list(t1_source_path)
        
        sub_list = np.array(sub_list) 
        sub_list.sort()
        
        t1_source_path.sort()
        
        if t1_source_path.size != sub_list.size:
            raise Warning, "sub list and T1 source list are not equal length"      
                  
        if start is None:
            start = 0
        
        if end is None:
            end = sub_list.size
        
        for i in range(start, end):
            sub = sub_list[i]                
            sub_dir = os.path.join(root_dir, sub)
            path = os.path.join(sub_dir, sub, 'mri', 'orig')
            print path
            os.makedirs(path)
        
            link_path = os.path.join(root_dir, sub)
            cmd = 'ln -s "${FREESURFER_HOME}/subjects/fsaverage"  %(link_path)s/fsaverage' 
            cmd = cmd %{'link_path':link_path}     
            self.call_shell_program(cmd)
              
            cmd = "mri_convert -i %(t1_source_path)s -o %(path)s/001.mgz"
            cmd = cmd %{'t1_source_path':t1_source_path[i],
                        'path':path
                        }
            self.call_shell_program(cmd)
  
        return sub_list
     
    def generate_FS_sub_list(self, t1_source_path):
        sub_list = []
        #try to parse the sub id from t1_source_path            
        re_obj = re.compile('(?P<site>sri|duke|ucsd|upmc|ohsu).*(?P<sub_id>NCANDA_S[0-9]{5})')
        for  entry in t1_source_path:
            sub = re_obj.search(os.path.basename(entry))
            if sub:
                sub_list.append(sub.group())
                
        sub_list =  np.array(sub_list)    
        sub_list.sort()
        
        return sub_list              
            
    def run_resting_state(self, sub_list, root_dir, physio=False, cluster=False, short=False, submit_delay=10):
        program = 'python /fs/cl10/dpc/CopyOfRepoForCluster/python/Pypelines/NCANDA_resting_state_pipeline.py  -r %(root_dir)s -t vox -c %(sub)s %(physio)s'
 
        if physio:
            physio = '-P'
        
        elif physio is False:
            physio = '-p'   
        
        else:
            raise Exception, "Not an understood physio option." 
 
        if short:
            options = ' -q short ' 
            
        else:
            options = ''          
   
        if cluster:
            for sub_id in sub_list:
                #self.validate_sub_id(sub_id)
                name = sub_id + '_rs'
                prog = program %{'sub':sub_id, 
                                 'physio':physio,
                                 'root_dir':root_dir
                                 }
                print prog
                
                self.qsub_10gb_mem(prog, name, options)
                sleep(submit_delay)              
            
        else:
            for sub_id in sub_list:
                try:
                    self.validate_sub_id(sub_id)
                    prog = program %{'sub':sub_id, 
                                     'physio':physio,
                                     'root_dir':root_dir
                                     }
                    print prog
                    self.call_shell_program(prog)    
                except:
                    traceback.print_exc()
                
    def run_ncanda_deg_map(self, path_list, root_dir):
            program = 'python /fs/cl10/dpc/CopyOfRepoForCluster/python/Scripts/NCANDA_resting_state_deg_map.py -r %(root_dir)s -s %(sub)s -i %(sub_id)s'
             
            for sub in path_list:
                sub_id = re.search('NCANDA_S[0-9]{5}', sub).group()
                name = sub_id + '_deg'
                prog = program %{'root_dir':root_dir,
                                 'sub':sub, 
                                 'sub_id':sub_id
                                 }
                print prog
                
                self.qsub_10gb_mem(prog, name)        
        
    def validate_sub_id(self, sub_id):
        known_list = pysql.get_500_sub_list()
        
        if sub_id not in known_list:
            raise ValueError, 'Not in database:%s' %sub_id    
        
    def run_ctb_setup_stepA(self, sub_ids, root_dir, cluster=False):
        program = "python /fs/cl10/dpc/CopyOfRepoForCluster/python/Modules/SetupCtbProcessing.py -r %(root_dir)s -s %(sub_id)s"
        
        for sub_id in sub_ids:
            prog = program %{'root_dir':root_dir,
                             'sub_id':sub_id
                             }
            
            if cluster:
            
                name = sub_id[9:] + '_CTB'
                self.qsub_generic(prog, name)
                
            else:
                try:
                    self.call_shell_program(prog, catch_errors=True, catch_warnings=False)
                except Exception:
                    traceback.print_exc() 
        
        
        
        
        
        
        
                        




