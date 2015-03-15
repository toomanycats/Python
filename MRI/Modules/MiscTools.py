import glob
import subprocess
import re
from os import path
import os
import logging
import numpy as np
import uuid #for random strints
import json

#log = logging.getLogger('module_logger')
log = logging.getLogger(__name__)

import inspect
from itertools import izip as itertools_izip
import decorator

def check_for_True_in_last_arg(*args):
    if len(args) > 0 and args[-1] is True:
        return True

@decorator.decorator
def log_method(func, *args, **kwargs):
    # allows me to put the wrapper in helper methods that are sometimes major steps
    if check_for_True_in_last_arg(*args):
        log.debug('''Last arg is "True" which could indicate the caller 
intends the log to work on this method. For that behaviour, use the keyword arg:  log_this=True''')

    if 'log_this' in kwargs and kwargs['log_this'] is not False: 
        args_name = inspect.getargspec(func)[0][1:] #drop the 'self' arg
        args_dict = dict(list(itertools_izip(args_name, args)) + list(kwargs.iteritems()))
        method_name = func.__name__
        args_dict_string = '\nMETHOD: ' + method_name + '\n'

        for k,v in args_dict.iteritems():
            if type(v) is not np.ndarray:
                args_dict_string += "%s: %s\n" %(k,v)

            elif type(v) is np.ndarray:
                args_dict_string += "NUMPY ARRAY:%s \nSHAPE:%s\n" %(k, str(v.shape))

        args_dict_string += '\n'
        log.info(args_dict_string)

    return func(*args, **kwargs) 



class MiscTools(object):
    def __init__(self):
        pass
 
    def mk_temp_file_name(self, dir='/tmp', suffix=''):
        random_string = str(uuid.uuid4()) + suffix
        return path.join( dir, random_string )
   
    @log_method
    def glob_for_files(self, root_dir, pattern, num_limit=1, log_this=False):
        files = glob.glob(path.join(root_dir, pattern))
        if len(files) == 0:
            log.debug("No files Found")
            return None
            #raise Exception, "No files founds with pattern: %s" %pattern
        elif len(files) > num_limit:
            raise Exception, "The number of files found were greater than the limit set: %i" %num_limit
        elif len(files) == 1 and num_limit ==1:
            return files[0]
        
        log.debug("File found with pattern: %s  In root dir: %s" %(pattern, root_dir))
        return files    

    def flatten(self, list_ob):
        if list_ob == []:
            return list_ob
        if isinstance(list_ob[0],list):
            return self.flatten(list_ob[0]) + self.flatten(list_ob[1:])
        return list_ob[:1] + self.flatten(list_ob[1:])
        
    def call_shell_program(self, cmd, catch_errors=True, catch_warnings=True):
        log.debug('command line used: %s' %cmd)
        
        process = subprocess.Popen(cmd, shell=True,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)

        out, err = process.communicate()
        errcode = process.returncode
        
        log.debug('Shell call output: %s' %out)
        
        warn_ob = re.compile('.*warn.*', flags=re.IGNORECASE | re.DOTALL)
        error_ob = re.compile('.*error.*', flags=re.I | re.DOTALL)
        stop_ob = re.compile('.*STOPPING PROGRAM.*', flags=re.I)
        
        if catch_errors  and (errcode or error_ob.match(err)): 
            print err
            log.debug('Shell call error mess: %s' %err)
            raise Exception, "%s %s" %(err, errcode)
        
        elif catch_warnings  and  ( warn_ob.match(err) or stop_ob.match(err) or error_ob.match(err) ):
            print err
            log.debug('Shell call warning mess: %s' %err)
            raise Exception, "%s %s" %(err, errcode)
        
        return out
    
    def write_shell_script(self, prog_name, template):
        f = open(prog_name, 'w')
        f.write(template)
        f.close()
        os.chmod(prog_name, 0755)

    @log_method
    def ungzip(self, File, force=False, log_this=False):
        split_list = path.splitext(File)
        if split_list[-1] != '.gz':
            return File
        
        if force is True:
            cmd = 'gunzip -f %s' %File
        else:
            cmd = 'gunzip %s' %File
        
        self.call_shell_program(cmd)
        return split_list[0]

    @log_method
    def gzip(self, File, log_this=False):
        cmd = 'gzip %s' %File
        self.call_shell_program(cmd)
        
        return File + '.gz'

    def which(self, name, flags=os.X_OK):
        """
        http://twistedmatrix.com/trac/browser/tags/releases/twisted-8.2.0/twisted/python/procutils.py
        
        Search PATH for executable files with the given name.
       
        On newer versions of MS-Windows, the PATHEXT environment variable will be
        set to the list of file extensions for files considered executable. This
        will normally include things like ".EXE". This fuction will also find files
        with the given name ending with any of these extensions.
    
        On MS-Windows the only flag that has any meaning is os.F_OK. Any other
        flags will be ignored.
       
        @type name: C{str}
        @param name: The name for which to search.
       
        @type flags: C{int}
        @param flags: Arguments to L{os.access}.
       
        @rtype: C{list}
        @param: A list of the full paths to files found, in the
        order in which they were found.
        """
        result = []
        exts = filter(None, os.environ.get('PATHEXT', '').split(os.pathsep))
        path = os.environ.get('PATH', None)
        if path is None:
            return []
        for pos in os.environ.get('PATH', '').split(os.pathsep):
            pos = os.path.join(pos, name)
            if os.access(pos, flags):
                result.append(pos)
            for e in exts:
                pext = pos + e
                if os.access(pext, flags):
                    result.append(pext)
        
        if len(result) == 0:
            raise Exception, "Program not found: %s" %name
        elif len(result) > 0:
            result = result[0]
        
        return result        

    def build_file_list(self, search_path, regex, string=True):
        out = []
        re_obj = re.compile(regex, flags=re.DOTALL)
                        
        files= os.listdir(search_path)
        
        for f in files:
            if re_obj.search(f):
                out.append(os.path.join(search_path, f))
            
        if string:
            out = " ".join(out)
        
        return out

    def check_if_gzipped(self, infile, unzip_if_true=True):
            
        if infile.endswith('.gz'):
            if unzip_if_true is True:
                infile = self.ungzip(infile) 
                return True, infile
            else:
                return True
                
        else:
            return False, infile
    
    @log_method    
    def untar_to_dir(self, tarfile, out_dir):
        cmd = "tar xvf  %(tarfile)s   --directory %(out_dir)s"

        cmd = cmd %{'tarfile':tarfile,
                    'out_dir':out_dir
                    } 
        
        self.call_shell_program(cmd)    

    def find_files(self, root, name, Type='f', mindepth=None, maxdepth=None):
        if mindepth is None and maxdepth is None:
            options = ''
         
        elif mindepth is not None:
            options = ' -mindepth %s ' %mindepth
            
        elif maxdepth is not None:
            options = ' -maxdepth %s ' %maxdepth
            
        else:
            options = " -mindepth %s -maxdepth %s  " %(mindepth, maxdepth)             
        
        if Type == 'f':
            Type = '-type f '
        
        elif Type == 'd':
            Type = '-type d '    
        
        else:
            raise Exception, "Type not understood"
        
        cmd = 'find %(root)s  %(depth)s %(type)s -name "%(name)s"'
        
        cmd = cmd %{"root":root,
                    "depth":options,
                    "name":name,
                    "type":Type
                    }
    
        out = self.call_shell_program(cmd)
        out = out.split('\n')
        out = np.array(out)
        out.sort()
        
        return out[1:]#first list item is '' b/c of split
    
    def load_dict_from_json(self, file_path):
            f = open(file_path, 'r')
            new_dict = json.load(f)
            f.close()
            
            return new_dict
    
    def save_dict_as_json(self, dict_obj, file_path):
        f = open(file_path, 'w')
        json.dump(dict_obj, f)
        f.close()

    def call_matlab_program(self, mfile):
        cmd = '/fs/p00/matlabr14sp3/bin/matlab -nodesktop -nojvm  -nosplash -r %s' %mfile
        
        self.call_shell_program(cmd, catch_errors=True, catch_warnings=False)        
        
        
             
        