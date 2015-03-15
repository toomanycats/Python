#!/usr/bin/python


'''
Created on Dec 16, 2013

@author: dpc
'''

from optparse import OptionParser
import glob
import filecmp
from os import path, walk, remove
from MiscTools import MiscTools
from time import sleep
import collections
import datetime
import PyXnatTools
import re

pyx = PyXnatTools.PyXnatTools()
misctools = MiscTools()

class Archive(object):
    def __init__(self, archive_source, temp_iso_dir, max_size, device):
        self.archive_souce = archive_source
        self.iso_path = temp_iso_dir
        self.test_mount_point = '/media/dvdrom'
        self.device = device
        self.max_size = max_size
        self.all_zip_size = 0
        self.all_zip_paths = []
        
        self.archive_group = []
        self.burn_speed = 0
    
        root = '/fs/u00/dpc/BartLaneArchiveRecords'
        today = datetime.date.today()
        filename = "%s.txt" %today.strftime('%Y-%m-%-d-%H:%M')
        self.arhcive_record_file = path.join(root, filename)
    
    def make_list(self, basename=False):
        file_list = ''
        
        if basename is False:
            for f in self.archive_group:
                file_list = file_list + '"%s"'%f + ' '
        
        elif basename is True:
            for f in self.archive_group:
                file_list = file_list + '"%s"'%path.basename(f) + ' ' 
        
        return file_list            

    def validate_only(self):
        self.get_all_zip_file_paths(self.archive_souce)
        self.choose_subset_of_zipfiles()
        self.mount_DVD()
        self.validate_files()
        sorted_pair_list = self.make_zip_path_sub_id_pair_list_and_sort(self.archive_group)
        self.keep_record_of_subjects_archived(sorted_pair_list)
        self.remove_arch_files_from_source()
        self.remove_disc()   
    
    def dry_run(self):
        self.get_all_zip_file_paths(self.archive_souce)   
        self.check_source_for_duplicates()
        
        for f in self.all_zip_paths:
            print f
    
        sorted_pair_list = self.make_zip_path_sub_id_pair_list_and_sort(self.all_zip_paths)
       
        for item in sorted_pair_list:
            print "%s:%s" %(item[0], item[1])
    
    def main(self):

        self.load_disc()
        
        self.get_all_zip_file_paths(self.archive_souce)
                
        self.get_source_size()
        
        self.choose_subset_of_zipfiles()
        
        self.check_source_for_duplicates()
        
        for f in self.archive_group:
            print f
        
        self.make_iso_container()
       
        self.burn_medium()
      
        self.mount_DVD()
       
        self.validate_files()
        
        sorted_pair_list = self.make_zip_path_sub_id_pair_list_and_sort(self.archive_group)
        
        self.keep_record_of_subjects_archived(sorted_pair_list)
        
        self.remove_disc()
        
        self.remove_arch_files_from_source()
        
        self.report_how_many_files_left()
    
    def load_disc(self):
        cmd = 'eject %(device)s' %{'device':self.device}
        misctools.call_shell_program(cmd)
        
        raw_input("load the blank disc...NOW! Then press RETURN")
        
        cmd = 'eject -t %(device)s' %{'device':self.device}
        misctools.call_shell_program(cmd)

    def remove_disc(self):
        cmd = 'eject'
        misctools.call_shell_program(cmd)
        
        raw_input("Remove disc...Then press ENTER")
        
        cmd = 'eject -t'
        misctools.call_shell_program(cmd)
        
    def validate_files(self):
        print "validating files on disc"
        
        new = sorted(glob.glob(path.join(self.test_mount_point, '*.zip')), key=lambda item: item.split('/')[-1])
        self.archive_group.sort(key=lambda item: item.split('/')[-1])
        
        if len(new) != len(self.archive_group):
            raise Exception, "number of new files on disc do not match length of archive group"
        
        for i in range(len(self.archive_group)):
            if not filecmp.cmp(new[i], self.archive_group[i]):
                raise Exception, "File comparison failed: %s and %s" % (self.archive_group[i], new[i])
    
    def get_all_zip_file_paths(self, start_path):
        print "Getting list of all files in source."
        
        for dirpath, _, filenames in walk(start_path):
            for f in filenames:
                self.all_zip_paths.append(path.join(dirpath, f))
                
        if len(self.all_zip_paths) == 0:
            raise Exception, "There are no files in the source dir: %s" %self.archive_souce        
  
    def get_source_size(self):
        print "computing size of all files in source."
        
        for f in self.all_zip_paths:
            self.all_zip_size += path.getsize(f)
    
    def choose_subset_of_zipfiles(self):
        print "Choosing subset of files to burn."
        
        aggregate = 0
        for f in self.all_zip_paths:
            aggregate += path.getsize(f)
            if aggregate < self.max_size:
                self.archive_group.append(f)
                
            else:
                self.archive_group
                return    

    def make_iso_container(self):
        print "Making an iso container."
        
        file_list = self.make_list()
        
        cmd = "genisoimage -file-mode 777  -J -hfs -o %(iso_path)s %(file_list)s"
        cmd = cmd %{'iso_path':self.iso_path,
                    'file_list':file_list
                    }        
            
        misctools.call_shell_program(cmd, catch_errors=True, catch_warnings=False)    
        
    def burn_medium(self):
        print "Burning iso to disc."
        
        cmd = "dvdrecord -data speed=%(speed)i -tao %(iso_file)s"
        cmd = cmd %{'iso_file':self.iso_path,
                    'speed':self.burn_speed
                    }        
            
        misctools.call_shell_program(cmd, catch_warnings=False)           
    
    def mount_DVD(self):    
        print "Mounting DVD for validation."
        
        cmd = "eject %(device)s && sleep 1 && eject -t %(device)s"
        cmd = cmd %{'device':self.device}
        
        misctools.call_shell_program(cmd)
        
        sleep(30)
    
        cmd = "mount | grep -i %(device)s" %{'device':self.device}
        misctools.call_shell_program(cmd, catch_errors=False)  
        cmd = "echo $?"
        std_out = misctools.call_shell_program(cmd)
        
        if std_out != '0':
            sleep(20)
            cmd = "mount /media/dvdrom"
            misctools.call_shell_program(cmd, catch_errors=False)   
        
    def remove_arch_files_from_source(self):
        print "Removing files from source."
       
        for f in self.archive_group:
            print "%s \n" %f
            remove(f)
            
    def check_source_for_duplicates(self):
        fnames = [ item.split('/')[-1] for item in self.archive_group ]
        dupes = []
        i = 0 # counter to link back to full path in self.archive_group
        for item, count in collections.Counter(fnames).items():
            if count > 1:
                print "Duplicate File: %s  Count: %i \n" % (item, count) 
                dupes.append(self.archive_group[i])
            i += 1
        
        if len(dupes) > 0:
            response = raw_input('Remove these files ? (y/n)')
        
            if response.lower() == 'y':
                for file in dupes:
                    pass #remove(file)               
            elif response.lower() == 'n':
                raise Exception, "Program terminated, duplicate items not removed."
            else:
                raise Exception, "Choice not understood."
        
    def report_how_many_files_left(self):
        remaining = len(self.all_zip_paths) - len(self.archive_group)
        print "There are:%i  file remaning to archive in the next run." %remaining

    def _get_sub_id_for_zip_path(self, zip_path):
        # /fs/ncanda-share/burn2dvd/2014-10-16/C-70087-M-8-20140917.zip
        name = path.basename(zip_path)
        pattern = '([A-E]{1}-[0-9]{5}-(M|F)-[0-9]{1})-[0-9]{8}.zip'
        re_obj = re.compile(pattern)
        
        match = re_obj.match(name)
        if match:
            sub_label_trun = match.group(1)
            sub_id =  pyx.get_subject_ids_like(sub_label_trun)
        
        else:
            sub_id = "sub_id not found"
        
        return sub_id    
    
    def make_zip_path_sub_id_pair_list_and_sort(self, list_of_archives):
        # build pairs of zip file path ( included the sub label )
        # and match it with the sub id ( aka 'S number' ), like NCANDA_S00042
        zip_path_id_pair = []
        
        for zip_path in list_of_archives:
            sub_id = self._get_sub_id_for_zip_path(zip_path)
            zip_path_id_pair.append((zip_path, sub_id))
            
        sorted(zip_path_id_pair, key=lambda x: x[1])    
        
        return zip_path_id_pair 
    
    def keep_record_of_subjects_archived(self, zip_path_id_pair):
                
        f = open(self.arhcive_record_file, 'a')# append just in case
        for tup in zip_path_id_pair:
            f.write("%s\n:%s\n" %(tup[0], tup[1]))
        
        f.close()
        
        print "list of subjects just archived, written to file:%s" %self.arhcive_record_file   
            
if __name__ == "__main__":
    parser = OptionParser()
    
    parser.add_option("-v", "--validate", dest="validate_only", action='store_true',
                      help="Run data validation only. Arg should be 'True'")
    
    parser.add_option("-s", "--archive-src", dest="archive_src", default='/fs/ncanda-share/burn2dvd/',
                      help="Path where the zipped images are stored.")

    parser.add_option("-t", "--temp-iso", dest="iso_path", default = '/var/tmp/image.iso',
                      help="Location of the temp iso file. Default is /tmp (RAM)")
    
    parser.add_option("-S", "--disc-size", dest="max_size", default=4e9, type='float',
                      help="Maximum size of the iso. Default is 4 GB.")

    parser.add_option("-d", "--device", dest="device", default='/dev/sr0',
                      help="Location of the DVD ROM drive. Default is /dev/sr0.")

    parser.add_option("--dry-run", dest="dry_run", action='store_true',
                      help="Gather the list of all files to be archived and print to screen.")
    
    (options, args) = parser.parse_args()
    
    if options.validate_only:
        Archive(options.archive_src, 
            options.iso_path, 
            options.max_size,
            options.device
            ).validate_only()
    
    elif options.dry_run:
        Archive(options.archive_src, 
            options.iso_path, 
            options.max_size,
            options.device
            ).dry_run()
        
    else:
        Archive(options.archive_src, 
                options.iso_path, 
                options.max_size,
                options.device
                ).main()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            