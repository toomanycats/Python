import os
import numpy as np
import ImageTools
from operator import itemgetter
from itertools import groupby

imagetools = ImageTools.ImageTools()

def get_contiguous(chunk):
    out = []
    for _, g in groupby(enumerate(chunk), lambda (i,x):i-x):
        out.append( map(itemgetter(1), g) )
       
    return out    

root = '/fs/cl10/dpc/Data/Test'

rem_files = imagetools.find_files(root, name='removed_ind.txt', maxdepth=2)

print "sub_id : our num of removed: 4min ? : # of motion > 0.2"

for f in rem_files[1:]:
    try:
        sub_id = f.split('/')[6]
        
        f = os.path.join(root, sub_id, 'removed_ind.txt')
        rem_ind = np.loadtxt(f, int) 
        
        ind = np.arange(260)
        ind = np.delete(ind, rem_ind)
        
        ind = get_contiguous(ind)
        for i in ind:
            if len(i) > 109:
                len_bool = 'True'
                break
            else:
                len_bool = 'False'
        
        
        traj_file = os.path.join(root, sub_id, 'motion_traj.txt')
        traj = np.loadtxt(traj_file, float)
        ind = np.where(traj > 0.2)[0]
        
            
        print sub_id, rem_ind.size, len_bool, ind.size    
    
    except:
        print sub_id, "Had to skip :( "
            
    
    