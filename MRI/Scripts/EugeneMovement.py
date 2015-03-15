import numpy as np
import os
from os import path
import matplotlib.pyplot as plt


def rms(data):
    rms = np.sqrt(np.mean(data**2, axis=0))
    
    return rms

def distance(der):
    trans = der[:, 0:3]
    trans_rms = rms(trans)
    trans_dist = np.sqrt(np.sum(trans_rms**2))
    
    rots = der[:, 3:]
    rot_dist = 100 * rots # 100mm radius
    rot_rms = rms(rot_dist)
    rot_dist = np.sqrt(np.sum(rot_rms**2))
    
    dist = trans_dist + rot_dist
    
    return dist
    
def process(Path):
    data = np.loadtxt(Path)
    der = np.diff(data, axis=0)
    
    dist = distance(der)
    
    return dist


########################## BEGIN #######################
root_dir = "/fs/corpus6/dpc/EvaProjects/EugeneMovement"

C_group = 'ctrl_N26'  
E_group = 'etoh_N26'


####### C ##########
subs = os.listdir(path.join(root_dir, C_group))

C_subs = []

for sub in subs:
    C_subs.append(path.join(root_dir, C_group, sub, 'run1', 'rp_V001.txt'))
    C_subs.append(path.join(root_dir, C_group, sub, 'run2', 'rp_V001.txt'))


C_summary = np.zeros((len(C_subs)), dtype=np.float64)

for i, item in enumerate(C_subs):
    C_summary[i]  = process(item)
    

####### E ##########
subs = os.listdir(path.join(root_dir, E_group))

E_subs = []

for sub in subs:
    E_subs.append(path.join(root_dir, E_group, sub, 'run1', 'rp_V001.txt'))
    E_subs.append(path.join(root_dir, E_group, sub, 'run2', 'rp_V001.txt'))

E_summary = np.zeros((len(E_subs)), dtype=np.float64)

for i, item in enumerate(E_subs):
    dist = process(item)
    
    E_summary[i] = dist


np.savetxt(path.join(root_dir, 'C_group_report.txt'), C_summary)
np.savetxt(path.join(root_dir, 'E_group_report.txt'), E_summary)

plt.hist(C_summary, 20, label="C group")
plt.hist(E_summary, 20, alpha=0.5, label="E group")
 
plt.title("Histogram of RMS Translation Distance: bin size = 10")
plt.xlabel("Distance (mm)", fontsize=15)
plt.ylabel("Number of subjects in distance bin.", fontsize=15)
plt.legend(loc='upper right')
 
string_c = "C trans mean dist: %f\n" %C_summary.mean()
string_e = "E trans mean dist: %f" %E_summary.mean()
string = string_c + string_e
 
plt.text(0.80, 15, string, fontsize=13)
 
 

 
plt.show()





























