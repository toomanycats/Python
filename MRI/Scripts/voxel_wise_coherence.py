from os import path
import numpy as np
from matplotlib import mlab
from multiprocessing import Pool
from itertools import product
from ImageTools import ImageTools
from optparse import OptionParser


def comp_coh(arg):
    row1 = arg[0]
    row2 = arg[1]
    mag, f = mlab.cohere(roi[row1,:], roi[row2,:], NFFT=64, Fs=sample_rate, noverlap=32, sides='onesided')
    ind = np.where((f > 0.009) & (f < 0.09))[0]
    coh = np.sqrt(2*mag[ind].mean())    
#    coh = np.sqrt(2*mag.mean())
    
    return coh    

### starts here ###
parser = OptionParser()

parser.add_option("-m", "--mask-path", dest="mask_path",
                  help="Path to mask.")

parser.add_option("-r", "--rs-path", dest="rs_path",
                  help="Path to rs data.")

parser.add_option("-o", "--output", dest="output")

parser.add_option("-c", "--num-cores", dest="num_cores", type=int)

(options, args) = parser.parse_args()
mask_path = options.mask_path 
rs_path = options.rs_path

imagetools = ImageTools()

TR = 2.2
sample_rate = 1 / TR

roi, ind, shape, coord = imagetools.get_roi(rs_path, mask_path)

length = roi.shape[0]    

# setup up row and col vector of indices
indices = np.arange(length)

args = product(indices, indices)# all possible combinations iterator

# Multi core
num_cores = options.num_cores
pool = Pool(num_cores)
coh = np.hstack(pool.map(comp_coh, args))

# coh = coh.ravel()

out_path = path.join(options.output)
np.save(out_path, coh)

    


























