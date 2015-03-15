import numpy as np
import InterVolMovDetection
from os.path import join

p = '/data/dpc/motion_control_rsfmri/Vols'
tools =  InterVolMovDetection.InterVolMovDet(p)

vol_list = tools.get_vol_list()


slice_diff_vec = tools.compute_one_subject(vol_list)

bad_slice_ind, mse_norm = tools.look_for_outliers_in_slice(slice_diff_vec)

#outfile = join('/data/dpc/motion_control_rsfmri', 'bad_motion_test')
#np.save(outfile, mse_norm)

print bad_slice_ind