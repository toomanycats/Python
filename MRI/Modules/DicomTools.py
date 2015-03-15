from os import path
import glob
import dicom
import numpy as np
import MiscTools

misc_tools = MiscTools.MiscTools()
# Seimens Mosiac slice timing from:
#       $Author: frederic $
#       $Date: 2013/09/09 13:38:56 $
#       $Id: getslicetimes,v 1.2 2013/09/09 13:38:56 frederic Exp $
#


class DicomTools(object):
    def __init__(self):
        pass
    
    def _get_slice_positions(self, the_plan):
        SiemensCSAHeader2 = the_plan[0x0029,0x1020].value
        start_position = SiemensCSAHeader2.find('### ASCCONV BEGIN ###')+len('### ASCCONV BEGIN ###')
        end_position = SiemensCSAHeader2.find('### ASCCONV END ###')
        InterestingSiemensHeader = SiemensCSAHeader2[start_position:end_position].splitlines()
        
        saglist = []
        tralist = []
        corlist = []
        
        for the_line in InterestingSiemensHeader[1:]:
            the_pair = the_line.split()
    
        if (the_pair[0][0:len('sSliceArray.asSlice')] == 'sSliceArray.asSlice'):
            if (the_pair[0][-len('sPosition.dSag'):] == 'sPosition.dSag'):
                saglist.append(the_pair[2])
        
        if (the_pair[0][-len('sPosition.dTra'):] == 'sPosition.dTra'):
            tralist.append(the_pair[2])
    
        if (the_pair[0][-len('sPosition.dCor'):] == 'sPosition.dCor'):
            corlist.append(the_pair[2])
        
        return saglist,tralist,corlist
    
    def _get_siemens_header(self, the_plan):
        SiemensCSAHeader2 = the_plan[0x0029,0x1020].value
        startposition = SiemensCSAHeader2.find('### ASCCONV BEGIN ###')+len('### ASCCONV BEGIN ###')
        endposition = SiemensCSAHeader2.find('### ASCCONV END ###')
        InterestingSiemensHeader = SiemensCSAHeader2[startposition:endposition].splitlines()
        datadict = {}
        
        for theline in InterestingSiemensHeader[1:]:
            thepair = theline.split()
            datadict[thepair[0]] = thepair[2]
        
        return datadict
    
    def _get_slice_times_mosaic_vol(self, file_vol_path):
        output_times_list = []
        
        plan = dicom.read_file(file_vol_path)
        timestr = float(plan.AcquisitionTime)
        thetr = float(plan.RepetitionTime)
        timeconv = 3600 * (int(timestr/10000)%100) + 60 * (int(timestr/100)%100) + timestr%100 + (timestr - int(timestr))
        acqnum = plan.AcquisitionNumber
        
        #thesiemensheader=_get_siemens_header(plan)
        #slicethickness=thesiemensheader['sSliceArray.asSlice[0].dThickness']
        #inplanerot=thesiemensheader['sSliceArray.asSlice[0].dInPlaneRot']
        
        sag_list, tra_list, cor_list = self._get_slice_positions(plan)
        
        num_slices = len(sag_list)
        locs = np.zeros((4, num_slices), dtype='float')
        locs[0,:] = tra_list
        locs[1,:] = sag_list
        locs[2,:] = cor_list
        
        try:
            mosaic_info = plan[0x0019,0x1029].value
        except KeyError:
            print 'key error getting mosaic_info'
        
        which_slice = 0
        for slicetime in mosaic_info:
            locs[3, which_slice] = (slicetime % thetr) / 1000.0
            which_slice = which_slice + 1
            output_times_list.append((slicetime % thetr) / 1000.0)
            
        return output_times_list    
            
    def generate_slice_times_files_4D(self, mosaics_directory, outfile_path, prefix='MR'):  
        pattern = path.join(mosaics_directory, '%s*' %prefix)
        mosaics = glob.glob(pattern)[0]
        
        if len(mosaics) < 10:
            raise Exception, "Less than ten mosiacs were found at :%s" %mosaics_directory
        
        slice_timings_list = []
        
        for vol in mosaics:
            slice_timings_list.append(self._get_slice_times_mosaic_vol(vol))
           
        out_list =  misc_tools.flatten_double_nested(slice_timings_list)      
        
        f = open(outfile_path, 'w')
        for time_info in out_list:
            f.write(time_info)
            
        f.close()
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
              
            
            
            
            
            
            
        