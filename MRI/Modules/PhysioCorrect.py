'''
Created on Sep 20, 2013

@author: dpc
'''

import re
import os
import numpy as np
import tempfile
from shutil import rmtree
import gzip
import logging

from scipy.signal import decimate
from scipy.signal import welch

import ImageTools
import SignalProcessTools
import MiscTools

log_method = MiscTools.log_method

log = logging.getLogger('module_logger') 
sig = SignalProcessTools.SignalProcessTools()   
imagetools = ImageTools.ImageTools()

class PreprocessBase(object):
    '''This base class setsup the directories and common variables used in all site verions. '''
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):

        self.physio_dir = physio_dir
        self.TR = self.get_TR(rs_path)
        self.rs_path = rs_path
        self.validate_trigger = False   
        self.output_dir = output_dir        
        
        self.trun_resp_path = os.path.join(self.output_dir, 'resp_filtered.txt')   
        self.trun_card_path = os.path.join(self.output_dir, 'card_filtered.txt')  

        # keep a copy here for other prog which might want to know this path
        self.trun_card_path_trig = self.trun_card_path.replace('.txt', '_trig.txt')
        
        self.num_vols_drop = num_vols_drop
        
        if num_vols_drop is not None:
            log.info('num vols worth of physio data, dropped: %i' %self.num_vols_drop)

        self.file_list_dict = file_list_dict
         
    def truncate_data(self, data, samp_rate, exp_len):
        #first I get rid of dummy scans, just make the data the right length for the full 274/275 vols
        data_length = data.size / samp_rate #seconds
        log.info('data length in seconds: %i' %data_length)
        remove = data_length - exp_len
        log.info('removing time from beginning of physio: %i' %remove)
        
        if remove < 0:
            log.exception("Length of physio data shorter than experiment length")
            raise Exception, "Length of physio data shorter than experiment length"
        
        index = samp_rate * remove
        log.info('number of points removed to make original num of vols and physio correct %i '%remove)
        #second, further reduce the length of the physio, b/c of dropped vols for steady state
        additional_drop_pts = samp_rate * self.num_vols_drop * 2.2
        log.info('Final number of points dropped from physio: %i' %index)
        index = index + additional_drop_pts
        trun_data = data[index+1:] 
              
        return trun_data   

    def save_data(self, fname, data):
        np.savetxt(fname, data)     
  
    def get_win(self, sample_rate, Type):
        if Type == 'card':
            time = 0.88
         
        elif Type == 'resp':
            time = 1.5    
        
        else:
            raise Exception, "Not a valid type."
        
        T = 1.0 / sample_rate
        win = int(time / T)
        ## keep the win always an odd int, so that win-1 = even
        ## this helps later when convoling and fixing the phase shift
        if np.mod(win, 2) == 0:
            win += 1
        
        log.debug('filter window in points: %i \n sample_rate: %i' %(win, sample_rate))
        
        return win      

    def bandpass(self, Type, sample_rate, data, order):
            #bandpass filter 
            if Type == 'card':
                lowcut = 0.75
                hicut  = 2.0
             
            elif Type == 'resp':
                lowcut = 0.166 # 6 sec period
                hicut  = 0.5   # 2 sec period 
                
            else:
                log.exception("Not a recognized type.")
                raise Exception, "Not a recognized type."
                
            out = sig.butter_bandpass(lowcut, hicut, sample_rate, data, order=order)
            
            return out
    
    def pre_trigger_filter(self, sample_rate, data, Type):
        '''This method smooths, bandpasses, and decimates the signal before
         using it to create the trigger file. '''
        # smoothing with gaussian kernel, adds phase shift = kernel size/2
        win = self.get_win(sample_rate, Type)
        log.info('Window length (pts) for %s kernel smooth filter: %i' %(Type, win))
        
        data_smoothed = sig.smooth(data, win, window='hanning', mode='valid')
        # removing extra points added by convolution filter
        
        amt = np.floor(data_smoothed.size / (win - 1.0))
        rem_ind = np.arange(0, data_smoothed.size, amt) 
        data_smoothed = np.delete(data_smoothed, rem_ind)
        
        delta_shape = data_smoothed.shape[0] - data.shape[0]
        
        if delta_shape < 0 :
            pad = np.zeros(np.abs(delta_shape))
            data_smoothed = np.concatenate((data_smoothed, pad), axis=0)
        
        elif delta_shape > 0:
            data_smoothed = data_smoothed[0:-delta_shape]
        
        shift_pts = int(np.around((win - 1) / 2.0, decimals=0)) + 2
        shift_pad = np.zeros(shift_pts)
        data_smoothed = data_smoothed[shift_pts:]
        data_smoothed = np.concatenate((data_smoothed, shift_pad), axis=0)
        
        data_filtered = self.bandpass(Type, sample_rate, data_smoothed, 2)
#        data_filtered = self.down_sample_ts(data_filtered, orig_sample_rate=sample_rate)

        return data_filtered

    def make_trigger(self, sample_rate, data):
            
            data_filtered = self.pre_trigger_filter(sample_rate, data, Type='card')
            #np.save('/fs/corpus6/dpc/python_conn/Human/Test1/card_fil.npy', data_filtered)
            
            trigger = self._create_trigger(data_filtered)
            if self.validate_trigger:
                self.validate_created_trigger(sample_rate, trigger)
                   
            self.trun_card_path = self.trun_card_path_trig
            self.save_data(self.trun_card_path, trigger)
            self.file_list_dict['card_trig'] = self.trun_card_path
            
    def validate_data_fft(self, Type, data, sample_rate):
        data -= data.mean()# center
        
        #smooth first 
        win = self.get_win(sample_rate, Type)
        data = sig.smooth(data, win)
         
        # truncate long data to 2**x for fft
        power = np.floor(np.log2(data.shape[0]))
        data = data[0:2**power]

        # PSD estimate
        freq, power = welch(data, fs=sample_rate, window='hanning', nperseg=2*win, noverlap=win, 
                            detrend='linear', scaling='spectrum', return_onesided=True )
        
        max_ind = np.where(power == power.max())[0]

        # test the range of acceptable freq
        if Type == 'card':
            if freq[max_ind] < 2.0  and freq[max_ind] > 0.75: #noise could have a max in the proper range, but gaussian noise would be low power
                    log.info('Valid card data. freq @ peak power: %f' %freq[max_ind])
                    self.file_list_dict['card'] = self.trun_card_path
            
            else:
                self.file_list_dict['card'] = None
                log.info('card failed validation. freq @ peak power: %f' %freq[max_ind])
          
        if Type == 'resp':
            if freq[max_ind] > 0.1 and freq[max_ind] < 0.5:
                log.info('Valid resp data. freq @ peak power: %f' %freq[max_ind])
                self.file_list_dict['resp'] = self.trun_resp_path
            
            else:
                self.file_list_dict['resp'] = None         
                log.info('resp failed validation.  freq @ peak power: %f' %freq[max_ind])

    def validate_data_threshold(self, Type, data, threshold):
        pass
        # should be re-written using RMS
        # a single peak in a non valid file
        # could pass the file, b/c the absolute range
        # is valid....       
  
    def down_sample_ts(self, data, orig_sample_rate):
        if orig_sample_rate == 25.0:
            return data
        
        q = orig_sample_rate / 25.0
        if q < 2:
            q = 2
        
        else:
            q = int(np.round(q, 0))# round to whole number    
        
        out = decimate(data, q)

        return out

    def _create_trigger(self, data):#TODO: use savitzy golay for der andfix concavity intersection
    
        data -= data.mean()
        
        der1 = np.diff(data)
        extrema_ind = np.where( (der1[0:-1] * der1[1:]) < 0)[0]
        
        der2 = np.diff(der1)
        
        concavedow = np.where(der2 < 0)[0]
        
        peaks = []
        for i in range(0, extrema_ind.shape[0]):
            if np.any(data[extrema_ind[i]] == data[concavedow]):
                peaks.append(extrema_ind[i])
        
        trigger = np.zeros(data.shape[0])
        trigger[peaks] = 10
        
        return trigger
        
    def validate_created_trigger(self, sample_rate, trigger):
        '''Try to validate the triggered physio using some assumptions
        about a typical cardiac and resp rate. '''
        
        one_sec =  sample_rate  
        threshold = 0.25 * one_sec
               
        trig_ind = np.where(trigger == 10)[0]
    
        test = trig_ind[1:] - trig_ind[0:-1]
        bad_ind = np.where(test < threshold)
        
        percent_bad = len(bad_ind) / float(len(trigger))
        log.info('Ratio of bad indices in created for %s trigger: %f' %(percent_bad))
        
        if percent_bad > 0.10:
            return  bad_ind
        
        else:
            return None   

    def calculate_exp_len_time(self):
        num_vols = imagetools.get_number_of_vols(self.rs_path)
        exp_len_time = np.round(num_vols * self.TR, decimals=2)
        log.info('Experiment length: %g' %exp_len_time)
        
        return exp_len_time

    def get_TR(self, infile):
        time_unit = imagetools.get_time_unit(infile)
        
        if time_unit != 'sec':
            log.exception("Time unit in TR is not seconds.")
            raise Exception, "Time unit in TR is not seconds."
        
        TR = imagetools.get_pixdims(infile)[4]
        if TR > 8.0:
            err = """Unsually high TR. It is likely that the unit is set to seconds, but the values is in
milliseconds, i.e. 2200"""
            log.exception(err)
            raise Exception, err
        
        log.info("TR from header:%f" %TR)
        return TR


class PreprocessSpiral(PreprocessBase):
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir)
        
        self.card_sample_rate = 100.0 #Hz
        self.resp_sample_rate = 25.0  #Hz    
       
        log.debug('card samp rate: %i' %self.card_sample_rate)
        log.debug('resp same rate: %i' %self.resp_sample_rate)
    
    def spiral_load_data(self):
        data = np.loadtxt(self.file_list_dict['single_file'])
        
        ind1 = np.where(data == -9999)[0]
        ind2 = np.where(data == -8888)[0] 
        
        if len(ind1) == 0:
            log.info("-9999 resp marker not found")
            self.file_list_dict['resp'] = None
        else:
            self.file_list_dict['resp'] = True
        
        if len(ind2) == 0:
            log.info("-8888 card marker not found")
            self.file_list_dict['card'] = None
        else:
            self.file_list_dict['card'] = True
        
        resp = data[ind1+1 : ind2-1]
        card = data[ind2+1:]
        
        return resp, card

    def main(self):
        self.exp_len_time = self.calculate_exp_len_time()
        log.info('Experiment length: %f' %self.exp_len_time)
        
        if self.file_list_dict['single_file'] is not None:
            resp, card = self.spiral_load_data()
        
        else:
            log.exception("Physio file not found.")
            raise Exception, "Physio file not found."
        
        if self.file_list_dict['card'] is not None:
            self.validate_data_fft('card', card, self.card_sample_rate)
        
        if self.file_list_dict['card'] is not None:   
            card_trun = self.truncate_data(card, self.card_sample_rate, self.exp_len_time)
            self.save_data(self.trun_card_path, card_trun)
        
        if self.file_list_dict['resp'] is not None:
            self.validate_data_fft('resp', resp, self.resp_sample_rate)
         
        if self.file_list_dict['resp'] is not None:    
            resp_trun = self.truncate_data(resp, self.resp_sample_rate, self.exp_len_time)
            self.save_data(self.trun_resp_path, resp_trun)
        
        if self.file_list_dict['card'] is not None:
            self.make_trigger(self.card_sample_rate, data=card_trun)
        
        if self.file_list_dict['resp'] is not None:
            resp_trun_fil = self.pre_trigger_filter(self.resp_sample_rate, resp_trun, Type='resp')
            self.save_data(self.trun_resp_path, resp_trun_fil)
        
        
class PreprocessUpmc(PreprocessBase):
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir)
        self.sample_rate = 50.0 #Hz
        
    def load_physio(self, Type, File):
        # gzip covers the case where the prog is only validating
        # doesn't break if the file has been unzipped
        f = gzip.open(File, 'r')
        data = f.readline()
        footer = f.readlines()
        f.close()
        
        data = data.split()
        data = np.array(data, dtype=float)
        
        #chop off the last data point which is 5003, marking the end
        if data[-1] == 5003:
            data = data[:-1]
        
        #some files have footers and some don't
        if len(footer) == 0:
            log.exception("No footer was found.")
            raise Exception, "No footer was found."
     
        return data, footer   
     
    def truncate_data(self, footer, data):
        log_start = int(footer[9].split()[1]) 
        log.debug('Physio log start: %i' %log_start)
        exp_start = int(footer[11].split()[1])
        log.debug('Physio experiment start: %i' %exp_start)
        
        # experiment start is when the fMRI scan starts. The log start, is
        # when the recording system beings, which SHOULD BE BEFORE the experiment.
        
        # deal with broken footers where the log start is after the exp starts
        start = (exp_start - log_start) / 20.0 # X milliseconds / 1000 * 50 Hz
        if start < 0 and (-20 * start) < (2.2 * self.num_vols_drop) : #20 is conversion factor to sec
            log.info("""Physio log start time is ahead of exp start time. HOWEVER, it's 
OK b/c the number of frames dropped makes up for that difference.""")
            start = np.abs(start)
        elif start < 0 and (-20 * start) > (2.2 * self.num_vols_drop):
            err =  """Physio footer has recording start after scan start and 
the diff is more than the time taken for the number of dropped volumes."""
            log.exception(err)
            raise Exception, err
        
        log.debug('Points to skip: %i' %start)
        #num_vols = imagetools.get_number_of_vols(self.rs_path)
        #stop = 2.2 * num_vols * self.sample_rate # TR * number vols * phy sample rate
        stop = self.exp_len_time * self.sample_rate
        log.debug('stopping point: %i' %stop)
        
        # additional dropped points from drop num
        additional_drop_pts = self.sample_rate * self.num_vols_drop * 2.2
        start = start + additional_drop_pts
        log.info('Final number of points dropped from physio: %i' %start)
        
        trun_data = data[start+1:stop] 
        
        return trun_data
    
    def remove_trigger(self, data):
        ind = np.where( data > 4500)[0]
        
        # in some cases the last indice is a trigger and 
        # this will blow up the ind + 1 algorithm
        if ind[-1] == data.shape[0] - 1:#shape is abs len, where ind starts at 0
            data[-1] = data[ind[-2]]
            ind = ind[0:-1]
        
        data[ind] =(data[ind + 1] + data[ind - 1]) / 2.0
   
        return data

    def main(self):
        self.exp_len_time = self.calculate_exp_len_time()
        log.info('Experiment length: %f' %self.exp_len_time)
        
        if self.file_list_dict['card'] is not None:
            card_path = self.file_list_dict['card']
            card_data, card_footer = self.load_physio('card', card_path)
            card_data = self.remove_trigger(card_data)
            card_trun = self.truncate_data(card_footer, card_data)
            self.validate_data_fft('card', card_trun, self.sample_rate)
           
        if self.file_list_dict['resp'] is not None:    
            resp_path = self.file_list_dict['resp']
            resp_data, resp_footer = self.load_physio('resp', resp_path)
            resp_data = self.remove_trigger(resp_data)
            resp_trun = self.truncate_data(resp_footer, resp_data)
            self.validate_data_fft('resp', resp_trun, self.sample_rate)
        
        if self.file_list_dict['card'] is not None:
            self.save_data(self.trun_card_path, card_trun)
            self.make_trigger(sample_rate=self.sample_rate, data=card_trun)
        
        if self.file_list_dict['resp'] is not None:
            resp_trun_fil = self.pre_trigger_filter(self.sample_rate, resp_trun, Type='resp')
            self.save_data(self.trun_resp_path, resp_trun_fil)

            
class PreprocessSriUcsd(PreprocessBase):
    '''This class contains the pre-processing work for physio files from SRI and UCSD.
    This is the only site that uses the triggered data as it is given. '''
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir)
        
        self.card_samp_rate = 100.0 #Hz
        self.resp_samp_rate = 25.0  #Hz   
        
        log.debug('card samp rate: %i' %self.card_samp_rate)
        log.debug('resp same rate: %i' %self.resp_samp_rate)
    
    def load_data(self, File, Type=None):   
        
        if Type is None:
            data = np.loadtxt(fname=File)
            log.debug('Loading non-trigger data for type: %s' %Type)
        
        elif Type == 'trigger':
            data = np.loadtxt(fname=File, dtype=int)
            log.debug('Loading trigger data for type: %s' %Type)
        
        else:
            log.exception("Type is not understood: %s" %Type)     
            raise Exception, "Type is not understood: %s" %Type         
        
        if data.size == 0:
            self.file_list_dict[Type] = None
            
        return data
    
    def make_trigger_vector(self, trig_pts, physio_data):
        log.debug('Making trigger vector from provided trigger points.')
        trigger_vec = np.zeros(physio_data.shape[0])
        trig_ind = trig_pts
        
        trigger_vec[trig_ind] = 1   
        
        return trigger_vec
    
    def main(self):
        self.exp_len_time = self.calculate_exp_len_time()
        log.info('Experiment length: %f' %self.exp_len_time)
        
        ### Cardiac
        card_path = self.file_list_dict['card']
        card_trig_path = self.file_list_dict['card_trig']
        
        if card_path is not None: 
            card_data = self.load_data(card_path)       
            #self.validate_data_threshold('card', card_data, 200)
            self.validate_data_fft('card', card_data, self.card_samp_rate)
        
        if card_trig_path is not None:    
                card_trig_data = self.load_data(card_trig_path, Type='trigger')

        ### use the scanner trig file ###
        if card_trig_path is not None and card_path is not None:
            log.info('Scanner made card trig file found. These are usually quite good. Using trig file.')
            trig_card_data = self.make_trigger_vector(card_trig_data, card_data)    
            # give trig data the same var name, trun_card_path, b/c it's easier logic later
            trun_card = self.truncate_data(trig_card_data, self.card_samp_rate, self.exp_len_time)
            self.trun_card_path = self.trun_card_path.replace('.txt', '_trig.txt')
            self.save_data(self.trun_card_path, trun_card)
            self.file_list_dict['card_trig'] = self.trun_card_path
         
        ### just use the card file only, make a trigger for it
        # will change if validation fails
        elif card_path is not None and card_trig_path is None:
            trun_card = self.truncate_data(card_data, self.card_samp_rate, self.exp_len_time)
            self.save_data(self.trun_card_path, trun_card)
            #trigger
            self.make_trigger(sample_rate=self.card_samp_rate, data=trun_card)
        
        else:
            trun_card = None
        
                
        ### Respirtory
        resp_path = self.file_list_dict['resp']
        
        if resp_path is not None:
            resp_data = self.load_data(resp_path)
            #self.validate_data_threshold('resp', resp_data, 2000)
            self.validate_data_fft('resp', resp_data, self.resp_samp_rate)

        if self.file_list_dict['resp'] is not None: # will change if validation fails
            trun_resp = self.truncate_data(resp_data, self.resp_samp_rate, self.exp_len_time)
            trun_resp_fil = self.pre_trigger_filter(self.resp_samp_rate, trun_resp, Type='resp')
            self.save_data(self.trun_resp_path, trun_resp_fil)
            #self.make_trigger(sample_rate=self.resp_samp_rate, data=trun_resp, Type='resp')
        
        else:
            trun_resp_fil = None
            
        return trun_card, trun_resp_fil


class PreprocessOhsu(PreprocessBase):
    '''This class contains the pre-processing work for physio files from OHSU. '''
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir)
        self.sample_rate = 1000.0#Hz
        
        log.info('Both resp and card has sample_rate: %i' %self.sample_rate)
        
    def load_physio(self):
        # trigger is col 0, but it has been unreliable.
        card, resp = np.loadtxt(fname=self.file_list_dict['single_file'], skiprows=7, usecols=[1,2], unpack=True)
        
        if card.size == 0:
            self.file_list_dict['card'] = None
        else:
            self.file_list_dict['card'] = True
        
        if resp.size == 0:
            self.file_list_dict['resp'] = None
        else:
            self.file_list_dict['resp'] = True
                
        return card, resp
 
    def main(self):
        self.exp_len_time = self.calculate_exp_len_time()
        log.info('Experiment length: %f' %self.exp_len_time)
        
        if self.file_list_dict['single_file'] is not None:
            card, resp = self.load_physio()
        
        elif self.file_list_dict['single_file'] is None:
            log.exception("No physio file found.")
            raise Exception, "No physio file found."
        
        if self.file_list_dict['card'] is not None:   
            self.validate_data_fft('card', card, self.sample_rate)
        
        if self.file_list_dict['resp'] is not None:
            self.validate_data_fft('resp', resp, self.sample_rate)
                     
        if self.file_list_dict['card'] is not None:
            card_trun = self.truncate_data(card, self.sample_rate, self.exp_len_time)
            self.save_data(self.trun_card_path, card_trun)
            
            # trigger data
            self.make_trigger(sample_rate=self.sample_rate, data=card_trun)
        
        if self.file_list_dict['resp'] is not None:
            resp_trun = self.truncate_data(resp, self.sample_rate, self.exp_len_time)
            self.save_data(self.trun_resp_path, resp_trun)
            
            resp_trun_fil = self.pre_trigger_filter(self.sample_rate, resp_trun, Type='resp')
            self.save_data(self.trun_resp_path, resp_trun_fil)
        

class PreprocessDuke(PreprocessBase):
    '''This class contains the pre-processing work for physio files from SRI and Duke '''
    def __init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir):
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, file_list_dict, output_dir)       
        self.sample_rate = 100.0
    
    def load_data(self, File):  
        data = np.loadtxt(fname=File, usecols=[1])
        
        if data.size == 0:
            log.exception("data file is empty:" %self.physio_file)
            raise Exception,  "data file is empty:" %self.physio_file
                
        return data    
        
    def main(self):
        self.exp_len_time = self.calculate_exp_len_time()
        log.info('Experiment length: %f' %self.exp_len_time)
        
        if self.file_list_dict['card']:
            card = self.load_data(self.file_list_dict['card'])
            self.validate_data_fft('card', card, self.sample_rate)
        
        if self.file_list_dict['resp']:
            resp = self.load_data(self.file_list_dict['resp'])
            self.validate_data_fft('resp', resp, self.sample_rate)
        
        if self.file_list_dict['card'] is not None:
            #trucate
            card_trun = self.truncate_data(card, self.sample_rate, self.exp_len_time)
            self.save_data(self.trun_card_path, card_trun)
            
            # trigger data
            self.make_trigger(sample_rate=self.sample_rate, data=card_trun)
        
        if self.file_list_dict['resp'] is not None:
            resp_trun = self.truncate_data(resp, self.sample_rate, self.exp_len_time)
            self.save_data(self.trun_resp_path, resp_trun)
            
            resp_trun_fil = self.pre_trigger_filter(self.resp_samp_rate, resp_trun, Type='resp')
            self.save_data(self.trun_resp_path, resp_trun_fil)


class Denoise(object):
    '''Runs the retroicor binary on the target 4D file. '''
    def __init__(self, file_list_dict, denoise_input, phy_denoised_out, output_dir):
        
        self.output_dir = output_dir
        self.misctools = MiscTools.MiscTools()
        
        self.file_list_dict = file_list_dict
        self.trun_card_path = self.file_list_dict['card_trig']
        self.trun_resp_path = self.file_list_dict['resp']
        self.denoise_input = denoise_input    
        self.denoise_output = phy_denoised_out
        
    def denoise_card_and_resp(self):
        cmd = '3dretroicor -resp %(resp)s -card %(card)s -prefix %(output)s -order 2 %(input)s'
        self.cmd = cmd %{'resp': self.trun_resp_path,
                         'card': self.trun_card_path,
                         'output':self.denoise_output,
                         'input':self.denoise_input
                          }
        
    def denoise_card_only(self):
        cmd = '3dretroicor -card %(card)s -prefix %(output)s -order 2 %(input)s'
        self.cmd = cmd %{'card': self.trun_card_path,
                         'output':self.denoise_output,
                         'input':self.denoise_input
                         }    
 
    def denoise_resp_only(self):
        cmd = '3dretroicor -resp %(resp)s  -prefix %(output)s -order 2 %(input)s'
        self.cmd = cmd %{'resp':self.trun_resp_path,
                         'output':self.denoise_output,
                         'input':self.denoise_input
                          }    
         
    def main(self):
        
        # BOTH card and resp
        if self.file_list_dict['card_trig'] is not None and self.file_list_dict['resp'] is not None:
            self.denoise_card_and_resp()
            log.info('Both card and resp files used for Denoise().')
        # Card only
        elif self.file_list_dict['card_trig'] is not None and self.file_list_dict['resp'] is None:
            self.denoise_card_only()
            log.info("Only card file used for Denoised()")
        # Resp only
        elif self.file_list_dict['card_trig'] is None and self.file_list_dict['resp'] is not None:
            self.denoise_resp_only()    
            log.info('Only resp file used for Denoised()')
        # None 
        else:
            log.exception("No card or resp data, cannot denoise.")
            raise Exception, "No card or resp data, cannot denoise."
        
        self.run_retroicor()
        
        # fix slice duration since it gets wiped
        # slice_dim should still == 3
        imagetools.fix_slice_dur(self.denoise_output)
        
    def run_retroicor(self):    
        shell_script_template = """#!%(bash_path)s
%(cmd)s
""" %{'bash_path':MiscTools.MiscTools().which('bash'), 'cmd':self.cmd}        
        
        shell_script_path = os.path.join(self.output_dir, '3dretroicor.sh')
        
        self.misctools.write_shell_script(shell_script_path, shell_script_template)
            
        self.misctools.call_shell_program(shell_script_path)    
               
               
class Main(PreprocessBase):
    
    def __init__(self, physio_dir, num_vols_drop, rs_path, output_dir, phy_denoised_out, preprocess_only=False):
        '''sites can be: duke, sri, ohsu, upmc or ucsd. '''
        self.preprocess_only = preprocess_only # all output to /dev/null, but allows data validation
        self.file_list_dict = {'card':None, 
                               'card_trig':None,
                               'resp':None,
                               'single_file':None
                              }
        
        # this instance if only for the truncated physio paths. That way changes only go in one place
        PreprocessBase.__init__(self, physio_dir, num_vols_drop, rs_path, self.file_list_dict, output_dir)
        
        self.output_dir = output_dir
        self.denoised_out = phy_denoised_out        
        self.num_vols_drop = num_vols_drop
        self.rs_data = rs_path

        self.prepro_dict = {'sri' :PreprocessSriUcsd, 
                            'duke':PreprocessDuke, 
                            'ohsu':PreprocessOhsu, 
                            'upmc':PreprocessUpmc,
                            'ucsd':PreprocessSriUcsd,
                            'spiral':PreprocessSpiral
                            } 
        
        self.physio_file_site_map = {'card_data_50hz' :'upmc',
                                     'resp_data_50hz' :'upmc', 
                                     'card_data_100hz':'sri',
                                     'resp_data_25hz':'sri',
                                     'card_trig_data_resp_data_1000hz':'ohsu',
                                     'card_time_100hz': 'duke',
                                     'resp_time_data_100hz':'duke',
                                     'Spiral.physio':'spiral'
                                     }
        
        self.physio_dir = physio_dir
        log.debug('Physio data search path: %s' %self.physio_dir)
        
    def preprocess(self):
        '''Only run the preprocess steps, not denoise. '''
        files = self.get_physio_file_list()
        self.populate_file_list_dict(files)
        
        site = self.get_site()

        # choose the preprocessing class based on site 
        PrePro = self.prepro_dict[site]
        preproc = PrePro(self.physio_dir, self.num_vols_drop, self.rs_data, self.file_list_dict, self.output_dir)
                        
        preproc.main()
    
    def denoise(self):
        '''Run denoise step only on preprocessed physio. '''
        
        self.repopulate_file_list_dict()
        
        denoise = Denoise(self.file_list_dict, self.rs_data, self.denoised_out, self.output_dir)        
        denoise.main()  
        
    def main(self): 
        '''Run preproces and denoise back to back '''
        # get the physio files and fill in the dict of files variable to pass into the object
        files = self.get_physio_file_list()
        self.populate_file_list_dict(files)
        
        site = self.get_site()

        # choose the preprocessing class based on site 
        PrePro = self.prepro_dict[site]
        
        if self.preprocess_only:
            self.output_dir = tempfile.mkdtemp(dir='/tmp', prefix='physio')
        
            try:
                preproc = PrePro(self.physio_dir, self.num_vols_drop, self.rs_data, self.file_list_dict, self.output_dir)  
                preproc.main()
            
                self.file_list_dict = preproc.file_list_dict # self exposes this critical var
            
            finally:
                rmtree(self.output_dir)
               
        else:
            preproc = PrePro(self.physio_dir, self.num_vols_drop, self.rs_data, self.file_list_dict, self.output_dir)  
            preproc.main()
            # Get a fresh copy of truncated data paths which is updated. 
            # An entry in the dict, will change to None, if the data returned after loading
            # has a length of 0 or the fft based validation fails.
            file_list_dict = preproc.file_list_dict
            # run the retroicor binary 
            # file_list_dict, denoise_input, phy_denoised_out, output_dir     
            denoise = Denoise(file_list_dict, self.rs_data ,self.denoised_out, self.output_dir)        
            denoise.main()        
    
    def populate_file_list_dict(self, file_list):
        basename = self.physio_dir
        
        for item in file_list:
            if re.match('card_data_50hz|card_data_50hz.gz', item):
                self.file_list_dict['card'] = os.path.join(basename, item) 
            
            if re.match('resp_data_50hz|resp_data_50hz.gz', item):
                self.file_list_dict['resp'] = os.path.join(basename, item)     
             
            if re.match('card_trig_data_resp_data_1000hz|card_trig_data_resp_data_1000hz.gz', item):
                self.file_list_dict['single_file'] = os.path.join(basename, item)
             
            if re.match('card_data_100hz|card_data_100hz.gz', item):
                self.file_list_dict['card'] = os.path.join(basename, item) 
             
            if re.match('resp_data_25hz|resp_data_25hz.gz', item):
                self.file_list_dict['resp'] = os.path.join(basename, item)   
    
            if re.match('Spiral.physio', item):
                self.file_list_dict['single_file'] = os.path.join(basename, item)
    
        dict_values = self.file_list_dict.values()
        if all( val is None for val in dict_values):
            log.exception("No physio files were found.")
            raise Exception, "No physio files were found."
                        
    def get_physio_file_list(self):
        file_list = os.listdir(self.physio_dir)
        log.debug('list of files found in the physio base path: %s' %str(file_list))
        
        return file_list                          
                          
    def get_site(self): 
        site = None                  
        keys = self.physio_file_site_map.keys()        
  
        for v in self.file_list_dict.itervalues():
            if v is not None:
                v  = os.path.splitext(v)[0] # when only running preproc, the physio are still gzipped
                if os.path.basename(v) in keys: 
                    site = self.physio_file_site_map[os.path.basename(v)]
                    break            
        
        if site is None:
            log.exception("Site was not detected from physio file name." )
            raise Exception, "Site was not detected from physio file name."                  
        else:
            return site                  
                                    
    def repopulate_file_list_dict(self):
        '''Use this when not allowing the full cycle of the Main() class, like in Torsten's
pipeline version that calls only the find_files() and preprocess().'''
    
        # check for: 
        # truncated_card_data_trig.txt
        # truncated_resp_data.txt
        if os.path.exists(self.trun_card_path_trig):
            self.file_list_dict['card_trig'] = self.trun_card_path_trig

        if os.path.exists(self.trun_resp_path):        
            self.file_list_dict['resp'] = self.trun_resp_path
            
        else:
            log.info("No truncated card or resp physio data was found.")                             
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                                            