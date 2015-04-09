import numpy as np
import scipy.signal as sig
import scipy
import logging
from math import factorial
from MiscTools import MiscTools
from ImageTools import ImageTools
from os import path
import nibabel.eulerangles as nie
from scipy import ndimage

log = logging.getLogger('module_logger')
from MiscTools import log_method

class SignalProcessTools(object):
    def __init__(self, PathType='production'):
        self.misctools = MiscTools()
        #self.imagetools = ImageTools(PathType)

    def low_pass_filter(self,input_signal, cutoff_freq, sample_rate_Hz, order):
        '''Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians / sample.  (`Wn` is thus in
        half-cycles / sample.)
        '''

        nyq = 0.5 * sample_rate_Hz
        Wn = (cutoff_freq / nyq)

        bev,alc = sig.butter(order, Wn, 'low')

        output = sig.filtfilt(bev, alc, input_signal)

        return output

    def hi_pass_filter(self, input_signal, cutoff_freq, sample_rate_Hz, order):
        nyq = 0.5 * sample_rate_Hz
        Wn = (cutoff_freq / nyq)
        bev, alc = sig.butter(order, Wn, btype='highpass' )

        out_put = sig.filtfilt(bev, alc, input_signal)

        return out_put

    def butter_bandstop(self, lowcut, highcut, sample_rate, data, order=5):
            nyq = 0.5 * sample_rate
            low = lowcut / nyq
            high = highcut / nyq
            bev, alc = sig.butter(order, [low, high], btype='stop', analog=False)#FIR filter

            out = sig.filtfilt(bev, alc, data)

            return out

    def fir_bandpass(self, lowcut_hz, highcut_hz, sample_rate, data, width=None, ripple=None, output_response=False):
        nyq_rate = sample_rate / 2.0

        width = 0.05/nyq_rate

        ripple_db = 10

        N, beta = sig.kaiserord(ripple_db, width)

        cutoff_hz = np.array((lowcut_hz, highcut_hz))

        taps = sig.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)

        filtered_data = sig.lfilter(taps, 1.0, data)

        if output_response:
            w, h = sig.freqz(taps, worN=8000)
            frq = ( w / np.pi ) * nyq_rate
            H = np.abs(h)

            return filtered_data, frq, H

        return filtered_data

    @log_method
    def butter_bandpass(self, lowcut, highcut, sample_rate_Hz, data, order=5, output_f_H=False):

        nyq = 0.5 * sample_rate_Hz
        low = lowcut / nyq
        high = highcut / nyq
        bev, alc = sig.butter(order, [low, high], btype='bandpass', analog=False)#FIR filter

        out = sig.filtfilt(bev, alc, data)

        if output_f_H:
            w, H = sig.freqz(bev, alc)
            frq =  nyq * w/np.pi
            return out, frq, H

        return out

    def periodogram_1D(self, data, sample_rate):
        data_fft = np.fft.fft(data)

        # throw out the dup and norm
        half_pt = np.floor(len(data_fft)/2.0)

        fft_uniq = data_fft[0:half_pt] / (half_pt - 1 )

        mag = 2 * np.abs(fft_uniq)

        # freq axis
        freq  = sample_rate # (s)
        nyquist = freq / 2.0
        num_pts = len(mag)
        freq_res = nyquist / float(num_pts)
        freq_vec = np.arange(0,nyquist, freq_res)

        return freq_vec, mag

    @log_method
    def flat_smooth(self, data, L=10):
        out = np.zeros(data.shape)
        for i in range(0, data.shape[0]-L):
            win = data[i:i+L]
            smoothed = np.mean(win)
            out[i:i+L] = smoothed

        return out

    @log_method
    def smooth(self, x, window_len=11, window='hanning', mode='valid', log_this=False):
        """smooth the data using alc window with requested size."""

        if x.ndim != 1:
            log.exception( "smooth only accepts 1 dimension arrays.")
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            log.exception("Input vector needs to be bigger than window size.")
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len<3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            log.exception("Windpw must be:'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
            raise ValueError, "Windpw must be:'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode=mode)

        return y

    @log_method
    def remove_global_drift(self, rs_path, mask, outfile, seg_len, deg=6, mask_output=True):
        #rs = self.imagetools.load_image(rs_path)

        roi, ind, shape, coord = self.imagetools.get_roi(rs_path, mask)
        roi = roi.astype(scipy.float32, copy=False)

        roi -= roi.mean(axis=1, keepdims=True)

        smooth = np.zeros((roi.shape[0], roi.shape[1] + seg_len - 1))
        for i in range(roi.shape[0]):
            smooth[i,:] = self.smooth(roi[i,:], seg_len, 'hanning', mode='valid')

        extra = np.float(seg_len - 1)

        ind_array = np.arange(smooth.shape[1], dtype=np.int16)
        np.random.seed(0)
        np.random.shuffle(ind_array)
        rem_ind = np.arange(extra, dtype=np.int16)
        smooth = np.delete(smooth, ind_array[rem_ind], axis=1)

#        if np.mod(smooth.shape[1], extra) != 0.0:
#             n = np.ceil(smooth.shape[1] / np.float(extra))
#             diff =  n * extra
#
#             # diff > extra always
#             pad_amt = diff - extra
#
#             smooth = np.pad(smooth, ((0, 0), (0, np.int(pad_amt))), mode='constant')
#            rem_ind = np.arange(0, smooth.shape[1], n)
#            smooth = np.delete(smooth, rem_ind, axis=1)
#
#         else:
#             rem_ind = np.arange(0, smooth.shape[1], n)

        t = np.arange(smooth.shape[1])
        t_arr = np.tile(t, (roi.shape[0],1)).T

        pos = scipy.polyfit(t, smooth.T, deg)
        fit = scipy.polyval(pos, t_arr)
        roi -=  fit.T

        if mask_output is True:
            new_image = np.zeros(shape, dtype=scipy.float32)

        else:
            new_image = self.imagetools.load_image(rs_path).get_data()


        new_image[ind] = roi

        log.info("Saving drift corrected image.")
        self.imagetools.save_new_image(new_image, outfile, coord)
        #self.imagetools.save_new_image_clone(new_image, rs.header, outfile, coord)

    def norm_centered_roi_to_zero_one(self, infile, mask, outfile=None):
            '''If you want to return only the normed roi, then do not use an
            outfile arg. If you do want alc normed image written, then include an
            outfile path in the args'''

            roi, ind, data_shape, coordmap = self.imagetools.get_roi(infile, mask)

            # un-center, make all positive
            # numpy is row major order, so reshape takes rows
            # and lays them down as cols
            mins = roi.min(axis=1,keepdims=True)
            roi -= mins

            maxs = roi.max(1, keepdims=True)
            roi /= maxs

            roi -= np.mean(roi, axis=1, keepdims=True)

            new_image = np.zeros(data_shape, dtype=float)
            new_image[ind,:] = roi

            if outfile is not None:
                self.imagetools.save_new_image(new_image, outfile, coordmap )

            else:
                return roi

    @log_method
    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters: Copied from http://wiki.scipy.org/Cookbook/SavitzkyGolay
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))

        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")

        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")

        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")

        order_range = range(order+1)
        half_window = (window_size -1) // 2

        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])

        y = np.concatenate((firstvals, y, lastvals))

        return np.convolve( m[::-1], y, mode='valid')

    @log_method
    def get_mean_signal_with_mask(self, time_series, tissue_mask, std=False, range_=False):

        roi, _,_,_ = self.imagetools.get_roi(time_series, tissue_mask)

        if std:
            return roi.mean(0), roi.std(0)

        if range_:
            range_ = roi.max(0)  - roi.min(0)
            return roi.mean, range_

        return roi.mean(0)

    def regress_out_confounds(self, signal, confounds, lower=None, upper=None, sample_rate=None):
        '''signal should be features:time-courses, like the output from imagetools.get_roi() and
        the confounds should be time-courses:features. Rank 1 arrays will be reshaped.'''
        # band pass filter the confounds per Hallquist before regressing out
        ## Transpose the confounds n/c filter uses row major order
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]

        if lower is not None and upper is not None:
            confounds = self.butter_bandpass(lower, upper, sample_rate, confounds.T, order=5)

            # back to col major order for clean method coming up
            confounds = confounds.T

            signal = self.butter_bandpass(lower, upper, sample_rate, signal, order=5)

        # col maj, detrend and norm along rows, axis=0
        signal = sig.detrend(signal, axis=-1, type='linear')

        if signal.size < 10:
            ddof = 0
        else:
            ddof = 1

        # normalize confounds
        sigma = np.std(confounds, ddof=ddof, keepdims=True, axis=0)
        sigma[sig == 0] = 1
        confounds /= sigma
        confounds -= np.mean(confounds, keepdims=True, axis=0)

        # normalize signal
        sigma = np.std(signal, ddof=ddof, keepdims=True, axis=1)
        sigma[sig == 0] = 1
        signal /= sigma
        signal -= np.mean(signal, keepdims=True, axis=1)

            # orthogonolize for 2D arays
        Q = scipy.linalg.qr(confounds, mode='economic')[0]

        # regress by projection
        signal = signal.T
        signal -= np.dot(Q, np.dot(Q.T, signal))
        signal = signal.astype(np.float32, copy=False)

        signal[np.isnan(signal)] = 0

        return signal.T

    def volterra_confound_for_motion(self, infile, mask, outfile, signal):
        from PyLysis.models.non_parametric.volterra_series  import VolterraLaguerre as volt

        roi, ind, shape, coord = self.imagetools.get_roi(infile, mask)

        y_est = np.zeros_like(roi)


        for j in range(roi.shape[0]):
            v = volt(signal, y=roi[j,:], order=2, memory=5, num_laguerres=3)
            v.train()
            v.estimate_y()
            y_est[j,:] = np.squeeze(v.y_est)

        roi = self.regress_out_confounds(signal=roi,
                                       confounds=y_est.T,
                                        lower=0.009,
                                        upper=0.09,
                                        sample_rate=1/2.2
                                        )

        new = np.zeros(shape)
        new[ind] = roi
        self.imagetools.save_new_image(new, outfile, coord)

        return y_est

    def examine_eigen_values_per_label(self, roi, label):
#         ts_data = self.imagetools.load_image(time_series).get_data()
#         parc_data = self.imagetools.load_image(parc_file).get_data()
#
#         ind = parc_data == label
#
#         roi = ts_data[ind, :]

        S = np.linalg.svd(roi, full_matrices=0, compute_uv=0)

        sig = np.zeros(S.size, np.float32)
        # running std
        for i in range(1, S.size):
            sig[i] = np.std(S[0:i], ddof=0)

        diff_sig = self.savitzky_golay(sig, window_size=3, order=1, deriv=1)
        diff_sig /= diff_sig.max()

        return np.abs(diff_sig)

    def get_motion_f2f_params_mcflirt(self, mats_path, data_dim):
        mats = self._load_mat_files(mats_path, data_dim)
        deltaMat = self._compute_temp_der(mats, data_dim)
        deltaTrans = self._get_delta_trans(mats, data_dim)
        thx, thy, thz = self._get_rots(deltaMat, data_dim)
        # rows=samples and col=features
        confounds = deltaTrans
        confounds = np.hstack((confounds, thx))
        confounds = np.hstack((confounds, thy))
        confounds = np.hstack((confounds, thz))

        return confounds

    def _load_mat_files(self, base_path, data_dim):

        mats = np.zeros((data_dim[3],4,4))
        # only load in the mats we need. dropping 1 to have 268 even num, means toss MAT_0000
        for i in range(0, data_dim[3]):
            index = i
            mat_path = "MAT_%04d" %index
            fp = path.join(base_path, mat_path)
            mats[i,:,:] = np.loadtxt(fp)

        log.info('mat files loaded.')
        log.debug('Number of mat files loaded: %i' %mats.shape[0])

        return mats

    def _compute_temp_der(self, mats, data_dim):
        deltaMat = np.zeros((data_dim[3],4,4))
        for i in range(1,len(mats)-1):
            deltaMat[i,:,:] =  np.dot( mats[i+1,:,:], np.linalg.inv(mats[i,:,:]) )

        log.info('Derivatives of affines computed.')
        log.debug('Number of affines derivatives: %i' %deltaMat.shape[0])

        log.debug("temp der computed")

        return deltaMat

    def _get_delta_trans(self, mats, data_dim):
        deltaTrans = np.zeros((data_dim[3], 3))
        for i in range(1, mats.shape[0]):
            deltaTrans[i,:] = mats[i,:3,3] - mats[i-1,:3,3]

        return deltaTrans

    def _get_trans(self, mats, data_dim):
        trans = np.zeros((data_dim[3], 3))
        for i in range(mats.shape[0]):
            trans[i,:] = mats[i,:3,3]

        return trans


    def _get_rots(self, deltaRotmat, data_dim):
        theta_x = np.zeros((data_dim[3],1))
        theta_y = theta_x.copy()
        theta_z = theta_x.copy()

        for i, mat in enumerate(deltaRotmat):
            rots = nie.mat2euler(mat[:3,:3])
            theta_z[i] = rots[0]
            theta_y[i] = rots[1]
            theta_x[i] = rots[2]

        log.debug('Rots computed.')
        return theta_x, theta_y, theta_z

    def calculate_trajectory_dpc(self, mask, motion_f2f_file):
        ''' motion file is from PyConn.Detrend. it is the frame to frame
        motion and is 6 cols with a row for each time point'''

        motion = np.loadtxt(motion_f2f_file)

        # vox dim in mm
        pixdims = self.imagetools.get_pixdims(mask)[1:4]

        mask_data = self.imagetools.load_image(mask).get_data()

        # distance map to closet vox outside mask
        distances = ndimage.distance_transform_edt(mask_data)

        dist = np.where(distances == distances.max())

        # max dist is worst case
        distx = dist[0].max()
        disty = dist[1].max()
        distz = dist[2].max()

        #largest diameter in each axis
        dia = 2 * np.array((distx, disty, distz)) * pixdims
        dia = np.tile(dia, (motion.shape[0],1))

        trans = motion[:,0:3]
        rots = dia * motion[:,3:]

        trans_rms = np.sqrt(np.mean(trans**2, axis=1))
        rots_rms = np.sqrt(np.mean(rots**2, axis=1))

        trajectory = trans_rms + rots_rms

        return trajectory

    def get_abs_motion_params(self, mats_path, data_dim):
        mats = self._load_mat_files(mats_path, data_dim)
        trans = self._get_trans(mats, data_dim)
        thx, thy, thz = self._get_rots(mats, data_dim)
        # rows=samples and col=features
        motion = trans
        motion = np.hstack((motion, thx))
        motion = np.hstack((motion, thy))
        motion = np.hstack((motion, thz))

        return motion





