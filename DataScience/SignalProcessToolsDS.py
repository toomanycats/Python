import numpy as np
import scipy.signal as sig
import scipy
from math import factorial
from MiscToolsDS import MiscTools
from os import path

class SignalProcessTools(object):
    def __init__(self, PathType='production'):
        self.misctools = MiscTools()

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

    def flat_smooth(self, data, L=10):
        out = np.zeros(data.shape)
        for i in range(0, data.shape[0]-L):
            win = data[i:i+L]
            smoothed = np.mean(win)
            out[i:i+L] = smoothed

        return out

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


