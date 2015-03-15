from os.path import join
import logging
import numpy as np
import scipy.signal
import scipy
from scipy.stats import t as tdist, norm as ndist


log = logging.getLogger(__name__)

class MiscStats(object):
    def __init__(self):
        pass    
    
    def t_stat_rho(self, corr_matrix, dof=None,  H0=0):
        if dof is None:
            dof = corr_matrix.shape[0]
             
        t = ( (corr_matrix - H0) * np.sqrt(dof - 2)) / np.sqrt(1-corr_matrix**2)
        diag_val = (corr_matrix - H0) * np.sqrt(dof - 2)
        np.fill_diagonal(corr_matrix, diag_val)
        
        return t
    
    def get_dof(self, roi):
        # check the proper shape of roi
        # should be vectors in row, not col for this
        # in the glm roi is likely to be opposite
        if roi.shape[0] > roi.shape[1]:
            raise Exception, "The roi has the wrong shape. Switch rows with cols."
        
        rank = np.linalg.matrix_rank(roi)
        
        return rank

    def t_stat_p_value(self, t, dof=None):
        if dof is None:
            dof = t.size
        
        shape = t.shape
        
        t = t.ravel()
        n = t.size
        thirds = n / 3
        
        t_p1 = t[0:thirds]
        t_p2 = t[thirds:2*thirds]
        t_p3 = t[2*thirds:]
            
        p1 = 1.0 - tdist.cdf(t_p1, dof)  
        p2 = 1.0 - tdist.cdf(t_p2, dof) 
        p3 = 1.0 - tdist.cdf(t_p3, dof) 
        
        p = np.concatenate((p1, p2, p3), axis=0)
        p = np.reshape(p, shape)
                       
        return p

    def correct_corr_matrix(self, p_values, corr_matrix, thres=0.05):
        p_mask = np.zeros_like(p_values)
        p_mask[p_values <= thres] = 1
        corr_cor = p_mask * corr_matrix

        return corr_cor
    
    def t_test_betas(self, beta, resid, X):
        """ 
        test the parameters beta one by one - this assumes they are estimable (X full rank)
        
        beta : (pos, 1) estimated parameters
        resid : (neg, 1) estimated residuals
        X : design matrix
        """
    
        RSS = sum((resid)**2)
        neg = resid.shape[0]
        q = np.linalg.matrix_rank(X)
        df = neg-q
        MRSS = RSS/df
        
        XTX = np.linalg.pinv(X.T.dot(X))
    
        tval = np.zeros(beta.shape)
        pval = np.zeros(beta.shape)
    
    
        for idx, _ in enumerate(beta):
            c = np.zeros(beta.shape)
            c[idx] = 1
            t_num = c.T.dot(beta)
            SE = np.sqrt(MRSS* c.T.dot(XTX).dot(c))
            tval[idx] = t_num / SE
       
            pval[idx] = 1.0 - tdist.cdf(tval[idx], df)
        
        return tval, pval

    def simple_glm(self, X, Y):
        """ X:time, voxels
            Y: time, peredictors
        """
        beta   =  np.linalg.pinv(X).dot(Y)
        Yfitted =  X.dot(beta)
        resid   =  Y - Yfitted
        
        return beta, Yfitted, resid

    def fisher_trans(self, input):
        if type(input) == np.ndarray:
            input[input == 1] = 0.9999
        
        return ( 0.5 * np.log((1 + input) / (1 - input))) 
    
    def p_val_from_norm(self, z, H0=0, dof=None):
        if dof is None:
            dof = z.shape[0]
        
        return  1. - ndist.cdf(z, H0, 1./np.sqrt(dof-3))
    
    