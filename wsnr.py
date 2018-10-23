# joao.paulo.f.guimaraes@gmail.com
# Weighed Signal to error Ratio
# 23.10.2018
from math import log10
import numpy as np


def wsnr(w,w_hat):

    w_t = np.transpose(w)

    (p,N) = w_hat.shape

    log_wsnr = np.matrix(np.zeros((1,N)))
    c = 0
    while c<N:
        e = w-w_hat[:,c]
        e_c = np.conjugate(e)
        e_h = np.transpose(e_c)
        log_wsnr[:,c] = 10*log10((w_t.dot(w))/(e_h.dot(e)))

        c = c+1

    
    return log_wsnr
        
