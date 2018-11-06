# joao.paulo.f.guimaraes@gmail.com
# Weighed Signal to error Ratio
# 23.10.2018
from math import log10
import numpy as np


def wsnr(w,w_hat):

    w_t = np.transpose(w); w_h = np.conjugate(w_t)

    (p,N) = w_hat.shape

    log_wsnr = np.matrix(np.zeros((1,N)))
    c = 0
    while c<N:
        e = w-w_hat[:,c]
        e_c = np.conjugate(e)
        e_h = np.transpose(e_c)
        # bug: w^h*w, e^h *e always real, i.e. [1+0j]
        a = np.real(w_h.dot(w))
        b = np.real(e_h.dot(e))
        log_wsnr[:,c] = 10*log10(a/b)

        c = c+1

    
    return log_wsnr
        
