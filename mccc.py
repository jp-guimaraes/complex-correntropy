# joao.paulo.f.guimaraes@gmail.com
# Fixed point using the Maximum Correntropy Criteria
# 05/10/2018
import random
import math
import numpy as np


def mccc(x,d,sigma,w0):

    N = d.size

    w = w0    
    wl,wc = w.shape

    # record the estimated w at each step
    h_w = np.matrix(np.zeros((wl,N)))
    h_w[:,0] = w0


    # auxiliar matrix
    R_ = 1e-4*np.matrix(np.eye(wl))
    P_ = 1e-4*np.matrix(np.ones((wl,1)))



   
    
    return h_w
        
