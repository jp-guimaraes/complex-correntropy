# joao.paulo.f.guimaraes@gmail.com
# Fixed point using the Maximum Correntropy Criteria
# 10/04.2018
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

    n=0
    while(n<N):
        
        # input data at iteration 'n'
        xn = x[:,n]; xn = xn.reshape(2,1)
        xn_H = np.conjugate(np.transpose(xn))

        # desired signal at iteration 'n'
        dn = d[:,n]; 
        dn_c = np.conjugate(dn)
        
        # transposing w
        w_t = np.transpose(w)
        
        # calculating error and the error conjugate
        e = dn - w_t.dot(xn); e_c = np.conjugate(e)
        
        # exponential part
        B = -0.5*(e*e_c)/(math.pow(sigma,2))
        expB = math.exp(B)       
        
            
        P = P_ + np.multiply(expB*dn_c,xn)
        R = R_ + expB*xn*xn_H; R_i = np.linalg.inv(R)
        
        w = R_i*P
        
        # old R and P updated with the new values
        R_ = R; P_ = P

        h_w[:,n] = w;

        n = n+1


   
    
    return h_w
        
