# -*- coding: utf-8 -*-
# João Paulo F. Guimarães
# joao.paulo.f.guimaraes@gmail.com || joao.guimaraes@ifrn.edu.br
# Correntropy tests using python
# 10.02.2018

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from bimodal import bimodal
from mccc import mccc
from wsnr import wsnr



# meta - input data
# number of samples
input_length = 300; mu = 0; sigma = 1


# filter weights
w1 = 1+2j; w2 = -3+4j;
w = np.matrix([ [w1], [w2]] ); w_t = w.transpose(); w_h = w_t.conjugate();



nlw,ncw = w.shape



# defining the input variable x
x = np.random.normal(mu,sigma,(nlw,input_length))


# generating clean output
y = matmul(w_h,x)


# noise data
mu1 = 0; mu2 = 10
sigma1 = 0.05; sigma2 = 5

# adding additive noise to the output
# noise = bimodal(0.90,mu1,sigma1,mu2,sigma2,input_length)

# desired
d = y #+ noise

# using Maximum Correntropy Criteria MCCC to estimate w
kernel_size = 0.5; w0 = np.zeros(w.shape)

w_hat = mccc(x,d,kernel_size,w0)

print(w_hat[:,-1])

h_wsnr = wsnr(w,w_hat); h_wsnr = np.transpose(h_wsnr)


# plotting routine


# tranposing results
y_t = y.transpose(); d_t = d.transpose()



fig1= plt.figure()
fig1.suptitle('input data')

plt.plot(y_t, 'o-',label = 'original signal')
plt.plot(d_t, 'x--',label = 'noisy signal')
plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('amplitude')


fig2 = plt.figure()
plt.plot(h_wsnr,label = ['MCCC with kernel size', kernel_size])
plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('WSNR')


plt.show()







