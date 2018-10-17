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



# meta - input data
# number of samples
input_length = 300; mu = 0; sigma = 1


# filter weights
w = np.matrix('1.2345;2.6789'); w_t = w.transpose()

nlw,ncw = w.shape



# defining the input variable x
x = np.random.normal(mu,sigma,(nlw,input_length))


# generating clean output
y = matmul(w_t,x)


# noise data
mu1 = 0; mu2 = 10
sigma1 = 0.05; sigma2 = 5

# adding additive noise to the output
noise = bimodal(0.90,mu1,sigma1,mu2,sigma2,input_length)

# desired
d = y + noise

# using Maximum Correntropy Criteria MCCC to estimate w
kernel_size = 0.5; w0 = np.zeros(w.shape)

w_hat = mccc(x,d,kernel_size,w0)

print(w_hat)


# plotting routine


# tranposing results
y_t = y.transpose(); d_t = d.transpose()


fig = plt.figure()
fig.suptitle('input data')

plt.plot(y_t, 'o-',label = 'original signal')
plt.plot(d_t, 'x--',label = 'noisy signal')
plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.show()







