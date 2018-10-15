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
input_length = 10; mu = 5; sigma = 10


# filter weights
w = np.matrix('1;2'); w_t = w.transpose()

nlw,ncw = w.shape



# defining the input variable x
x = np.random.normal(mu,sigma,(nlw,input_length))


# generating clean output
y = matmul(w_t,x)


# noise data
mu1 = 0; mu2 = 30
sigma1 = 5; sigma2 = 1

# adding additive noise to the output
noise = bimodal(0.9,mu1,sigma1,mu2,sigma2,input_length)

# desired
d = y + noise

# using Maximum Correntropy Criteria MCCC to estimate w
sigma = 1; w0 = np.zeros(w.shape);

w_hat = mccc(x,d,sigma,w0);

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







