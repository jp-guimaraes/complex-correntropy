# João Paulo F. Guimarães
# joao.paulo.f.guimaraes@gmail.com || joao.guimaraes@ifrn.edu.br
# Correntropy tests using python


# from pylab import *
import numpy as np
import matplotlib.pyplot as plt
# from bimodal import bimodal
from mccc import mccc
from wsnr import wsnr
from bimodal import bimodal

# seed
np.random.seed(seed=1)

# meta - input data

# filter weights
w1 = 1+2j; w2 = -3+4j


w = np.matrix([ [w1], [w2]] ); w_t = w.transpose(); w_h = w_t.conjugate()
nlw,ncw = w.shape
print('Right values:')
print(w1)
print(w2)


# number of samples; mean and var from the input data
input_length = 500; mu = 0; sigma = 1

# defining the input variable x
x = np.random.normal(mu,sigma,(nlw,input_length)) + 1j*np.random.normal(mu,sigma,(nlw,input_length))


# generating clean output
y = np.matmul(w_h,x)


# noise data real part
mu1 = 0; mu2 = 10
sigma1 = 0.05; sigma2 = 5

# adding additive noise to the output
real_noise = bimodal(0.95,mu1,sigma1,mu2,sigma2,input_length)


# noise data imaginary part
mu1 = 0; mu2 = 1
sigma1 = 0.05; sigma2 =25


imag_noise = bimodal(0.90,mu1,sigma1,mu2,sigma2,input_length)
noise = real_noise + 1j*imag_noise

# desired
d = y + noise

# using Maximum Correntropy Criteria MCCC to estimate w
kernel_size = 1; w0 = np.zeros(w.shape)

w_hat = mccc(x,d,kernel_size,w0)

print('Estimated values: ')
print(w_hat[:,-1])

h_wsnr = wsnr(w,w_hat); h_wsnr = np.transpose(h_wsnr)


# plotting routine


# tranposing results (just for visualization)
y_t = y.transpose(); d_t = d.transpose()


fig1= plt.figure()
fig1.suptitle('input data: real part')
plt.plot(np.real(y_t), 'o-',label = 'original signal - real part')
plt.plot(np.real(d_t), 'x--',label = 'noisy signal - real part')

plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('amplitude')


fig2= plt.figure()
fig2.suptitle('input data: imaginary part')
plt.plot(np.imag(y_t), 'o-',label = 'original signal - imaginary part')
plt.plot(np.imag(d_t), 'x--',label = 'noisy signal - imaginary part')

plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('amplitude')


fig3 = plt.figure()
plt.plot(h_wsnr,label = ['MCCC with kernel size', kernel_size])
plt.grid(True)
plt.legend()
plt.xlabel('sample')
plt.ylabel('WSNR')


plt.show()






