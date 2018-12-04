# joao.paulo.f.guimaraes@gmail.com
# bimodal noise function
# 04/10/2018
import random
import math
import numpy as np

# perc * N(mu1,var1) + (1-per)* N(mu2,var2)
def bimodal(perc,mu1,sigma1,mu2,sigma2,input_length):


	# cast necessary when using linux (?)
    size1 = int(math.floor(perc*input_length))
    size2 = int(input_length - size1)

    mode1 = np.random.normal(mu1,sigma1,size1)
    mode2 = np.random.normal(mu2,sigma2,size2)    
    noise = np.concatenate((mode1,mode2),axis=0)    
    np.random.shuffle(noise)
    return noise
        
