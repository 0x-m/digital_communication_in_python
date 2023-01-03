import numpy as np
from modem import Modem

class QAMModem(Modem):
    def __init__(self, M):
        if (M == 1) or (np.mod(np.log2(M)) != 0): # M is not a even power of 2
            raise ValueError('Only square MQAM supported. M must be even power of 2')
        
        n = np.arange(0, M) # Sequential address from 0 to M - 1
        a = np.asarray([x ^ (x >> 1) for x in n]) # convert linear addresses to Gray code
        D = np.sqrt(M).astype(int) # Dimension of k-map NxN matrix
        a = np.reshape(a, (D,D)) # NxN gray coded matrix
        oddRows = np.arange(start=1, stop=D, step=2)
        a[oddRows, :] = np.fliplr(a[oddRows,:]) # Flip rows - KMap representaion!?
        nGray = np.reshape(a, (M)) # reshape to 1xM gray code walk on KMap ??
        
        #Construction of ideal M-QAM constellation from sqrt(M)-PAM
        (x, y) = np.divmod(nGray, D) #element-wise quotient and remainder
        Ax = 2*x+1-D # PAM Amplitudes 2d + 1 - D real axis
        Ay = 2*y+1-D # PAM Ampitudes 2d + 1 - D imag axis
        constellation = Ax + 1j*Ay
        Modem.__init__(self, M, constellation, name='QAM') # set the modem attributes
        
            