import numpy as np

from modem import Modem

class PAMModem(Modem):
    # Dervided class
    def __init__(self,M):
        m = np.arange(0, M) # all information symbols 
        constellation = 2*m+1 - M + 1j*0 # refernce constellation
        Modem.__init__(self, M, constellation, name='PAM')
    