import numpy as np
from modem import Modem

class PSKModem(Modem):
    def __init__(self, M):
        m = np.arange(0, M)
        I = (1/ np.sqrt(2))*np.cos(m/M*2*np.pi)
        Q = 1/np.sqrt(2)*np.sin(m/M*2*np.pi)
        constellation = I + 1j*Q # reference constellation
        Modem.__init__(self, M, constellation, name='PSK') # set the modem attribute


'''
In any qam constellation in order to restrict the erroneous symbol to signle bit errors the adjacent
symbols in the trasmitter constellation should not differ by more than one bit this is usually
achieved by converting the input symbols to gray coded 
'''