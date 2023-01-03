import numpy as np
from modem import Modem

class FSKModem(Modem):
    def __init__(self, M, coherence='coherent'):
        if coherence.lower() == 'coherent':
            phi = np.zeros(M) # phase = 0 for coherent detection
        elif coherence.lower() == 'noncoherent':
            phi = 2*np.random.rand(M) # M random phases in the (0, 2pi)
        else:
            raise ValueError('Coherence must be ...')
        Modem.__init__(self, M, constellation, name='FSK', coherece=coherence)
    


'''
the modulated signal from the transmitter needs to be added with random specific noise
of specific strength the strength of the generated noise depends on the desired snr level 
which usually is an input in such a simulation in practice snrs are specific in dB given a
specific snr point for simulation let's see how we can simulate an awgn channel that adds
correct level
for generating gaussian random noise is given by
deonting the symbol error rate as Ps snr per bit as yb and snr per symbol as 
the symbol rates for varous modulation schmese over awgns 
'''
            
        