import numpy as np
import matplotlib.pylab as plt
from passband_modulations import qpsk_mod, oqpsk_mod, piBy4_dqpsk_mod, msk_mod
from pulseshapers import raisedCosineDesign

N = 1000 # Number of symbols to transmit keep it small and adequate
fc = 10 #carrier frequency
L = 8 # oversampling factor
a = np.random.randint(2, size=N)  # uniform random symbols from 0' and 1's

# modulate the source symbols using QPSK, pi/4-dqpsk, MSK
qpsk_result = qpsk_mod(a, fc, L)
oqpsk_result = oqpsk_mod(a, fc, L)
piBy4_dqpsk_result = piBy4_dqpsk_mod(a, fc, L)
msk_result = msk_mod(a, fc, L)

# Pulse shape the modulated waveform by convolving with RC filter
alpha = 0.3
span = 10

