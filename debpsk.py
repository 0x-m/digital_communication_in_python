import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import bpsk_demod, bpsk_mod
from channels import awgn
from scipy.signal import lfilter
from scipy.special import erfc


N = 100_000 # nubmer of symbols to transmit
EBN0dB = np.arange(start=-4, stop=11, step=2) # Eb/N0 range in dB for simulation
L = 16 # ovesampling factor L = Tb/Ts 
# if a carrier is used use L = Fs/Fc where Fs >> 2*Fc

Fc = 800 # carrier fequency
Fs = L*Fc # sampling frequency

SER = np.zeros(len(EBN0dB)) # for SER values fro each Eb/N0

ak = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0], [1.0,-1.0], ak) # IIR filter for differential encoding
bk = bk % 2 # XOR operation is equivalent to modulo 2 