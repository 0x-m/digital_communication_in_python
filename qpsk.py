import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import qpsk_demod, qpsk_mod
from channels import awgn
from scipy.special import erfc

N = 100_000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4, stop=11, step=2) 
fc = 100 # carrier frequency in hertz
OF = 8 # oversampling factor sampling frequency will be fs=OF*fc
BER = np.zeros(len(EbN0dB)) # For BER values for each 
a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
result = qpsk_mod(a, fc, OF, enable_plot=True)
s = result['s(t)'] # get values from returned dictionary

for i, EbN0 in enumerate(EbN0dB):
    r = awgn(s, EbN0, OF) 
    a_hat = qpsk_demod(r, fc, OF)
    BER[i] = np.sum(a != a_hat) / N

# Theoretical bit error rate --------------
theoreticalBER = 00.5*erfc(np.sqrt(10**(EbN0dB/10)))
# -----------------------------------------

#-------------- Plot performance curve ----
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.semilogy(EbN0dB, BER, 'k*', label='Simulated')
axs.semilogy(EbN0dB, theoreticalBER, 'r-', 'Theoretical')
axs.set_tile("Probablity of Bit Error Rate for QPSK ")
axs.set_xlabel(r"$E_b/N_0$ (dB)")
axs.set_ylabel(r'Probability of Bit Error - $P_b$')
axs.legend()
fig.show()
