import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import oqpsk_demod, oqpsk_mod
from channels import awgn
from scipy.special import erfc


N = 100_000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4, stop=11, step=2) # Eb/N0 range in dB for simulation
fc = 100 # carrier frequnecy in hertz
OF = 8 # oversampling factor sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # for BER values for each Eb/N0

a = np.random.randint(2, size=N) # uniform symbols from 0's and 1's
result = oqpsk_mod(a, fc, OF, enable_plot=False) # QPSK modulation
s = result['s(t)']

for i, EbN0 in enumerate(EbN0dB):
    # compute and add AWGN noise
    r = awgn(s, EbN0, OF) 
    a_hat = oqpsk_demod(r, N, OF, enable_plot=True) # QPSK demodulation
    BER[i] = np.sum(a != a_hat) / N # bit Error rate
    
# ------------ Theoretical Bit Error Rate ------------
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/ 10)))
# ----------------------------------------------------

#------------------ Plot performanece curve ----------
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.semilogx(EbN0dB, BER, 'k*', label='Simulated')
axs.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
axs.set_title("Probability of Bit Error for OQPSK")
axs.set_xlabel(r"$E_b/N_0$ (dB)")
axs.set_ylabel(r'Probablity of Bit error - $P_b$')
axs.legend()
fig.show()
