import numpy as np
import matplotlib.pylab as plt 
from passband_modulations import msk_demod, msk_mod
from channels import awgn
from scipy.special import erfc

N = 100_000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4, stop=11, step=2) 
fc = 800
OF = 32
BER = np.zeros(len(EbN0dB))

a = np.random.randint(2, size=N) #uniform random symbols from 0's and 1's
result = msk_mod(a, fc, OF, enable_plot=True)
s = result['s(t)']

for i, EbN0 in enumerate(EbN0dB):
    # compute and add AWGN
    r = awgn(s, EbN0, OF)
    a_hat = msk_demod(r, N, fc, OF) # receiver
    BER[i] = np.sum(a != a_hat) / N # bit error rate
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB / 10))) 

#----------- Plots -------------------
fig, ax = plt.subplots(1, 1)
ax.semilogy(EbN0dB, BER, 'k*', label='Simulated') # simulated BER
ax.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
ax.set_xlabel(r'$E_b/N_0$ (dB)')
ax.set_ylabel(r'Probability of Bit Error - $P_b$')
ax.set_title("Probability of Bit error for MSK modulation")
ax.legend()
plt.show()
