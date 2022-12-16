import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import bpsk_demod, bpsk_mod
from channels import awgn
from scipy.signal import lfilter
from scipy.special import erfc

N = 100_000 # number of symbols to transmit
EbN0dB = np.arange(start=4, stop=11, step=2) # Eb/N0 range in dB for simulation
L = 10 # oversampling factor L=Tb/Ts (Tb= bit period, Ts= sampling period)
Fc = 800 # carrier frequency
Fs = L*Fc # sampling frequncy
SER = np.zeros(len(EbN0dB)) # for SER values for each Eb/N0
ak = np.random.randint(2,size=N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0], [1.0, -1.0], ak) # IIR filter for differential encoding
bk = bk % 2 # XOR operation is equivalent to module 2
[s_bb, t] = bpsk_mod(bk, L) # BPSK modulation - baseband
s = s_bb * np.cos(2*np.pi*Fc*t/Fs) # DEBPSK with carrier
for i, EbN0 in enumerate(EbN0dB):
    # compute and add AWGN noise
    r = awgn(s, EbN0, L) 
    
    phaseAmbiguity = np.pi # 180 phase ambiguity of costas loop
    r_bb = r*np.cos(2*np.pi*Fc*t/Fs + phaseAmbiguity) # recovered signal
    b_hat = bpsk_demod(r_bb, L) # base band correlation detector
    a_hat = lfilter([1.0, 1.0], [1.0], b_hat) # FIR for differential decoding
    SER[i] = np.sum(ak != a_hat) / N # symbol error rate computation
    
    #------------- Theoretical Bit/Symbol error rate ------------------
    EbN0lins = 10**(EbN0dB / 10) # converting dB values to linear scale
    theorySER_DPSK = erfc(np.sqrt(EbN0lins)) * (1 - 0.5*erfc(np.sqrt(EbN0lins)))
    theorySER_BPSK = 0.5*erfc(np.sqrt(EbN0lins))
    
#-------------plots-------------------------
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(EbN0dB, SER, 'k*', label='Coherent DEBPSK(sim)')
ax.semilogy(EbN0dB, theorySER_DPSK, 'r-', label='Coherent DEBPSK(theory)')
ax.semilogy(EbN0dB, theorySER_BPSK, 'b-', label='Conventional BPSK')
ax.set_title('Probability of Bit Error for BPSK over AWGN')
ax.set_xlabel(r"$E_b/N_0$ (dB)")
ax.set_ylabel(r"Probability of Bit Error - $P_b$")
ax.legend()
plt.show()
