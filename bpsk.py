import numpy as np
import matplotlib.pyplot as plt # for plotting functions
from passband_modulations import bpsk_demod, bpsk_mod
from channels import awgn
from scipy.special import erfc


N = 100_000 # Number of symbols to transmit
EbN0dB = np.arange(start=4, stop=11, step=2) #Eb/N0 range in dB for simulation
L = 16 #oversampling factor L= Tb/Ts (Tb=bit period, Ts=sampling period)
# if a carrier is used use L = Fs/Fc where Fs >> 2*Fc
Fc = 800 # carrier frequency
Fs = L*Fc # sampling frequnecy
BER = np.zeros(len(EbN0dB)) # for BER values for each Eb/N0
ak = np.random.randint(2, size=N) # uniform symbol from 0's and 1's
(s_bb, t) = bpsk_mod(ak, L) # BPSK modulation (waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # with carrier
#waveforms at the transmitter
fig1, axs = plt.subplots(2, 2)

axs[0, 0].plot(t, s_bb) # baseband wfm zoomed to first 10 bits
axs[0, 0].set_xlabel("t(s)")
axs[0,1].plot(t, s) # transmitted wfm zoomed to first 10 bits
axs[0, 1].set_xlabel("t(s)") 
axs[0, 1].set_ylabel('s(t)-with carrier')
axs[0, 0].set_xlim(0, 10*L)
axs[0, 1].set_xlim(0, 10*L)
#signal constellation at trasmitter
axs[1, 0].plot(np.real(s_bb), np.imag(s_bb), 'o')
axs[1,0].set_xlim(-1.5, 1.5)
axs[1, 0].set_ylim(-1.5, 1.5)

for i, EbN0 in enumerate(EbN0dB):
    # compute and add AWGN noise
    r = awgn(s, EbN0, L) 
    
    r_bb = r*np.cos(2*np.pi*Fc*t/Fs) # recovered baseband signal 
    ak_hat = bpsk_demod(r_bb, L) # baseband correlation demodulator
    BER[i] = np.sum(ak != ak_hat)/ N # bit error rate computation
    
    # received signal waveform zoomed to first 1o bits
    axs[1,1].plot(t, r) # received signal with noise
    axs[1, 1].set_xlabel("t(s)")
    axs[1,1].set_xlim(0, 10*L)
    
# Theoretical bit/symbol error rates ---------
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/ 10))) # Theoretical bit error rate
#--------------------- plots -----------------
fig2, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.semilogy(EbN0dB, BER, 'k*', label='Simulated') # simulated BER
ax1.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
ax1.set_xlabel(r"$E_b/N_0 (db)")
ax1.set_ylabel(r"Probabilty of bit error - $P_b$")
ax1.set_title("Probability of BER for BPSK modulation")
ax1.legend()
   
   
plt.show()
