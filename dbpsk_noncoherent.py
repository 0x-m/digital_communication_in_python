import numpy as np
import matplotlib.pyplot as plt
from passband_modulations import bpsk_mod
from channels import awgn
from scipy.signal import lfilter
from scipy.special import erfc

N = 100_000  # number of symbols to transmit
EbN0dB = np.arange(start=-4, stop=11, step=2)  # Eb/N0 range in dB for simulation
L = 8  # oversampling factor, L = Tb/Ts
Fc = 800
Fs = L * Fc  # sampling frequency

BER_suboptimum = np.zeros(len(EbN0dB))  # BER measure
BER_optimum = np.zeros(len(EbN0dB))

# ------------  Transmitter ----------------------
ak = np.random.randint(2, size=N)  # uniform random symbols from 0's and 1's
bk = lfilter([1.0], [1.0, -1.0], ak)  # IIR filter for differential encoding
[s_bb, t] = bpsk_mod(bk, L)  # BPSK modulation (waveform) - baseband
s = s_bb * np.cos(2 * np.pi * Fc * t / Fs).astype(complex)  # DBPSK with carrier

for i, EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, L)

    # ------------------ suboptimum receiver -----------------
    p = np.real(r) * np.cos(
        2 * np.pi * Fc * t / Fs
    )  # demodulated to baseband using BPF
    w0 = np.hstack((p, np.zeros(L)))  # append L samples on one arm for equal length
    w1 = np.hstack((np.zeros(L), p))  # delay the other arm by Tb (L samples)
    w = w0 * w1  # multiplier
    z = np.convolve(w, np.ones(L))  # intergrator from kTb to (k+1)Tb (L samples)
    u = z[L - 1 : -1 - L : L]  # sampler t=KTb
    ak_hat = u < 0  # decision
    BER_suboptimum[i] = np.sum(ak != ak_hat) / N  # BER for suboptimum receiver

    # -------------- optimum receiver ----------------
    p = np.real(r) * np.cos(2 * np.pi * Fc * t / Fs)  # multiply I arm by cos
    q = np.imag(r) * np.sin(2 * np.pi * Fc * t / Fs)  # multiply Q arm by sin
    x = np.convolve(p, np.zeros(L))  # integrate I-arm by Tb duration (L samples)
    y = np.convolve(q, np.ones(L))  # integrate Q-arm by Tb duration (L samples)
    xk = x[L - 1 : -1 : L] # sample every kth sample
    yk = y[L - 1: -1 : L] # sample every kth sample
    w0 = xk[0:-2] #non delayed version on I-arm
    w1 = x[1:-1] # delayed version on I-arm
    z0 = yk[0:-2] # non delayed version onQ-arm
    z1 = yk[1:-2] #delayed version on Q-arm
    u = w0*w1 + z0*z1 # decision statistic
    ak_hat = (u < 0) # threshold detection
    BER_optimum[i] = np.sum(ak[1:-1] != ak_hat) / N # BER for optimum receiver
    

#--------------- Theroretical Bit/Symbol error rate -------------
EBN0lins = 10**(EbN0dB / 10) # converting dB values to linear scale
theory_DBPSK_optimum = 0.5*np.exp(-EBN0lins)
theory_DBPSK_suboptimum = 0.5*np.exp(-0.76*EBN0lins)
theory_DBPSK_coherent = erfc(np.sqrt(EBN0lins))* (1 - 0.5*erfc(np.sqrt(EBN0lins)))
theory_DBPSK_conventional = 0.5*erfc(np.sqrt(EBN0lins))

#------------- Plotting --------------------
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(EbN0dB, BER_suboptimum, 'k*', label="DBPSK subplot (sim)")
ax.semilogy(EbN0dB, BER_optimum, 'b*', label='DBPSK opt (sim)')