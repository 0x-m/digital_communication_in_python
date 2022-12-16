import numpy as np
import matplotlib.pyplot as plt


def bpsk_qpsk_ms_psd():
    from passband_modulations import bpsk_mod, qpsk_mod, msk_mod
    from essentials import plotWelchPSD

    N = 100_000  # Number of symbols to transmit
    fc = 800  # carrier frequency
    OF = 8  # oversampling factor
    fs = fc * OF  # sampling frequency

    a = np.random.randint(2, size=N)  # uniform symbols from 0's and 1's
    (s_bb, t) = bpsk_mod(a, OF)
    s_bpsk = s_bb * np.cos(2 * np.pi * fc * t / fs)
    s_qpsk = qpsk_mod(a, fc, OF)["s(t)"]
    s_msk = msk_mod(a, fc, OF)["s(t)"]

    fig, ax = plt.subplots(1, 1)
    plotWelchPSD(s_bpsk, fs, fc, ax=ax, color="b", label="BPSK")
    plotWelchPSD(s_qpsk, fs, fc, ax=ax, color="r", label="QPSK")
    plotWelchPSD(s_msk,fs,fc,ax=ax, color='k', label='MSK')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.legend()
    plt.show()

bpsk_qpsk_ms_psd()
