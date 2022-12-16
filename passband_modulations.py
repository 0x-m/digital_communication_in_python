import numpy as np
import matplotlib.pyplot as plt


def bpsk_mod(ak, L):
    """
    Function to modulate an incoming binary stream using BPSK (baseband)
    Parameters:
        ak: input binary data stream to modulate
        L: oversampling factor (Tb/Ts)
    Returns:
        (s_bb, t) : tuple of following variables
            s_bb: BPSK modulated signal (baseband)
            t: generated time base for the modulated signal

    """

    from scipy.signal import upfirdn

    s_bb = upfirdn(h=[1] * L, x=2 * ak - 1, up=L)  # NZR encoder
    t = np.arange(start=0, stop=len(ak) * L)  # discrete time base
    return (s_bb, t)


def bpsk_demod(r_bb, L):
    """
    Function to demodulate a BPSK signal
    Parameters:
        r_bb: received signal at the receiver front end (baseband)
        L: oversampling factor(Tsym/ Ts)
    Return:
        ak_hat: detected/ estimated binary stream

    """
    x = np.convolve(r_bb, np.ones(L))  # integrate for Tb duration (L samples!)
    x = x[L - 1 : -1 : L]  # I arm - samle at every L
    ak_hat = (x > 0).transpose()  # threshold detector
    return ak_hat


def qpsk_mod(a, fc, OF, enable_plot=False):
    """
    Modulate an incoming binary stream using conventional QPSK
    parameters:
        a: input arra data stream
        fc: carrier frequency in Hertz
        OF: oversampling factor - at least 4 is better
        enable_plot: True = plot transmitter waveform (default False)
    Returns:
        result: Dictionary contaning the following keyword entries:
        s(t): QPSK modulated signal vector with carrier s(t)
        I(t): baseband I channel waveform (no carrier)
        Q(t): baseband Q channel waveform (no carrier)
        t: time base for the carrier modulate signal

    """
    L = 2 * OF  # samples in each symbol (QPSK has 2 bits in each symbol)
    I, Q = a[0::2], a[1::2]  # even and odd bit streams

    from scipy.signal import upfirdn

    I = upfirdn(h=[1] * L, x=2 * I - 1, up=L)  # NRZ encoder
    Q = upfirdn(h=[1] * L, x=2 * Q - 1, up=L)
    fs = OF * fc  # sampling frequency
    t = np.arange(0, len(I) / fs, 1 / fs)  # time base
    I_t = I * np.cos(2 * np.pi * fc * t)
    Q_t = -Q * np.sin(2 * np.pi * fc * t)
    s_t = I_t + Q_t  # QPSK modulated baseband signal
    result = {"s(t)": s_t, "I(t)": I, "Q(t)": Q, "t": t}
    return result


def qpsk_demod(r, fc, OF, enable_plot=False):
    """
    Demodulate a conventional QPSK signal
    Parameters:
        r: receive signal at the receiver front end
        fc: carrier frequency (Hz)
        OF: oversampling factor (at least 4 is better)
        enable_plot: True = plot receiver waveforms (default False)
    Returns:
        a_hat: detected binary stream
    """
    fs = OF * fc  # sampling frequency
    L = 2 * OF  # number of samples in 2Tb duration
    t = np.arange(start=0, stop=len(r) / fs, step=1 / fs)  # time base
    x = r * np.cos(2 * np.pi * fc * t)  # I arm
    y = -r * np.sin(2 * np.pi * fc * t)  # Q arm
    x = np.convolve(x, np.ones(L))  # intergrate for L
    y = np.concatenate(y, np.ones(L))  # intergrate for L

    x = x[L - 1 :: L]  # Ia rm - sample at every symbol instant
    y = y[L - 1 :: L]  # Q arm sample at every L
    a_hat = np.zeros(2 * len(x))
    a_hat[0::2] = x > 0  # even bits
    a_hat[1::2] = y > 0  # odd bits
    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(x[0:200], y[0:200], "o")
        fig.show()
    return a_hat


def oqpsk_mod(a, fc, OF, enable_plot=False):
    """
    Modulate an incoming binary stream using qpsk
    Parameters:
        a: input binary frequency
        fc: carrier frequency in hertz
        enable_plot: true= plot transmitter waveform
    Returns:
        result:
            s(t): QPSK modulated signal vector
            I(t): baseband I channel waveform (no carrier)
            Q(t): baseband Q channel waveform (no carrier)
            t: time base for the carrier modulated signal
    """
    L = 2 * OF  # samples in each symbol
    I = a[0::2]  # even bit stream
    Q = a[1::2]  # odd bit stream
    # even/odd bit stream at 1/2Tb baud
    from scipy.signal import upfirdn

    I = upfirdn(h=[1] * L, x=2 * I - 1, up=L)
    Q = upfirdn(h=[1] * L, x=2 * Q - 1, up=L)

    I = np.hstack((I, np.zeros(L // 2)))  # padding at end
    Q = np.hstack((np.zeros(L // 2), Q))  # padding at start

    fs = OF * fc  # sampling frequency
    t = np.arange(0, len(I) / fs, 1 / fs)  # time base
    I_t = I * np.cos(2 * np.pi * fc * t)
    Q_t = -Q * np.sin(2 * np.pi * fc * t)
    s = I_t + Q_t  # QPSK modulated baseband signal

    if enable_plot:
        fig = plt.figure(constrained_layout=True)

        from matplotlib.gridspec import GridSpec

        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1, :])

        # show first few symbols
        ax1.plot(t, I)
        ax1.set_title("I(t)")
        ax2.plot(t, Q)
        ax2.set_title("Q(t)")
        ax3.plot(t, I_t, "r")
        ax3.set_title(r"$I(t) cos(2 \pi f_c t)$")
        ax4.plot(t, Q_t, "r")
        ax4.set_title(r"$Q(t) sin(2 \pi f_c t)")
        ax1.set_xlim(0, 20 * L / fs)
        ax2.set_xlim(0, 20 * L / fs)
        ax5.plot(t, s)
        ax5.set_xlabel(0, 20 * L / fs)
        fig.show()
        ax5.set_title(r"$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$")

        fig, axs = plt.subplots(1, 1)
        axs.plot(I, Q)
        fig.show()

    return {"s(t)": s, "I(t)": I, "Q(t)": Q, "t": t}


def oqpsk_demod(r, N, fc, OF, enable_plot):
    """
    Demodulate a QPSK signal
    Parameters:
        r: received signal at the receiver front end
        N: Number of OQPSK symbols transimitted
        fc: carrier frequency(hz)
        OF: oversampling frequency
        enable_plot: True = plot receiver waveform (default False)
    Return:
        a_hat: detected binary stream
    """
    fs = OF * fc  # sampling frequency
    L = 2 * OF  # number of samples in 2Tb duration
    t = np.arange(0, (N + 1) * OF / fs, 1 / fs)  # time base
    x = r * np.cos(2 * np.pi * fc * t)  # I arm
    y = -r * np.sin(2 * np.pi * fc * t)  # Q arm
    x = np.convolve(x, np.ones(L))  # intergrate for L (Tsymb = 2 * Tb) duration
    y = np.convolve(y, np.ones(L))  # intergrate for L (Tsym=2*Tb) duration

    x = x[L - 1 : -1 - L : L]  # I arm - sample at every symbol instant Tsym
    y = y[L + L // 2 - 1 : -1 - L : L]  # Q arm sample at every symbol
    a_hat = np.zeros(N)
    a_hat[0::2] = x > 0  # even bits
    a_hat[1::2] = y > 0  # odd bits

    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(x[0:200], y[0:200], "o")
        fig.show()
    return a_hat


def piBy4_dqpsk_diff_encoding(a, enable_plot=False):
    """
    Phase mapper for pi/4 dqpsk modulation
    Parameters:
        a: input stream of bianry bits
    Returns:
        (u, v): tuple, where
            u: differentially coded I-channel bits
            v: differentially coded Q-channel bits
    """
    from numpy import pi, cos, sin

    if len(a) % 2:
        raise ValueError("Length of binary stream must be even.")

    I = a[0::2]  # odd bit stream
    Q = a[1::2]  # even bit stream

    # club t-bits to form a symbol and use it as index for dTheta table
    m = 2 * I + Q
    dTheta = np.array([-3 * pi / 4, 3 * pi / 4, -pi / 4, pi / 4])  # LUT for pi/4 dqpsk
    u = np.zeros(len(m) + 1)
    v = np.zeros(len(m) + 1)
    u[0] = 1
    v[0] = 0
    for k in range(len(m)):
        u[k + 1] = u[k] * cos(dTheta[m[k]]) - v[k] * sin(dTheta[m[k]])
        v[k + 1] = u[k] * sin(dTheta[m[k]]) + v[k] * cos(dTheta[m[k]])

    if enable_plot:  # constellation plot
        fig, axs = plt.subplots(1, 1)
        axs.plot(u, v, "o")
        axs.set_title("Constellation")
        fig.show()
    return (u, v)


def piBy4_dqpsk_mod(a, fc, OF, enable_plot=False):
    """
    Modulate a binary stream using pi/4 DQPSK
    Parameters:
        a: input binary data stream (0's and 1's) to modulate
        fc: carrier frequency in Hertz
        OF: oversampling factor
    Returns:
        result: Dictionary containing the following keyword entries:
            s(t): pi/4 QPSK modulated signal vector with carrier
            u(t): differentially coed I-channel waveform (no carrier)
            v(t): differentially coded Q-channel waveform (no carrier)
            t: time base
    """
    (u, v) = piBy4_dqpsk_diff_encoding(a)  # Differentially Encoding for pi/4 QPSK
    # waveform formation (similiar to conventional QPSK)
    L = 2 * OF  # Number of samples in each symbol (QPSK has 2 bits/sybmol)
    U = np.tile(u, (L, 1)).flatten("F")  # odd bits steram at 1/2Tb baud
    V = np.tile(v, (L, 1)).flatten("F")  # even bit steream at 1/2Tb baud

    fs = OF * fc  # sampling frequency
    t = np.arange(0, len(U) / fs, 1 / fs)  # time base
    U_t = U * np.cos(2 * np.pi * fc * t)
    V_t = -V * np.sin(2 * np.pi * fc * t)
    s_t = U_t + V_t

    if enable_plot:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[-1, :])
        ax1.plot(t, U)
        ax2.plot(t, V)
        ax3.plot(t, U_t, "r")
        ax4.plot(t, V_t, "r")
        ax5.plot(t, s_t)  # QPSK waveform zoomed to first few symbols
        ax1.set_ylabel("U(t)-baseband")
        ax2.set_ylabel("V(t)-baseband")
        ax3.set_ylabel("U(t)-with carrier")
        ax4.set_ylabel("V(t)-with carrier")
        ax5.set_ylabel("s(t)")
        ax5.set_xlim([0, 10 * L / fs])
        ax1.set_xlim([0, 10 * L / fs])
        ax2.set_xlim([0, 10 * L / fs])
        ax3.set_xlim([0, 10 * L / fs])
        ax4.set_xlim([0, 10 * L / fs])
        fig.show()
    return {"s(t)": s_t, "U(t)": U, "V(t)": V, "t": t}


def piBy4_dpsk_diff_decoding(w, z):
    """
    Phase Mapper for pi/4 DQPSK modulation
    Parameters:
        w - differentially coded I-channel bits the receiver
        z - differentially coded Q-channel bits at the receiver
    Returns:
        a_hat - binary bit stream after differentially decoding
    """
    if len(w) != len(z):
        raise ValueError("Length mismatch between w and z")
    x = np.zeros(len(w) - 1)
    y = np.zeros(len(w) - 1)
    for k in range(0, len(w) - 1):
        x[k] = w[k + 1] * w[k] + z[k + 1] * z[k]
        y[k] = z[k + 1] * w[k] - w[k + 1] * w[k]

    a_hat = np.zeros(2 * len(x))  # odd bits
    a_hat[0::2] = x > 0  # odd bits
    a_hat[1::2] = y > 0  # even bits
    return a_hat


def piBy4_dqpsk_demod(r, fc, OF, enable_plot=False):
    """
    Differentially coherent demodulation of pi/4 dqpsk
    Parameters:
        r: received signal at the receiver front end
        fc: carrier frequency in hertz
        OF: oversampling factor (multipliers of fc) - at least 4 is better
    Retursn:
        a_cap: detected binary stream
    """

    fs = OF * fc  # sampling frequnecy
    L = 2 * OF  # samples in 2Tb duration
    t = np.arange(0, len(r) / fs, 1 / fs)
    w = r * np.cos(2 * np.pi * fc * t)  # I arm
    z = -r * np.sin(2 * np.pi * fc * t)  # Q arm
    W = np.convolve(w, np.ones(L))  # intergate for L
    Z = np.convolve(z, np.ones(L))
    w = W[L - 1 :: L]  # I arm sample at every symbol instant tsym
    z = Z[L - 1 :: L]  # Q arm sample at every symbol instant Tsym
    a_cap = piBy4_dpsk_diff_decoding(w, z)

    if enable_plot:
        fig, axs = plt.subplots(1, 1)
        axs.plot(w, z, "o")
        axs.set_title("Constellation")
        fig.show()
    return a_cap


def msk_mod(a, fc, OF, enable_plot=False):
    """
    Modulate an incoming binary stream using MSK
    Parameters:
        a: input binary data stream (0's and 1's) to modulate
        fc: carrier frequency in Hertz
        OF: ovesampling factor (at least 4 is better)
    Results:
        result: dictionary containing the following keyword entries:
            s(t): MSK modulated signal
            sI(t): baseband I channel waveform (no carrier)
            sQ(t): base bad Q channel waveform (no carrier)
            t: time base
    """
    ak = 2 * a - 1  # NRZ encoding 0-> -1 , 1 -> 1
    ai = ak[0::2]
    aq = ak[1::2]
    L = 2 * OF  # represent one symbol duration Tsym = 2*Tb

    # upsample by L the bits streams in I and Q arm
    from scipy.signal import upfirdn, lfilter

    ai = upfirdn(h=[1], x=ai, up=L)
    aq = upfirdn(h=[1], x=aq, up=L)

    aq = np.pad(aq, (L // 2, 0), "constant")  # delay aq by Tb
    ai = np.pad(ai, (0, L // 2), "constant")  # padding at end to equl length of Q

    # construct Low-pass filter and filter the I/Q samples through it
    Fs = OF * fc
    Ts = 1 / Fs
    Tb = OF * Ts
    t = np.arange(0, 2 * Tb + Ts, Ts)
    h = np.sin(np.pi * t / (2 * Tb))  # LPF filter
    sI_t = lfilter(b=h, a=[1], x=ai)  # baseband I-channel
    sQ_t = lfilter(b=h, a=[1], x=aq)  # baseband Q-channel

    t = np.arange(0, Ts * len(sI_t), Ts)  # for RF carrier
    sIc_t = sI_t * np.cos(2 * np.pi * fc * t)  # with carrier
    sQc_t = sQ_t * np.sin(2 * np.pi * fc * t)  # with carrier
    s_t = sIc_t - sQc_t  # Bandpass MSK modulated signal

    if enable_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(t, sI_t)
        ax1.plot(t, sIc_t, "r")
        ax2.plot(t, sQ_t)
        ax2.plot(t, sQc_t, "r")
        ax3.plot(t, s_t, "--")
        ax1.set_xlabel("$s_I(t)$")
        ax2.set_xlabel("$s_Q(t)$")
        ax3.set_ylabel("s(t)")
        ax1.set_xlim([-Tb, 20 * Tb])
        ax2.set_xlim([-Tb, 20 * Tb])
        ax3.set_xlim([-Tb, 20 * Tb])
        fig.show()
    return {"s(t)": s_t, "sI(t)": sI_t, "sQ(t)": sQ_t, "t": t}


def msk_demod(r, N, fc, OF):
    """
    MSK demodulator
    Parameters:
        r: received signal at the receiver front end
        N: number of symbols transmitted
        fc: carrier frequency
        OF: oversampling factor
    Returns:
        a_hat: detected binary stream
    """

    L = 2 * OF  # samples in 2Tb duration
    Fs = OF * fc
    Ts = 1 / Fs
    Tb = OF * Ts
    t = np.arange(-OF, len(r) - OF) / Fs  # time base

    # cosine and sine function for half-sinusoid shaping
    x = abs(np.cos(np.pi * t / (2 * Tb)))
    y = abs(np.sin(np.pi * t / (2 * Tb)))

    u = r*x*np.cos(2*np.pi*fc*t) # multiply I by half cosines and cos(2pifct)
    v = -r*y*np.sin(2*np.pi*fc*t) # multiply Q by half sines and sin(2pifct)
    
    iHat = np.convolve(u, np.ones(L)) # integrate for L 
    qHat = np.convolve(v, np.ones(L))

    iHat = iHat[L-1:-1-L:L] # I sample at the end of every symbol
    qHat = qHat[L + L//2 - 1: -1 - L//2: L] # Q sample from L + L//2th sample

    a_hat = np.zeros(N)
    a_hat[0::2] = iHat > 0 # thresholding even bits
    a_hat[1::2] = qHat > 0 # thesholding odd bits
    
    return a_hat
