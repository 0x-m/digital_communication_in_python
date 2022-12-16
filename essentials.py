import numpy as np


def plotWelchPSD(x, fs, fc, ax=None, color="b", label=None):
    """
    Plot PSD of a carrier modulated signal using Welch estiamte
    Parameters:
        x: signal vector (numpy array) for which the PSD is plotted
        fs: sampling frequency
        fc: center carrier frequency of the signal
        ax: Matplotlib axes object refernce for plotting
        color: color character

    """

    from scipy.signal import hanning, welch
    from numpy import log10

    nx = max(x.shape)
    na = 16  # averaging factor to plot averaged welch spectrum
    w = hanning(nx // na)
    # Welch PSD estimate with Hanning window and no overlap
    f, Pxx = welch(x, fs, window=w, noverlap=0)
    indices = (f >= fc) & (f < 4 * fc)  # To plot PSD from Fc to 4*Fc
    Pxx = Pxx[indices] / Pxx[indices][0]  # normalized psd w.r.t Fc
    ax.plot(f[indices] - fc, 10 * log10(Pxx), color, label=label)

    """
    energy electrical signal per unit of time this quantity is useful if the energy of the signl
    goes to infinty or the signal is not for no 
    a signal can not be both an energy signal and and power signal 
    strongly rooted in operations on polynomials
    this operation is reffered as linear convolution denoted by the symbol * it is very
    closely relatd to other operations on vectors like cross correlation auto correlation
    and moving average computation thus when we are computig convolution we are actualy multiplying two polynomials 
    note that if the polynomials have N and M terms their multiplication products N + M -1 terms
    
    """


def analytic_signal(x):
    """
    Generate analytic signal using frequency domain approach
    Parameters:
        x: signal data. must be real
    returns:
        z: analytic signal of x
    """

    from scipy.fftpack import fft, ifft

    N = len(x)
    X = fft(x, N)
    Z = np.hstack((X[0], 2 * X[1 : N // 2], X[N // 2], np.zeros(N // 2 - 1)))
    z = ifft(Z, N)
    return z