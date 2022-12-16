import numpy as np
import matplotlib.pyplot as plt  # libray for plotting
from scipy import signal

from signalgen import sine_wave, rect_pulse, gaussian_pulse


def sine_wave_demo():
    """
    simulate a sinuisoidal signal with given sampling rate.
    """
    f = 10  # freqeuncy
    overSampRate = 30
    phase = (1 / 3) * np.pi
    ncy1 = 5 # desired number of cycles of the sine wave
    (t, g) = sine_wave(f, overSampRate, phase, ncy1)
    
    plt.plot(t, g) # plot using pyplot library from matplot lib package
    plt.title("sine wave f=" + str(f) + 'hz')
    plt.xlabel("time (s)") # x-axis label
    plt.ylabel("Amplitude" ) # y-axis label
    plt.show()


def gaus_demo():
    (t, g) = gaussian_pulse(100, 0.1)
    plt.plot(t, g)
    plt.show()


def analytic_signal_demo():
    '''
    Investigaet components of an analytic signal
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from essentials import analytic_signal
    
    t = np.arange(start=0, stop=0.5, step=0.001) # time base
    x = np.sin(2*np.pi*10*t) # real valued f = 10 hz
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    
    ax1.plot(t, x) # plot the original signal
    ax1.set_title("x[n] - real valued signal")
    ax1.set_xlabel("n")
    ax1.set_ylabel("x[n]")

    z = analytic_signal(x) # construct analytic signal
    
    ax2.plot(t, np.real(z), 'k', label="Real(z[n])")
    ax2.plot(t, np.imag(z), 'r', label='Imag(z[n])')
    ax2.set_title("Components of analytic signal")
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$z_r[n]$ and $z_i[n]$")
    ax2.legend()
    fig.show()
    plt.show()
    
    
def extract_envelope_phase():
    '''
    demostrate extraction of instantaneous amplitude and phase from the analytic signal constrcuted
    from a real valued modulated signal
    '''    
    import numpy as np
    from scipy.signal import chirp
    import matplotlib.pyplot as plt
    from essentials import analytic_signal
    
    fs = 600 # sampling frequency in hz
    t = np.arange(start=0, stop=1, step=1/fs) # time base
    a_t = 1.0 + 0.7 * np.sin(2.0*np.pi*3.0*t) # information signal
    c_t = chirp(t, f0=20, t1=t[-1], f1=80, phi=0, method='linear')
    x = a_t * c_t # modualted signal
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(x) # plot the modulaed signal
    z = analytic_signal(x) # form the analytical signal
    
    inst_amplitude = abs(z)  #envelope extraction
    inst_phase = np.unwrap(np.angle(z)) #inst phase
    inst_freq = np.diff(inst_phase / (2*np.pi) * fs) # inst frequency
    
    # Regenerate the carrier from the instantaneous phase
    extracted_carrier = np.cos(inst_phase) 
    ax1.plot(inst_amplitude, 'r') # overlay the extracte envelope
    ax1.set_title("modulated signal and extracted envelope")
    ax1.set_xlabel("n")
    ax1.set_ylabel(r"x(t) and $|z(t)|$")
    ax2.plot(extracted_carrier)
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$cos[\omega(t)]$")
    fig.show()
    plt.show()
    

def hilbert_phase_demod():
    '''
    Demonstrate simple phase demoudlation using hilbert transform
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    
    fc = 210 # carrier frequency
    fm = 10 # frequency of modulating signal
    alpha = 1 # amplitude of modulating signal
    theta = np.pi / 4 # phase offset of modulating signal
    beta = np.pi / 5 # constant carrier phase offset
    # Set true if receiver knows carrier frequency & phase offset
    receiverKnowCarrier = False
    
    fs = 8*fc # sampling frequency
    duration = 0.5 # duration of the signal
    t = np.arange(start=0, stop=duration, step=1/fs)
    
    # Phase modulation
    m_t = alpha*np.sin(2*np.pi*fm*t + theta) # modulating signal
    x =  np.cos(2*np.pi*fc*t + beta + m_t) # modulated signal
    
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t,m_t) # plot modulating signal
    ax1.set_title("Modulating signal")
    ax1.set_xlabel("t")
    ax1.set_ylabel("m(t)")

    ax2.plot(t, x) # plot modulated signal
    ax2.set_title("Modulated signal")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x(t)")
    fig1.show()
    
    # add AWGN to the transmitted signal
    mu, sigma = 0, 0.1 # noise mean and sigma
    n = mu + sigma*np.random.normal(len(t)) #awgn noise
    r = x + n # noisy received signal
    
    # demodulated of the noisy phase modulated signal
    z = hilbert(r) # form the analytical modulated signal
    inst_phase = np.unwrap(np.angle(z)) # instantaneous phase
    
    if receiverKnowCarrier: # if receiver knows the carrier freq/phase perfectly
        offsetTerm = 2*np.pi*fc*t+ beta
    else: # estimate the subtraction term
        p = np.polyfill(x=t, y=inst_phase, deg=1) # linear instantaneous phase
        # re-evaluate the offset term using the fitted values
        estimated = np.polyval(p, t)
        offsetTerm = estimated
    
    demoudlated = inst_phase - offsetTerm
    fig2, ax3 = plt.subplots()
    ax3.plot(t, demoudlated) # demodulated signal
    ax3.set_title("Demoduated signal")
    ax3.set_xlabel("n")
    ax3.set_ylabel(r"$\hat{m(t)}$")
    fig2.show()


    


if __name__ == "__main__":
    extract_envelope_phase()