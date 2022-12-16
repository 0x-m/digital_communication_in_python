import numpy as np

def sine_wave(f, overSampRate, phase, nCy1):
    '''
    Generate sine wave with teh following parameters
    f: frequency of sine wave in hertz
    overSampRate: oversampling rate (integer)
    phase: desired phase shift in radians
    ncy1: number of cycles of sine wave to generate

    returns:
    (t, g): time base (t) and the signal g(t) as tuple
    
    '''

    fs = overSampRate*f # sampling frequency
    t = np.arange(0, nCy1 * 1/f - 1/fs, 1/fs) #time base
    g = np.sin(2*np.pi*f*t + phase) # replace with cos if a cosine wave is desired
    return (t, g)
    
    

def square_wave(f, overSampRate, nCy1):
    '''
    Generate square wave signal with the following parameters
    Parameters:
        f: frequency of square wave in hertz
        overSampRate: oversampling rate (integer)
        nCy1: nubmer of cycles of square wave to generate
    Returns:
        (t, g) : time base (t) and the signal g(t) as tuple
    '''
    
    fs = overSampRate*f # sampling frequency
    t = np.arange(0, nCy1 * 1/f - 1/fs, 1/fs) # time base
    g = np.sign(np.sin(2*np.pi*f*t)) # replace cos if a cosine wave is desired
    
    return (t, g) # return time base signal g(t) as tuple


def rect_pulse(A, fs, T):
    '''
    Generate isolated rectangular pulse with the following parameters
    
    Parameters:
        A: amplitude of the rectangular pulse
        fs: sampling frequecy in Hz
        T: duration of the pulse in seconds
    Returns:
        (t, g): time base (t) and the signal g(t) as tuple
        
    '''
    
    t = np.arange(-0.5, 0.5, 1/fs) # time base 
    rect = (t > -T/2) * (t < T/2) + 0.5*(t == T/2) + 0.5*(t == -T/2)
    g = A*rect
    return (t, g)


def gaussian_pulse(fs, sigma):
    '''
    Generate isolated Gaussian pulse with the following parameters:
    Parameters:
        fs: sampling frequency in Hz
        sigma: pulse with in seconds
    Returns:
        (t, g): time base (t) and the signal g(t) as tuple
    '''
    
    t = np.arange(-0.5, 0.5, 1/fs) # time base
    g = 1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-t**2/(2*sigma**2)))
    return (t, g)
