import numpy as np
import abc
import matplotlib.pyplot as plt

class Modem:
    __metadata__ = abc.ABCMeta
    # Base class: Modem
    # Attribute definitions:
    #   self.M: number of points in the MPSK constellation
    #   self.name: name of the modem 
    #   self.constellation: refernce constellation
    #   self.coherence: only for coherent or noncoherent FSK
    
    def __init__(self, M, constellation, name, coherece=None) -> None:
        if (M < 2) or ((M & (M - 1)) != 0): # if M not power of 2:
            raise ValueError("M should be a power of 2")
        if name.lower() == 'fsk':
            if (coherece.lower() == 'coherent') or (coherece.lower() == 'noncoherent'):
                self.coherence = coherece
            else:
                raise ValueError('Coherence must be \'coerent\' or ...')
        else:
            self.coherence = None
        self.M = M # number of poitns in the constellation
        self.name = name # name of the modem : psk, qam, pam, fsk
        self.constellation: constellation # refernce constellation
    
    def plotConstellation(self):
        '''
        Plot the reference constellation points for the selected modem
        '''
        if self.name.lower() == 'fsk':
            return 0 # FSK is multi-dimentional difficult to visualize
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(np.real(self.constellation), np.imag(self.constellation), 'o')
        for i in range(0, self.M):
            axs.annotate("{0:0{1}}".format(i, self.M), (np.real(self.constellation[i]), np.imag(self.constellation[i])))

        axs.set_title('Constellation')
        axs.set_xlabel('I')
        axs.set_ylabel('Q')
        fig.show()
    
    def modulate(self, inputSymbol):
        '''
        Modulate a vector of input symbols (numpy array format) using 
        the choosen modem Input symbol take integer values in the range 0 to M - 1
        '''
        
        if isinstance(inputSymbol, list):
            inputSymbol = np.array(inputSymbol)
        
        if not (0 <= inputSymbol.all() < self.M-1):
            raise ValueError('inputSymbols values are beyond the range 0 to m -1')
        
        modulatedVec = self.constellation[inputSymbol]
        return modulatedVec # return modulated vector
    
    def demodulate(self, receivedSymbs):
        '''
        Demodulated a vector of received symbols using the chosen modem
        '''
        
        if isinstance(receivedSymbs, list):
            receivedSymbs = np.array(receivedSymbs)
        
        detectedSymbs = self.iqDetector(receivedSymbs)
        return detectedSymbs

    def iqDetector(self, receivedSymbs):
        '''
        Optimum detector for 2-dim signal (ex: MQAM, MPSK, MPAM) in IQ plane
        Note: MPAM/BPSK are one dimensional modulations. The same function can be 
        applied for these modulations sice quadrature is zero (Q = 0)
        
        The function computes the pair-wise Euclidean distance of each point in the
        received vector against every point in the reference constellation It then 
        returns the symbols from the reference constellation that providess the minimum Euclidean distance

        Parameters:
            receivedSymbs: received symbol vector of complex form
        Returns:
            detectedSyms: decoded symbols that provides minimum Euclidean distance
        '''

        from scipy.spatial.distance import cdist
        
        # received vector and refernce in cartesian form
        XA = np.column_stack((np.real(receivedSymbs), np.imag(receivedSymbs)))
        XB = np.column_stack((np.real(self.constellation), np.imag(self.constellation)))

        d = cdist(XA, XB, metric='euclidean') # compute pair-wise Eulidean distances
        detectedSyms = np.argmin(d, axis=1) # indices corresponding minimum Euclid dist
        return detectedSyms
        
        
        