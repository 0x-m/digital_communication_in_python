import numpy as np
import matplotlib.pylab as plt
from passband_modulations import qpsk_mod, oqpsk_mod, piBy4_dqpsk_mod, msk_mod
from pulseshapers import raisedCosineDesign

N = 5000 # Number of symbols to transmit keep it small and adequate
fc = 10 #carrier frequency
L = 8 # oversampling factor
a = np.random.randint(2, size=N)  # uniform random symbols from 0' and 1's

# modulate the source symbols using QPSK, pi/4-dqpsk, MSK
qpsk_result = qpsk_mod(a, fc, L)
oqpsk_result = oqpsk_mod(a, fc, L)
piBy4_dqpsk_result = piBy4_dqpsk_mod(a, fc, L)
msk_result = msk_mod(a, fc, L)

# Pulse shape the modulated waveform by convolving with RC filter
alpha = 0.3
span = 10
b = raisedCosineDesign(alpha,span, L) # RC pulse shaper
iRC_qpsk = np.convolve(qpsk_result['I(t)'],b, mode='valid') # RC - QPSK I(t)
qRC_qpsk = np.convolve(qpsk_result['Q(t)'], b, mode='valid') # Rc - QPSK Q(t)
iRC_oqpsk = np.convolve(oqpsk_result['I(t)'],b, mode='valid') 
qRC_oqpsk = np.convolve(oqpsk_result['Q(t)'],b, mode='valid') 
iRC_piby4qpsk = np.convolve(piBy4_dqpsk_result['U(t)'],b,mode='vaid')
qRC_piby4qpsk = np.convolve(piBy4_dqpsk_result['V(t)'],b,mode='vaid')
i_msk = msk_result['sI(t)']
q_msk = msk_result['sQ(t)']

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(iRC_qpsk,qRC_qpsk)
axs[0,1].plot(iRC_oqpsk,qRC_oqpsk)
axs[1,0].plot(iRC_piby4qpsk,qRC_piby4qpsk)
axs[1,1].plot(i_msk[20:-20],q_msk[20:-20])
axs[0,0].set_title(r'QPSK, RC $\alpha$='+str(alpha))
axs[0,0].set_xlabel('I(t)')
axs[0,0].set_ylabel('Q(t)');
axs[0,1].set_title(r'OQPSK, RC $\alpha$='+str(alpha))
axs[0,1].set_xlabel('I(t)')
axs[0,1].set_ylabel('Q(t)');
axs[1,0].set_title(r'$\pi$/4 - QPSK, RC $\alpha$='+str(alpha))
axs[1,0].set_xlabel('I(t)')
axs[1,0].set_ylabel('Q(t)')
axs[1,1].set_title('MSK')
axs[1,1].set_xlabel('I(t)');axs[1,1].set_ylabel('Q(t)');
plt.show()

