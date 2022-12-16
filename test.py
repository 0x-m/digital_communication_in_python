from pulseshapers import raisedCosineDemo, raisedCosineDesign
import numpy as np
from scipy.signal import upfirdn
import matplotlib.pyplot as plt

a = [1, -1, 1, 1,-1,1,1,-1,-1,1,-1,1,1,1]
b = raisedCosineDesign(0.3, 10, 10)
c = upfirdn(b, a, up=20)
t = np.arange(start=0, stop=len(c) / 2000, step=1 / 2000)
d = c* np.cos(2*np.pi*100*t)
plt.plot(t,d)
plt.show()

