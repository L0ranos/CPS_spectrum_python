#here will live the Python code for Antoni FACP.
import numpy as np

def Fast_CSC(x, winlen=256, alharange=[1], fs=1, nfft=256, noverlap=128, window=None, form="symmetric"):
    '''
    Implementation of the FACP algorithm for estimation of cyclic spectral coherence of signal x
    The method is based on the paper "Fast Algorithm for Cyclic Spectral Analysis" by Antoni, F. (2017).'''
    if window is None:
        window = np.hanning(winlen)
    if max(alharange) >= fs/2:
        raise ValueError("Cyclic frequencies must be less than Nyquist frequency (fs/2)")
    #TODO
