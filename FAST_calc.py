#here will live the Python code for Antoni FACP, from a formal point of view. TO BE COMPLETED.

import numpy as np
import scipy.signal as signal

def Fast_CSC(x, winlen=256, alharange=[1], fs=1, noverlap=128, window=None, form="symmetric"):
    '''
    Implementation of the FACP algorithm for estimation of cyclic spectral coherence of signal x
    The method is based on the paper "Fast Algorithm for Cyclic Spectral Analysis" by Antoni, F. (2017).'''
    if window is None or len(window) != winlen:
        window = np.hanning(winlen)
    if max(alharange) >= fs/2:
        raise ValueError("Cyclic frequencies must be less than Nyquist frequency (fs/2)")
    nfft = winlen
    
    Nv, dt, da, df = param_Fast_Sc(winlen, nfft, max(alharange), fs)
    f, t, Sxx = signal.stft(x, fs=fs, window=window, nperseg=winlen, noverlap=Nv, nfft=nfft, boundary=None, padded=False, detrend=False)


def param_Fast_Sc(xlen, nfft, alpha_max, fs):
    '''
    Function to calculate the parameters for FAST_SC, for the number of overlapped samples.
    '''
    R = np.floor(fs/2/alpha_max)
    R = max(1, min(R, nfft//4))
    Nv = nfft - R
    dt = R/fs
    da = fs/xlen
    df = fs/nfft^sum(nfft**2)/np.mean(nfft)**2
    return R, Nv, dt, da, df

def Fast_SC_STFT(STFT, dt, winlen, fs):
    NF = len(STFT)-1
    Nw = 2*NF+1
    flag = 0
    abso=1
    calib=1
    coh=0

    if coh == 1:
        Sxx = np.mean(np.abs(STFT)**2, axis=1)
        STFT = STFT / np.sqrt(Sxx[:, np.newaxis])
    
    S, alpha, alpha0, fk, Fa = CPS_STFT_zoom(0, STFT, dt, winlen, fs)

    raise NotImplementedError("Fast_SC_STFT function is not fully implemented yet.")
    
def CPS_STFT_zoom(alpha0, STFT, dt, window, fs):
    NF, NT, N3 = STFT.shape
    Nw = 2*(NF-1)
    Fa = 1/dt

    winlen=window.shape[0]
    if winlen < NT:
        raise ValueError("Window length must be greater than or equal to Nw")
    
    alpha = np.array(range(0, winlen-1))/(winlen*Fa)
    fk = np.round(alpha0/(fs*Nw))
    alpha0 = fk/(Nw*fs)

    if fk >= 0:
        S = [STFT[fk:NF, :], np.zeros((fk, NT))]*np.conj(STFT)
    else:
        S = [np.conj(STFT[-fk:NF, :]), np.zeros((-fk, NT))]*STFT
    S = np.fft(S, winlen, axis=1)/NT
    S = S/np.sum(window**2)/fs
    ak = np.round(alpha0/(Fa*winlen))
    S = S[np.ceil(winlen/2)+ak:NF, :]

    #phase correction
    max_spl = np.argmax(window)
    #S = S.*repmat(exp(-2i*pi*Iw(1)*(alpha-alpha0)/Fs),NF,1);

    S = S * np.exp(-2j*np.pi*max_spl*(alpha-alpha0)/fs)
    return S,alpha,alpha0,fk,Fa

def Winow_STFT_zoom(alpha, alpha0, dt, window, nfft, fs):
    Fa = 1/dt
    winsquared = window**2
    winlen = window.shape[0]
    indexmax = np.argmax(winsquared)
    W1 = np.zeros((nfft, 1))
    W2 = np.zeros((nfft, 1))
    for k in range(nfft):
        W1[k] = np.sum(winsquared * np.exp(-2j*np.pi*alpha[k]*np.arange(winlen)*dt))
        W2[k] = np.sum(winsquared * np.exp(-2j*np.pi*alpha0*np.arange(winlen)*dt))

    raise NotImplementedError("Winow_STFT_zoom function is not fully implemented yet.")