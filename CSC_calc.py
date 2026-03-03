import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.io.matlab import loadmat

def hanning(window_length):
    m = int((window_length)/2)
    w = 0.5*(1 - np.cos(2*np.pi*np.arange(1, m+1)/(window_length+1)))
    wfull = np.concatenate((w, np.flip(w)))
    print(np.shape(wfull))
    return wfull


def compute_CPS_cross(y, x, nfft, noverlap, window):
    '''
    Compute the Cyclostationary Cross Power Spectrum of two given signals y and x.
    Parameters:
    y: First input signal (numpy array).
    x: Second input signal (numpy array).
    nfft: Number of FFT points (int).
    noverlap: Number of overlapping samples (int).
    window: Window function to apply to the segments (numpy array).
    Returns:
    cps_cross: Cyclostationary Cross Power Spectrum (numpy array).
    '''

    if len(window) == 1:
        window = hanning(window)

    winlen = len(window)
    n = len(y)
    step = winlen - noverlap
    numwindows = (n - noverlap) // step

    CPS_accum = np.zeros(nfft, dtype=np.complex128)

    for i in range(numwindows):
        start = i * step
        end = start + winlen
        y_win = y[start:end] * window
        x_win = x[start:end] * window
        Y = np.fft.fft(y_win, n=nfft)
        X = np.fft.fft(x_win, n=nfft)
        CPS = Y * np.conj(X)
        # Accumulate CPS for each segment
        if i == 0:
            CPS_accum = CPS
        else:
            CPS_accum += CPS

    # scale factor for normalisation (numwindows by square norm of window)
    scale_factor = numwindows * np.sum(np.abs(window)**2)
    CPS_accum /= scale_factor
    return CPS_accum

def compute_CSC(y, x, alpha, nfft=256, noverlap=128, window=None, form="symmetric"):
    '''
    Compute the Cyclostationary Spectral Coherence of two given signals y and x. For auto-cyclostationarity, pass y = x.
    Parameters:
    y: First input signal (numpy array).
    x: Second input signal (numpy array).
    nfft: Number of FFT points (int).
    noverlap: Number of overlapping samples (int).
    window: Window function to apply to the segments (numpy array).
    Returns:
    csc: Cyclostationary Spectral Coherence (numpy array).
    '''
    if window is None:
        window = hanning(nfft)

    n = len(y)
    if form == "symmetric":
        x_shft = x * np.exp(1j*np.pi*alpha*np.arange(n))
        y_shft = y * np.exp(-1j*np.pi*alpha*np.arange(n))
    elif form == "asymmetric":
        x_shft = x * np.exp(1j*2*np.pi*alpha*np.arange(n))
        y_shft = y.copy()

    CPS_cross = compute_CPS_cross(y_shft, x_shft, nfft, noverlap, window)
    CPS_y = compute_CPS_cross(y_shft, y_shft, nfft, noverlap, window)
    CPS_x = compute_CPS_cross(x_shft, x_shft, nfft, noverlap, window)
    
    eps = np.finfo(float).eps
    csc = CPS_cross / np.sqrt(CPS_y * CPS_x + eps)

    return csc
    
def compute_CSC_matrix(y, x, alpharange, fs=1, nfft=256, noverlap=128, window=None, form="symmetric", matrix="onesided"):
    '''
    Compute the Cyclostationary Spectral Coherence Matrix of two given signals y and x over a range of cyclic frequencies.
    Parameters:
    y: First input signal (numpy array).
    x: Second input signal (numpy array).
    alpharange: Array of cyclic frequencies [Hz] to compute CSC for (numpy array).
    fs: Sampling frequency (float).
    nfft: Number of FFT points (int).
    noverlap: Number of overlapping samples (int).
    window: Window function to apply to the segments (numpy array).
    form: Form of cyclostationarity ("symmetric" or "asymmetric").
    Returns:
    csc_matrix: Cyclostationary Spectral Coherence Matrix (2D numpy array).
    '''

    alpharange = alpharange / fs  

    csc_matrix = np.zeros((nfft, len(alpharange)), dtype=np.complex64)
    
    for i, alpha in tqdm(enumerate(alpharange), total=len(alpharange), desc="Computing CSC Matrix"):
        csc_matrix[:, i] = compute_CSC(y, x, alpha, nfft, noverlap, window, form)
    
    if matrix == "onesided":
        csc_matrix = csc_matrix[:nfft//2, :]

    return csc_matrix

if __name__ == "__main__":
    # Generate test signals
    fs = 1000  
    timelength = 120  
    t = np.arange(0, timelength, 1/fs)  
    fc = 150  
    f_m= 10  
    fd = 1

    y = np.cos(2 * np.pi * (fc + f_m*np.cos(2 * np.pi *2 * t)) * t)
    modulating_signal = np.cos(2 * np.pi * f_m * t)

    integral_mod = np.cumsum(modulating_signal) / fs 
    fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * fd * integral_mod)

    sum_signal = fm_signal + 0.05 * np.random.normal(size=t.shape)  

    alpharange = np.linspace(1, 50, 50)
    csc_matrix = compute_CSC_matrix(sum_signal, sum_signal, alpharange, fs=fs, nfft=512, noverlap=170, window=hanning(256), form="symmetric", matrix="onesided")

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(csc_matrix), aspect='auto', origin='lower',
            extent=[alpharange[0], alpharange[-1], 0, fs/2],)
    plt.colorbar(label='Cyclic Coherence magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Cyclic Frequency (Hz)')
    plt.title('Cyclostationary Spectral Coherence Matrix')
    plt.set_cmap('jet')
    plt.show()