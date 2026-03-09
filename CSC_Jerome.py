import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#here implemented the faster computation by Jerome Antoni
# Roger Boustany, Jérôme Antoni,
# CYCLIC SPECTRAL ANALYSIS FROM THE AVERAGED CYCLIC PERIODOGRAM,
# 


def hanning(window_length):
    m = int((window_length)/2)
    w = 0.5*(1 - np.cos(2*np.pi*np.arange(1, m+1)/(window_length+1)))
    return np.concatenate((w, np.flip(w)))

def get_spectra(sig1, sig2, n_frames, nfft, step, window):
    """Helper to compute the mean cross-spectrum of STFT blocks."""
    shape = (n_frames, nfft)
    
    strides1 = (step * sig1.itemsize, sig1.itemsize)
    seg1 = np.lib.stride_tricks.as_strided(sig1, shape=shape, strides=strides1)
    
    strides2 = (step * sig2.itemsize, sig2.itemsize)
    seg2 = np.lib.stride_tricks.as_strided(sig2, shape=shape, strides=strides2)
    
    S1 = np.fft.fft(seg1 * window, axis=1)
    S2 = np.fft.fft(seg2 * window, axis=1)
    
    return np.mean(S1 * np.conj(S2), axis=0)

def compute_CSC_matrix_antoni(y, x, alpharange, fs=1, nfft=256, noverlap=128, window=None, form="symmetric", matrix="onesided", verbose=False):
    '''
    Compute the CSC Matrix using highly optimized STFT block-processing.
    '''
    alpha_norm = alpharange / fs  
    n = len(y)
    n_idx = np.arange(n)
    
    if window is None:
        window = hanning(nfft)
        
    step = nfft - noverlap
    n_frames = (n - nfft) // step + 1
    

    csc_matrix = np.zeros((nfft, len(alpharange)), dtype=np.complex64)
    eps = np.finfo(float).eps

    if form == "asymmetric":
        CPS_y = np.abs(get_spectra(y, y, n_frames, nfft, step, window))

    for i, alpha in tqdm(enumerate(alpha_norm), total=len(alpharange), desc="Computing CSC Matrix", disable=not verbose):
        if form == "symmetric":
            shift_x = np.exp(1j * np.pi * alpha * n_idx)
            shift_y = np.exp(-1j * np.pi * alpha * n_idx)
            x_shft = x * shift_x
            y_shft = y * shift_y
            
            CPS_cross = get_spectra(y_shft, x_shft, n_frames, nfft, step, window)
            CPS_y = np.abs(get_spectra(y_shft, y_shft, n_frames, nfft, step, window))
            CPS_x = np.abs(get_spectra(x_shft, x_shft, n_frames, nfft, step, window))
            
        elif form == "asymmetric":
            shift_x = np.exp(1j * 2 * np.pi * alpha * n_idx)
            x_shft = x * shift_x
            
            CPS_cross = get_spectra(y, x_shft, n_frames, nfft, step, window)
            CPS_x = np.abs(get_spectra(x_shft, x_shft, n_frames, nfft, step, window))
            
        csc_matrix[:, i] = CPS_cross / np.sqrt(CPS_y * CPS_x + eps)

    if matrix == "onesided":
        csc_matrix = csc_matrix[:nfft//2, :]

    return csc_matrix

if __name__ == "__main__":

    fs = 1000  
    timelength = 8
    t = np.arange(0, timelength, 1/fs)  
    fc = 150  
    f_m= 10  
    fd = 1

    y = np.cos(2 * np.pi * (fc + f_m*np.cos(2 * np.pi * 2 * t)) * t)
    modulating_signal = np.cos(2 * np.pi * f_m * t)

    integral_mod = np.cumsum(modulating_signal) / fs 
    fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * fd * integral_mod)

    sum_signal = fm_signal + 0.4 * np.random.normal(size=t.shape)  

    alpharange = np.linspace(1, 50, 50)
    
    csc_matrix = compute_CSC_matrix_antoni(
        sum_signal, sum_signal, alpharange, 
        fs=fs, nfft=256, noverlap=170, 
        window=hanning(256), form="symmetric", 
        matrix="onesided", verbose=True
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(csc_matrix), aspect='auto', origin='lower',
               extent=[alpharange[0], alpharange[-1], 0, fs/2])
    plt.colorbar(label='Cyclic Coherence magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Cyclic Frequency (Hz)')
    plt.title('Cyclostationary Spectral Coherence Matrix')
    plt.set_cmap('jet')
    plt.show()