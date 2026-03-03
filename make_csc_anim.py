import numpy as np
import scipy.io
from CSC_calc import compute_CSC_matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt

#This is a WIP script for creating an animation of the CSC changing over time for a real signal.

sig_folder = "test_audio_concat/"
#find all wav files in sig_folder and sort by time of recording

import os
wav_files = [f for f in os.listdir(sig_folder) if f.endswith('.wav')]
wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(sig_folder, x)))
print(wav_files)

fig, ax = plt.subplots(figsize=(10, 6))
frames = []

for wav_file in wav_files:
    mat = scipy.io.wavfile.read(os.path.join(sig_folder, wav_file))
    
    fs = mat[0]
    signal = mat[1][:, 0]
    #resample to 2000 Hz using numpy
    a, b = butter(2, 80/fs, btype='highpass')
    signal = filtfilt(a, b, signal)
    signal = np.interp(np.linspace(0, len(signal), int(len(signal) * 2000 / fs)), np.arange(len(signal)), signal)
    signal = signal - np.mean(signal)  # Remove DC offset
    
    fs = 2000
    overlap = 0.5
    minute_windows = range(0, len(signal)-int(fs*1*10), int(fs * 1 * 10 * (1 - overlap)))
    alpharange = np.linspace(1, 40, 40)
    time_recording = f"{wav_file[16:18]}:{wav_file[18:20]}"
    print("Processing recording from " + time_recording)
    #make animation of CSC changes each minute, save as mp4 using matplotlib.animation

    for minute_window in minute_windows:
        signal_window = signal[minute_window:(minute_window+int(fs*1*10))]
        # print(f"Processing minute window {minute_window+1}/{minute_windows}..., starting at sample {minute_window*fs*10*60} and ending at sample {(minute_window+1)*fs*10*60}")
        print("Computing CSC Matrix for real signal..." + str(np.shape(signal_window)))
        csc_matrix = compute_CSC_matrix(signal_window, signal_window, alpharange, fs=fs, nfft=8192, noverlap=2048, window=np.hanning(4096), form="symmetric", matrix="onesided")
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Cyclic Frequency (Hz)')
        ax.set_title(f'CSC Matrix for Real Signal, 1 minute window, 15.18.2025')

        im = ax.imshow(np.abs(csc_matrix), aspect='auto', origin='lower',
                        extent=[alpharange[0], alpharange[-1], 0, fs/2],
                        cmap='jet')
        frames.append([im])

    print("numframes: " + str(len(frames)))
    cbar = plt.colorbar(im, ax=ax, label='Cyclic Coherence magnitude')
    anim = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat=True)
    anim.save('csc_animation.mp4', writer='ffmpeg', fps=0.5)
    plt.close()