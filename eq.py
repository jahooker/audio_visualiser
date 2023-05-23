#!python3
import os, sys
import time
import threading
import pandas as pd
import numpy as np
import scipy as sp
# import matplotlib.pylab as plt
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
import seaborn as sns
import pyaudio
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import timedelta
from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

''' User provides filename of .wav
    e.g. '/Users/james/Desktop/ESC2023/SamoMiSeSpava.wav'
'''
_, audiofn = sys.argv

# ipd.Audio(audiofn)
# playsound(audiofn)

y, sr = librosa.load(audiofn, sr=None)
print(f'y: {y[:10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')

sound = AudioSegment.from_wav(audiofn)
# sr = sound.frame_rate
song_length = sound.duration_seconds
# y = sound.get_array_of_samples()
n = len(y)

plt.style.use('bmh')

# Short-time Fourier transform
if False:
    D = librosa.stft(y[:n], n_fft=4092)
    hop_length = 4092 // 4
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)

# Plot the transformed audio data
if False:
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, n_fft=4092, x_axis='s', y_axis='log', ax=ax)
    ax.set_title(f'Spectogram of {audiofn}', fontsize=20)
    fig.colorbar(img, ax=ax, format='%0.2f')
    plt.show()

SAMPLESIZE = 4096 # // 8 # number of data points to read at a time

f, t, Z = sp.signal.stft(y, sr, nperseg=SAMPLESIZE)
print(f'{Z.shape = }')
# plt.pcolormesh(t, f, np.log(np.abs(Z)), shading='gouraud')
# plt.show()

skip = 1
Z = Z[...,::skip]

# set up plotting
maxamp = 1.0
fig = plt.figure()
# ax = plt.axes(xlim=(0, SAMPLESIZE - 1), ylim=(-maxamp, +maxamp))
# ax = plt.axes(xlim=(0, SAMPLESIZE - 1), ylim=(-20, 0))
ax = plt.axes(xlim=(f.min(), f.max()), ylim=(-16, 0))
line, = ax.plot([], [], lw=1)


# x axis data points
t = np.arange(SAMPLESIZE)
# plt.plot(y) and plt.show()

# methods for animation
title = ax.text(0.5, 0.85, '', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                transform=ax.transAxes, ha='center')

music_thread = threading.Thread(target=play, args=(sound,))

def init():
    np.seterr(divide='ignore')
    line.set_data([], [])
    return line,

def animate_real(i):
    j = i * SAMPLESIZE
    data = y[j:j + SAMPLESIZE]
    line.set_data(t, data)
    title.set_text(f'{j / sr:.2f} s')
    return line, title

# frames = len(y) // SAMPLESIZE  # Real-space
frames = Z.shape[1]

def animate_comp(i):
    global music_thread, t0

    if i == 0:
        music_thread.start()
        t0 = time.perf_counter()

    elapsed = time.perf_counter() - t0
    i = round(frames * elapsed / (n / sr))
    if i > 10: i -= 10
    xi = np.log(np.abs(Z[:,i]))
    line.set_data(f, xi)
    title.set_text(f'{i * SAMPLESIZE * skip / sr / 2:.2f} s')

    if i >= frames - 1:
        plt.close(fig)
        return line, title

    '''
    xi = np.log(np.abs(Z[:,i]))
    line.set_data(f, xi)
    title.set_text(f'{i * SAMPLESIZE * skip / sr / 2:.2f} s')
    '''
    return line, title

print(f'Animation will have {frames} frames.')
print(f'At a sr of {sr}/s, a stretch of {SAMPLESIZE} samples lasts {SAMPLESIZE/sr * 1000} ms')
anim = FuncAnimation(fig, animate_comp, init_func=init,
                     frames=frames, interval=SAMPLESIZE/sr/2 * 1000 * skip,
                     blit=True, repeat=False)

# os.system(f'afplay \'{audiofn}\' &')
tic = time.time()
plt.show()
toc = time.time()
print(f'{toc - tic}')
# os.system('pkill afplay')
music_thread.join()


