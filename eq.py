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

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix = np.arange(n) / (n - 1)
    rgb = [(1 - m) * c1_rgb + (m * c2_rgb) for m in mix]
    return ['#' + ''.join([format(int(round(val * 255)), '02x') for val in item]) for item in rgb]

''' User provides filename of .wav
    e.g. '/Users/james/Desktop/ESC2023/SamoMiSeSpava.wav'
'''
_, audiofn = sys.argv

# ipd.Audio(audiofn)
# playsound(audiofn)

y, sr = librosa.load(audiofn, sr=None)
print(f'{y.shape = }')
print(f'{sr = }')

sound = AudioSegment.from_wav(audiofn)
# sr = sound.frame_rate
song_length = sound.duration_seconds
# y = sound.get_array_of_samples()
n = len(y)

# plt.style.use('bmh')

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

SAMPLESIZE = 128 # number of data points to read at a time

f, t, Z = sp.signal.stft(y, sr, nperseg=SAMPLESIZE)
print(f'{Z.shape = }')
# plt.pcolormesh(t, f, np.log(np.abs(Z)), shading='gouraud')
# plt.show()

skip = 1
Z = Z[...,::skip]

# set up plotting
maxamp = 1.0
colors = get_color_gradient('#ff0000', '#0000ff', len(f))
fig = plt.figure()
ax = plt.axes(xlim=(f.min() / 1000, f.max() / 1000), ylim=(-16, 0))
ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('Amplitude [dB]')
bars = plt.bar(f / 1000, np.zeros(len(f)),
               bottom=-16, width=1.5 * f.max() / 1000 / SAMPLESIZE,
               snap=False, color=colors, edgecolor='none')
# line, = ax.plot([], [], lw=1)


t = np.arange(SAMPLESIZE)

# methods for animation
title = ax.text(0.5, 0.85, '',
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                transform=ax.transAxes, ha='center')

music_thread = threading.Thread(target=play, args=(sound,))

def init():
    np.seterr(divide='ignore')
    # line.set_data([], [])
    # return line,
    for bar in bars:
        bar.set_height(-16)
    return bars

def animate_real(i):
    j = i * SAMPLESIZE
    data = y[j:j + SAMPLESIZE]
    line.set_data(t, data)
    title.set_text(f'{j / sr:.2f} s')
    return line, title

# frames = len(y) // SAMPLESIZE  # Real-space
frames = Z.shape[1]

'''
delay = {
    64: 800,
    128: 400,
    256: 200,
    512: 100,
    1024: 50,
    2048: 25,
    4096: 12,
}[SAMPLESIZE]
'''
delay = int(400 * 128 / SAMPLESIZE)

def animate_comp(i):
    global music_thread, t0

    if i == 0:
        music_thread.start()
        t0 = time.perf_counter()

    elapsed = time.perf_counter() - t0
    i = round(frames * elapsed / (n / sr))
    i -= delay
    if i < 0:
        return (*bars, title)
    elif i >= Z.shape[1]:
        plt.close(fig)
        return (*bars, title)

    xi = np.log(np.abs(Z[:,i]))
    # line.set_data(f, xi)
    for bar, a in zip(bars, xi):
        bar.set_height(16 + a)
    title.set_text(f'{i * SAMPLESIZE * skip / sr / 2:.2f} s')

    if i >= frames - 1:
        plt.close(fig)
        # return line, title
        return (*bars, title)

    # return line, title
    return (*bars, title)

def animate_bars():
    plt.clf()
    fig = plt.figure()
    n = 4096
    ax = plt.axes(xlim=(0, n - 1), ylim=(-3, +3))
    bars = plt.bar(np.arange(n), np.zeros(n), snap=False, width=1.0, edgecolor='none')
    def draw(i):
        heights = np.random.randn(n)
        for h, bar in zip(heights, bars):
            bar.set_height(h)
        return bars
    anim = FuncAnimation(fig, draw, frames=int(10 * 1000 / 60), interval=60, blit=True, repeat=False)
    plt.show()
    del anim

print(f'Animation will have {frames} frames.')
print(f'At a sr of {sr}/s, {SAMPLESIZE} samples lasts {SAMPLESIZE/sr * 1000} ms')
interval = 40  # ms
print(f'Animation will target {1000 / interval} fps')
anim = FuncAnimation(fig, animate_comp, init_func=init,
                     # frames=frames, interval=SAMPLESIZE/sr/2 * 1000 * skip,
                     frames=int(len(y) / sr * interval), interval=interval,
                     blit=True, repeat=False)

# os.system(f'afplay \'{audiofn}\' &')
tic = time.time()
plt.show()
toc = time.time()
print(f'{toc - tic}')
# os.system('pkill afplay')
music_thread.join()


