#!python3
import os, sys
import time
import threading
import pandas as pd
import numpy as np
import scipy as sp
# import matplotlib.pylab as plt
import matplotlib as mpl
from pydub import AudioSegment
import pydub.playback
from playsound import playsound
#  import sounddevice
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

audiofn = sys.argv[1]
dashed = [arg.removeprefix('-') for arg in sys.argv]

time_domain = 't' in dashed or 'tf' in dashed or 'ft' in dashed
freq_domain = 'f' in dashed or 'tf' in dashed or 'ft' in dashed

# ipd.Audio(audiofn)
# playsound(audiofn)

speed = 1.0

y, sr = librosa.load(audiofn, sr=None)
print(f'{y.shape = }')
print(f'{sr = }')
# y = sp.signal.resample(y, int(len(y) / speed))

sound = pydub.AudioSegment.from_wav(audiofn)
# sound.frame_rate = int(sound.frame_rate * speed)
song_length = sound.duration_seconds
# sr = sound.frame_rate
# y = sound.get_array_of_samples()
n = len(y)

# Alternatively:
'''
sound = pydub.AudioSegment(
    y.tobytes(),
    frame_rate=int(sr / speed),
    sample_width=y.dtype.itemsize,
    channels=1)
'''


# plt.style.use('bmh')

if False:
    D = librosa.stft(y[:n], n_fft=4092)
    hop_length = 4092 // 4
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, n_fft=4092, x_axis='s', y_axis='log', ax=ax)
    ax.set_title(f'Spectogram of {audiofn}', fontsize=20)
    fig.colorbar(img, ax=ax, format='%0.2f')
    plt.show()

# How many data points to read at a time
SAMPLESIZE_t = 4096
SAMPLESIZE_f = 128

f, t, Z = sp.signal.stft(y, sr, nperseg=SAMPLESIZE_f)
print(f)
# print(t)
print(f'{Z.shape = }')
# plt.pcolormesh(t, f, np.log(np.abs(Z)), shading='gouraud')

skip = 1
Z = Z[...,::skip]

# set up plotting
colors = get_color_gradient('#ff0000', '#0000ff', len(f))
if time_domain and freq_domain:
    fig, (axt, axf) = plt.subplots(1, 2, figsize=(12, 6))
elif time_domain:
    fig, axt = plt.subplots(1, 1, figsize=(6, 6))
elif freq_domain:
    fig, axf = plt.subplots(1, 1, figsize=(6, 6))

num_waves = 64

if time_domain:
    axt.set_xlim(0, SAMPLESIZE_t)
    axt.set_ylim(-1, +1)
    line, = axt.plot([], [], lw=1)
if freq_domain:
    axf.set_xlim(f.min() / 1000, f.max() / 1000)
    axf.set_ylim(-16, 0)
    axf.set_xlabel('Frequency [kHz]')
    axf.set_ylabel('Amplitude [dB]')
    bars = axf.bar(f / 1000, np.zeros(len(f)), 
                   bottom=-16, width=1.5 * f.max() / 1000 / SAMPLESIZE_f, 
                   snap=False, color=colors, edgecolor='none')
if time_domain and freq_domain:
    empties = ([] for i in range(2 * num_waves))
    # waves = axt.plot(*empties, lw=2)
    waves = axt.plot([], [], lw=2)


# t = np.arange(SAMPLESIZE)

if time_domain:
    title = axt.text(0.5, 0.85, '',
                     bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                     transform=axt.transAxes, ha='center')
else:
    title = axf.text(0.5, 0.85, '',
                     bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                     transform=axf.transAxes, ha='center')

def init_real():
    np.seterr(divide='ignore')
    line.set_data([], [])
    return line,

def init_comp():
    np.seterr(divide='ignore')
    for bar in bars:
        bar.set_height(-16)
    return bars

def animate_real(elapsed: float):
    i = round(elapsed * sr)
    delay = round(sr * 0.6)
    i -= delay
    if i < 0:
        return line, title
    elif i >= y.size - SAMPLESIZE_t:
        plt.close(fig)
        return line, title

    line.set_data(np.arange(SAMPLESIZE_t), y[i:i+SAMPLESIZE_t])
    hsv = np.array(mpl.colors.rgb_to_hsv(line.get_color()))
    hsv[0] += np.random.randn() * 0.01
    hsv = np.clip(hsv, 0, 1)
    line.set_color(mpl.colors.hsv_to_rgb(hsv))
    title.set_text(f'{i / sr:.2f} s')

    return line, title


def animate_comp(elapsed):
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
    delay = int(400 * 128 / SAMPLESIZE_f)
    i = round(Z.shape[1] * elapsed / (n / sr))
    i -= delay
    if i < 0:
        return (*bars, title)
    elif i >= Z.shape[1]:
        plt.close(fig)
        return (*bars, title)

    xi = np.log(np.abs(Z[:,i]))
    for bar, a in zip(bars, xi):
        bar.set_height(16 + a)
    title.set_text(f'{i * SAMPLESIZE_f * skip / sr / 2:.2f} s')

    if i >= Z.shape[1] - 1:
        plt.close(fig)
        return (*bars, title)

    return (*bars, title)

def animate_waves(elapsed):
    delay = int(400 * 128 / SAMPLESIZE_f)
    i = round(Z.shape[1] * elapsed / (n / sr))
    i -= delay
    if i < 0:
        return waves
    elif i >= Z.shape[1]:
        plt.close(fig)
        return waves

    ws = Z[1:65, i]  # Ignore the DC component
    x = np.arange(SAMPLESIZE_t)
    b = np.zeros(SAMPLESIZE_t)
    for j, w in enumerate(ws):
        A = np.abs(w)
        phi = 0
        phi = np.angle(w)
        freq = f[j + 1]
        b += A * np.cos((x * 2 * np.pi * freq / sr + phi) / 32)
    waves[0].set_data(x, b)

    if i >= Z.shape[1] - 1:
        plt.close(fig)
        return waves

    return waves

# print(f'Animation will have {Z.shape[1]} frames.')
print(f'At a sr of {sr}/s, {SAMPLESIZE_t} samples lasts {SAMPLESIZE_t/sr * 1000} ms')
interval = 40  # ms
print(f'Animation will target {1000 / interval} fps')
print(f'There will be approximately {int(len(y) / sr * interval)} frames.')

def frames():
    audio_thread = threading.Thread(target=pydub.playback.play, args=(sound,))
    audio_thread.start()
    t0 = time.perf_counter()
    end = n / sr
    while (elapsed := time.perf_counter() - t0) < end:
        yield elapsed
    audio_thread.join()

if time_domain and freq_domain:
    anim = FuncAnimation(
            fig, lambda x: (*animate_real(x), *animate_comp(x)),
            init_func=lambda: (*init_real(), *init_comp()),
            frames=frames(), interval=interval,
            blit=True, repeat=False)
elif time_domain:
    anim = FuncAnimation(
            fig, animate_real, init_func=init_real,
            frames=frames(), interval=interval,
            blit=True, repeat=False)
else:
    anim = FuncAnimation(
            fig, animate_comp, init_func=init_comp,
            frames=frames(), interval=interval,
            blit=True, repeat=False)


tic = time.time()
plt.show()
toc = time.time()
print(f'{toc - tic}')

