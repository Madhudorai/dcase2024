import numpy as np
import soundfile as sf
from scipy import signal
import os
import scipy.special as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.special import factorial, lpmv
import pyroomacoustics as pra

import MicrophoneArray
import plot_tools
from tools import sph2cart, soundread, setRoom, ssl_SHmethod_broad, ssl_SHmethod_broad2
from plot_tools import plot_wave, plot_stft, plot_option

MicArray32 = MicrophoneArray.EigenmikeEM32()
name = MicArray32.getname()
array_type = MicArray32.gettype()
thetas = MicArray32._thetas
phis = MicArray32._phis
radius = MicArray32._radius
weights = MicArray32._weights
num_elements = MicArray32._numelements
directivity = MicArray32._directivity

# Initialize the list of mic_position in spherical harmonic domain
mic_pos_sph = []

# for each theta and phi, create a new coordinate which contains all parameters
# to the list of mic_pos
for theta, phi in zip(thetas, phis):
    mic_pos_sph.append([radius, theta, phi])
mic_pos_sph = np.array(mic_pos_sph)
print("Name:", name)
print("Array Type:", array_type)
print("Thetas:", thetas)
print("Phis:", phis)
print("Radius:", radius)
print("Weights:", weights)
print("Number of Elements:", num_elements)
print("Directivity:", directivity)

ambisonic_signals, fs, num_channels = soundread('foa/fold4_room23_mix001.wav')

# Run SSL on all frames, plot only for frame 10
doa_results = ssl_SHmethod_broad2(
    ambisonic_signals,
    fs,
    resolution=1,
    method="DAS",
    plot_method="2D"
)

# Save to CSV
import csv
with open("estimated_doa_allframes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_id", "azimuth_deg", "elevation_deg"])
    writer.writerows(doa_results)

print("Saved estimated DOA to estimated_doa_allframes.csv")
