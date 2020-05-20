import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import pathlib
import os

def read_audio_file(path, fs):
    audio_sample = librosa.load(str(path), sr=fs)
    return audio_sample

sr = 16000
hop_length = 256
n_fft = 512

data_dir = pathlib.Path("../melgan-neurips/output")
_, ext = os.path.splitext(os.listdir(data_dir)[3])
paths = list(data_dir.glob("*" + ext))
cats = np.load("sc09.npy")

rnd_paths = np.random.choice(paths, size=9)
images = []
for rp in rnd_paths:
    audio = read_audio_file(rp, 16000)
    D = np.abs(librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    images.append(DB)
"""
indices = np.random.choice(np.arange(cats.shape[0]), size=9)
images = []
for rp in indices:
    D = np.abs(librosa.stft(cats[rp].reshape(-1), n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    images.append(DB)
"""
fig, ax = plt.subplots(nrows=3, ncols=3)
c = 0
for row in ax:
    for col in row:
        col.imshow(images[c], aspect="auto", cmap="gnuplot")
        col.set_axis_off()
        c += 1
plt.subplots_adjust(hspace=0.02, wspace=0.02)
plt.savefig("../results/melgan_gen_random_spectrogram_test", bbox_inches="tight", pad_inches=0)