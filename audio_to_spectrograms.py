import numpy as np
import librosa
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.misc import imsave

wav_folder = './data/sad_splits_wavs_npys/'
npy_folder = './data/sad_splits_tags_npys/'

specs_folder = './data/sad_splits_spectrograms/'

N_FFT = 256
HOP_LEN = N_FFT // 8

if not os.path.exists(specs_folder):
    os.makedirs(specs_folder)


for top, dirs, files in os.walk(wav_folder):
    for nm in tqdm(files):       
        file = os.path.join(top, nm)

        X = librosa.stft(np.load(file), n_fft = N_FFT, hop_length = HOP_LEN)
        D = librosa.amplitude_to_db(np.abs(X))
        data = np.flipud(D)
        # plt.imshow(D, cmap='gray')
        # plt.show()
        imsave(specs_folder + nm.split(npy_folder)[-1].split(".npy")[0] + '.png', D)