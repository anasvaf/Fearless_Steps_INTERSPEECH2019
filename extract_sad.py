
from __future__ import division

import numpy as np
seed = 1988
np.random.seed(seed)

import os, sys

import librosa
#from python_speech_features.sigproc import preemphasis

import pandas as pd

#from matplotlib import pyplot as plt


args = sys.argv

SR = 8000


if 'train' in args:
    train_folder = './data/Audio/Tracks/Dev/'

    X_wavs = []
    dfs = []

    for path, subdirs, files in os.walk(train_folder):
        for file_name in files:
            X_wavs.append(path + file_name)

            dfs.append(pd.read_csv('./data/Transcripts/SAD/Dev/' + file_name.split('.wav')[0] + '.txt', 
                sep = '\t',
                names = ["filename", "null", "start_s", "end_s", "target", "null", "null", "null", "null", "null", "null", "null"]))


    for i in range(len(X_wavs)):
        data, rate = librosa.load(X_wavs[i], sr = SR, mono = True)

        print
        print 'file_name', X_wavs[i]
        print 'file_size', data.size, 'chunks', data.size / SR

        labels = []

        for j in range(len(dfs[i])):
            if j == len(dfs[i]) - 1:
                current_seg = data[int(np.round(dfs[i].start_s[j] / 0.000125)):]
            
            else:
                current_seg = data[int(np.round(dfs[i].start_s[j] / 0.000125)):int(np.round(dfs[i].end_s[j] / 0.000125))]
            
            if dfs[i].target[j] == 'S':
                for k in range(current_seg.size):
                    labels.append(1.)
            
            elif dfs[i].target[j] == 'NS':
                for k in range(current_seg.size):
                    labels.append(0.)

        labels = np.array(labels, np.float32)

        start_sample = 0

        for l in range(int(np.ceil(labels.size / SR))):
            end_sample = start_sample + SR
            
            if labels[start_sample:end_sample].size != SR:
                data_chunk = data[start_sample:end_sample]
                labels_chunk = labels[start_sample:end_sample]

                data_chunk = np.pad(data_chunk, (0, SR - data_chunk.size), 'constant', constant_values = 0.)
                labels_chunk = np.pad(labels_chunk, (0, SR - labels_chunk.size), 'constant', constant_values = 0.)

            else:
                data_chunk = data[start_sample:end_sample]
                labels_chunk = labels[start_sample:end_sample]

            # # D = preemphasis(data[start_sample:end_sample], coeff = 0.97)
            
            # D = librosa.feature.mfcc(y = data_chunk, sr = SR, n_mfcc = 20)
            # # #D = librosa.feature.delta(D)
            # D = librosa.feature.delta(D, order = 2)

            # D = librosa.feature.melspectrogram(y = data_chunk, sr = SR, S = None, n_fft = 2048, 
            #     hop_length = 512, power = 2.0, n_mels = 64)

            # D = np.flipud(D)

            # np.save('./data/sad_data_chunks/' + X_wavs[i].split('./data/Audio/Tracks/Dev/')[-1].split('.wav')[0] + '_' + str(l) + '.npy', 
            #     D)
            
            np.save('./data/sad_data_chunks/' + X_wavs[i].split('./data/Audio/Tracks/Dev/')[-1].split('.wav')[0] + '_' + str(l) + '.npy', 
              data_chunk)

            np.save('./data/sad_labels_chunks/' + X_wavs[i].split('./data/Audio/Tracks/Dev/')[-1].split('.wav')[0] + '_' + str(l) + '.npy', 
              labels_chunk)
            
            start_sample += SR


if 'test' in args:
    test_folder = './data/Audio/Tracks/Eval/'

    X_test_wavs = []

    for path, subdirs, files in os.walk(test_folder):
        for file_name in files:
            X_test_wavs.append(path + '/' + file_name)


    for i in range(len(X_test_wavs)):
        data, rate = librosa.load(X_test_wavs[i], sr = SR, mono = True)

        start_sample = 0

        for l in range(int(np.ceil(data.size / SR))):
            end_sample = start_sample + SR

            if data[start_sample:end_sample].size != SR:
                data_chunk = data[start_sample:end_sample]

                data_chunk = np.pad(data_chunk, (0, SR - data_chunk.size), 'constant', constant_values = 0.)
                
                print data_chunk
                print data_chunk.size

            else:
                data_chunk = data[start_sample:end_sample]

            np.save('./data/test_sad_data_chunks/' + X_test_wavs[i].split('./data/Audio/Tracks/Eval/')[-1].split('.wav')[0] + '_' + str(l) + '.npy', 
                data_chunk)

            start_sample += SR
