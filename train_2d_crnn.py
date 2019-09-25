
from __future__ import division

import numpy as np
seed = 1988
np.random.seed(seed)

import os, sys, math
import librosa
#from python_speech_features.sigproc import preemphasis

from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import CuDNNGRU
from keras.layers import Bidirectional
#from keras.layers import Dropout

from keras.callbacks import CSVLogger
from keras.callbacks import Callback

from keras_tqdm import TQDMCallback

from keras import Model
from keras import backend as K


args = sys.argv

try:
    os.mkdir('./output')
except:
    pass

model_name = './output/basic_crnn_2d_sad'

best_weights_path = model_name + '.h5'
log_path = model_name + '.log'
extra_log = model_name + '_extra.log'

opt = 'adam'
batch_size = 32
epochs = 100
rlr_patience = 5

SR = 8000

input_shape = (129, 126, 1)     

N_FFT = 256
HOP_LEN = int(N_FFT / 4)

wav_folder = './data/sad_splits_wavs_npys/'
npy_folder = './data/sad_splits_tags_npys/'


X = []
y = []

for path, subdirs, files in os.walk(wav_folder):
        for file_name in files:
            X.append(path + file_name)
            y.append(npy_folder + file_name)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
    test_size = 0.20, shuffle = True, random_state = seed)

print
print 'train samples', len(y_train)
print 'valid samples', len(y_valid)
print


def random_data_shift(data, time_tags, u):
    if np.random.random() < u:
        random_num = int(np.round(np.random.uniform(-(data.size), (data.size))))
        
        data = np.roll(data, random_num)
        time_tags = np.roll(time_tags, random_num)
    
    return data, time_tags


def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_train))
            
            current_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(current_batch)):
                data = np.load(current_batch[i])
                time_tags = np.load(labels_batch[i])

                data, time_tags = random_data_shift(data, time_tags, u = 1.0)
                
                #data = preemphasis(data, coeff = 0.97)

                X = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN, window = 'hann')
                D = librosa.amplitude_to_db(X)
                D = librosa.feature.melspectrogram(y = data, sr = SR, S = None, n_fft = 2048, hop_length = 64, n_mels = 129)

                # print D.shape[0]
                # print D.shape[1]
                # input("wait")


                data = np.flipud(D)

                # ## LOW CUT
                # for i in range(12):
                #     data = np.delete(data, (data.shape[0]-1), axis = 0)

                # ## HIGH CUT
                # for i in range(12):
                #     data = np.delete(data, (0), axis = 0)

                # plt.imshow(data)
                # plt.show()

                x_batch.append(data)
                y_batch.append(time_tags)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            x_batch = np.expand_dims(x_batch, axis = -1)
           
            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_valid), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_valid))
            
            current_batch = X_valid[start:end]
            labels_batch = y_valid[start:end]
            
            for i in range(len(current_batch)):
                data = np.load(current_batch[i])
                time_tags = np.load(labels_batch[i])

                #data = preemphasis(data, coeff = 0.97)

                X = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN, window = 'hann')
                D = librosa.amplitude_to_db(X)
                # D = librosa.feature.melspectrogram(y = data, sr = SR, S = None, n_fft = 512, hop_length = 64, n_mels = 64)
                D = librosa.feature.melspectrogram(y = data, sr = SR, S = None, n_fft = 2048, hop_length = 64, n_mels = 129)

                data = np.flipud(D)

                # ## LOW CUT
                # for i in range(12):
                #     data = np.delete(data, (data.shape[0]-1), axis = 0)

                # ## HIGH CUT
                # for i in range(12):
                #     data = np.delete(data, (0), axis = 0)

                x_batch.append(data)
                y_batch.append(time_tags)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)

            x_batch = np.expand_dims(x_batch, axis = -1)
           
            yield x_batch, y_batch


def dcf_with_thresh(u = 0.5):
    def dcf(y_true, y_pred):
        y_pred = K.cast(K.greater_equal(y_pred, u), K.floatx())

        val_false = 1 - y_true

        pred_false = 1 - y_pred

        true_total_ns = K.sum(val_false)
        true_total_s = K.sum(y_true)

        fp = K.sum(y_pred * val_false)
        fn = K.sum(pred_false * y_true)

        pfp = fp / true_total_ns
        pfn = fn / true_total_s

        return (0.75 * (pfn)) + (0.25 * (pfp))

    return dcf


acc_dcf_metric_list = []
acc_dcf_metric_list.append('accuracy')
totry = np.arange(0, 1, 0.01)
for t in totry:
    acc_dcf_metric_list.append(dcf_with_thresh(u = float("{0:.2f}".format(t))))


class get_best_dcf_and_thresh(Callback):
    best_dcf_thresh = 0.5
    best_dcf_score = float("inf")
    best_epoch = 1

    reduce_lr = 0

    def on_epoch_begin(self, epoch, logs = {}):
        print
        
        print '\033[92m' + 'Epoch ' + str(epoch + 1) + '\033[0m', 'at learning_rate', K.get_value(model.optimizer.lr)
        
        if (epoch + 1) != 1:
            print 'current best_dcf_thresh', '\033[95m' + str(self.best_dcf_thresh) + '\033[0m'
            print 'current best_dcf_score', '\033[95m' + str(self.best_dcf_score) + '\033[0m'
            print 'current best_epoch', self.best_epoch

        return
 
    def on_epoch_end(self, epoch, logs = {}):
        self.log_epochs_dcfs = []

        self.log_epochs_dcfs.append(logs.get('val_dcf'))

        for i in range(1, 100):
            self.log_epochs_dcfs.append(logs.get('val_dcf_' + str(i)))

        if min(self.log_epochs_dcfs) < self.best_dcf_score:
            self.old_best_dcf_score = self.best_dcf_score

            self.best_dcf_thresh = self.log_epochs_dcfs.index(min(self.log_epochs_dcfs)) / 100.0
            self.best_dcf_score = min(self.log_epochs_dcfs)
            self.best_epoch = epoch + 1

            self.reduce_lr = 0
            
            print
            print
            print '----------> ' + '\033[91m' + 'dcf_score improved from', str(self.old_best_dcf_score), 'to', str(self.best_dcf_score) + ' with dcf_thresh ' + str(self.best_dcf_thresh) + '\033[0m'
            print '----------> ' + '\033[93m' + 'val_loss', str(logs.get('val_loss')) + '\033[0m'
            print '----------> ' + '\033[93m' + 'val_acc', str(logs.get('val_acc')) + '\033[0m'
            print '----------> ' + '\033[93m' + 'val_dcf_0.5', str(logs.get('val_dcf_50')) + '\033[0m'
            
            print
            print 'saving best weights...'

            self.model.save_weights(best_weights_path)

            with open(extra_log, 'a') as my_file:
                my_file.write("\nbest_epoch: " + str(self.best_epoch)
                + "\nbest_dcf_score: " + str(self.best_dcf_score) + ' with dcf_thresh ' + str(self.best_dcf_thresh) + ' at learning_rate ' + str(K.get_value(model.optimizer.lr))
                + " \nval_loss: " + str(logs.get('val_loss'))
                + " \nval_acc: " + str(logs.get('val_acc'))
                + " \nval_dcf_0.5: " + str(logs.get('val_dcf_50'))
                + "\n\n")
        else:
            print
            print
            print 'min_dcf_score', min(self.log_epochs_dcfs), 'with dcf_thresh', self.log_epochs_dcfs.index(min(self.log_epochs_dcfs)) / 100.0
            print 'val_loss', logs.get('val_loss')
            print 'val_acc', logs.get('val_acc')
            print 'val_dcf_0.5', logs.get('val_dcf_50')

            self.reduce_lr += 1
            
            if self.reduce_lr == rlr_patience:
                self.reduce_lr = 0

                print
                print 'reducing learning_rate to', K.get_value(model.optimizer.lr) * 0.1, 'from', K.get_value(model.optimizer.lr)

                K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.1)
        
        return


def basic_crnn_2d(rows, cols, channels, num_classes):
    kernel_size_7 = (7, 7)
    kernel_size_5 = (5, 5)
    kernel_size_3 = (3, 3)
    
    pool_size = (3, 3)
    
    activ = 'relu'

    input_1 = Input(shape = [rows, cols, channels])

    input_2 = Input(shape = [row, cols, channels])

    print input_1.shape

    print input_2.shape

    x = Conv2D(16, kernel_size = kernel_size_7, padding = 'same') (input_1)
    x = BatchNormalization() (x)
    x = Activation(activ) (x)
    x = MaxPooling2D(pool_size, strides = (2, 1), padding = 'same') (x)

    print x.shape

    x = Conv2D(32, kernel_size = kernel_size_5, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation(activ) (x)
    x = MaxPooling2D(pool_size, strides = (2, 1), padding = 'same') (x)

    print x.shape

    x = Conv2D(32, kernel_size = kernel_size_3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation(activ) (x)
    x = MaxPooling2D(pool_size, strides = (2, 1), padding = 'same') (x)

    print x.shape

    x = Conv2D(32, kernel_size = kernel_size_3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation(activ) (x)
    x = MaxPooling2D(pool_size, strides = (2, 1), padding = 'same') (x)

    print x.shape

    x = Conv2D(32, kernel_size = kernel_size_3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation(activ) (x)
    x = MaxPooling2D(pool_size, strides = (2, 1), padding = 'same') (x)

    print x.shape

    x = Permute((2, 1, 3)) (x)
    x = Reshape((126, 5 * 32)) (x)

    print x.shape

    x = Bidirectional(CuDNNGRU(126, return_sequences = True)) (x)
    x = Bidirectional(CuDNNGRU(126, return_sequences = False)) (x)

    print x.shape

    #x = Dropout(0.25) (x)
  
    final = Dense(num_classes) (x)
    
    outputs = Activation('sigmoid', name = 'target') (final)

    model = Model([input_1], [outputs])

    model.compile(optimizer = opt, loss = ['binary_crossentropy'], metrics = acc_dcf_metric_list)

    return model


model = basic_crnn_2d(input_shape[0], input_shape[1], input_shape[2], SR)

callbacks_list = [get_best_dcf_and_thresh(),
                    TQDMCallback(outer_description = "", inner_description_initial = "", 
                        inner_description_update = "", metric_format = "", 
                        separator = "", leave_inner = False, leave_outer = False, show_inner = True, 
                        show_outer = False),
                    CSVLogger(filename = log_path)]

if 'train' in args:
    with open(extra_log, 'wb') as my_file:
        my_file.write("\nseed: " + str(seed) 
            + " \ninput_shape: " + str(input_shape)
            + " \nbatch_size: " + str(batch_size)
            + " \noptimizer: " + str(opt)
            + " \nepochs: " + str(epochs)
            + " \nrlr_patience: " + str(rlr_patience)
            + "\n\n")

    model.fit_generator(train_generator(),
        steps_per_epoch = int(np.ceil(float(len(X_train)) / float(batch_size))),
        validation_data = valid_generator(),
        validation_steps = int(np.ceil(float(len(X_valid)) / float(batch_size))),
        epochs = epochs,
        callbacks = callbacks_list,
        shuffle = False,
        verbose = 0)


K.clear_session()
