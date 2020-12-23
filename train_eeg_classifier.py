import glob
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import itertools
import sys

from scipy.fftpack import fft
from scipy.signal import butter, lfilter, iirnotch
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sobi import sobi
from hyperopt import hp, fmin, tpe, Trials
from mne.decoding import CSP
import mne
from math import inf

mne.set_log_level('warning')

pathfile = os.getcwd() + "\\path.txt"
f = open(pathfile, "r")
lines = f.readlines()
f.close()

model_path = lines[0]

n = 0

with open(model_path + '\\global_settings.json') as inputfile:
    settings = json.load(inputfile)
sampling_rate = settings['sampling_rate']
classes = settings['eeg_classes']
num_classes = len(classes)
columns = settings['channels']
not_engaged_index = classes.index("not_engaged")
engaged_index = classes.index("engaged")
non_railed_channels = settings['non_railed_channels']
rejected_sources = settings['rejected_sources']
column_indices = [columns.index(channel) for channel in columns if channel in non_railed_channels and channel not in rejected_sources]  # todo: put second condition into tuning algorithm itself if ica can be False; for now, only use rejected_sources json field if setting ica to True by default
column_indices = tuple(column_indices)

channel_subsets = []
for i in range(1, len(column_indices) + 1):
    channel_subsets.extend(itertools.combinations(column_indices, i))

if not os.path.exists(model_path + "\\EEG_CV"):
    os.makedirs(model_path + "\\EEG_CV")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y


def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    [b, a] = butter(order, low, btype='high')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y


def read_data(show_plots=False):
    extension = 'csv'
    os.chdir(model_path)

    files = glob.glob('*.{}'.format(extension))
    eeg_data = {}
    for file in files:
        if "OpenBCI_labeled" in file:
            key = file[-5]
            eeg_data[key] = pd.read_csv(file, header=0, delimiter=',', engine='c')

            if show_plots:

                X = np.array(eeg_data[key][non_railed_channels].T)

                for i in range(0, X.shape[0]):
                    pad = 20  # padding signal with zeros prior to bandpass filtering
                    padded_column = np.pad(X[i, :], (pad, 0), mode='mean')
                    X[i, :] = butter_highpass_filter(padded_column, 0.1, sampling_rate, 3)[pad:]

                s, a, w = sobi(X, num_lags=None, eps=1.0e-6, random_order=True)
                contributions = np.mean(np.abs(a) / np.sum(np.abs(a), axis=1), axis=0)  # ordering components by average contribution to channels, based on mixing matrix
                s = s[contributions.argsort()]

                for i in range(0, s.shape[0]):

                    ax1 = plt.subplot(4, 4, i + 1)
                    ax2 = ax1.twinx()
                    # ax1.plot(filtered_column[pad:], 'g-')
                    ax1.plot(s[i, :], 'g-')
                    ax2.plot(eeg_data[key]['label'], 'b-')

                    ax1.set_xlabel('')
                    ax1.set_ylabel('')
                    ax2.set_ylabel('')
                    plt.ylim(5, -5)

                plt.show()

                for i in range(0, s.shape[0]):

                    N = np.size(X[i, :])
                    yf = fft(X[i, :])
                    Yf = np.power(np.abs(yf[:N // 2]), 2)
                    scale_factor = sampling_rate / (2 * np.size(Yf))
                    xf = np.linspace(0, np.size(Yf), np.size(Yf))
                    xf = scale_factor * xf

                    ax = plt.subplot(4, 4, i + 1)
                    # ax.plot(filtered_column[pad:], 'g-')
                    ax.plot(xf, Yf, 'g-')

                    ax.set_xlabel('')
                    ax.set_ylabel('')

                plt.show()

                for i, channel in enumerate(columns):
                    pad = 20  # padding signal with zeros prior to bandpass filtering
                    padded_column = np.pad(eeg_data[key][channel], (pad, 0), mode='mean')
                    filtered_column = butter_highpass_filter(padded_column, 0.1, sampling_rate, 3)

                    ax1 = plt.subplot(4, 4, i + 1)
                    ax2 = ax1.twinx()
                    # ax1.plot(filtered_column[pad:], 'g-')
                    ax1.plot(eeg_data[key][channel], 'g-')
                    # ax2.plot(eeg_data[key]['label'], 'b-')

                    ax1.set_xlabel('')
                    ax1.set_ylabel('')
                    ax2.set_ylabel('')
                    # ax1.set_xticks([])
                    # ax1.set_yticks([])
                    # ax2.set_yticks([])
                    plt.ylim(5, -5)

                plt.show()

                for i, channel in enumerate(columns):
                    pad = 20  # padding signal with zeros prior to bandpass filtering
                    padded_column = np.pad(eeg_data[key][channel], (pad, 0), mode='mean')
                    filtered_column = butter_highpass_filter(padded_column, 0.1, sampling_rate, 3)
                    # filtered_column = padded_column[pad:]

                    N = np.size(filtered_column)
                    yf = fft(filtered_column)
                    Yf = np.power(np.abs(yf[:N // 2]), 2)
                    scale_factor = sampling_rate / (2 * np.size(Yf))
                    xf = np.linspace(0, np.size(Yf), np.size(Yf))
                    xf = scale_factor * xf

                    ax = plt.subplot(4, 4, i + 1)
                    # ax.plot(filtered_column[pad:], 'g-')
                    ax.plot(xf, Yf, 'g-')

                    ax.set_xlabel('')
                    ax.set_ylabel('')

                plt.show()

    return eeg_data


def split_indices(period_array, labels, n_folds=3):
    if n_folds == -1:  # maximum number of folds, not fully debugged
        sizes = []
        for period in np.unique(np.array(period_array)):
            indices = np.where(np.array(period_array) == period)[0]
            sizes.append(np.size(indices))
        n_folds = np.min(sizes)

    n_periods = np.size(np.unique(np.array(period_array)))
    fold_indices = []
    for i in range(0, n_folds):
        fold_indices.append([])
    for i in range(0, n_periods):
        for k in range(0, len(classes)):
            period_indices = np.where(np.array(period_array) == i)[0]
            class_indices = np.where(np.array(labels) == i)[0]
            period_indices = np.intersect1d(period_indices, class_indices)
            split_interval = int(np.floor(np.size(period_indices) / n_folds))
            start_index = 0
            used_folds = [-1]
            for j in range(0, n_folds):
                random_fold_index = -1
                while random_fold_index in used_folds:  # randomly adding samples in this region of the given period to a fold that doesn't have samples of this period yet
                    random_fold_index = np.random.randint(0, n_folds)
                used_folds.append(random_fold_index)
                fold_indices[random_fold_index].extend(period_indices[start_index:start_index + split_interval].tolist())
                start_index += split_interval

    return fold_indices


def epoch_data(eeg_data, bin_size=1000, overlap=0.5, low_cutoff=1, high_cutoff=50, filter_order=5, ica=True,
               ica_pre_filter="bandpass"):
    length = int((bin_size / 1000) * sampling_rate)
    if bin_size < 1000:
        length += 1
    epoch_matrix_shape = (len(columns), length)
    # print(epoch_matrix_shape)

    epochs_train = []
    labels_train = []
    periods_train = []  # tracking period of each training sample for cv splitting based on time, to prevent training
    epochs_test = []    # on samples close in time to test samples and thus to prevent overfitting to shared noise in
    labels_test = []    # these close-in-time samples

    unmixing_matrices = []
    for key, eeg_frame in eeg_data.items():
        maximum_sample_gap = np.max(np.diff(eeg_frame['timestamp']))
        for channel in columns:
            pad = 20  # padding signal with mean prior to bandpass filtering
            padded_column = np.pad(eeg_frame[channel], (pad, 0), mode='mean')

            f0 = 60.0  # Frequency to be removed from signal (Hz)
            Q = 100.0  # Quality factor
            # Design notch filter
            [b, a] = iirnotch(f0, Q, sampling_rate)
            padded_column = lfilter(b, a, padded_column)
            padded_column = np.pad(padded_column[pad:], (pad, 0), mode='mean')

            if ica:
                if ica_pre_filter is None:
                    filtered_column = padded_column
                elif ica_pre_filter == "bandpass":
                    filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                             filter_order)
                elif ica_pre_filter == "highpass":
                    filtered_column = butter_highpass_filter(padded_column, low_cutoff, sampling_rate, filter_order)
            else:
                filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate, filter_order)
            eeg_frame[channel] = filtered_column[pad:]
            eeg_frame[channel].replace([np.inf, -np.inf, inf, -inf], np.nan)  # replacing NaN's and inf in signal with signal mean
            mean = np.mean(eeg_frame[channel][eeg_frame[channel] != np.nan])
            eeg_frame[channel].fillna(mean)

        # if ica:
        #     s, a, w = sobi(eeg_frame[non_railed_channels].T, num_lags=None, eps=1.0e-6, random_order=True)
        #     eeg_frame[non_railed_channels] = s.T

        # print("Period {}".format(key))
        for tag in range(0, 2):     # 0 = train, 1 = test
            # print("Tag {}".format(tag))
            if ica:
                eeg_sub_frame = eeg_frame[(eeg_frame['evaluation_tag'] == tag)]
                if tag == 0:
                    s, a, w = sobi(eeg_sub_frame[non_railed_channels].T, num_lags=None, eps=1.0e-6, random_order=True)
                    eeg_frame.loc[(eeg_frame['evaluation_tag'] == tag), non_railed_channels] = s.T
                    contributions = np.mean(np.abs(a) / np.sum(np.abs(a), axis=1), axis=0)  # ordering components by average contribution to channels, based on mixing matrix
                    w = w[contributions.argsort()]
                    unmixing_matrices.append(w)
                else:
                    sources = np.dot(unmixing_matrices[int(key)], eeg_sub_frame[non_railed_channels].T)
                    eeg_frame.loc[(eeg_frame['evaluation_tag'] == tag), non_railed_channels] = sources.T

            for label in range(0, len(classes)):
                # print("Label {}".format(label))
                # print(classes[label])
                eeg_sub_frame = eeg_frame[(eeg_frame['evaluation_tag'] == tag) & (eeg_frame['label'] == label)]
                if len(eeg_sub_frame) == 0:     # if there are no samples of the given label and tag in this period, continue to next iteration of loop
                    continue
                diff = np.diff(eeg_sub_frame['timestamp'])
                # split_indices = np.where(diff >= np.max(diff))[0] + 1
                split_indices = np.where(diff > maximum_sample_gap)[0] + 1
                split_indices = np.concatenate((split_indices, [-1]))
                start_index = 0
                for index in split_indices:
                    eeg_region = eeg_sub_frame.iloc[start_index:index]
                    start_index = index
                    time_index = 0
                    start_time = eeg_region['timestamp'].iloc[time_index]
                    for i, time in enumerate(eeg_region['timestamp']):
                        if 1000 * (time - start_time) >= bin_size:
                            epoch = eeg_region[(eeg_region['timestamp'] >= start_time) & (eeg_region['timestamp'] < time)]
                            epoch = epoch[columns]
                            epoch = np.array(epoch).T
                            # print(epoch.shape)
                            if epoch.shape == epoch_matrix_shape:
                                if tag == 0:
                                    epochs_train.append(epoch)
                                    labels_train.append(label)
                                    periods_train.append(int(key))
                                elif tag == 1:
                                    epochs_test.append(epoch)
                                    labels_test.append(label)

                            if tag == 0:  # overlap is a form of data augmentation, so only applying to training data here
                                time_index = np.where(1000*(eeg_region['timestamp'] - start_time) >= bin_size - overlap*bin_size)[0]
                                time_index = time_index[0]
                            elif tag == 1:
                                time_index = i
                            start_time = eeg_region['timestamp'].iloc[time_index]

    epochs_train = np.stack(epochs_train, axis=0)
    labels_train = np.array(labels_train)
    epochs_test = np.stack(epochs_test, axis=0)
    labels_test = np.array(labels_test)

    return epochs_train, labels_train, epochs_test, labels_test, periods_train, unmixing_matrices


def extract_scores(epochs):
    features = []
    for i, epoch in enumerate(epochs):
        pad = 20
        padded = np.pad(epoch, ((0, 0), (pad, 0)))

        filtered = butter_bandpass_filter(padded, 4, 8, sampling_rate, 3)
        theta = np.sum(np.square(np.array(filtered[:, pad:])), axis=1)
        filtered = butter_bandpass_filter(padded, 7, 13, sampling_rate, 3)
        alpha = np.sum(np.square(np.array(filtered[:, pad:])), axis=1)
        filtered = butter_bandpass_filter(padded, 13, 30, sampling_rate, 3)
        beta = np.sum(np.square(np.array(filtered[:, pad:])), axis=1)

        score = beta / (alpha + theta)

        features.append(score)
    return np.array(features)


def extract_powers(epochs):
    powers = np.sum(np.square(np.array(epochs)), axis=-1)/np.size(epochs, axis=-1)
    return powers


def apply_csp(epochs_train, labels_train, epochs_test, n_filters, return_features=False):
    result = 'csp_space'
    log = None
    if return_features:
        result = 'average_power'
        log = True
    csp = CSP(n_components=n_filters, reg='shrinkage', log=log, cov_est="concat",
              transform_into=result, norm_trace=False,
              cov_method_params=None, rank=None)

    for i, epoch in enumerate(epochs_train):
        for k, channel in enumerate(epoch):
            for j, sample in enumerate(channel):
                if sample > sys.float_info.max or sample == np.nan or sample == inf:
                    print(sample)
                    # epochs_train[i][k][j] = 0
                    # print(epochs_train[i][k][j])

    csp.fit(epochs_train, labels_train)

    if return_features:
        features_train = csp.transform(epochs_train)
        features_test = csp.transform(epochs_test)
        return features_train, features_test, csp
    return csp


def extract_features(epochs_train, labels_train, epochs_test, n_filters, processing, features, channels):
    if processing == 'csp':
        epochs_train = epochs_train[:, channels]
        epochs_test = epochs_test[:, channels]

        if features == 'average_power':
            features_train, features_test, csp = apply_csp(epochs_train, labels_train, epochs_test, n_filters,
                                                           return_features=True)
        elif features == 'scores':
            csp = apply_csp(epochs_train, labels_train, epochs_test, n_filters, return_features=False)
            epochs_train = csp.transform(epochs_train)
            epochs_test = csp.transform(epochs_test)

            features_train = extract_scores(epochs_train)
            features_test = extract_scores(epochs_test)
    else:
        if features == 'average_power':
            features_train = extract_powers(epochs_train)
            features_test = extract_powers(epochs_test)
        elif features == 'scores':
            features_train = extract_scores(epochs_train)
            features_test = extract_scores(epochs_test)

        features_train = features_train[:, channels]
        features_test = features_test[:, channels]
        csp = None

    return features_train, features_test, csp


def resample_epochs(epochs, labels):
    epochs_ = {}
    labels_ = {}
    sizes = []
    for i, c in enumerate(classes):
        indices = np.where(labels == i)[0]
        sizes.append(len(indices))
        epochs_[c] = epochs[indices]
        labels_[c] = labels[indices]

    min_ = np.min(sizes)

    for i, key in enumerate(epochs_):
        epochs_[key], labels_[key] = resample(epochs_[key], labels_[key], n_samples=min_, replace=False)

        if i == 0:
            epochs = epochs_[key]
            labels = labels_[key]
        else:
            epochs = np.concatenate((epochs, epochs_[key]))
            labels = np.append(labels, labels_[key])

    return epochs, labels


def train_model(epochs_train, labels_train, epochs_test, labels_test, model, certainty_threshold=0.5,
                index=str(0), final=False, n_filters=4, resample_flag=True, processing='csp', features='average_power',
                pca=False, channels=column_indices):

    if final:
        index = "Final"

    model_type = model['type']

    features_train, features_test, csp = extract_features(epochs_train, labels_train, epochs_test, n_filters, processing, features, channels)

    if resample_flag:
        features_train, labels_train = resample_epochs(features_train, labels_train)
        features_test, labels_test = resample_epochs(features_test, labels_test)

    features_train = np.nan_to_num(features_train, posinf=0, neginf=0, nan=0)
    features_test = np.nan_to_num(features_test, posinf=0, neginf=0, nan=0)

    scaler = StandardScaler().fit(features_train)
    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)

    if pca:
        p = PCA(n_components='mle')
        test = p.fit_transform(features_train)
        if np.size(test, -1) == 0:
            p = PCA()
        p.fit(features_train)
        features_train = p.transform(features_train)
        features_test = p.transform(features_test)

    clf = []
    if model_type == "NN":
        alpha0 = model['alpha0']
        decay = model['decay']

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: alpha0 * decay ** epoch)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_filters, activation='relu', input_shape=[n_filters]),
            tf.keras.layers.Dense(len(classes), activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=alpha0, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             amsgrad=False)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        EPOCHS = 10
        BATCH_SIZE = 8
        model.fit(features_train, labels_train, validation_data=(features_test, labels_test), epochs=EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stopping, lr_schedule])

        clf.append(model)
    elif model_type == "LDA":
        clf.append(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=None, n_components=None,
                                              store_covariance=False, tol=0.0001))
    elif model_type == "ADB":
        clf.append(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=model['criterion'],
                                                                            max_depth=model['max_depth'],
                                                                            min_samples_leaf=model['min_samples_leaf'],
                                                                            min_samples_split=model['min_samples_split']),
                                      n_estimators=model['n_estimators'], learning_rate=model['learning_rate'],
                                      algorithm='SAMME.R', random_state=None))
    elif model_type == "RF":
        if model['max_depth']:
            max_depth = int(model['max_depth'])
        else:
            max_depth = None

        global n
        if pca:
            n = p.n_components_
        else:
            if processing == 'csp':
                n = n_filters
            else:
                n = len(channels)

        if 'max_features_fraction' in model.keys():
            model['max_features'] = int(model['max_features_fraction'] * (n - 1) + 1)
        else:
            n = int(np.random.randint(1, n + 1))
            model['max_features'] = n

        clf.append(RandomForestClassifier(n_estimators=model['n_estimators'], criterion=model['criterion'],
                                          max_depth=max_depth, min_samples_leaf=int(model['min_samples_leaf']),
                                          min_samples_split=int(model['min_samples_split']),
                                          max_features=model['max_features']))
    elif model_type == "SVM":
        clf.append(SVC(C=model['C'], kernel=model['kernel'], degree=model['degree'], gamma='scale', max_iter=1000000, probability=True))
    elif model_type == "KNN":
        clf.append(KNeighborsClassifier(n_neighbors=model['n_neighbors'], p=model['p'], metric=model['metric']))
    elif model_type == "LR":
        clf.append(LogisticRegression(solver='lbfgs', C=model['c'], max_iter=100000))
    elif model_type == "NB":
        clf.append(GaussianNB())

    if model_type != "NN":
        clf[0].fit(features_train, labels_train)
        # train_results = clf[0].predict(features_train)
        train_probs = clf[0].predict_proba(features_train)
        # test_results = clf[0].predict(features_test)
        test_probs = clf[0].predict_proba(features_test)
    else:
        train_probs = clf[0].predict(features_train)
        test_probs = clf[0].predict(features_test)

    train_results = []
    for prob in train_probs[:, engaged_index]:
        if prob > certainty_threshold:
            train_results.append(engaged_index)
        else:
            train_results.append(not_engaged_index)

    test_results = []
    for prob in test_probs[:, engaged_index]:
        if prob > certainty_threshold:
            test_results.append(engaged_index)
        else:
            test_results.append(not_engaged_index)

    conf_mx = confusion_matrix(labels_train, train_results)
    f1 = f1_score(labels_train, train_results, average='binary')
    [fpr, tpr, _] = roc_curve(labels_train, train_probs[:, engaged_index])
    AUC = roc_auc_score(labels_train, train_probs[:, engaged_index])

    figure_size = (14, 12)
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure_size)
    sns.heatmap(conf_mx, annot=True, ax=ax, cbar=False, cmap='binary')
    ax.set_ylim(conf_mx.shape[0] - 0, -0.5)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    s = 0
    for i in range(0, np.size(np.unique(classes))):
        s = s + conf_mx[i, i]
    accuracy = s / sum(sum(conf_mx)) * 100
    ax.set_title(index + ' Training Accuracy: {0:.3f}%, F1 Score: {1:.3f}'.format(accuracy, f1))
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    ax3.plot(fpr, tpr)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Training Area Under Curve: {0:.3f}'.format(AUC))

    print("{} Training Accuracy: {}".format(index, accuracy))
    # print("{} Training F-1 Score: {}".format(index, f1))

    conf_mx = confusion_matrix(labels_test, test_results)
    f1 = f1_score(labels_test, test_results, average='binary')
    [fpr, tpr, _] = roc_curve(labels_test, test_probs[:, engaged_index])
    AUC = roc_auc_score(labels_test, test_probs[:, engaged_index])

    sns.heatmap(conf_mx, annot=True, ax=ax2, cbar=False, cmap='binary')
    ax2.set_ylim(conf_mx.shape[0] - 0, -0.5)
    ax2.set_xlabel('Predicted labels')
    ax2.set_ylabel('True labels')
    s = 0
    for i in range(0, np.size(np.unique(classes))):
        s = s + conf_mx[i, i]
    accuracy = s / sum(sum(conf_mx)) * 100
    ax2.set_title(index + ' Test Accuracy: {0:.3f}%, F1 Score: {1:.3f}'.format(accuracy, f1))
    ax2.xaxis.set_ticklabels(classes)
    ax2.yaxis.set_ticklabels(classes)

    ax4.plot(fpr, tpr)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('Test Area Under Curve: {0:.3f}'.format(AUC))

    print("{} Test Accuracy: {}".format(index, accuracy))
    # print("{} Test F-1 Score: {}".format(index, f1))

    if not final:
        if model_type == "NN":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_a={}_d={}_fold_{}_{}.png'.format(n_filters, model_type, model['alpha0'],
                                                                                        model['decay'], index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
        if model_type == "LDA":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_fold_{}_{}.png'.format(n_filters, model_type, index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
        if model_type == "ADB":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_n={}_l={}_fold_{}_{}.png'.format(n_filters, model_type, model['n_estimators'],
                                                                                model['learning_rate'], index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
        if model_type == "RF":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_crit={}_num={}_max={}_minsplit={}_minleaf={}_maxf={}_fold_{}_{}.png'.format(n_filters, model_type,
                                                                                                           model['criterion'],
                                                                                                           model['n_estimators'],
                                                                                                           model['max_depth'],
                                                                                                           model['min_samples_split'],
                                                                                                           model['min_samples_leaf'],
                                                                                                           model['max_features'],
                                                                                                           index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
        if model_type == "SVM":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_c={}_k={}_d={}_fold_{}_{}.png'.format(n_filters, model_type,
                                                                                             model['C'],
                                                                                             model['kernel'],
                                                                                             model['degree'], index,
                                                                                             time.time()),
                        dpi=fig.dpi,
                        figsize=figure_size)
        if model_type == "KNN":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_n={}_p={}_metric={}_fold_{}_{}.png'.format(n_filters, model_type,
                                                                                             model['n_neighbors'],
                                                                                             model['p'],
                                                                                             model['metric'], index,
                                                                                             time.time()),
                        dpi=fig.dpi,
                        figsize=figure_size)
        if model_type == "LR":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_c={}_fold_{}_{}.png'.format(n_filters, model_type, model['c'], index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
        if model_type == "NB":
            plt.savefig('./EEG_CV/eeg_nfilters={}_m={}_fold_{}_{}.png'.format(n_filters, model_type, index, time.time()),
                        dpi=fig.dpi, figsize=figure_size)
    else:
        plt.savefig('eeg_test_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    metric = accuracy/100
    if processing == 'csp':
        feature_extractor = csp
    else:
        feature_extractor = None

    if pca:
        pc = p
    else:
        pc = None

    return feature_extractor, scaler, clf[0], pc, metric


def cross_val(args):
    print(args)
    processing = args['processing']
    if processing['type'] == 'csp':
        selected_channels = processing['selected_channels0']
        n_filters = int(np.max((1, int(processing['n_filters_fraction'] * len(selected_channels)))))
    elif processing['type'] is None:
        n_filters = None
        selected_channels = processing['selected_channels1']
    features = args['features']
    pca = args['pca']
    ica = args['ica_settings']['ica']
    ica_pre_filter = args['ica_settings']['ica_pre_filter']
    bin_size = int(args['bin_size'])
    overlap = args['overlap']
    low_cutoff = float(args['low_cutoff'])
    high_cutoff = float(args['high_cutoff_fraction']*(max_high_cutoff - (low_cutoff + 10)) + low_cutoff + 10)
    filter_order = int(args['filter_order'])
    certainty_threshold = args['certainty_threshold']
    n_folds = args['n_folds']
    random_split = args['random_split']

    model = args['model']

    epochs_train, labels_train, _, _, periods_train, _ = epoch_data(eeg_data, bin_size, overlap, low_cutoff, high_cutoff, filter_order, ica, ica_pre_filter)

    splits = []
    if random_split:
        kf = KFold(n_splits=n_folds, shuffle=True)
        splits = kf.split(epochs_train, y=labels_train)
    else:
        fold_indices = split_indices(periods_train, labels_train, n_folds=n_folds)  # splitting periods into equivalence sets (folds) to prevent overfitting during cv
        number_of_folds = len(fold_indices)
        all_folds = np.arange(0, number_of_folds)
        for fold, indices in enumerate(fold_indices):
            train_folds = np.where(all_folds != fold)[0]
            train_indices = []
            for train_fold in train_folds:
                train_indices.extend(fold_indices[train_fold])
            test_indices = indices
            splits.append((train_indices, test_indices))

    metrics = []
    k = 1
    for train_indices, test_indices in splits:
        _, _, _, _, metric = train_model(epochs_train[train_indices], labels_train[train_indices],
                                   epochs_train[test_indices], labels_train[test_indices], certainty_threshold=certainty_threshold,
                                   model=model, index=str(k), n_filters=n_filters, resample_flag=True,
                                   processing=processing['type'], features=features, pca=pca, channels=selected_channels)
        metrics.append(1-metric)
        k += 1

    cv_mean = np.mean(metrics)

    return cv_mean


def random_search(n_iterations=10, n_folds=10, random_split=False):
    best_hyperparameters = {}
    best_mean = 0
    model_types = ["SVM"]  # ["NN", "LDA", "ADB", "RF", "SVM", "KNN", "LR", "NB"]
    cv_means = []
    best_means = []
    # n_estimators = []
    # min_samples_leaf = []
    # min_samples_split = []
    # max_features = []
    # max_depths = []
    processing_methods = ['csp', None]
    features_list = ['average_power', 'scores']
    pca_choice = [True, False]
    ica_choice = [True]  # , False]
    ica_pre_filter_choice = [None, "bandpass", "highpass"]
    for j in range(0, n_iterations):
        processing = processing_methods[np.random.randint(0, len(processing_methods))]
        features = features_list[np.random.randint(0, len(features_list))]
        pca = pca_choice[np.random.randint(0, len(pca_choice))]
        ica = ica_choice[np.random.randint(0, len(ica_choice))]
        if ica:
            ica_pre_filter = ica_pre_filter_choice[np.random.randint(0, len(ica_pre_filter_choice))]
        else:
            ica_pre_filter = None
        # bin_sizes = [1000, 1500, 2000, 2500, 3000]
        bin_sizes = [1000]
        bin_size = bin_sizes[np.random.randint(0, len(bin_sizes))]
        overlap = 0.99  # np.random.uniform(0, 0.5)
        low_cutoff = 1  # np.random.randint(1, 21)
        high_cutoff = max_high_cutoff  # np.random.randint(low_cutoff + 10, max_high_cutoff)
        filter_order = 3  # np.random.randint(3, 7)
        certainty_threshold = 0.5  # np.random.uniform(0.5, 1)
        selected_channels = channel_subsets[np.random.randint(0, len(channel_subsets))]
        if processing != 'csp':
            n_filters = None
        else:
            n_filters = np.random.randint(1, len(selected_channels) + 1)

        selected_model = model_types[np.random.randint(0, len(model_types))]
        model = {'model': {'type': selected_model}}
        if selected_model == "NN":
            model['model']['alpha0'] = 10 ** np.random.uniform(-5, -3)
            model['model']['decay'] = np.random.uniform(0.9, 1.0)
        if selected_model == "ADB":
            criteria = ['gini', 'entropy']
            max_depth_choice = [None, np.random.randint(1, 100)]
            model['model']['criterion'] = criteria[np.random.randint(0, len(criteria))]
            model['model']['n_estimators'] = np.random.randint(1, 1001)
            model['model']['learning_rate'] = 10 ** np.random.uniform(0, 3)
            model['model']['max_depth'] = max_depth_choice[np.random.randint(0, len(max_depth_choice))]
            model['model']['min_samples_leaf'] = np.random.randint(1, 100)
            model['model']['min_samples_split'] = np.random.randint(2, 100)
        if selected_model == "RF":
            criteria = ['gini', 'entropy']
            max_depth_choice = [None, np.random.randint(1, 100)]
            model['model']['criterion'] = criteria[np.random.randint(0, len(criteria))]
            model['model']['n_estimators'] = np.random.randint(1, 1001)
            model['model']['max_depth'] = max_depth_choice[np.random.randint(0, len(max_depth_choice))]
            model['model']['min_samples_leaf'] = np.random.randint(1, 100)
            model['model']['min_samples_split'] = np.random.randint(2, 100)
        if selected_model == "SVM":
            kernels = ['rbf']#['linear', 'poly', 'rbf']
            model['model']['kernel'] = kernels[np.random.randint(0, len(kernels))]
            model['model']['degree'] = np.random.randint(1, 6)
            model['model']['C'] = 10**np.random.uniform(-3, 3)
            print(model['model']['C'])
        if selected_model == "KNN":
            metrics = ['manhattan', 'chebyshev', 'minkowski']
            model['model']['n_neighbors'] = np.random.randint(1, 11)
            model['model']['p'] = np.random.randint(2, 6)
            model['model']['metric'] = metrics[np.random.randint(0, len(metrics))]
        if selected_model == "LR":
            model['model']['c'] = 10**np.random.lognormal(0, 1)

        epochs_train, labels_train, _, _, periods_train, _ = epoch_data(eeg_data, bin_size, overlap, low_cutoff,
                                                                     high_cutoff, filter_order, ica, ica_pre_filter)

        splits = []
        if random_split:
            kf = KFold(n_splits=n_folds, shuffle=True)
            splits = kf.split(epochs_train, y=labels_train)
        else:
            fold_indices = split_indices(periods_train, labels_train,
                                         n_folds=n_folds)  # splitting periods into equivalence sets (folds) to prevent overfitting during cv
            number_of_folds = len(fold_indices)
            all_folds = np.arange(0, number_of_folds)
            for fold, indices in enumerate(fold_indices):
                train_folds = np.where(all_folds != fold)[0]
                train_indices = []
                for train_fold in train_folds:
                    train_indices.extend(fold_indices[train_fold])
                test_indices = indices
                splits.append((train_indices, test_indices))

        metrics = []
        k = 1
        for train_indices, test_indices in splits:
            _, _, _, _, metric = train_model(epochs_train[train_indices], labels_train[train_indices],
                                       epochs_train[test_indices], labels_train[test_indices], certainty_threshold=certainty_threshold,
                                       index=str(k), n_filters=n_filters, model=model['model'], resample_flag=True,
                                       processing=processing, features=features, pca=pca, channels=selected_channels)
            metrics.append(metric)
            k += 1

        cv_mean = np.mean(metrics)
        cv_means.append(1-cv_mean)

        # n_estimators.append(model['model']['n_estimators'])
        # max_depths.append(model['model']['max_depth'])
        # min_samples_leaf.append(model['model']['min_samples_leaf'])
        # min_samples_split.append(model['model']['min_samples_split'])
        # max_features.append(model['model']['max_features'])
        if cv_mean > best_mean:
            if processing == 'csp':
                processing = {'type': processing, 'n_filters': n_filters, 'selected_channels': selected_channels}
            elif processing is None:
                processing = {'type': processing, 'n_filters': None, 'selected_channels': selected_channels}

            ica_settings = {'ica_settings': {'ica': ica, 'ica_pre_filter': ica_pre_filter}}

            best_hyperparameters = {'processing': processing, 'features': features, 'pca': pca, 'ica_settings': ica_settings,
                                    'bin_size': bin_size, 'overlap': overlap, 'low_cutoff': low_cutoff,
                                    'high_cutoff': high_cutoff, 'filter_order': filter_order,
                                    'certainty_threshold': certainty_threshold, 'model': model['model']}
            best_mean = cv_mean
        best_means.append(1-best_mean)

        print("{}/{}, best score: {}".format(j + 1, n_iterations, best_mean))

    figure_size = (14, 8)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.plot(cv_means, label='Iteration Loss')
    ax.plot(best_means, label='Best Loss')
    plt.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.savefig('eeg_tuning_loss_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    # figure_size = (14, 8)
    # fig, ax = plt.subplots(1, 1, figsize=figure_size)
    # ax.scatter(n_estimators, cv_means)
    # ax.set_xlabel('N Estimators')
    # ax.set_ylabel('Average Loss')
    #
    # figure_size = (14, 8)
    # fig, ax = plt.subplots(1, 1, figsize=figure_size)
    # ax.scatter(min_samples_leaf, cv_means)
    # ax.set_xlabel('Min Samples Leaf')
    # ax.set_ylabel('Average Loss')
    #
    # figure_size = (14, 8)
    # fig, ax = plt.subplots(1, 1, figsize=figure_size)
    # ax.scatter(min_samples_split, cv_means)
    # ax.set_xlabel('Min Samples Split')
    # ax.set_ylabel('Average Loss')
    #
    # figure_size = (14, 8)
    # fig, ax = plt.subplots(1, 1, figsize=figure_size)
    # ax.scatter(max_depths, cv_means)
    # ax.set_xlabel('Max Depth')
    # ax.set_ylabel('Average Loss')
    #
    # figure_size = (14, 8)
    # fig, ax = plt.subplots(1, 1, figsize=figure_size)
    # ax.scatter(max_features, cv_means)
    # ax.set_xlabel('Max Features')
    # ax.set_ylabel('Average Loss')
    #
    # plt.show()

    if best_hyperparameters['model']['type'] == "RF":
        global n
        best_hyperparameters['model']['max_features'] = n

    return best_hyperparameters


def bayesian_search(n_iterations=10, n_folds=10, random_split=False):
    # bin_sizes = [1000, 1500, 2000, 2500, 3000]
    bin_sizes = [2000]
    features = ['average_power']  # , 'scores']
    pca = [True, False]
    ica_pre_filter = ["bandpass"]  # [None, "bandpass", "highpass"]
    space = {
             'processing': hp.choice('processing', [
                 {
                     'type': 'csp',
                     'n_filters_fraction': hp.uniform('n_filters_fraction', 0, 1),  # 1 + hp.randint('n_filters', len(column_indices)),
                     'selected_channels0': hp.choice('selected_channels0', channel_subsets),
                 },
                 {
                     'type': None,
                     'selected_channels1': hp.choice('selected_channels1', channel_subsets),
                 },
             ]),
             'features': hp.choice('features', features),
             'pca': hp.choice('pca', pca),
             'ica_settings': hp.choice('ica_settings', [
                 {
                     'ica': True,
                     'ica_pre_filter': hp.choice('ica_pre_filter', ica_pre_filter)
                 },
                 # {
                 #     'ica': False,
                 #     'ica_pre_filter': None
                 # },
             ]),
             'bin_size': hp.choice('bin_size', bin_sizes),
             'overlap': 0.9,  # hp.uniform('overlap', 0, 0.5),
             'low_cutoff': hp.randint('low_cutoff', 20) + 1,
             'high_cutoff_fraction': hp.uniform('high_cutoff_fraction', 0, 1),
             'filter_order': 5,  # + hp.randint('filter_order', 4),
             'certainty_threshold': 0.5,  # hp.uniform('certainty_threshold', 0.5, 1),
             'n_folds': n_folds,
             'random_split': random_split,
             'model': hp.choice('model', [
                        # {
                        #     'type': 'NN',
                        #     'alpha0': 10 ** hp.uniform('alpha0', -3, -1),
                        #     'decay': hp.uniform('decay', 0.9, 1.0),
                        # },
                        {
                            'type': 'LDA',
                        },
                        # {
                        #     'type': 'ADB',
                        #     'criterion': hp.choice('criterion', ['gini', 'entropy']),
                        #     'n_estimators': 1 + hp.randint('n_estimators', 1000),
                        #     'max_depth': 1 + hp.randint('max_depth_select', 20),  # hp.choice('max_depth', [None, 1 + hp.randint('max_depth_select', 99)]),
                        #     'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 30),
                        #     'min_samples_split': 2 + hp.randint('min_samples_split', 19),
                        #     'learning_rate': 10 ** hp.uniform('learning_rate', -2, 1)
                        # },
                        # {
                        #     'type': 'RF',
                        #     'criterion': hp.choice('criterion', ['gini', 'entropy']),
                        #     'n_estimators': 1 + hp.randint('n_estimators', 1000),
                        #     'max_depth': 1 + hp.randint('max_depth', 20),  # hp.choice('max_depth', [None, 1 + hp.randint('max_depth', 99)]),
                        #     'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 5),
                        #     'min_samples_split': 2 + hp.randint('min_samples_split', 5),
                        #     'max_features_fraction': hp.uniform('max_features_fraction', 0, 1),
                        # },
                        # {
                        #     'type': 'SVM',
                        #     'C': 10 ** hp.uniform('C', -3, 3),
                        #     'kernel': hp.choice('kernel', ['rbf']),  # ['linear', 'poly', 'rbf']),
                        #     'degree': 0 + hp.randint('degree', 1),
                        # },
                        # {
                        #     'type': 'KNN',
                        #     'n_neighbors': 1 + hp.randint('n_neighbors', 10),
                        #     'p': 2 + hp.randint('p', 4),
                        #     'metric': hp.choice('metric', ['manhattan', 'chebyshev', 'minkowski']),
                        # },
                        # {
                        #     'type': 'LR',
                        #     'c': 10 ** hp.lognormal('c', 0, 1),
                        # },
                        # {
                        #     'type': 'NB',
                        # },
                    ])
            }

    # minimize the objective over the space
    trials = Trials()
    best_hyperparameters = fmin(cross_val, space, algo=tpe.suggest, max_evals=n_iterations, return_argmin=False, trials=trials)

    figure_size = (14, 8)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.plot(trials.losses(), label='Iteration Loss')
    trial_mins = [np.min(trials.losses()[:i+1]) for i in range(0, len(trials.losses()))]
    ax.plot(trial_mins, label='Best Loss')
    plt.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.savefig('eeg_tuning_loss_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    if best_hyperparameters['processing']['type'] == 'csp':
        best_hyperparameters['processing']['selected_channels'] = best_hyperparameters['processing']['selected_channels0']
        del best_hyperparameters['processing']['selected_channels0']
    else:
        best_hyperparameters['processing']['selected_channels'] = best_hyperparameters['processing']['selected_channels1']
        del best_hyperparameters['processing']['selected_channels1']

    if best_hyperparameters['processing']['type'] == 'csp':
        best_hyperparameters['processing']['n_filters'] = int(np.max((1, best_hyperparameters['processing']['n_filters_fraction']*len(best_hyperparameters['processing']['selected_channels']))))
        # best_hyperparameters['processing']['selected_channels'] = None
    elif best_hyperparameters['processing']['type'] is None:
        best_hyperparameters['processing']['n_filters'] = None
    best_hyperparameters['bin_size'] = int(best_hyperparameters['bin_size'])
    best_hyperparameters['low_cutoff'] = float(best_hyperparameters['low_cutoff'])
    best_hyperparameters['high_cutoff'] = float((max_high_cutoff - (best_hyperparameters['low_cutoff'] + 10))*best_hyperparameters['high_cutoff_fraction'] + best_hyperparameters['low_cutoff'] + 10)
    best_hyperparameters['filter_order'] = int(best_hyperparameters['filter_order'])
    if best_hyperparameters['model']['type'] == "RF" or best_hyperparameters['model']['type'] == "ADB":
        best_hyperparameters['model']['n_estimators'] = int(best_hyperparameters['model']['n_estimators'])
        best_hyperparameters['model']['min_samples_leaf'] = int(best_hyperparameters['model']['min_samples_leaf'])
        best_hyperparameters['model']['min_samples_split'] = int(best_hyperparameters['model']['min_samples_split'])
        if best_hyperparameters['model']['type'] == "RF":
            global n
            best_hyperparameters['model']['max_features'] = int(best_hyperparameters['model']['max_features_fraction']*(n - 1) + 1)
        if best_hyperparameters['model']['max_depth']:
            best_hyperparameters['model']['max_depth'] = int(best_hyperparameters['model']['max_depth'])
    if best_hyperparameters['model']['type'] == "ADB":
        best_hyperparameters['model']['n_estimators'] = int(best_hyperparameters['model']['n_estimators'])
    if best_hyperparameters['model']['type'] == "SVM":
        best_hyperparameters['model']['degree'] = int(best_hyperparameters['model']['degree'])
    if best_hyperparameters['model']['type'] == "KNN":
        best_hyperparameters['model']['n_neighbors'] = int(best_hyperparameters['model']['n_neighbors'])
        best_hyperparameters['model']['p'] = int(best_hyperparameters['model']['p'])

    return best_hyperparameters


if __name__ == '__main__':
    max_high_cutoff = 50

    eeg_data = read_data(show_plots=False)

    n_iterations = 30
    n_folds = 5
    random_split = False

    best_hyperparameters = bayesian_search(n_iterations, n_folds, random_split)
    # best_hyperparameters = random_search(n_iterations, n_folds, random_split)

    epochs_train, labels_train, epochs_test, labels_test, _, unmixing_matrices = epoch_data(eeg_data, best_hyperparameters['bin_size'],
                                                                         best_hyperparameters['overlap'],
                                                                         best_hyperparameters['low_cutoff'],
                                                                         best_hyperparameters['high_cutoff'],
                                                                         best_hyperparameters['filter_order'],
                                                                         best_hyperparameters['ica_settings']['ica'],
                                                                         best_hyperparameters['ica_settings']['ica_pre_filter'])

    print(epochs_train.shape)
    print(best_hyperparameters)
    if best_hyperparameters['ica_settings']['ica']:
        print(unmixing_matrices[0])
        print(unmixing_matrices[1])
        print(unmixing_matrices[2])

    feature_extractor, scaler, clf, pc, _ = train_model(epochs_train, labels_train, epochs_test, labels_test, final=True,
                                                        certainty_threshold=best_hyperparameters['certainty_threshold'],
                                                        model=best_hyperparameters['model'], n_filters=best_hyperparameters['processing']['n_filters'],
                                                        resample_flag=True, processing=best_hyperparameters['processing']['type'],
                                                        features=best_hyperparameters['features'], pca=best_hyperparameters['pca'],
                                                        channels=best_hyperparameters['processing']['selected_channels'])

    filename = "eeg_feature_extractor_{}.pkl".format(time.time())
    pickle.dump(feature_extractor, open(filename, 'wb'))

    filename = "eeg_pca_{}.pkl".format(time.time())
    pickle.dump(pc, open(filename, 'wb'))

    filename = "eeg_ica_{}.pkl".format(time.time())
    pickle.dump(unmixing_matrices, open(filename, 'wb'))

    filename = "eeg_scaler_{}.pkl".format(time.time())
    pickle.dump(scaler, open(filename, 'wb'))

    if best_hyperparameters['model']['type'] != "NN":
        filename = "eeg_model_{}.pkl".format(time.time())
        pickle.dump(clf, open(filename, 'wb'))
    else:
        filename = "eeg_model_{}.h5".format(time.time())
        clf.save(filename)

    if 'n_filters_fraction' in best_hyperparameters['processing'].keys():
        del best_hyperparameters['processing']['n_filters_fraction']
    if 'high_cutoff_fraction' in best_hyperparameters.keys():
        del best_hyperparameters['high_cutoff_fraction']
    if 'max_features_fraction' in best_hyperparameters['model'].keys():
        del best_hyperparameters['model']['max_features_fraction']
    if 'n_folds' in best_hyperparameters.keys():
        del best_hyperparameters['n_folds']
    if 'random_split' in best_hyperparameters.keys():
        del best_hyperparameters['random_split']
    with open('eeg_best_hyperparameters_{}.json'.format(time.time()), 'w', encoding='utf-8') as outfile:
        json.dump(best_hyperparameters, outfile, ensure_ascii=False, indent=2)
