import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import butter, lfilter, iirnotch
import cv2
import os
import glob
import pickle
import json
from math import inf

pathfile = os.getcwd() + "\\path.txt"
cascadefile = os.getcwd() + "\\haarcascade_frontalface_alt.xml"
f = open(pathfile, "r")
lines = f.readlines()
f.close()

path = lines[0]
os.chdir(path)

extension = 'pkl'
files = glob.glob('*.{}'.format(extension))
eeg_clf_file = ""
csp_file = ""
scaler_file = ""
pca_file = ""
ica_file = ""
for file in files:
    if "model" in file:
        eeg_clf_file = file
    if "feature_extractor" in file:
        csp_file = file
    if "scaler" in file:
        scaler_file = file
    if "pca" in file:
        pca_file = file
    if "ica" in file:
        ica_file = file
        unmixing_matrices = pickle.load(open(ica_file, 'rb'))
csp = pickle.load(open(csp_file, 'rb'))
scaler = pickle.load(open(scaler_file, 'rb'))
pca = pickle.load(open(pca_file, 'rb'))
eeg_classifier = pickle.load(open(eeg_clf_file, 'rb'))

extension = 'mp4'
files = glob.glob('*.{}'.format(extension))
file_video = files[0]

extension = 'csv'
files = glob.glob('*.{}'.format(extension))
files_eeg = []
for file in files:
    if "OpenBCI_labeled" in file:
        files_eeg.append(file)

with open('global_settings.json') as inputfile:
    settings = json.load(inputfile)
sampling_rate = settings['sampling_rate']
fps = settings['fps']
time_step = settings['time_step']
size = settings['face_box_size']
emotion_classes = settings['emotion_classes']
class_to_remove = settings['class_to_remove']
emotion_classes.remove(class_to_remove)
eeg_classes = settings['eeg_classes']
not_engaged_index = eeg_classes.index("not_engaged")
engaged_index = eeg_classes.index("engaged")
non_railed_channels = settings['non_railed_channels']
columns = settings['channels']
time_gaps = settings['time_gaps']

extension = 'json'
files = glob.glob('*.{}'.format(extension))
eeg_hp_file = ""
cnn_settings_file = ""
for file in files:
    if 'eeg_best_hyperparameters' in file:
        eeg_hp_file = file
    if 'cnn_settings' in file:
        cnn_settings_file = file

with open(eeg_hp_file) as inputfile:
    settings = json.load(inputfile)
model_type = settings['model']['type']
eeg_buffer_size = settings['bin_size']
# eeg_buffer_overlap = settings['overlap']
processing = settings['processing']['type']
features = settings['features']
pca_setting = settings['pca']
ica = settings['ica_settings']['ica']
ica_pre_filter = settings['ica_settings']['ica_pre_filter']
eeg_buffer_overlap = 0
low_cutoff = settings['low_cutoff']
high_cutoff = settings['high_cutoff']
filter_order = settings['filter_order']
eeg_threshold = settings['certainty_threshold']
channels = settings['processing']['selected_channels']

with open(cnn_settings_file) as inputfile:
    settings = json.load(inputfile)
model = settings['model']
IMAGE_RES = settings['image_res']
certainty_threshold = settings['certainty_threshold']
feature_mean = settings['feature_mean']
feature_std = settings['feature_std']

extension = 'h5'
files = glob.glob('*.{}'.format(extension))
cnn_file = ""
eeg_file = ""
for file in files:
    if "cnn" in file:
        cnn_file = file
    if "eeg" in file:
        eeg_file = file
cnn_model = tf.keras.models.load_model(cnn_file)
cnn_model.summary()
if eeg_file != "":
    eeg_classifier = tf.keras.models.load_model(eeg_file)
    eeg_classifier.summary()

preprocess = {'emopy': lambda image: ((tf.image.rgb_to_grayscale(image) / 255.0) - feature_mean)/feature_std,
              'vgg16': tf.keras.applications.vgg16.preprocess_input,
              'vgg19': tf.keras.applications.vgg19.preprocess_input,
              'vggface2': tf.keras.applications.resnet50.preprocess_input,
              'inception_resnet': tf.keras.applications.inception_resnet_v2.preprocess_input,
              'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input}


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


def load_eeg(filenames):
    df = pd.DataFrame()
    eeg_start_time = []
    eeg_end_time = []
    for i, file in enumerate(filenames):
        df_temp = pd.read_csv(file, header=0, delimiter=',', engine='c')
        for channel in columns:
            pad = 20  # padding signal with mean prior to bandpass filtering
            padded_column = np.pad(df_temp[channel], (pad, 0), mode='mean')

            f0 = 60.0  # Frequency to be removed from signal (Hz)
            Q = 100.0  # Quality factor
            # Design notch filter
            [b, a] = iirnotch(f0, Q, sampling_rate)
            padded_column = lfilter(b, a, padded_column)
            padded_column = np.pad(padded_column[pad:], (pad, 0), mode='mean')

            filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                     filter_order)

            if ica:
                if ica_pre_filter is None:
                    filtered_column = padded_column
                elif ica_pre_filter == "bandpass":
                    filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                             filter_order)
                elif ica_pre_filter == "highpass":
                    filtered_column = butter_highpass_filter(padded_column, low_cutoff, sampling_rate, filter_order)
            else:
                filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                         filter_order)
            df_temp[channel] = filtered_column[pad:]
            df_temp[channel].replace([np.inf, -np.inf, inf, -inf], np.nan)  # replacing NaN's and inf in signal after filtering with signal mean
            mean = np.mean(df_temp[channel].loc[df_temp[channel] != np.nan])
            df_temp[channel].fillna(mean)

        if ica:
            print(unmixing_matrices[i])
            print(df_temp[non_railed_channels].head())
            df_temp[non_railed_channels] = np.dot(unmixing_matrices[i], df_temp[non_railed_channels].T).T
            print(df_temp[non_railed_channels].head())

        df_temp['timestamp'] = df_temp['timestamp'] * 1000
        eeg_start_time.append(df_temp['timestamp'].iloc[0])
        eeg_end_time.append(df_temp['timestamp'].iloc[-1])
        df = pd.concat((df, df_temp), axis=0, ignore_index=True)
    return df, eeg_start_time, eeg_end_time


def load_video(filename):
    classifier = cv2.CascadeClassifier(cascadefile)
    vid = cv2.VideoCapture(filename)
    return vid, classifier


def time_in_milliseconds(dt):
    return 1000*(dt.hour*3600 + dt.minute*60 + dt.second)


def format_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.reshape(image, (1, image.shape[0], image.shape[1], 3))
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))  # / 255.0
    image = preprocess[model](image)
    return image


def extract_scores(buffer):
    pad = 20
    padded = np.pad(buffer, ((0, 0), (0, 0), (pad, 0)))

    filtered = butter_bandpass_filter(padded, 4, 8, sampling_rate, 3)
    theta = np.sum(np.square(np.array(filtered[:, :, pad:])), axis=-1)
    filtered = butter_bandpass_filter(padded, 7, 13, sampling_rate, 3)
    alpha = np.sum(np.square(np.array(filtered[:, :, pad:])), axis=-1)
    filtered = butter_bandpass_filter(padded, 13, 30, sampling_rate, 3)
    beta = np.sum(np.square(np.array(filtered[:, :, pad:])), axis=-1)

    score = beta / (alpha + theta)
    return score


def extract_powers(buffer):
    powers = np.sum(np.square(np.array(buffer)), axis=-1)/np.size(buffer, axis=-1)
    return powers


def extract_features(buffer):
    buffer = buffer[columns]
    buffer = np.array(buffer).T
    buffer = buffer.reshape((1, buffer.shape[0], buffer.shape[1]))
    buffer = buffer[:, channels, :]
    if processing == 'csp':
        if features == 'average_power':
            feature_vector = csp.transform(buffer)
        elif features == 'scores':
            buffer = csp.transform(buffer)
            feature_vector = extract_scores(buffer)
    else:
        if features == 'average_power':
            feature_vector = extract_powers(buffer)
        elif features == 'scores':
            feature_vector = extract_scores(buffer)

    feature_vector = np.nan_to_num(feature_vector, posinf=0, neginf=0, nan=0)
    feature_vector = scaler.transform(feature_vector)

    if pca_setting:
        feature_vector = pca.transform(feature_vector)

    return feature_vector


if __name__ == "__main__":
    f = open("obs_timestamps.txt", "r")
    lines = f.readlines()
    f.close()

    obs_start_time = time_in_milliseconds(datetime.strptime(lines[0], '%H:%M:%S\n'))
    morae_start_time = time_in_milliseconds(datetime.strptime(lines[1], '%H:%M:%S\n'))
    current_start_time = obs_start_time

    obs_end_time = time_in_milliseconds(datetime.strptime(lines[-1], '%H:%M:%S'))
    morae_end_time = time_in_milliseconds(datetime.strptime(lines[-2], '%H:%M:%S\n'))

    test = pd.read_csv("test_start_end_times.csv", header=None, names=['start_time', 'end_time'])
    period = pd.read_csv("period_start_end_times.csv", header=None, names=['start', 'end'])

    eeg, eeg_start_time, eeg_end_time = load_eeg(files_eeg)

    [video, face_classifier] = load_video(file_video)

    sub_dict = {'n_eeg': 0, 'n_engaged': 0, 'n_happy': 0, 'n_sad': 0, 'n_emotion': 0}
    period_rewards = {'0': sub_dict.copy(), '1': sub_dict.copy(), '2': sub_dict.copy()}

    for j in range(0, len(test)):
        # start_time = np.max((eeg_start_time, obs_start_time))
        start_time = 1000*test['start_time'].iloc[j]
        # end_time = np.min((eeg_end_time, obs_end_time))
        end_time = 1000*test['end_time'].iloc[j]

        gaps = []
        for time_gap in time_gaps:              # finding previous gaps in video to correct number of frames to skip ahead
            if start_time > time_in_milliseconds(datetime.strptime(time_gap['time'], '%H:%M:%S')):   # (a gap implies less frames need to be skipped to reach a certain testing time)
                gaps.append(time_in_milliseconds(datetime.strptime(time_gap['gap'], '%H:%M:%S'))/1000)

        frame = np.floor((start_time - obs_start_time)/(1000/fps))
        for gap in gaps:
            frame -= gap*fps
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)

        n_emotion_predictions = 0
        n_happy_predictions = 0
        n_sad_predictions = 0
        n_eeg_predictions = 0
        n_engaged_predictions = 0
        emotion_reward = 0
        eeg_buffer = pd.DataFrame()
        previous_end_time = start_time/1000

        for i in range(int(start_time), int(end_time)+1):
            # print(i)

            i1 = np.where(1000*period['start'] <= i)[0]
            i2 = np.where(1000*period['end'] >= i)[0]
            period_index = np.intersect1d(i1, i2)[0]
            period_start_time = period['start'].iloc[period_index]

            if (i - eeg_start_time[int(period_index)]) % (1000/sampling_rate) == 0 and (i - start_time) > 0:
                # print("EEG")
                # do eeg processing
                # add to buffer, if buffer is full then extract features and classify
                eeg_buffer = pd.concat((eeg_buffer, eeg[round(eeg['timestamp']) == i]), ignore_index=True, axis=0)
                if not eeg_buffer.empty:
                    if eeg_buffer['timestamp'].iloc[-1] - eeg_buffer['timestamp'].iloc[0] >= eeg_buffer_size:
                        eeg_channel_buffer = eeg_buffer.iloc[:-1]
                        n_eeg_predictions += 1
                        period_rewards[str(period_index)]['n_eeg'] += 1
                        # for channel in columns:
                        #     pad = 20  # padding signal with zeros prior to bandpass filtering
                        #     padded_column = np.pad(eeg_channel_buffer[channel], (pad, 0), mode='mean')
                        #     filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate, filter_order)
                        #     eeg_channel_buffer[channel] = filtered_column[pad:]
                        #     eeg_channel_buffer[channel].replace([np.inf, -np.inf, inf, -inf], np.nan)  # replacing NaN's and inf in signal after filtering with signal mean
                        #     mean = np.mean(eeg_channel_buffer[channel].loc[eeg_channel_buffer[channel] != np.nan])
                        #     eeg_channel_buffer[channel].fillna(mean)

                        feature_vector = extract_features(eeg_channel_buffer)
                        # predicted_state = eeg_classes[eeg_classifier.predict(feature_vector)[0]]
                        if model_type != "NN":
                            probs = eeg_classifier.predict_proba(feature_vector).flatten()
                            # print(probs)
                        else:
                            probs = eeg_classifier.predict(feature_vector)
                        if probs[engaged_index] > eeg_threshold:
                            predicted_state = eeg_classes[engaged_index]
                            n_engaged_predictions += 1
                            period_rewards[str(period_index)]['n_engaged'] += 1
                        else:
                            predicted_state = eeg_classes[not_engaged_index]
                        # print(predicted_state)

                        keep_indices = np.where(eeg_buffer['timestamp'] >= eeg_buffer['timestamp'].iloc[-1] -
                                                eeg_buffer_overlap*(eeg_buffer['timestamp'].iloc[-1] - eeg_buffer['timestamp'].iloc[0]))[0]
                        eeg_buffer = eeg_buffer.iloc[keep_indices[0]:]

            if ((i - current_start_time) >= 1000/fps) and (i - start_time) > 0:  # classify any faces as happy/neutral/sad
                current_start_time = i
                # print("IMAGE")
                (rval, im) = video.read()
                # im = cv2.flip(im, 1, 0)  # Flip to act as a mirror

                mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))
                # Resize the image to speed up detection

                faces = face_classifier.detectMultiScale(mini)  # detect MultiScale / faces

                for f in faces:  # Draw rectangles around each face
                    n_emotion_predictions += 1
                    period_rewards[str(period_index)]['n_emotion'] += 1
                    (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    sub_face = im[y:y + h, x:x + w]  # Save just the rectangle faces in SubRecFaces
                    img = format_image(sub_face)
                    prob = cnn_model.predict(img)
                    # print(prob)
                    if np.max(prob) <= certainty_threshold:
                        predicted_emotion = class_to_remove
                    else:
                        predicted_emotion = emotion_classes[np.argmax(prob, axis=1)[0]]
                    # print(predicted_emotion)
                    if predicted_emotion == "Happy":
                        n_happy_predictions += 1
                        period_rewards[str(period_index)]['n_happy'] += 1
                    if predicted_emotion == "Sad":
                        n_sad_predictions += 1
                        period_rewards[str(period_index)]['n_sad'] += 1

                # cv2.imshow('Capture', im)
                # key = cv2.waitKey(10)
                # # if Esc key is pressed then break out of the loop
                # if key == 27:  # The Esc key
                #     break

            if ((i - period_start_time*1000) % (time_step*1000) == 0 or i == end_time) and (i - start_time) > 0:  # calculate reward at each time step
                print("REWARD")
                if n_emotion_predictions != 0:
                    emotion_reward = (n_happy_predictions - n_sad_predictions)/n_emotion_predictions
                else:
                    emotion_reward = 0

                if n_eeg_predictions != 0:
                    eeg_reward = n_engaged_predictions/n_eeg_predictions
                else:
                    eeg_reward = 0

                # reward = np.max((emotion_reward + eeg_reward, 0))
                reward = emotion_reward + eeg_reward
                print(reward)

                if i == end_time:
                    start = previous_end_time
                else:
                    start = i/1000 - time_step
                f = open("reward_scores_{}.csv".format(period_index), "a")
                # save reward scores for comparison to clinical measures
                f.write(str(start) + "," + str(i/1000) + "," + str(reward) + "," + str(emotion_reward) +
                        "," + str(eeg_reward) + '\n')
                f.close()
                previous_end_time = i/1000

                n_emotion_predictions = 0
                n_happy_predictions = 0
                n_sad_predictions = 0
                n_eeg_predictions = 0
                n_engaged_predictions = 0

    final_df = pd.DataFrame(period_rewards, index=['n_engaged', 'n_eeg', 'n_happy', 'n_sad', 'n_emotion'])
    final_df.to_csv('total_reward.csv')
