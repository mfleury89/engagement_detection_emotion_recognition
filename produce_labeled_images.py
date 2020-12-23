import os
import glob
import json
import cv2
import numpy as np
import pandas as pd

from datetime import datetime

pathfile = os.getcwd() + "\\path.txt"
cascadefile = os.getcwd() + "\\haarcascade_frontalface_alt.xml"
f = open(pathfile, "r")
lines = f.readlines()
f.close()

path = lines[0]
os.chdir(path)

extension = 'mp4'
files = glob.glob('*.{}'.format(extension))

with open('global_settings.json') as inputfile:
    settings = json.load(inputfile)
fps = settings['fps']
size = settings['face_box_size']
emotion_classes = settings['emotion_classes']
time_gaps = settings['time_gaps']


def time_in_seconds(dt):
    return dt.hour*3600 + dt.minute*60 + dt.second


def load_video(filename):
    classifier = cv2.CascadeClassifier(cascadefile)
    vid = cv2.VideoCapture(filename)
    return vid, classifier


for c in emotion_classes:
    if not os.path.exists("./Faces/{}".format(c)):
        os.makedirs("./Faces/{}".format(c))

periods = ["resting_before", "intervention", "resting_after"]

f = "video_labels_{}.csv".format(0)
df = pd.read_csv(f, header=None, names=['start_time', 'end_time', 'label', 'evaluation_tag'])
for i in range(1, len(periods)):
    f = "video_labels_{}.csv".format(i)
    df_temp = pd.read_csv(f, header=None, names=['start_time', 'end_time', 'label', 'evaluation_tag'])
    df = pd.concat([df, df_temp], ignore_index=True)

f = open("obs_timestamps.txt", "r")
lines = f.readlines()
f.close()

obs_start_time = time_in_seconds(datetime.strptime(lines[0], '%H:%M:%S\n'))
morae_start_time = time_in_seconds(datetime.strptime(lines[1], '%H:%M:%S\n'))

obs_end_time = time_in_seconds(datetime.strptime(lines[-1], '%H:%M:%S'))
morae_end_time = time_in_seconds(datetime.strptime(lines[-2], '%H:%M:%S\n'))

if __name__ == '__main__':
    for file in files:
        i = 0
        j = 0
        k = 0
        [video, face_classifier] = load_video(file)

        period = pd.read_csv("period_start_end_times.csv", header=None, names=['start', 'end'])

        for j in range(0, len(period)):
            start_time = period['start'].iloc[j]
            end_time = period['end'].iloc[j]

            gaps = []
            for time_gap in time_gaps:  # finding previous gaps in video to correct number of frames to skip ahead
                if start_time > time_in_seconds(datetime.strptime(time_gap['time'],
                                                                  '%H:%M:%S')):  # (a gap implies less frames need to be skipped to reach a certain testing time)
                    gaps.append(time_in_seconds(datetime.strptime(time_gap['gap'], '%H:%M:%S')))

            frame = np.floor(start_time - obs_start_time) * fps
            for gap in gaps:
                frame -= gap * fps
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)

            time = start_time

            while time <= end_time:
                print(time)
                (rval, im) = video.read()

                # im = cv2.flip(im, 1, 0)  # Flip to act as a mirror

                try:
                    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))  # Resize the image to speed up detection
                except AttributeError:
                    break

                i0 = np.where(df['start_time'] <= time)[0]
                i1 = np.where(df['end_time'] > time)[0]
                indices = np.intersect1d(i0, i1)

                labels = []
                tags = []
                for index in indices:
                    labels.append(df['label'].iloc[index])
                    tags.append(df['evaluation_tag'].iloc[index])

                if morae_start_time <= time <= morae_end_time and np.size(indices) > 0 and ("positive emotion" or
                                                                                            "negative emotion" or
                                                                                            "visual regard" in labels):

                    faces = face_classifier.detectMultiScale(mini)  # detect MultiScale / faces

                    for f in faces:  # Draw rectangles around each face
                        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 4)

                        sub_face = im[y:y + h, x:x + w]  # Save just the rectangle faces in SubRecFaces

                        if tags != []:
                            tag = tags[0]   # if manual coding has been performed correctly (keeping whole behavior time ranges within a given evaluation region), then tags should be identical here
                            if np.size(np.unique(np.array(tags))) > 1:
                                print("Warning: overlapping behavior time ranges are coded as being part of different evaluation regions.  Consider recoding to improve estimates of generalization.")

                        if "positive emotion" in labels:   # NOTE: This set of if blocks must be hardcoded to capture emotional meaning for a given set of behaviors
                            i += 1
                            cv2.imwrite("./Faces/Happy/happy{}_{}.jpg".format(i, tag), sub_face)
                        if "negative emotion" in labels:
                            j += 1
                            cv2.imwrite("./Faces/Sad/sad{}_{}.jpg".format(j, tag), sub_face)
                        if "visual regard" in labels and "positive emotion" not in labels and "negative emotion" not in labels:
                            k += 1
                            cv2.imwrite("./Faces/Neutral/neutral{}_{}.jpg".format(k, tag), sub_face)
                        # if not labels:
                        #     k += 1
                        #     cv2.imwrite("./Faces/Neutral/neutral{}_{}.jpg".format(k, tag), sub_face)

                # cv2.imshow('Capture', im)
                # key = cv2.waitKey(10)
                # # if Esc key is pressed then break out of the loop
                # if key == 27:  # The Esc key
                #     break

                # print(video.get(cv2.CAP_PROP_POS_MSEC))
                for time_gap in time_gaps:
                    if np.floor(time) == time_in_seconds(datetime.strptime(time_gap['time'], '%H:%M:%S')):
                        time = np.floor(time)
                        time += time_in_seconds(datetime.strptime(time_gap['gap'], '%H:%M:%S'))
                        continue

                time += 1/fps
