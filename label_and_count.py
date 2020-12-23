import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

pathfile = os.getcwd() + "\\path.txt"
f = open(pathfile, "r")
lines = f.readlines()
f.close()

path = lines[0]
os.chdir(path)

with open('global_settings.json') as inputfile:
    settings = json.load(inputfile)
sampling_frequency = settings['sampling_rate']
second_adjust = settings['second_adjust']
time_step = settings['time_step']
channels = settings['channels']
behavior_list = settings['behaviors']
engagement_behaviors = settings['engagement_behaviors']
positive_behaviors = settings['positive_behaviors']
negative_behaviors = settings['negative_behaviors']


def time_in_seconds(dt):
    return dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1000000


def time_in_milliseconds(dt, hour_adjust=0, second_adjust=0):
    return (dt.hour + hour_adjust)*3600 + dt.minute*60 + (dt.second + second_adjust) + dt.microsecond/1000000


def calculate_score(behaviors, i, start_time):
    #score = (behaviors[0].sub_count + behaviors[1].sub_count - behaviors[3].sub_count)/(i - start_time)  # Note: this score is hardcoded for adult test, should be changed for study
    score = 0
    for behavior in behaviors:
        if behavior.name in engagement_behaviors or behavior.name in positive_behaviors:
        # if behavior.name in positive_behaviors:
            score += behavior.sub_count
        if behavior.name in negative_behaviors:
            score -= behavior.sub_count

    score /= (i - start_time)
    # score = np.max((score, 0))
    return score


class Behavior:
    def __init__(self, name):
        self.present = False
        self.current = False
        self.name = name
        self.count = 0
        self.sub_count = 0
        self.start_times = []
        self.end_times = []
        self.evaluation_tags = []


class TimeStep:
    def __init__(self, start, end, count, tag):
        self.start_time = start
        self.end_time = end
        self.count = count
        self.tag = tag


# import information from event_timestamps (manually coded using Morae)
behavior_columns = ['timestamp']
behavior_columns.extend(behavior_list)
behavior_columns.extend(['period', 'evaluation_tag'])
df = pd.read_excel("event_timestamps.xlsx", header=4, names=behavior_columns, skip_blank_lines=False)

extension = 'txt'
files2 = glob.glob('*.{}'.format(extension))
df2 = pd.DataFrame()
for f in files2:    # import OpenBCI data
    if "OpenBCI" in f:
        columns = ['index']
        columns.extend(channels)
        columns.extend(['ax', 'ay', 'az', 'timestamp', 'timestamp2'])

        df_eeg = pd.read_csv(f, skiprows=6, names=columns, delimiter=',', engine='c')

        eeg_start_time = time_in_milliseconds(datetime.strptime(df_eeg['timestamp'].iloc[0], ' %H:%M:%S.%f'), hour_adjust=1, second_adjust=second_adjust)

        df_eeg['index'] = [eeg_start_time + (1 / sampling_frequency) * i for i in df_eeg.index]   # assuming constant sampling rate
        df_eeg['timestamp'] = df_eeg['index']
        del df_eeg['index']
        columns = ['timestamp']
        columns.extend(channels)
        columns.extend(['ax', 'ay', 'az'])
        df_eeg = df_eeg[columns]
        df2 = pd.concat([df2, df_eeg], axis=0)

if __name__ == '__main__':
    test_start_times = []
    test_end_times = []
    period_start_times = []
    period_end_times = []
    for period in np.unique(df['period']):  # label and count for each experimental period
        df_current = df[df['period'] == period]

        behaviors = []
        for behavior in behavior_list:
            behaviors.append(Behavior(behavior))
        no_behavior = Behavior("none")

        start_time = time_in_milliseconds(df_current['timestamp'].iloc[0])
        start = start_time
        end_time = time_in_milliseconds(df_current['timestamp'].iloc[-1])

        period_start_times.append(start_time)
        period_end_times.append(end_time)

        time_step_count = 0
        time_step_scores = []
        times = [time_in_milliseconds(time) for time in df_current['timestamp'].values]
        time_index = 0

        current_tag = 0
        start_time_milli = int(start_time*1000)
        end_time_milli = int(end_time*1000)
        for i in range(start_time_milli, end_time_milli + 1):
            if i/1000 - start_time >= time_step or i/1000 == end_time:    # calculate overall engagement score for given time step,
                time_step_count += 1                                      # and tally up behavior counts for period engagement scores
                total_count = 0
                score = calculate_score(behaviors, i, start_time*1000)
                for behavior in behaviors:
                    behavior.sub_count = 0
                    if behavior.present:
                        behavior.count += 1
                        total_count += 1
                    behavior.present = False
                    behavior.current = False
                time_step_scores.append(TimeStep(start_time, i/1000, score, current_tag))
                start_time = i/1000

            if i/1000 in times:      # find behavior time ranges and evaluation tags, and log test start/end times
                if df_current['evaluation_tag'].iloc[time_index] == 2 and current_tag != 2:
                    test_start_times.append(start_time)
                if (df_current['evaluation_tag'].iloc[time_index] != 2 or i/1000 == end_time) and current_tag == 2:
                    test_end_times.append(i/1000)
                current_tag = df_current['evaluation_tag'].iloc[time_index]

                for behavior in behaviors:
                    if df_current[behavior.name].iloc[time_index] == 1 or df_current[behavior.name].iloc[time_index] == 0:
                        behavior.present = True
                        behavior.current = True
                        if df_current[behavior.name].iloc[time_index] == 1:
                            behavior.start_times.append(time_in_seconds(df_current['timestamp'].iloc[time_index]))
                            behavior.evaluation_tags.append(df_current['evaluation_tag'].iloc[time_index])
                    elif df_current[behavior.name].iloc[time_index] == -1:
                        behavior.end_times.append(time_in_seconds(df_current['timestamp'].iloc[time_index]))
                        behavior.current = False
                time_index += 1

            for behavior in behaviors:  # increment behavior count for current time step (not whole period)
                if behavior.current:
                    behavior.sub_count += 1

        file = open("time_step_scores_{}.csv".format(period), "w+")  # save overall engagement score for each time step
        for i, score in enumerate(time_step_scores):
            print("Time Step {}: {}, {}, {}".format(i, score.start_time, score.end_time, score.count, score.tag))
            file.write(str(score.start_time) + ", " + str(score.end_time) + ", " + str(score.count) + ", " + str(score.tag) + '\n')
        file.close()

        df2_current = df2[(df2['timestamp'] >= start) & (df2['timestamp'] <= end_time)]  # OpenBCI data within period

        all_ranges = []
        engagement_labels = np.zeros((len(df2_current), 1))
        file = open("engagement_scores_{}.csv".format(period), "w+")
        for behavior in behaviors:
            print(behavior.name + ": " + str(behavior.count/time_step_count))
            file.write(behavior.name + ", " + str(behavior.count/time_step_count) + '\n')   # save engagement scores for each behavior in the period

            name_array = [behavior.name for i in range(0, len(behavior.start_times))]
            evaluation_tag_array = [behavior.evaluation_tags[i] for i in range(0, len(behavior.start_times))]
            ranges = list(zip(behavior.start_times, behavior.end_times, name_array, evaluation_tag_array))
            all_ranges.extend(ranges)   # set up time ranges for video labeling
            if behavior.name in engagement_behaviors:
                for r in ranges:    # set engagement labels for OpenBCI data, honing in on engagement behaviors of interest
                    i0 = np.where(df2_current['timestamp'] >= r[0])[0]
                    i1 = np.where(df2_current['timestamp'] <= r[1])[0]
                    indices = np.intersect1d(i0, i1)
                    engagement_labels[indices] = 1
        total_engagement_score = 0
        for ts in time_step_scores:
            if ts.count > 0:
                total_engagement_score += 1
        print("total: " + str(total_engagement_score / time_step_count))
        file.write("total, " + str(total_engagement_score / time_step_count) + '\n')  # save total engagement score for the period
        file.close()

        file = open("video_labels_{}.csv".format(period), "w+")     # save behavior time ranges and labels for image labeling
        for r in all_ranges:
            file.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]) + '\n')
        file.close()

        evaluation_labels = np.zeros((len(df2_current), 1))     # translate evaluation tags to OpenBCI evaluation labels
        evaluation_tags = df_current['evaluation_tag']
        for i in range(0, len(evaluation_tags)-1):
            i0 = np.where(df2_current['timestamp'] >= time_in_seconds(df_current['timestamp'].iloc[i]))[0]
            i1 = np.where(df2_current['timestamp'] < time_in_seconds(df_current['timestamp'].iloc[i+1]))[0]
            indices = np.intersect1d(i0, i1)
            evaluation_labels[indices] = evaluation_tags.iloc[i]
        indices = np.where(df2_current['timestamp'] >= time_in_seconds(df_current['timestamp'].iloc[len(evaluation_tags)-1]))[0]
        evaluation_labels[indices] = evaluation_tags.iloc[-1]

        df2_current['label'] = engagement_labels        # save labeled OpenBCI data
        df2_current['evaluation_tag'] = evaluation_labels
        columns = df2_current.columns
        df2_current.to_csv("OpenBCI_labeled_{}.csv".format(period), header=columns)

    file = open("test_start_end_times.csv", "w+")     # save final test start and end times
    for i in range(0, len(test_start_times)):
        file.write(str(test_start_times[i]) + ", " + str(test_end_times[i]) + '\n')
    file.close()

    file = open("period_start_end_times.csv", "w+")     # save period start and end times
    for i in range(0, len(period_start_times)):
        file.write(str(period_start_times[i]) + ", " + str(period_end_times[i]) + '\n')
    file.close()
