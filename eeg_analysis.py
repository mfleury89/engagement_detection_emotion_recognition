import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, iirnotch
import os
import time
import glob
import pickle
import json

import scipy.interpolate
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from math import inf

pathfile = os.getcwd() + "\\all_paths.txt"
f = open(pathfile, "r")
lines = f.read().splitlines()
f.close()

paths = []
for line in lines:
    paths.append(line)

subjects = []  # grabbing subject ID's
for path in paths:
    subjects.append(path[-4:])


# Note: this is really the least polished and cleaned-up script of the set of scripts, so plots don't necessarily look nice, etc.
# Plots are all within-subject and include: across-period across-label, across-period within-label, within-period across-label, and within-period within-label;
# however the most informative slice of the data here is the across-period within-label one.


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


def read_data(path):
    extension = 'csv'
    os.chdir(path)

    files = glob.glob('*.{}'.format(extension))
    eeg_data = {}
    for file in files:
        if "OpenBCI_labeled" in file:
            key = file[-5]
            eeg_data[key] = pd.read_csv(file, header=0, delimiter=',', engine='c')

    return eeg_data


def preprocess_data(eeg_data, unmixing_matrices, low_cutoff=1, high_cutoff=50, filter_order=5, ica=True, ica_pre_filter=None):
    for key, eeg_frame in eeg_data.items():
        for channel in channels:
            pad = 20  # padding signal with mean prior to bandpass filtering
            padded_column = np.pad(eeg_frame[channel], (pad, 0), mode='mean')

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
                    # filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                    #                                          filter_order)
                    filtered_column = padded_column
                elif ica_pre_filter == "bandpass":
                    filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                             filter_order)
                elif ica_pre_filter == "highpass":
                    filtered_column = butter_highpass_filter(padded_column, low_cutoff, sampling_rate, filter_order)
            else:
                filtered_column = butter_bandpass_filter(padded_column, low_cutoff, high_cutoff, sampling_rate,
                                                         filter_order)
            eeg_frame[channel] = filtered_column[pad:]
            eeg_frame[channel].replace([np.inf, -np.inf, inf, -inf], np.nan)  # replacing NaN's and inf in signal with signal mean
            mean = np.mean(eeg_frame[channel][eeg_frame[channel] != np.nan])
            eeg_frame[channel].fillna(mean)

        # print("Period {}".format(key))

        if ica:
            sources = np.dot(unmixing_matrices[int(key)], eeg_frame[non_railed_channels].T)
            eeg_frame.loc[:, non_railed_channels] = sources.T

        eeg_data[key] = eeg_frame

    return eeg_data


def average_plot(c, eeg_data, sampling_rate, non_railed_channels, channels, plot_title, save_name):

    N = 300  # number of points for interpolation
    xy_center = [2, 2]  # center of the plot
    radius = 2  # radius

    if subjects[k] == 'TD03':  # fixing channel labeling to account for wiring error with TD03; remove in general
        eeg_data.rename(columns={"ch4": "ch8", "ch5": "ch4", "ch6": "ch5", "ch7": "ch6", "ch8": "ch7"}, inplace=True)

    means = np.mean(eeg_data[channels], axis=0)

    x, y = [], []
    for i in c:
        x.append(i[0])
        y.append(i[1])

    z = means

    xi = numpy.linspace(-2, 6, N)
    yi = numpy.linspace(-2, 6, N)
    zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = numpy.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
            if (r - dr / 2) > radius:
                zi[j, i] = "nan"

    # make figure
    # figure_size = (14, 8)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figure_size)

    figure_size = (14, 8)
    fig = plt.figure(figsize=figure_size)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)

    # set aspect = 1 to make it a circle
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    # use different number of levels for the fill and the lines
    if subjects[k] == 'TD04':  # fixing scaling for better visualization with artifacts of TD04's data; remove in general
        CS = ax.contourf(xi, yi, zi, 60, vmin=-220000, vmax=30000, norm=colors.SymLogNorm(linthresh=50000.0, linscale=1.0), cmap=plt.cm.jet, zorder=1)
    else:
        CS = ax.contourf(xi, yi, zi, 60, cmap=plt.cm.jet, zorder=1)
    ax.contour(xi, yi, zi, 15, colors="grey", zorder=2)

    # make a color bar
    cbar = fig.colorbar(CS, ax=ax, orientation='horizontal')

    # add the data points
    # I guess there are no data points outside the head...
    ax.scatter(x, y, marker='o', c='k', s=15, zorder=3)

    # draw a circle
    # change the linewidth to hide the
    circle = matplotlib.patches.Circle(xy=xy_center, radius=radius, edgecolor="k", facecolor="none")
    ax.add_patch(circle)

    # make the axis invisible
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy=[0, 2], width=0.5, height=1.0, angle=0, edgecolor="k", facecolor="w",
                                        zorder=0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy=[4, 2], width=0.5, height=1.0, angle=0, edgecolor="k", facecolor="w",
                                        zorder=0)
    ax.add_patch(circle)
    # add a nose
    xy = [[1.5, 3], [2, 4.5], [2.5, 3]]
    polygon = matplotlib.patches.Polygon(xy=xy, edgecolor="k", facecolor="w", zorder=0)
    ax.add_patch(polygon)

    # set axes limits
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title('Topographical Map (microvolts)')

    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.plot(np.mean(eeg_data[non_railed_channels], axis=1), 'g-')
    #
    # ax2.set_xlabel('milliseconds')
    # ax2.set_ylabel('microvolts')
    #
    # ax3 = fig.add_subplot(1, 3, 3)
    #
    # N = np.size(np.mean(eeg_data[non_railed_channels], axis=1))
    # yf = fft(np.mean(eeg_data[non_railed_channels], axis=1))
    # Yf = np.power(np.abs(yf[:N // 2]), 2)
    # scale_factor = sampling_rate / (2 * np.size(Yf))
    # xf = np.linspace(0, np.size(Yf), np.size(Yf))
    # xf = scale_factor * xf
    #
    # ax3.plot(xf, np.log(Yf), 'g-')
    #
    # ax3.set_xlabel('Frequency (Hz)')
    # ax3.set_ylabel('ln(microvolts^2)')

    plt.savefig("EEG_plots\\" + save_name + ".png", dpi=fig.dpi, figsize=figure_size)


def component_plot(c, eeg_data, mixing_matrix, sampling_rate, non_railed_channels, plot_title, save_name):

    N = 300  # number of points for interpolation
    xy_center = [2, 2]  # center of the plot
    radius = 2  # radius

    x, y = [], []
    for i in c:
        x.append(i[0])
        y.append(i[1])

    figure_size = (14, 8)
    fig = plt.figure(figsize=figure_size)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)

    for n, channel in enumerate(non_railed_channels):
        eeg_projection_df = eeg_data.copy()

        component = eeg_projection_df[non_railed_channels].iloc[:, n]
        projection = np.outer(mixing_matrix[:, n], component.T).T
        eeg_projection_df[non_railed_channels] = projection

        if subjects[k] == 'TD03':  # fixing channel labeling to account for wiring error with TD03; remove in general
            eeg_projection_df.rename(columns={"ch4": "ch8", "ch5": "ch4", "ch6": "ch5", "ch7": "ch6", "ch8": "ch7"}, inplace=True)

        z = np.mean(eeg_projection_df[channels], axis=0)

        xi = numpy.linspace(-2, 6, N)
        yi = numpy.linspace(-2, 6, N)
        zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

        # set points > radius to not-a-number. They will not be plotted.
        # the dr/2 makes the edges a bit smoother
        dr = xi[1] - xi[0]
        for i in range(N):
            for j in range(N):
                r = numpy.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
                if (r - dr / 2) > radius:
                    zi[j, i] = "nan"

        # set aspect = 1 to make it a circle
        ax = fig.add_subplot(4, 4, n + 1, aspect=1)

        # use different number of levels for the fill and the lines
        CS = ax.contourf(xi, yi, zi, 60, cmap=plt.cm.jet, zorder=1)
        ax.contour(xi, yi, zi, 15, colors="grey", zorder=2)

        # make a color bar
        cbar = fig.colorbar(CS, ax=ax, orientation='vertical')

        # add the data points
        # I guess there are no data points outside the head...
        ax.scatter(x, y, marker='o', c='k', s=15, zorder=3)

        # draw a circle
        # change the linewidth to hide the
        circle = matplotlib.patches.Circle(xy=xy_center, radius=radius, edgecolor="k", facecolor="none")
        ax.add_patch(circle)

        # make the axis invisible
        for loc, spine in ax.spines.items():
            spine.set_linewidth(0)

        # remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add some body parts. Hide unwanted parts by setting the zorder low
        # add two ears
        circle = matplotlib.patches.Ellipse(xy=[0, 2], width=0.5, height=1.0, angle=0, edgecolor="k", facecolor="w",
                                            zorder=0)
        ax.add_patch(circle)
        circle = matplotlib.patches.Ellipse(xy=[4, 2], width=0.5, height=1.0, angle=0, edgecolor="k", facecolor="w",
                                            zorder=0)
        ax.add_patch(circle)
        # add a nose
        xy = [[1.5, 3], [2, 4.5], [2.5, 3]]
        polygon = matplotlib.patches.Polygon(xy=xy, edgecolor="k", facecolor="w", zorder=0)
        ax.add_patch(polygon)

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        # ax.set_title('Topographical Map (microvolts)')

    plt.savefig("EEG_plots\\" + save_name + "_topographical_projections.png", dpi=fig.dpi, figsize=figure_size)

    figure_size = (14, 8)
    fig = plt.figure(figsize=figure_size)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)

    for i, channel in enumerate(non_railed_channels):
        ax2 = fig.add_subplot(4, 4, i + 1)
        ax2.plot(eeg_data[channel], 'g-')
        ax2.set_xlabel('milliseconds')
        ax2.set_ylabel('microvolts')

    plt.savefig("EEG_plots\\" + save_name + "_time_domain.png", dpi=fig.dpi, figsize=figure_size)

    figure_size = (14, 8)
    fig = plt.figure(figsize=figure_size)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)

    for i, channel in enumerate(non_railed_channels):
        ax3 = fig.add_subplot(4, 4, i + 1)

        N = np.size(eeg_data[channel])
        yf = fft(eeg_data[channel])
        Yf = np.power(np.abs(yf[:N // 2]), 2)
        scale_factor = sampling_rate / (2 * np.size(Yf))
        xf = np.linspace(0, np.size(Yf), np.size(Yf))
        xf = scale_factor * xf

        ax3.plot(xf, np.log(Yf), 'g-')

        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('ln(microvolts^2)')

    plt.savefig("EEG_plots\\" + save_name + "_frequency_domain.png", dpi=fig.dpi, figsize=figure_size)


if __name__ == '__main__':
    for k, path in enumerate(paths[:-1]):
        if k < 4:
            continue
        os.chdir(path)

        if not os.path.exists("EEG_plots"):
            os.makedirs("EEG_plots")

        eeg_data = read_data(path)

        extension = 'pkl'
        files = glob.glob('*.{}'.format(extension))
        eeg_clf_file = ""
        csp_file = ""
        scaler_file = ""
        pca_file = ""
        ica_file = ""
        unmixing_matrices = []
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

        extension = 'csv'
        files = glob.glob('*.{}'.format(extension))
        files_eeg = []
        for file in files:
            if "OpenBCI_labeled" in file:
                files_eeg.append(file)

        with open('global_settings.json') as inputfile:
            settings = json.load(inputfile)
        sampling_rate = settings['sampling_rate']
        eeg_classes = settings['eeg_classes']
        not_engaged_index = eeg_classes.index("not_engaged")
        engaged_index = eeg_classes.index("engaged")
        channels = settings['channels']
        non_railed_channels = settings['non_railed_channels']
        time_gaps = settings['time_gaps']

        extension = 'json'
        files = glob.glob('*.{}'.format(extension))
        eeg_hp_file = ""
        cnn_settings_file = ""
        for file in files:
            if 'eeg_best_hyperparameters' in file:
                eeg_hp_file = file

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
        low_cutoff = settings['low_cutoff']
        high_cutoff = settings['high_cutoff']
        filter_order = settings['filter_order']
        selected_channels = settings['processing']['selected_channels']
        selected_channels = ['ch' + str(selected_channel + 1) for selected_channel in selected_channels]
        railed_channels = list(set(channels) - set(non_railed_channels))

        if csp:
            filters = csp.filters_
            patterns = csp.patterns_

            print(filters)
            print(patterns)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.scatter(filters)
            # plt.show()

        # close old plots
        plt.close("all")

        # coord = [[1, 4], [3, 4], [1, 3], [3, 3], [2, 3], [1, 2], [3, 2],
        #          [2, 2], [1, 1], [3, 1], [2, 1], [1, 0], [3, 0],
        #          [0, 3], [4, 3], [0, 2], [4, 2], [0, 1], [4, 1]]

        coord = [[1.5, 4], [2.5, 4], [1, 2], [3, 2],
                 [0.2, 1], [3.8, 1], [1.5, 0], [2.5, 0],
                 [0.2, 3], [3.8, 3], [1.25, 2.75], [2.75, 2.75],
                 [0, 2], [4, 2], [1.25, 1.25], [2.75, 1.25]]  # ch1-16

        eeg_data = preprocess_data(eeg_data, unmixing_matrices, low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                   filter_order=filter_order, ica=ica, ica_pre_filter=ica_pre_filter)
        eeg_data_ica = eeg_data.copy()

        if ica:
            mixing_matrices = {}
            for period in eeg_data.keys():
                mixing_matrix = np.linalg.pinv(unmixing_matrices[int(period)])
                mixing_matrices[period] = mixing_matrix
                indices = [non_railed_channels.index(c) for c in selected_channels]
                eeg_data[period][non_railed_channels] = np.dot(mixing_matrix[:, indices],
                                                               eeg_data[period][selected_channels].T).T  # projecting selected sources back to channels

        all_eeg = pd.DataFrame()
        for period in eeg_data.keys():
            all_eeg = pd.concat((all_eeg, eeg_data[period]), axis=0)

        m = np.mean(all_eeg[non_railed_channels], axis=0)
        s = np.std(all_eeg[non_railed_channels], axis=0)

        # eeg_data[period][channels] -= m
        # eeg_data[period][channels] /= s

        # columns = eeg_data[period].columns
        # indices = [eeg_data[period].columns.get_loc(c) - 2 for c in columns if c in non_railed_channels]
        # means[period] = mean[non_railed_channels]
        # c = [coord[i] for i in indices]
        c = coord

        for period in eeg_data.keys():
            eeg_data[period][railed_channels] = np.mean(m)
            eeg_data_ica[period][railed_channels] = np.mean(m)

        average_plot(c, all_eeg, sampling_rate, non_railed_channels, channels,
                     '{}'.format(subjects[k]),
                     'eeg_average_plots_{}_all_periods_all_labels_{}'.format(subjects[k], time.time()))

        for label in all_eeg['label'].unique():
            eeg_data_sub_df = all_eeg[all_eeg['label'] == label]
            eeg_label = eeg_classes[int(label)]

            # eeg_data[period][channels] -= m
            # eeg_data[period][channels] /= s

            # columns = eeg_data[period].columns
            # indices = [eeg_data[period].columns.get_loc(c) - 2 for c in columns if c in non_railed_channels]
            # means[period] = mean[non_railed_channels]
            # c = [coord[i] for i in indices]
            c = coord

            average_plot(c, eeg_data_sub_df, sampling_rate, non_railed_channels, channels,
                         '{}, {}'.format(subjects[k], eeg_label),
                         'eeg_average_plots_{}_all_periods_{}_{}'.format(subjects[k], eeg_label, time.time()))

        for period in eeg_data.keys():
            eeg_data_df = eeg_data[period]
            eeg_data_df_ica = eeg_data_ica[period]

            # eeg_data[period][channels] -= m
            # eeg_data[period][channels] /= s

            # columns = eeg_data[period].columns
            # indices = [eeg_data[period].columns.get_loc(c) - 2 for c in columns if c in non_railed_channels]
            # means[period] = mean[non_railed_channels]
            # c = [coord[i] for i in indices]
            c = coord

            # average_plot(c, eeg_data_df, sampling_rate, non_railed_channels, channels,
            #              '{} Channels, Period {}'.format(subjects[k], period),
            #              'eeg_average_plots_{}_period_{}_all_labels_{}'.format(subjects[k], period, time.time()))

            if ica:
                c = coord
                # component_plot(c, eeg_data_df_ica, mixing_matrices[period], sampling_rate, non_railed_channels,
                #                '{} Components, Period {}'.format(subjects[k], period),
                #                'eeg_component_plots_{}_period_{}_all_labels_{}'.format(subjects[k], period, time.time()))

            for label in eeg_data[period]['label'].unique():
                eeg_data_sub_df = eeg_data[period][eeg_data[period]['label'] == label]
                eeg_data_sub_df_ica = eeg_data_ica[period][eeg_data_ica[period]['label'] == label]
                eeg_label = eeg_classes[int(label)]

                # eeg_data[period][channels] -= m
                # eeg_data[period][channels] /= s

                # columns = eeg_data[period].columns
                # indices = [eeg_data[period].columns.get_loc(c) - 2 for c in columns if c in non_railed_channels]
                # means[period] = mean[non_railed_channels]
                # c = [coord[i] for i in indices]
                c = coord

                # average_plot(c, eeg_data_sub_df, sampling_rate, non_railed_channels, channels,
                #              '{} Channels, Period {} {}'.format(subjects[k], period, eeg_label),
                #              'eeg_average_plots_{}_period_{}_{}_{}'.format(subjects[k], period, eeg_label, time.time()))

                if ica:
                    c = coord
                    # component_plot(c, eeg_data_sub_df_ica, mixing_matrices[period], sampling_rate, non_railed_channels,
                    #                '{} Components, Period {} {}'.format(subjects[k], period, eeg_label),
                    #                'eeg_component_plots_{}_period_{}_{}_{}'.format(subjects[k], period, eeg_label,
                    #                                                                time.time()))

        # plt.show()
