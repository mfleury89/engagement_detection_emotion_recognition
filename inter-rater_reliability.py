import os
import time
import itertools
import json
import krippendorff

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import choice

a_min = 0.8

pathfile = os.getcwd() + "\\test_paths.txt"
f = open(pathfile, "r")
lines = f.read().splitlines()
f.close()

paths = []
for line in lines:
    paths.append(line)

n_periods = 3
n_raters = 2
n_subsamples = 10000

# types = ['coding']
types = ['unitizing']


def plot_distribution(plot_title, save_name, alpha_dict):

    alpha = alpha_dict['alpha']
    distribution = alpha_dict['distribution']

    n_bins = 100

    a = 0.05
    h = np.histogram(distribution, bins=n_bins, density=False)
    i = np.where(h[1] < alpha)[0]
    height = h[0][i[-1]]

    sorted_dist = np.sort(distribution)
    alpha_low = sorted_dist[int(np.ceil(len(sorted_dist)*a/2) - 1)]
    alpha_high = sorted_dist[int(np.ceil(len(sorted_dist)*(1 - a/2)) - 1)]
    ci = (alpha_low, alpha_high)
    print(ci)

    q = np.size(np.where(np.array(distribution) < a_min)[0])/len(distribution)
    print(q)

    figure_size = (14, 8)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)
    ax.hist(distribution, bins=n_bins, density=False)
    # ax.vlines(a_min, label='minimum alpha cutoff', ymin=0, ymax=max(h[0]))
    ax.vlines(alpha, label='estimated alpha', ymin=0, ymax=height)
    ax.set_xlabel("Krippendorff's alpha")
    ax.set_ylabel('Frequency')
    ax.set_title('95% CI: ({0:.11f}, {1:.11f}), q = {2:.3f}'.format(ci[0], ci[1], q))
    ax.set_xlim(min(distribution), np.max((1, max(distribution))))
    # ax.set_ylim(0, 2000)
    plt.legend()
    # plt.yscale('log')

    plt.savefig('inter-rater reliability\\' + save_name + ".png", dpi=fig.dpi, figsize=figure_size)

    return ci, q


def bootstrap_coding(matrix):
    pairs = []  # generate all reliability value pairs
    coordinates = list(itertools.product(range(matrix.shape[0]), range(matrix.shape[1])))
    for c in coordinates:
        for k in coordinates:
            if c == k or c[1] != k[1] or matrix[c] == np.nan or matrix[k] == np.nan:  # only want distinct, well-defined pairs within a column
                continue
            pairs.append([c, k])

    alpha_distribution = []  # bootstrapping data to build alpha distribution
    for i in range(n_subsamples):
        all_coordinates = []  # sampling coordinates of pairs of values with replacement
        for j in range(len(pairs)):
            sample = choice(pairs)
            all_coordinates.append(sample[0])  # flattening pair coordinates to array of indices
            all_coordinates.append(sample[1])

        matrix_sample_parts = []  # stacking values by rater
        for k in range(n_raters):
            matrix_sample_parts.append([matrix[j] for j in all_coordinates if j[0] == k])
        matrix_sample = np.vstack(matrix_sample_parts)

        alpha = krippendorff.alpha(matrix_sample, level_of_measurement='interval')
        print("{}: ALPHA: {}".format(i + 1, alpha))
        alpha_distribution.append(alpha)

    return alpha_distribution


def bootstrap_unitizing(grid, length):
    max_n_units = 0
    for k in range(n_raters):
        n = len(grid[k])
        if n > max_n_units:
            max_n_units = n

    pairs = []  # generate all reliability value pairs
    coordinates = list(itertools.product(range(n_raters), range(max_n_units)))
    for c in coordinates:
        if c[1] >= len(grid[c[0]]):  # if coordinate falls outside of the sparse grid
            continue
        for k in coordinates:
            if c == k or c[0] == k[0] or k[1] >= len(grid[k[0]]):  # if coordinates are equal or coordinate falls outside of the sparse grid
                continue
            if grid[c[0]]['behavior'].iloc[c[1]] != grid[k[0]]['behavior'].iloc[k[1]]:  # only interested in pairs within a category
                continue
            pairs.append([c, k])

    alpha_distribution = []  # bootstrapping data to build alpha distribution
    for i in range(n_subsamples):
        all_coordinates = []  # sampling coordinates of pairs of values with replacement
        for j in range(len(pairs) // 1000):
            sample = choice(pairs)
            all_coordinates.append(sample[0])  # flattening pair coordinates to array of indices
            all_coordinates.append(sample[1])

        grid_sample = []  # create new grid
        for k in range(n_raters):
            new_df = pd.DataFrame()
            for c in all_coordinates:
                if c[0] != k:
                    continue
                new_df = new_df.append(grid[c[0]].iloc[c[1]])  # add sampled unit/gap to rater df
            grid_sample.append(new_df)  # add df to grid

        alpha = alpha_unitizing(grid_sample, length)
        print("{}: ALPHA: {}".format(i + 1, alpha))
        alpha_distribution.append(alpha)

    return alpha_distribution


def delta_unitizing(unit0, unit1):
    w0 = unit0['w']
    w1 = unit1['w']
    b0 = int(unit0['start'])
    b1 = int(unit1['start'])
    l0 = int(unit0['end']) - b0
    l1 = int(unit1['end']) - b1

    if w0 == 1 and w1 == 1 and -l0 < b0 - b1 < l1:  # if two units are non-equal and overlapping, i.e. b0 < b1 + l1 and b1 < b0 + l0
        return (b0 - b1)**2 + (b0 + l0 - (b1 + l1))**2
    elif w0 == 1 and w1 == 0 and l1 - l0 >= b0 - b1 >= 0:  # if unit is contained within gap
        return l0**2
    elif w0 == 0 and w1 == 1 and l1 - l0 <= b0 - b1 <= 0:  # if unit is contained within gap
        return l1**2
    return 0  # otherwise


def observed_disagreement(grid, category, length):
    summed_distances = 0
    for rater_df in grid:  # for each rater
        sub_df = rater_df[rater_df['behavior'] == category]
        for g in range(len(sub_df)):  # for each unit/gap by rater
            for rater_df2 in grid:  # for each rater not equal to first rater
                if rater_df2.equals(rater_df):
                    continue
                sub_df2 = rater_df2[rater_df2['behavior'] == category]
                for h in range(len(sub_df2)):  # for each unit/gap by second rater
                    summed_distances += delta_unitizing(sub_df.iloc[g], sub_df2.iloc[h])

    d_oc = summed_distances/(n_raters*(n_raters - 1)*(length**2))
    return d_oc


def expected_disagreement(grid, category, length):
    n_c = 0
    correction = 0
    numerator = 0
    for rater_df in grid:
        sub_df = rater_df[rater_df['behavior'] == category]
        n_c += len(sub_df[sub_df['w'] == 1])

    for rater_df in grid:
        sub_df = rater_df[rater_df['behavior'] == category]
        for g in range(len(sub_df)):
            w = sub_df['w'].iloc[g]
            l_ = sub_df['end'].iloc[g] - sub_df['start'].iloc[g]
            correction += w*l_*(l_ - 1)

            all_overlaps = ((n_c - 1)/3)*(2*l_**3 - 3*l_**2 + l_)
            gap_differences = 0
            for rater_df2 in grid:
                sub_df2 = rater_df2[rater_df2['behavior'] == category]
                for h in range(len(sub_df2)):
                    w2 = sub_df2['w'].iloc[h]
                    l_2 = sub_df2['end'].iloc[h] - sub_df2['start'].iloc[h]
                    if l_2 >= l_:
                        gap_differences += (1 - w2)*(l_2 - l_ + 1)
            gap_differences *= l_**2

            numerator += all_overlaps + gap_differences

    numerator = (2/length)*numerator
    denominator = n_raters*length*(n_raters*length - 1) - correction

    d_ec = numerator/denominator
    return d_ec


def alpha_unitizing(grid, length):
    behaviors = []  # grab all behaviors from all rater dataframes
    for d in grid:
        behaviors.extend(list(d['behavior'].unique()))
    behaviors = np.unique(behaviors)

    total_observed_disagreement = 0
    total_expected_disagreement = 0
    for b in behaviors:
        total_observed_disagreement += observed_disagreement(grid, b, length)
        total_expected_disagreement += expected_disagreement(grid, b, length)

    a = 1 - total_observed_disagreement/total_expected_disagreement
    return a


if __name__ == '__main__':
    alphas = {}
    for t in types:
        alphas[t] = {}
        print("Type:", t)
        rater_arrays = []
        if t == 'coding':
            for j in range(0, n_raters):
                df = pd.DataFrame()
                for path in paths[:-1]:
                    os.chdir(path)
                    for i in range(0, n_periods):
                        df_temp = pd.read_csv("raters\\time_step_scores_{}_{}.csv".format(i, j), header=None,
                                              names=['start', 'end', 'score', 'tag'], skip_blank_lines=False)

                        if i == 1:
                            if "CP01" in path:
                                df_temp = df_temp[df_temp['start'] <= 51933.23]  # last segment of period 1 removed, due to incomplete of reach- remove for future work
                            if "TD04" in path:
                                df_temp = df_temp[(df_temp['start'] >= 53035.63)
                                                  | (df_temp['start'] <= 52928.57)]  # gap in period 1 removed, due to incomplete coding of behaviors other than positive emotion- remove for future work

                        df = pd.concat((df, df_temp), ignore_index=True)

                rater_arrays.append(np.array(df['score']).T)

            matrix = np.vstack(rater_arrays)

            alpha_distribution = bootstrap_coding(matrix)
            alpha = krippendorff.alpha(matrix, level_of_measurement='interval')
            alphas[t]['alpha'] = alpha
            alphas[t]['distribution'] = alpha_distribution
            print(alpha)

        elif t == 'unitizing':
            total_length = 0
            previous_end = 0
            for j in range(0, n_raters):
                df = pd.DataFrame()
                for path in paths[:-1]:
                    os.chdir(path)

                    with open('global_settings.json') as inputfile:
                        settings = json.load(inputfile)
                    engagement_behaviors = settings['engagement_behaviors']
                    positive_behaviors = settings['positive_behaviors']
                    negative_behaviors = settings['negative_behaviors']
                    all_behaviors = engagement_behaviors + positive_behaviors + negative_behaviors

                    period = pd.read_csv("period_start_end_times.csv", header=None, names=['start', 'end'])
                    period['start'] *= 100  # converting to centiseconds
                    period['end'] *= 100

                    if j == 0:
                        total_length += np.sum(period['end'] - period['start'])  # accumulating total length of continuum

                    period['end'] -= period['start'].iloc[0]
                    period['end'] += previous_end + 1
                    period['start'] -= period['start'].iloc[0]  # adjusting relative times of sessions to form a concatenated "continuum"
                    period['start'] += previous_end + 1

                    previous_end = period['end'].iloc[-1]

                    for i in range(0, n_periods):
                        df_temp = pd.read_csv("raters\\video_labels_{}_{}.csv".format(i, j), header=None,
                                              names=['start', 'end', 'behavior', 'tag'], skip_blank_lines=False)

                        if i == 1:
                            if "CP01" in path:
                                df_temp = df_temp[((df_temp['start'] <= 51933.23) & (df_temp['behavior'] == 'reach'))
                                                  | (df_temp['behavior'] != 'reach')]  # last segment of period 1 removed, due to incomplete coding of reach- remove for future work
                            elif "TD04" in path:
                                df_temp = df_temp[(((df_temp['start'] >= 53035.63)
                                                  | (df_temp['start'] <= 52928.57)) & (df_temp['behavior'] == 'positive_emotion'))
                                                  | (df_temp['behavior'] != 'positive_emotion')]  # gap in period 1 removed, due to incomplete coding of behaviors other than positive_emotion- remove for future work

                        df_temp['start'] *= 100  # converting to centiseconds
                        df_temp['end'] *= 100

                        df_temp['start'] -= period['start'].iloc[i]
                        df_temp['start'] += previous_end + 1
                        df_temp['end'] -= period['start'].iloc[i]
                        df_temp['end'] += previous_end + 1

                        behaviors = np.intersect1d(np.unique(df_temp['behavior']), all_behaviors)

                        df_period = pd.DataFrame()
                        for behavior in behaviors:
                            df_behavior = df_temp[df_temp['behavior'] == behavior]
                            df_behavior = df_behavior.sort_values(by=['start'])
                            df_behavior['w'] = 1
                            if df_behavior.empty:
                                df_behavior = df_behavior.append({'start': period['start'].iloc[i],
                                                                  'end': period['end'].iloc[i],  # whole period is gap
                                                                  'behavior': behavior, 'tag': -1, 'w': 0},
                                                                 ignore_index=True)
                                continue

                            if df_behavior['start'].iloc[0] > period['start'].iloc[i]:
                                df_behavior = df_behavior.append({'start': period['start'].iloc[i],
                                                                  'end': df_behavior['start'].iloc[0],  # first gap in period
                                                                  'behavior': behavior, 'tag': -1, 'w': 0},
                                                                 ignore_index=True)
                            for k in range(len(df_behavior) - 1):
                                if df_behavior['end'].iloc[k] < df_behavior['start'].iloc[k+1]:
                                    df_behavior = df_behavior.append({'start': df_behavior['end'].iloc[k],  # gaps in between behavior units in period
                                                                      'end': df_behavior['start'].iloc[k+1],
                                                                      'behavior': behavior, 'tag': -1, 'w': 0}, ignore_index=True)
                            if period['end'].iloc[i] > df_behavior['end'].iloc[-1]:
                                df_behavior = df_behavior.append({'start': df_behavior['end'].iloc[-1],  # last gap in period
                                                                  'end': period['end'].iloc[i],
                                                                  'behavior': behavior, 'tag': -1, 'w': 0}, ignore_index=True)

                            df_period = pd.concat((df_period, df_behavior), ignore_index=True)

                        df = pd.concat((df, df_period), ignore_index=True)

                rater_arrays.append(df)  # sparse grid in the form of a list of df's of unequal lengths,
                                         # one df per rater consisting of units/gaps for all subjects across all periods

            alpha_distribution = bootstrap_unitizing(rater_arrays, total_length)
            alpha = alpha_unitizing(rater_arrays, total_length)
            alphas[t]['alpha'] = alpha
            alphas[t]['distribution'] = alpha_distribution
            print(alpha)

    os.chdir(paths[-1])
    for t in types:
        ci, q = plot_distribution('Reliability Data Type: {}'.format(t),
                                  'alpha_distribution_{}_{}'.format(t, time.time()), alphas[t])
        alphas[t]['ci'] = ci
        alphas[t]['q'] = q

        distribution_df = pd.DataFrame(alphas[t]['distribution'])
        distribution_df.to_csv('inter-rater reliability\\distribution_{}_{}.csv'.format(t, time.time()))

    with open('inter-rater reliability\\krippendorffs_alpha_{}.csv'.format(time.time()), 'w') as f:
        f.write("%s,%s,%s,%s,%s\n" % ('type', 'alpha', '95% CI low', '95% CI high', 'q'))
        for key in alphas.keys():
            f.write("%s,%s,%s,%s,%s\n" % (key, alphas[key]['alpha'], alphas[key]['ci'][0], alphas[key]['ci'][1], alphas[key]['q']))

    #################################################################

    # os.chdir(paths[-1])
    #
    # pd.set_option('precision', 16)
    # df = pd.read_csv('inter-rater reliability\\distribution_unitizing_1590730960.7846453.csv')
    #
    # alpha_distribution = list(df['0'])
    #
    # print(df['0'])
    #
    # print(np.size(np.where(df['0'] < 0)))
    #
    # alphas = {}
    # for t in types:
    #     alphas = {t: {'distribution': alpha_distribution}}
    #     alphas[t]['alpha'] = 0.999998949285053
    #     ci, q = plot_distribution('Reliability Data Type: {}'.format(t),
    #                               'alpha_distribution_{}_{}'.format(t, time.time()), alphas[t])
    #
    #     alphas[t]['ci'] = ci
    #     alphas[t]['q'] = q
    #
    # with open('inter-rater reliability\\krippendorffs_alpha_{}.csv'.format(time.time()), 'w') as f:
    #     f.write("%s,%s,%s,%s,%s\n" % ('type', 'alpha', '95% CI low', '95% CI high', 'q'))
    #     for key in alphas.keys():
    #         f.write("%s,%s,%s,%s,%s\n" % (key, alphas[key]['alpha'], alphas[key]['ci'][0], alphas[key]['ci'][1], alphas[key]['q']))
