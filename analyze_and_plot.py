import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.power import FTestAnovaPower
from sklearn.linear_model import LinearRegression

significance_level = 0.05
desired_power = 0.8

pathfile = os.getcwd() + "\\all_paths.txt"
f = open(pathfile, "r")
lines = f.read().splitlines()
f.close()

paths = []
for line in lines:
    paths.append(line)

n_unique_periods = 0

subjects = []  # grabbing subject ID's
for path in paths:
    subjects.append(path[-4:])


def grubbs_test(data):
    a = 0.05
    n = np.size(data)
    if n >= 4:
        t = stats.t.ppf(1 - (a / (2*n)), n - 2)

        centered_data = np.abs(data - np.mean(data))
        max_index = np.argmax(centered_data)
        outlier = data[max_index]

        g = centered_data[max_index]/np.std(data)
        threshold = ((n-1)/np.sqrt(n))*np.sqrt((t**2)/(n - 2 + t**2))

        if g > threshold:
            return True, outlier

    return False, None


def grubbs_period_analysis(period):
    remove_indices = {}
    for p in period.keys():
        remove_indices[p] = {}
        for subcomponent in subcomponents:
            remove_indices[p][subcomponent] = []
            find_outliers = True
            while find_outliers:
                keep_indices = list(set(range(len(period[p]['clinical']))) - set(remove_indices[p][subcomponent]))
                data = [period[p][subcomponent]['scores'][m] for m in keep_indices]
                data_ = [period[p]['clinical'][m] for m in keep_indices]
                outlier_detected, outlier = grubbs_test(data)  # checking for reward component outliers
                outlier_detected_, outlier_ = grubbs_test(data_)  # checking for clinical score outliers
                if outlier_detected or outlier_detected_:
                    if outlier_detected:
                        # outlier_indices = period[p][subcomponent]['scores'].index(outlier)
                        outlier_indices = [ind for ind, x in enumerate(period[p][subcomponent]['scores']) if x == outlier]  # dealing with potentially multiple equal outliers
                        for outlier_index in outlier_indices:
                            remove_indices[p][subcomponent].append(outlier_index)
                    if outlier_detected_:
                        # outlier_index = period[p]['clinical'].index(outlier_)
                        outlier_indices = [ind for ind, x in enumerate(period[p]['clinical']) if x == outlier_]  # dealing with potentially multiple equal outliers
                        for outlier_index in outlier_indices:
                            remove_indices[p][subcomponent].append(outlier_index)
                else:
                    find_outliers = False

            keep_indices = list(set(range(len(period[p]['clinical']))) - set(remove_indices[p][subcomponent]))
            period[p][subcomponent]['scores'] = [period[p][subcomponent]['scores'][m] for m in keep_indices]

        for subcomponent in subcomponents:
            keep_indices = list(set(range(len(period[p]['clinical']))) - set(remove_indices[p][subcomponent]))
            clinical_scores = np.array(period[p]['clinical'])[keep_indices].T
            reward_scores = np.array(period[p][subcomponent]['scores']).T
            if np.size(reward_scores) >= 3:
                period[p][subcomponent]['correlation'], period[p][subcomponent]['p_value'] = stats.pearsonr(
                    clinical_scores, reward_scores)
            else:
                period[p][subcomponent]['correlation'], period[p][subcomponent]['p_value'] = (np.nan, np.nan)

    return period, remove_indices


def grubbs_analysis(rewards, clinicals):
    remove_indices = {}
    for subcomponent in subcomponents:
        remove_indices[subcomponent] = []
        find_outliers = True
        while find_outliers:
            keep_indices = list(set(range(len(clinicals))) - set(remove_indices[subcomponent]))
            data = [rewards[subcomponent]['scores'][m] for m in keep_indices]
            data_ = [clinicals[m] for m in keep_indices]
            outlier_detected, outlier = grubbs_test(data)  # checking for reward component outliers
            outlier_detected_, outlier_ = grubbs_test(data_)  # checking for clinical score outliers
            if outlier_detected or outlier_detected_:
                if outlier_detected:
                    # outlier_index = rewards[subcomponent]['scores'].index(outlier)
                    outlier_indices = [ind for ind, x in enumerate(rewards[subcomponent]['scores']) if x == outlier]  # dealing with potentially multiple equal outliers
                    for outlier_index in outlier_indices:
                        remove_indices[subcomponent].append(outlier_index)
                if outlier_detected_:
                    # outlier_index = clinicals.index(outlier_)
                    outlier_indices = [ind for ind, x in enumerate(clinicals) if x == outlier_]  # dealing with potentially multiple equal outliers
                    for outlier_index in outlier_indices:
                        remove_indices[subcomponent].append(outlier_index)
            else:
                find_outliers = False

        keep_indices = list(set(range(len(clinicals))) - set(remove_indices[subcomponent]))
        rewards[subcomponent]['scores'] = [rewards[subcomponent]['scores'][m] for m in keep_indices]

    for subcomponent in subcomponents:
        keep_indices = list(set(range(len(clinicals))) - set(remove_indices[subcomponent]))
        clinical_scores = np.array(clinicals)[keep_indices].T
        reward_scores = np.array(rewards[subcomponent]['scores']).T
        rewards[subcomponent]['correlation'], rewards[subcomponent][
            'p_value'] = stats.pearsonr(clinical_scores,
                                        reward_scores)

    return rewards, remove_indices


def plot_period_correlations(plot_title, save_name, period, remove_indices=None):
    for subcomponent in subcomponents:
        clfs = []
        clinicals = {}
        for p in period.keys():
            if remove_indices:
                keep_indices = list(set(range(len(period[p]['clinical']))) - set(remove_indices[p][subcomponent]))
            else:
                keep_indices = list(range(len(period[p]['clinical'])))
            clinicals[p] = [period[p]['clinical'][m] for m in keep_indices]

            clf = LinearRegression().fit(np.array(clinicals[p]).reshape(-1, 1),
                                         np.array(period[p][subcomponent]['scores']).reshape(-1, 1))
            clfs.append(clf)

        figure_size = (14, 8)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figure_size)
        fig.canvas.set_window_title(plot_title + " {}".format(subcomponent))
        fig.suptitle(plot_title + " {}".format(subcomponent))
        ax1.scatter(clinicals['0'], period['0'][subcomponent]['scores'])
        r = np.linspace(np.min(clinicals['0']), np.max(clinicals['0']))
        ax1.plot(r, clfs[0].predict(r.reshape(-1, 1)))
        ax1.set_xlabel('Clinical Score')
        ax1.set_ylabel('Reward Score')
        ax1.set_title('Resting Before Correlation: {0:.3f}, p = {1:.3f}'.format(period['0'][subcomponent]['correlation'], period['0'][subcomponent]['p_value']))

        ax2.scatter(clinicals['1'], period['1'][subcomponent]['scores'])
        r = np.linspace(np.min(clinicals['1']), np.max(clinicals['1']))
        ax2.plot(r, clfs[1].predict(r.reshape(-1, 1)))
        ax2.set_xlabel('Clinical Score')
        # ax2.set_ylabel('Reward Score')
        ax2.set_title('Intervention Correlation: {0:.3f}, p = {1:.3f}'.format(period['1'][subcomponent]['correlation'], period['1'][subcomponent]['p_value']))

        ax3.scatter(clinicals['2'], period['2'][subcomponent]['scores'])
        r = np.linspace(np.min(clinicals['2']), np.max(clinicals['2']))
        ax3.plot(r, clfs[2].predict(r.reshape(-1, 1)))
        ax3.set_xlabel('Clinical Score')
        # ax3.set_ylabel('Reward Score')
        ax3.set_title('Resting After Correlation: {0:.3f}, p = {1:.3f}'.format(period['2'][subcomponent]['correlation'], period['2'][subcomponent]['p_value']))

        plt.savefig(save_name + "_{}.png".format(subcomponent), dpi=fig.dpi, figsize=figure_size)


def plot_correlation(plot_title, save_name, all_clinicals, all_rewards, remove_indices=None):
    for subcomponent in subcomponents:

        if remove_indices:
            keep_indices = list(set(range(len(all_clinicals))) - set(remove_indices[subcomponent]))
        else:
            keep_indices = list(range(len(all_clinicals)))
        clinicals = [all_clinicals[m] for m in keep_indices]

        clf = LinearRegression().fit(np.array(clinicals).reshape(-1, 1),
                                     np.array(all_rewards[subcomponent]['scores']).reshape(-1, 1))

        figure_size = (14, 8)
        fig, ax1 = plt.subplots(1, 1, figsize=figure_size)
        fig.canvas.set_window_title(plot_title + " {}".format(subcomponent))
        fig.suptitle(plot_title + " {}".format(subcomponent))
        ax1.scatter(clinicals, all_rewards[subcomponent]['scores'])
        r = np.linspace(np.min(clinicals), np.max(clinicals))
        ax1.plot(r, clf.predict(r.reshape(-1, 1)))
        ax1.set_xlabel('Clinical Score')
        ax1.set_ylabel('Reward Score')
        ax1.set_title('All Periods Correlation: {0:.3f}, p = {1:.3f}'.format(all_rewards[subcomponent]['correlation'],
                                                                             all_rewards[subcomponent]['p_value']))

        plt.savefig(save_name + "_{}.png".format(subcomponent), dpi=fig.dpi, figsize=figure_size)


subcomponents = ['reward', 'emotion_reward', 'eeg_reward']

total_engagement = []   # Lists to be set up for repeated measures ANOVA
overall_reward = []
subject_labels = []
period_labels = []

all_reward_scores = {}
for subcomponent in subcomponents:
    all_reward_scores[subcomponent] = {'0': [], '1': [], '2': []}
all_clinical_scores = {'0': [], '1': [], '2': []}

if __name__ == '__main__':
    for k, path in enumerate(paths[:-1]):
        os.chdir(path)

        total_reward_df = pd.read_csv('total_reward.csv', index_col=0)

        with open('global_settings.json') as inputfile:
            settings = json.load(inputfile)

        behavior_of_interest = settings['behavior_of_interest']

        f = "period_start_end_times.csv"
        periods = pd.read_csv(f, header=None, names=['start', 'end'])

        if len(periods) > n_unique_periods:
            n_unique_periods = len(periods)

        period = {}
        for i in range(0, len(periods)):
            f = "engagement_scores_{}.csv".format(i)    # Read in overall engagement scores for calculation of repeated measures ANOVA
            engagement_scores = pd.read_csv(f, header=None, names=['behavior', 'fraction'])

            behaviors = engagement_scores['behavior'].unique().tolist()
            index = behaviors.index(behavior_of_interest)
            total_engagement.append(engagement_scores.loc[index, 'fraction'])  # Adding subject's total engagement score for period
            subject_labels.append(k)
            period_labels.append(i)

            f = "time_step_scores_{}.csv".format(i)     # Read in time step scores and reward scores to calculate correlation for each period of each subject
            time_step_scores = pd.read_csv(f, header=None, names=['start', 'end', 'score', 'tag'])
            time_step_scores = time_step_scores[time_step_scores['tag'] == 2]   # Pull out only scores from the testing regions

            f = "reward_scores_{}.csv".format(i)
            reward_scores = pd.read_csv(f, header=None, names=['start', 'end', 'reward', 'emotion_reward', 'eeg_reward'])

            time_steps = {'clinical': []}
            for subcomponent in subcomponents:
                time_steps[subcomponent] = {'scores': []}
            for j in range(0, len(time_step_scores)):
                time_steps['clinical'].append(time_step_scores['score'].iloc[j])
                for subcomponent in subcomponents:
                    time_steps[subcomponent]['scores'].append(reward_scores[subcomponent].iloc[j])
            all_clinical_scores[str(i)].extend(time_steps['clinical'])

            clinical_scores = np.array(time_steps['clinical']).T
            reward_scores = {}
            for subcomponent in subcomponents:
                all_reward_scores[subcomponent][str(i)].extend(time_steps[subcomponent]['scores'])
                reward_scores = np.array(time_steps[subcomponent]['scores']).T

                # stack = np.vstack((clinical_scores, reward_scores))
                # time_steps['correlation'] = np.corrcoef(stack)[0, 1]  # off-diagonal element for correlation between two variables
                # r = time_steps['correlation']
                # F = (1/2)*np.log((1+r)/(1-r))  # Fisher transformation to determine significance (null = no correlation)
                # n = len(time_step_scores)
                # z_score = F*np.sqrt(n-3)
                # time_steps['p_value'] = stats.norm.sf(abs(z_score))*2

                if np.size(reward_scores) >= 3:
                    time_steps[subcomponent]['correlation'], time_steps[subcomponent]['p_value'] = stats.pearsonr(clinical_scores, reward_scores)
                else:
                    time_steps[subcomponent]['correlation'], time_steps[subcomponent]['p_value'] = (np.nan, np.nan)

            period[str(i)] = time_steps

            # Calculate overall reward for given period (analogous to total_engagement, the clinical measure for the whole period)
            n_happy = total_reward_df[str(i)].loc['n_happy']
            n_sad = total_reward_df[str(i)].loc['n_sad']
            n_emotion = total_reward_df[str(i)].loc['n_emotion']
            n_engaged = total_reward_df[str(i)].loc['n_engaged']
            n_eeg = total_reward_df[str(i)].loc['n_eeg']

            if n_emotion > 0:
                emotion_reward = (n_happy - n_sad)/n_emotion
            else:
                emotion_reward = 0

            if n_eeg > 0:
                eeg_reward = n_engaged/n_eeg
            else:
                eeg_reward = 0

            overall_reward_period = emotion_reward + eeg_reward
            overall_reward.append(overall_reward_period)

        all_subject_clinicals = []
        all_subject_rewards = {}
        for subcomponent in subcomponents:
            all_subject_rewards[subcomponent] = {'scores': []}
        for p in period.keys():
            all_subject_clinicals.extend(period[p]['clinical'])
            for subcomponent in subcomponents:
                all_subject_rewards[subcomponent]['scores'].extend(period[p][subcomponent]['scores'])

        clinical_scores = np.array(all_subject_clinicals).T
        for subcomponent in subcomponents:
            reward_scores = np.array(all_subject_rewards[subcomponent]['scores']).T
            all_subject_rewards[subcomponent]['correlation'], all_subject_rewards[subcomponent]['p_value'] = stats.pearsonr(clinical_scores,
                                                                                                            reward_scores)

        # plot_period_correlations('{}'.format(subjects[k]), 'correlation_plots_{}_{}'.format(subjects[k], time.time()),
        #                          period)
        # plot_correlation('{}, All Periods'.format(subjects[k], k),
        #                  'correlation_plots_{}_all_periods_{}'.format(subjects[k], time.time()),
        #                  all_subject_clinicals, all_subject_rewards)
        #
        # period, remove_indices = grubbs_period_analysis(period)  # repeating correlation analysis after performing Grubbs's test for outliers
        # plot_period_correlations('{}, Outliers Removed'.format(subjects[k]), 'correlation_plots_grubbs_{}_{}'.format(subjects[k], time.time()),
        #                          period, remove_indices)
        #
        # all_subject_rewards, remove_indices = grubbs_analysis(all_subject_rewards, all_subject_clinicals)
        # plot_correlation('{}, All Periods, Outliers Removed'.format(subjects[k], k),
        #                  'correlation_plots_grubbs_{}_all_periods_{}'.format(subjects[k], time.time()),
        #                  all_subject_clinicals, all_subject_rewards, remove_indices)

    period = {}
    for i in range(0, n_unique_periods):
        clinical_scores = np.array(all_clinical_scores[str(i)]).T
        time_steps = {'clinical': all_clinical_scores[str(i)]}
        for subcomponent in subcomponents:
            time_steps[subcomponent] = {}
            time_steps[subcomponent]['scores'] = all_reward_scores[subcomponent][str(i)]
            reward_scores = np.array(all_reward_scores[subcomponent][str(i)]).T
            if np.size(reward_scores) >= 3:
                time_steps[subcomponent]['correlation'], time_steps[subcomponent]['p_value'] = stats.pearsonr(clinical_scores, reward_scores)
            else:
                time_steps[subcomponent]['correlation'], time_steps[subcomponent]['p_value'] = (np.nan, np.nan)

        period[str(i)] = time_steps

    os.chdir(paths[-1])

    plot_period_correlations('All Subjects', 'correlation_plots_all_subjects_{}'.format(time.time()), period)

    all_clinicals = all_clinical_scores['0'].copy()
    all_clinicals.extend(all_clinical_scores['1'])
    all_clinicals.extend(all_clinical_scores['2'])
    clinical_scores = np.array(all_clinicals).T

    all_rewards = {}
    for subcomponent in subcomponents:
        all_rewards[subcomponent] = {'scores': []}
        all_rewards[subcomponent]['scores'] = all_reward_scores[subcomponent]['0'].copy()
        all_rewards[subcomponent]['scores'].extend(all_reward_scores[subcomponent]['1'])
        all_rewards[subcomponent]['scores'].extend(all_reward_scores[subcomponent]['2'])

        reward_scores = np.array(all_rewards[subcomponent]['scores']).T
        all_rewards[subcomponent]['correlation'], all_rewards[subcomponent]['p_value'] = stats.pearsonr(clinical_scores, reward_scores)

    plot_correlation('All Subjects, All Periods',
                     'correlation_plots_all_subjects_all_periods_{}'.format(time.time()),
                     all_clinicals, all_rewards)

    period, remove_indices = grubbs_period_analysis(period)  # repeating correlation analysis after performing Grubbs's test for outliers
    plot_period_correlations('All Subjects, Outliers Removed', 'correlation_plots_grubbs_all_subjects_{}'.format(time.time()), period, remove_indices)

    all_rewards, remove_indices = grubbs_analysis(all_rewards, all_clinicals)
    plot_correlation('All Subjects, All Periods, Outliers Removed',
                     'correlation_plots_grubbs_all_subjects_all_periods_{}'.format(time.time()),
                     all_clinicals, all_rewards, remove_indices)

    score_types = [total_engagement, overall_reward]  # repeated measures ANOVA using both types of measures
    type_strings = ['clinical', 'reward']
    final_df = pd.DataFrame()
    for i, scores in enumerate(score_types):
        columns = ['score', 'subject', 'period']
        data_dict = {columns[0]: scores, columns[1]: subject_labels, columns[2]: period_labels}
        engagement_df = pd.DataFrame(data_dict, columns=columns)

        n = np.size(np.unique(engagement_df['subject']))
        total_mean = np.mean(engagement_df['score'])

        ss_between = 0
        for period in np.unique(engagement_df['period']):
            mean = np.mean(engagement_df['score'][engagement_df['period'] == period])
            ss_between += (mean - total_mean)**2

        ss_total = 0
        for j in range(len(engagement_df)):
            ss_total += (engagement_df['score'].iloc[j] - total_mean)**2

        eta_squared = ss_between/ss_total   # effect size

        aovrm = AnovaRM(engagement_df, 'score', 'subject', within=['period'])
        results = aovrm.fit()
        f_value = results.anova_table['F Value'].iloc[0]
        p_value = results.anova_table['Pr > F'].iloc[0]

        ftestpower = FTestAnovaPower()

        power = ftestpower.power(effect_size=eta_squared, nobs=n, alpha=significance_level, k_groups=n_unique_periods)
        N = ftestpower.solve_power(effect_size=eta_squared, nobs=None, alpha=significance_level, power=desired_power, k_groups=n_unique_periods)

        # f_value, p_value = stats.f_oneway(total_engagement['0'], total_engagement['1'], total_engagement['2'])
        print("Effect size: {}".format(eta_squared))
        print("F value: {}, p value: {}".format(f_value, p_value))
        print("Power: {}, Required N: {}".format(power, N))

        t_01 = None
        p_01 = None
        t_02 = None
        p_02 = None
        t_12 = None
        p_12 = None
        if p_value < significance_level:
            indices0 = np.where(np.array(period_labels) == 0)[0]
            indices1 = np.where(np.array(period_labels) == 1)[0]
            indices2 = np.where(np.array(period_labels) == 2)[0]
            t_01, p_01 = stats.ttest_rel(np.array(total_engagement)[indices0], np.array(total_engagement)[indices1])
            t_02, p_02 = stats.ttest_rel(np.array(total_engagement)[indices0], np.array(total_engagement)[indices2])
            t_12, p_12 = stats.ttest_rel(np.array(total_engagement)[indices1], np.array(total_engagement)[indices2])
        else:
            print("No significant differences.")

        test_dict = {type_strings[i]: {'eta_squared': eta_squared, 'F_value': f_value, 'p_value': p_value, 'power': power, 'required_n': N,
                                       't_01': t_01, 'p_01': p_01, 't_02': t_02,
                                       'p_02': p_02, 't_12': t_12, 'p_12': p_12}}
        df = pd.DataFrame(test_dict, index=['eta_squared', 'F_value', 'p_value', 'power', 'required_n', 't_01', 'p_01',
                                            't_02', 'p_02', 't_12', 'p_12'])
        final_df = pd.concat((final_df, df), axis=1)

    final_df.to_csv('Repeated Measures ANOVA {}.csv'.format(time.time()), na_rep='NaN')
    print(final_df)
