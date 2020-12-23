After event_timestamps and obs_timestamps (the master record documents) have been completed, in order:

For each subject:
0: Set the given subject's path in path.txt and add the subject's path to all_paths.txt.  Set global_settings.json in this path.
   Ensure the frontal video (mp4) and the OpenBCI data file are in this path.
1: label_and_count.py
2: produce_labeled_images.py
3: train_emotion_network.py and train_eeg_classifier.py
4: reward_function.py

For all subjects:
5: analyze_and_plot.py

For a selected subset of subjects, as specified in test_paths.txt:
6: inter-rater_reliability.py

Note: Coding event_timestamps properly is critical.  If a testing region ends within a period, it MUST end at the end of a time step,
otherwise the next time step will contain testing and non-testing samples, causing the time_step_scores tag to be
non-testing, and the reward_scores tag to be testing; in contrast, if it ends at the end of a period, it does
not need to end at the end of a time step, as there is no following time step for the rest of the period.